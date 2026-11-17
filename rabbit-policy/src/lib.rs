// SPDX-License-Identifier: GPL-2.0-only
//! The capability policy of a program: which files it may open and which
//! servers it may connect to.
//!
//! A [`Policy`] is plain data with a fixed capacity, so it needs no allocator
//! and the same code decides in user space and in the kernel. The decision is
//! a pure function of the policy and the request: [`Policy::check_open`] and
//! [`Policy::check_connect`] touch no hardware and no file system, which is
//! what lets the host unit tests below cover the enforcement logic.
//!
//! The rule is deny by default. An empty policy refuses everything, a request
//! needs the capability that names it and a rule that lists it, and a path is
//! never resolved: [`normalize`] refuses any `..` component instead of
//! computing where it would lead, because a policy that resolves a path can be
//! made to resolve it differently from the file system.
//!
//! [`Policy::encode`] and [`Policy::decode`] are the wire format, so the
//! policy of a program can cross from the agent to the kernel. The decoder
//! validates everything and trusts nothing: the bytes come from a program that
//! the policy is meant to restrain.

#![cfg_attr(not(test), no_std)]
#![deny(unsafe_op_in_unsafe_fn)]

/// Longest path [`normalize`] accepts, in bytes.
pub const MAX_PATH_LEN: usize = 255;

/// Size of the buffer [`normalize`] writes into: a path and room to spare.
pub const PATH_BUFFER_LEN: usize = 256;

/// Longest root a rule can hold, in bytes.
pub const MAX_ROOT_LEN: usize = 64;

/// Roots a policy holds for reading, and as many for writing.
pub const MAX_ROOTS: usize = 4;

/// Endpoints a policy lists.
pub const MAX_ENDPOINTS: usize = 4;

/// Magic bytes at the start of an encoded policy.
const MAGIC: [u8; 4] = *b"RBPL";

/// Version of the wire format.
const VERSION: u8 = 1;

/// Bytes of one encoded endpoint: protocol, address and port.
const ENDPOINT_WIRE_LEN: usize = 1 + 4 + 2;

/// Size of the largest encoded policy, so a caller can size its buffer.
///
/// The header is the magic, the version and the capability byte. Each of the
/// two root lists is a count and then a length byte and the bytes of every
/// root, and the endpoint list is a count and then the endpoints.
pub const MAX_ENCODED_LEN: usize = MAGIC.len()
    + 2
    + 2 * (1 + MAX_ROOTS * (1 + MAX_ROOT_LEN))
    + 1
    + MAX_ENDPOINTS * ENDPOINT_WIRE_LEN;

/// Why a request or a policy was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    /// The policy does not grant the capability the request needs.
    NotGranted,
    /// The path is under none of the roots the policy lists.
    OutsideRoots,
    /// The path is not absolute, is too long or has a NUL byte or a `..`.
    BadPath,
    /// No endpoint of the policy matches the protocol, address and port.
    EndpointNotListed,
    /// The bytes or the values are not what the format allows.
    Malformed,
    /// A fixed capacity was exceeded: a list, or the buffer of an encoding.
    Capacity,
}

/// The decision about one request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// The request is inside the policy.
    Allow,
    /// The request is refused, for this reason.
    Deny(Reason),
}

/// One thing a program may be granted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Capability {
    /// Open files for reading.
    FsRead,
    /// Open files for writing.
    FsWrite,
    /// Send datagrams of UDP.
    NetUdp,
    /// Open connections of TCP.
    NetTcp,
}

impl Capability {
    const fn bit(self) -> u8 {
        match self {
            Self::FsRead => 1,
            Self::FsWrite => 2,
            Self::NetUdp => 4,
            Self::NetTcp => 8,
        }
    }
}

/// The set of capabilities a policy grants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Capabilities(u8);

impl Capabilities {
    /// The set that grants nothing.
    pub const EMPTY: Self = Self(0);

    /// Every bit that names a capability.
    const KNOWN: u8 = 0b1111;

    /// Whether `capability` is in the set.
    pub const fn contains(self, capability: Capability) -> bool {
        self.0 & capability.bit() != 0
    }

    /// Add `capability` to the set.
    pub const fn insert(&mut self, capability: Capability) {
        self.0 |= capability.bit();
    }

    /// The set as the byte of the wire format.
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// The set of a wire byte, or `None` when a bit names no capability.
    pub const fn from_bits(bits: u8) -> Option<Self> {
        if bits & !Self::KNOWN == 0 {
            Some(Self(bits))
        } else {
            None
        }
    }
}

/// Reduce an absolute path to its canonical form in `out`.
///
/// Repeated slashes and `.` components collapse and a trailing slash goes,
/// except on the root itself. The path is refused, not resolved, when it is
/// not absolute, is longer than [`MAX_PATH_LEN`], has a NUL byte or has a `..`
/// component: a name that only means something once `..` is resolved is a name
/// the policy does not need to understand. Returns the length of the result.
pub fn normalize(path: &[u8], out: &mut [u8; PATH_BUFFER_LEN]) -> Result<usize, Reason> {
    if path.first() != Some(&b'/') || path.len() > MAX_PATH_LEN || path.contains(&0) {
        return Err(Reason::BadPath);
    }
    let mut len = 0;
    for component in path.split(|&byte| byte == b'/') {
        match component {
            b"" | b"." => {}
            b".." => return Err(Reason::BadPath),
            name => {
                // Every component after the first, which is the empty one
                // before the leading slash, follows a slash of the input, so
                // the result is never longer than the path and fits.
                out[len] = b'/';
                out[len + 1..len + 1 + name.len()].copy_from_slice(name);
                len += 1 + name.len();
            }
        }
    }
    if len == 0 {
        out[0] = b'/';
        len = 1;
    }
    Ok(len)
}

/// An absolute path that a request may be under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PathRule {
    len: u8,
    bytes: [u8; MAX_ROOT_LEN],
}

impl PathRule {
    const EMPTY: Self = Self {
        len: 0,
        bytes: [0; MAX_ROOT_LEN],
    };

    /// The rule for `path`, which must already be in the form [`normalize`]
    /// produces and fit in [`MAX_ROOT_LEN`] bytes.
    ///
    /// A root is never rewritten: the person who wrote `/data/../etc` meant
    /// something, and silently turning it into another root would grant a
    /// different tree than the one on the page.
    pub fn new(path: &[u8]) -> Result<Self, Reason> {
        if path.len() > MAX_ROOT_LEN {
            return Err(Reason::BadPath);
        }
        let mut buffer = [0; PATH_BUFFER_LEN];
        let len = normalize(path, &mut buffer)?;
        if buffer[..len] != *path {
            return Err(Reason::BadPath);
        }
        let mut bytes = [0; MAX_ROOT_LEN];
        bytes[..len].copy_from_slice(path);
        let len = u8::try_from(len).map_err(|_| Reason::BadPath)?;
        Ok(Self { len, bytes })
    }

    /// The bytes of the root.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..usize::from(self.len)]
    }

    /// Whether the normalized `path` is the root or lies below it.
    ///
    /// The match is on a whole component: `/data` covers `/data/x` and does
    /// not cover `/database`.
    fn covers(&self, path: &[u8]) -> bool {
        let root = self.as_bytes();
        if root == b"/" {
            return true;
        }
        path.starts_with(root) && matches!(path.get(root.len()), None | Some(b'/'))
    }
}

/// The transport of an endpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Protocol {
    /// Datagrams.
    Udp,
    /// Connections.
    Tcp,
}

impl Protocol {
    const fn wire(self) -> u8 {
        match self {
            Self::Udp => 0,
            Self::Tcp => 1,
        }
    }

    const fn from_wire(byte: u8) -> Option<Self> {
        match byte {
            0 => Some(Self::Udp),
            1 => Some(Self::Tcp),
            _ => None,
        }
    }

    /// The capability a program needs to use this transport.
    pub const fn capability(self) -> Capability {
        match self {
            Self::Udp => Capability::NetUdp,
            Self::Tcp => Capability::NetTcp,
        }
    }
}

/// A server a program may talk to: the whole triple has to match.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Endpoint {
    /// Transport of the server.
    pub protocol: Protocol,
    /// IPv4 address of the server.
    pub ip: [u8; 4],
    /// Port of the server, never zero.
    pub port: u16,
}

impl Endpoint {
    const EMPTY: Self = Self {
        protocol: Protocol::Udp,
        ip: [0; 4],
        port: 0,
    };
}

/// A list of roots with a fixed capacity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Roots {
    rules: [PathRule; MAX_ROOTS],
    len: u8,
}

impl Roots {
    const EMPTY: Self = Self {
        rules: [PathRule::EMPTY; MAX_ROOTS],
        len: 0,
    };

    fn as_slice(&self) -> &[PathRule] {
        &self.rules[..usize::from(self.len)]
    }

    fn push(&mut self, path: &[u8]) -> Result<(), Reason> {
        let rule = PathRule::new(path)?;
        let slot = self
            .rules
            .get_mut(usize::from(self.len))
            .ok_or(Reason::Capacity)?;
        *slot = rule;
        self.len += 1;
        Ok(())
    }

    fn covers(&self, path: &[u8]) -> bool {
        self.as_slice().iter().any(|rule| rule.covers(path))
    }
}

/// What a program may do: the capabilities it holds and the rules that narrow
/// each of them.
///
/// A policy starts empty and grants nothing. The methods that add a rule
/// return an error when the rule is malformed or the list is full, and leave
/// the policy as it was: a rule that does not fit is never cut to fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Policy {
    granted: Capabilities,
    read: Roots,
    write: Roots,
    endpoints: [Endpoint; MAX_ENDPOINTS],
    endpoint_count: u8,
}

impl Policy {
    /// A policy that grants `granted` and lists no rule yet.
    pub const fn new(granted: Capabilities) -> Self {
        Self {
            granted,
            read: Roots::EMPTY,
            write: Roots::EMPTY,
            endpoints: [Endpoint::EMPTY; MAX_ENDPOINTS],
            endpoint_count: 0,
        }
    }

    /// The capabilities the policy grants.
    pub const fn granted(&self) -> Capabilities {
        self.granted
    }

    /// Allow reads under `path`, a normalized absolute path.
    pub fn add_read_root(&mut self, path: &[u8]) -> Result<(), Reason> {
        self.read.push(path)
    }

    /// Allow writes under `path`, a normalized absolute path.
    pub fn add_write_root(&mut self, path: &[u8]) -> Result<(), Reason> {
        self.write.push(path)
    }

    /// Allow the server `endpoint`. Port zero names no server and is refused.
    pub fn add_endpoint(&mut self, endpoint: Endpoint) -> Result<(), Reason> {
        if endpoint.port == 0 {
            return Err(Reason::Malformed);
        }
        let slot = self
            .endpoints
            .get_mut(usize::from(self.endpoint_count))
            .ok_or(Reason::Capacity)?;
        *slot = endpoint;
        self.endpoint_count += 1;
        Ok(())
    }

    fn endpoints(&self) -> &[Endpoint] {
        &self.endpoints[..usize::from(self.endpoint_count)]
    }

    /// Decide whether the program may open `path`, for writing when `write`.
    ///
    /// A read needs [`Capability::FsRead`] and a read root, a write needs
    /// [`Capability::FsWrite`] and a write root; one does not imply the other.
    pub fn check_open(&self, path: &[u8], write: bool) -> Verdict {
        let (capability, roots) = if write {
            (Capability::FsWrite, &self.write)
        } else {
            (Capability::FsRead, &self.read)
        };
        if !self.granted.contains(capability) {
            return Verdict::Deny(Reason::NotGranted);
        }
        let mut buffer = [0; PATH_BUFFER_LEN];
        let len = match normalize(path, &mut buffer) {
            Ok(len) => len,
            Err(reason) => return Verdict::Deny(reason),
        };
        if roots.covers(&buffer[..len]) {
            Verdict::Allow
        } else {
            Verdict::Deny(Reason::OutsideRoots)
        }
    }

    /// Decide whether the program may reach `ip` on `port` over `protocol`.
    ///
    /// The protocol needs its capability and the endpoint has to be listed
    /// exactly: the same protocol, address and port.
    pub fn check_connect(&self, protocol: Protocol, ip: [u8; 4], port: u16) -> Verdict {
        if !self.granted.contains(protocol.capability()) {
            return Verdict::Deny(Reason::NotGranted);
        }
        if port == 0 {
            return Verdict::Deny(Reason::Malformed);
        }
        let wanted = Endpoint { protocol, ip, port };
        if self.endpoints().contains(&wanted) {
            Verdict::Allow
        } else {
            Verdict::Deny(Reason::EndpointNotListed)
        }
    }

    /// Write the policy into `out` and return the bytes used.
    ///
    /// The layout is the magic, the version, the capability byte, then the
    /// read roots, the write roots and the endpoints, each list as a count
    /// followed by its entries; a root is a length byte and its bytes and an
    /// endpoint is the protocol, the four bytes of the address and the port
    /// in big endian. [`MAX_ENCODED_LEN`] bytes always suffice.
    pub fn encode(&self, out: &mut [u8]) -> Result<usize, Reason> {
        let mut writer = Writer { out, len: 0 };
        writer.put(&MAGIC)?;
        writer.put(&[VERSION, self.granted.bits()])?;
        for roots in [&self.read, &self.write] {
            writer.put(&[roots.len])?;
            for rule in roots.as_slice() {
                writer.put(&[rule.len])?;
                writer.put(rule.as_bytes())?;
            }
        }
        writer.put(&[self.endpoint_count])?;
        for endpoint in self.endpoints() {
            writer.put(&[endpoint.protocol.wire()])?;
            writer.put(&endpoint.ip)?;
            writer.put(&endpoint.port.to_be_bytes())?;
        }
        Ok(writer.len)
    }

    /// Read a policy from `bytes`, which must be exactly one encoding.
    ///
    /// Anything the encoder cannot have produced is [`Reason::Malformed`]: a
    /// wrong magic or version, a bit that names no capability, a count or a
    /// length over its capacity, a truncated entry, a byte left over. A root
    /// that is not a normalized absolute path is [`Reason::BadPath`].
    pub fn decode(bytes: &[u8]) -> Result<Self, Reason> {
        let mut reader = Reader { bytes };
        if reader.take(MAGIC.len())? != MAGIC || reader.byte()? != VERSION {
            return Err(Reason::Malformed);
        }
        let granted = Capabilities::from_bits(reader.byte()?).ok_or(Reason::Malformed)?;
        let mut policy = Self::new(granted);
        for roots in [&mut policy.read, &mut policy.write] {
            let count = usize::from(reader.byte()?);
            if count > MAX_ROOTS {
                return Err(Reason::Malformed);
            }
            for _ in 0..count {
                let len = usize::from(reader.byte()?);
                if len > MAX_ROOT_LEN {
                    return Err(Reason::Malformed);
                }
                roots.push(reader.take(len)?)?;
            }
        }
        let count = usize::from(reader.byte()?);
        if count > MAX_ENDPOINTS {
            return Err(Reason::Malformed);
        }
        for _ in 0..count {
            let protocol = Protocol::from_wire(reader.byte()?).ok_or(Reason::Malformed)?;
            let ip = reader.take(4)?.try_into().map_err(|_| Reason::Malformed)?;
            let port =
                u16::from_be_bytes(reader.take(2)?.try_into().map_err(|_| Reason::Malformed)?);
            policy.add_endpoint(Endpoint { protocol, ip, port })?;
        }
        if !reader.bytes.is_empty() {
            return Err(Reason::Malformed);
        }
        Ok(policy)
    }
}

/// Appends to a buffer and refuses to go past its end.
struct Writer<'a> {
    out: &'a mut [u8],
    len: usize,
}

impl Writer<'_> {
    fn put(&mut self, bytes: &[u8]) -> Result<(), Reason> {
        let end = self.len + bytes.len();
        self.out
            .get_mut(self.len..end)
            .ok_or(Reason::Capacity)?
            .copy_from_slice(bytes);
        self.len = end;
        Ok(())
    }
}

/// Consumes a buffer from the front and refuses to go past its end.
struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8], Reason> {
        let (head, tail) = self.bytes.split_at_checked(len).ok_or(Reason::Malformed)?;
        self.bytes = tail;
        Ok(head)
    }

    fn byte(&mut self) -> Result<u8, Reason> {
        match self.take(1)? {
            [byte] => Ok(*byte),
            _ => Err(Reason::Malformed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(read: &[&str], write: &[&str]) -> Policy {
        let mut granted = Capabilities::EMPTY;
        granted.insert(Capability::FsRead);
        granted.insert(Capability::FsWrite);
        let mut policy = Policy::new(granted);
        for root in read {
            policy.add_read_root(root.as_bytes()).unwrap();
        }
        for root in write {
            policy.add_write_root(root.as_bytes()).unwrap();
        }
        policy
    }

    fn open(policy: &Policy, path: &str, write: bool) -> Verdict {
        policy.check_open(path.as_bytes(), write)
    }

    fn normalized(path: &str) -> Result<String, Reason> {
        let mut out = [0; PATH_BUFFER_LEN];
        let len = normalize(path.as_bytes(), &mut out)?;
        Ok(String::from_utf8(out[..len].to_vec()).unwrap())
    }

    fn server_policy() -> Policy {
        let mut granted = Capabilities::EMPTY;
        granted.insert(Capability::NetUdp);
        granted.insert(Capability::NetTcp);
        let mut policy = Policy::new(granted);
        policy
            .add_endpoint(Endpoint {
                protocol: Protocol::Tcp,
                ip: [10, 0, 2, 2],
                port: 80,
            })
            .unwrap();
        policy
    }

    #[test]
    fn capabilities_insert_and_contain() {
        let mut set = Capabilities::EMPTY;
        assert!(!set.contains(Capability::NetUdp));
        set.insert(Capability::NetUdp);
        assert!(set.contains(Capability::NetUdp));
        assert!(!set.contains(Capability::NetTcp));
        assert_eq!(Capabilities::from_bits(set.bits()), Some(set));
        assert_eq!(Capabilities::from_bits(0b1_0000), None);
    }

    #[test]
    fn normalize_collapses_what_is_harmless() {
        assert_eq!(normalized("/data/x").as_deref(), Ok("/data/x"));
        assert_eq!(normalized("//data//x").as_deref(), Ok("/data/x"));
        assert_eq!(normalized("/data/./x").as_deref(), Ok("/data/x"));
        assert_eq!(normalized("/data/").as_deref(), Ok("/data"));
        assert_eq!(normalized("/").as_deref(), Ok("/"));
        assert_eq!(normalized("///").as_deref(), Ok("/"));
        assert_eq!(normalized("/.").as_deref(), Ok("/"));
        assert_eq!(normalized("/...").as_deref(), Ok("/..."));
    }

    #[test]
    fn normalize_refuses_instead_of_resolving() {
        for path in [
            "",
            "data",
            "./data",
            "/data/../etc",
            "/..",
            "/a/..",
            "/a/../",
        ] {
            assert_eq!(normalized(path), Err(Reason::BadPath), "{path}");
        }
        let mut out = [0; PATH_BUFFER_LEN];
        assert_eq!(normalize(b"/da\0ta", &mut out), Err(Reason::BadPath));
        assert_eq!(normalize(b"\0", &mut out), Err(Reason::BadPath));
    }

    #[test]
    fn normalize_length_limit_is_255_bytes() {
        let mut path = vec![b'/'];
        path.resize(MAX_PATH_LEN, b'a');
        let mut out = [0; PATH_BUFFER_LEN];
        assert_eq!(normalize(&path, &mut out), Ok(MAX_PATH_LEN));
        path.push(b'a');
        assert_eq!(normalize(&path, &mut out), Err(Reason::BadPath));
    }

    #[test]
    fn traversal_attempts_never_escape_the_root() {
        let policy = policy(&["/data"], &[]);
        assert_eq!(
            open(&policy, "/data/../etc", false),
            Verdict::Deny(Reason::BadPath)
        );
        assert_eq!(
            open(&policy, "/data/../data/x", false),
            Verdict::Deny(Reason::BadPath)
        );
        assert_eq!(open(&policy, "/data/./x", false), Verdict::Allow);
        assert_eq!(open(&policy, "//data//x", false), Verdict::Allow);
        assert_eq!(open(&policy, "/data/", false), Verdict::Allow);
        assert_eq!(open(&policy, "/data", false), Verdict::Allow);
        assert_eq!(
            open(&policy, "data/x", false),
            Verdict::Deny(Reason::BadPath)
        );
        assert_eq!(
            open(&policy, "/etc/passwd", false),
            Verdict::Deny(Reason::OutsideRoots)
        );
    }

    #[test]
    fn a_root_matches_on_a_component_boundary() {
        let policy = policy(&["/data"], &[]);
        assert_eq!(
            open(&policy, "/database", false),
            Verdict::Deny(Reason::OutsideRoots)
        );
        assert_eq!(
            open(&policy, "/data2/x", false),
            Verdict::Deny(Reason::OutsideRoots)
        );
        assert_eq!(
            open(&policy, "/dat", false),
            Verdict::Deny(Reason::OutsideRoots)
        );
        assert_eq!(open(&policy, "/data/base", false), Verdict::Allow);
    }

    #[test]
    fn the_root_of_the_tree_covers_every_path() {
        let policy = policy(&["/"], &[]);
        assert_eq!(open(&policy, "/anything/at/all", false), Verdict::Allow);
        assert_eq!(open(&policy, "/", false), Verdict::Allow);
        assert_eq!(
            open(&policy, "/../etc", false),
            Verdict::Deny(Reason::BadPath)
        );
    }

    #[test]
    fn reads_and_writes_do_not_imply_each_other() {
        let policy = policy(&["/in"], &["/out"]);
        assert_eq!(open(&policy, "/in/a", false), Verdict::Allow);
        assert_eq!(
            open(&policy, "/in/a", true),
            Verdict::Deny(Reason::OutsideRoots)
        );
        assert_eq!(open(&policy, "/out/a", true), Verdict::Allow);
        assert_eq!(
            open(&policy, "/out/a", false),
            Verdict::Deny(Reason::OutsideRoots)
        );
    }

    #[test]
    fn a_root_without_its_capability_grants_nothing() {
        let mut granted = Capabilities::EMPTY;
        granted.insert(Capability::FsRead);
        let mut policy = Policy::new(granted);
        policy.add_read_root(b"/in").unwrap();
        policy.add_write_root(b"/out").unwrap();
        assert_eq!(open(&policy, "/in/a", false), Verdict::Allow);
        assert_eq!(
            open(&policy, "/out/a", true),
            Verdict::Deny(Reason::NotGranted)
        );

        let mut policy = Policy::new(Capabilities::EMPTY);
        policy.add_read_root(b"/in").unwrap();
        assert_eq!(
            open(&policy, "/in/a", false),
            Verdict::Deny(Reason::NotGranted)
        );
    }

    #[test]
    fn an_empty_policy_denies_everything() {
        let policy = Policy::new(Capabilities::EMPTY);
        assert_eq!(open(&policy, "/", false), Verdict::Deny(Reason::NotGranted));
        assert_eq!(open(&policy, "/", true), Verdict::Deny(Reason::NotGranted));
        assert_eq!(
            policy.check_connect(Protocol::Udp, [10, 0, 2, 2], 9999),
            Verdict::Deny(Reason::NotGranted)
        );
        assert_eq!(
            policy.check_connect(Protocol::Tcp, [10, 0, 2, 2], 80),
            Verdict::Deny(Reason::NotGranted)
        );

        let mut granted = Capabilities::EMPTY;
        granted.insert(Capability::FsRead);
        granted.insert(Capability::NetTcp);
        let policy = Policy::new(granted);
        assert_eq!(
            open(&policy, "/", false),
            Verdict::Deny(Reason::OutsideRoots)
        );
        assert_eq!(
            policy.check_connect(Protocol::Tcp, [10, 0, 2, 2], 80),
            Verdict::Deny(Reason::EndpointNotListed)
        );
    }

    #[test]
    fn an_endpoint_matches_exactly() {
        let policy = server_policy();
        let ip = [10, 0, 2, 2];
        assert_eq!(policy.check_connect(Protocol::Tcp, ip, 80), Verdict::Allow);
        assert_eq!(
            policy.check_connect(Protocol::Tcp, ip, 81),
            Verdict::Deny(Reason::EndpointNotListed)
        );
        assert_eq!(
            policy.check_connect(Protocol::Udp, ip, 80),
            Verdict::Deny(Reason::EndpointNotListed)
        );
        assert_eq!(
            policy.check_connect(Protocol::Tcp, [10, 0, 2, 3], 80),
            Verdict::Deny(Reason::EndpointNotListed)
        );
        assert_eq!(
            policy.check_connect(Protocol::Tcp, ip, 0),
            Verdict::Deny(Reason::Malformed)
        );
    }

    #[test]
    fn a_listed_endpoint_needs_its_protocol_capability() {
        let mut granted = Capabilities::EMPTY;
        granted.insert(Capability::NetUdp);
        let mut policy = Policy::new(granted);
        for protocol in [Protocol::Udp, Protocol::Tcp] {
            policy
                .add_endpoint(Endpoint {
                    protocol,
                    ip: [1, 2, 3, 4],
                    port: 53,
                })
                .unwrap();
        }
        assert_eq!(
            policy.check_connect(Protocol::Udp, [1, 2, 3, 4], 53),
            Verdict::Allow
        );
        assert_eq!(
            policy.check_connect(Protocol::Tcp, [1, 2, 3, 4], 53),
            Verdict::Deny(Reason::NotGranted)
        );
    }

    #[test]
    fn capacity_is_an_error_and_changes_nothing() {
        let mut policy = Policy::new(Capabilities::EMPTY);
        for root in ["/a", "/b", "/c", "/d"] {
            policy.add_read_root(root.as_bytes()).unwrap();
            policy.add_write_root(root.as_bytes()).unwrap();
        }
        let full = policy;
        assert_eq!(policy.add_read_root(b"/e"), Err(Reason::Capacity));
        assert_eq!(policy.add_write_root(b"/e"), Err(Reason::Capacity));
        assert_eq!(policy, full);

        for port in 1..=4 {
            policy
                .add_endpoint(Endpoint {
                    protocol: Protocol::Udp,
                    ip: [1, 1, 1, 1],
                    port,
                })
                .unwrap();
        }
        let full = policy;
        let extra = Endpoint {
            protocol: Protocol::Udp,
            ip: [1, 1, 1, 1],
            port: 5,
        };
        assert_eq!(policy.add_endpoint(extra), Err(Reason::Capacity));
        assert_eq!(policy, full);
    }

    #[test]
    fn a_root_is_never_truncated_or_rewritten() {
        let mut policy = Policy::new(Capabilities::EMPTY);
        let mut longest = vec![b'/'];
        longest.resize(MAX_ROOT_LEN, b'a');
        assert_eq!(policy.add_read_root(&longest), Ok(()));
        longest.push(b'a');
        assert_eq!(policy.add_read_root(&longest), Err(Reason::BadPath));
        for root in [
            "",
            "data",
            "/data/",
            "//data",
            "/data/./x",
            "/data/../x",
            "/da\0ta",
        ] {
            assert_eq!(
                policy.add_read_root(root.as_bytes()),
                Err(Reason::BadPath),
                "{root}"
            );
        }
        assert_eq!(policy.add_write_root(b"/"), Ok(()));
        assert_eq!(
            policy.add_endpoint(Endpoint {
                protocol: Protocol::Udp,
                ip: [0; 4],
                port: 0,
            }),
            Err(Reason::Malformed)
        );
    }

    fn full_policy() -> Policy {
        let mut granted = Capabilities::EMPTY;
        granted.insert(Capability::FsRead);
        granted.insert(Capability::FsWrite);
        granted.insert(Capability::NetUdp);
        granted.insert(Capability::NetTcp);
        let mut policy = Policy::new(granted);
        policy.add_read_root(b"/data").unwrap();
        policy.add_read_root(b"/").unwrap();
        policy.add_write_root(b"/tmp").unwrap();
        policy
            .add_endpoint(Endpoint {
                protocol: Protocol::Udp,
                ip: [10, 0, 2, 2],
                port: 9999,
            })
            .unwrap();
        policy
            .add_endpoint(Endpoint {
                protocol: Protocol::Tcp,
                ip: [10, 0, 2, 2],
                port: 0x1234,
            })
            .unwrap();
        policy
    }

    fn encoded(policy: &Policy) -> Vec<u8> {
        let mut out = [0; MAX_ENCODED_LEN];
        let len = policy.encode(&mut out).unwrap();
        out[..len].to_vec()
    }

    #[test]
    fn encode_then_decode_returns_the_same_policy() {
        for policy in [
            Policy::new(Capabilities::EMPTY),
            full_policy(),
            policy(&["/a"], &["/b"]),
        ] {
            assert_eq!(Policy::decode(&encoded(&policy)), Ok(policy));
        }
    }

    #[test]
    fn the_largest_policy_fits_the_public_constant() {
        let mut root = vec![b'/'];
        root.resize(MAX_ROOT_LEN, b'a');
        let mut policy = Policy::new(full_policy().granted());
        for _ in 0..MAX_ROOTS {
            policy.add_read_root(&root).unwrap();
            policy.add_write_root(&root).unwrap();
        }
        for port in 1..=MAX_ENDPOINTS as u16 {
            policy
                .add_endpoint(Endpoint {
                    protocol: Protocol::Tcp,
                    ip: [9; 4],
                    port,
                })
                .unwrap();
        }
        assert_eq!(encoded(&policy).len(), MAX_ENCODED_LEN);
        assert_eq!(Policy::decode(&encoded(&policy)), Ok(policy));
    }

    #[test]
    fn encode_into_a_small_buffer_is_an_error() {
        let policy = full_policy();
        let size = encoded(&policy).len();
        for len in 0..size {
            let mut out = vec![0; len];
            assert_eq!(policy.encode(&mut out), Err(Reason::Capacity), "{len}");
        }
    }

    #[test]
    fn decode_refuses_every_truncation() {
        let bytes = encoded(&full_policy());
        for len in 0..bytes.len() {
            assert!(Policy::decode(&bytes[..len]).is_err(), "{len}");
        }
    }

    #[test]
    fn decode_refuses_a_wrong_header() {
        let bytes = encoded(&full_policy());

        let mut wrong = bytes.clone();
        wrong[0] = b'X';
        assert_eq!(Policy::decode(&wrong), Err(Reason::Malformed));

        let mut wrong = bytes.clone();
        wrong[4] = VERSION + 1;
        assert_eq!(Policy::decode(&wrong), Err(Reason::Malformed));

        let mut wrong = bytes;
        wrong[5] |= 0b1_0000;
        assert_eq!(Policy::decode(&wrong), Err(Reason::Malformed));
    }

    #[test]
    fn decode_refuses_a_trailing_byte() {
        let mut bytes = encoded(&full_policy());
        bytes.push(0);
        assert_eq!(Policy::decode(&bytes), Err(Reason::Malformed));
    }

    #[test]
    fn decode_refuses_lengths_over_the_capacity() {
        let empty = encoded(&Policy::new(Capabilities::EMPTY));
        assert_eq!(empty.len(), 6 + 3);

        for index in [6, 7, 8] {
            let mut wrong = empty.clone();
            wrong[index] = MAX_ROOTS as u8 + 1;
            assert_eq!(Policy::decode(&wrong), Err(Reason::Malformed), "{index}");
        }

        let mut wrong = empty[..6].to_vec();
        wrong.push(1);
        wrong.push(MAX_ROOT_LEN as u8 + 1);
        wrong.resize(wrong.len() + MAX_ROOT_LEN + 1, b'a');
        wrong.extend_from_slice(&[0, 0]);
        assert_eq!(Policy::decode(&wrong), Err(Reason::Malformed));

        let mut wrong = empty[..6].to_vec();
        wrong.extend_from_slice(&[1, 200]);
        assert_eq!(Policy::decode(&wrong), Err(Reason::Malformed));
    }

    #[test]
    fn decode_refuses_a_root_that_is_not_normalized() {
        for root in ["data", "/data/", "//data", "/data/../x", "/a/./b"] {
            let mut bytes = encoded(&Policy::new(Capabilities::EMPTY))[..6].to_vec();
            bytes.push(1);
            bytes.push(root.len() as u8);
            bytes.extend_from_slice(root.as_bytes());
            bytes.extend_from_slice(&[0, 0]);
            assert_eq!(Policy::decode(&bytes), Err(Reason::BadPath), "{root}");
        }
    }

    #[test]
    fn decode_refuses_a_bad_endpoint() {
        let header = encoded(&Policy::new(Capabilities::EMPTY))[..6].to_vec();

        let mut wrong = header.clone();
        wrong.extend_from_slice(&[0, 0, 1, 2, 1, 2, 3, 4, 0, 80]);
        assert_eq!(Policy::decode(&wrong), Err(Reason::Malformed));

        let mut wrong = header.clone();
        wrong.extend_from_slice(&[0, 0, 1, 0, 1, 2, 3, 4, 0, 0]);
        assert_eq!(Policy::decode(&wrong), Err(Reason::Malformed));

        let mut wrong = header;
        wrong.extend_from_slice(&[0, 0, MAX_ENDPOINTS as u8 + 1]);
        assert_eq!(Policy::decode(&wrong), Err(Reason::Malformed));
    }

    #[test]
    fn the_wire_format_is_these_exact_bytes() {
        const GOLDEN: [u8; 25] = [
            b'R', b'B', b'P', b'L',   // magic
            1,      // version
            0b1011, // FsRead, FsWrite, NetTcp
            1, 3, b'/', b'i', b'n', // one read root
            1, 4, b'/', b'o', b'u', b't', // one write root
            1,    // one endpoint
            1, 10, 0, 2, 2, 0x1f, 0x90, // TCP 10.0.2.2:8080, port in big endian
        ];
        let mut granted = Capabilities::EMPTY;
        granted.insert(Capability::FsRead);
        granted.insert(Capability::FsWrite);
        granted.insert(Capability::NetTcp);
        let mut expected = Policy::new(granted);
        expected.add_read_root(b"/in").unwrap();
        expected.add_write_root(b"/out").unwrap();
        expected
            .add_endpoint(Endpoint {
                protocol: Protocol::Tcp,
                ip: [10, 0, 2, 2],
                port: 8080,
            })
            .unwrap();
        assert_eq!(encoded(&expected), GOLDEN);
        assert_eq!(Policy::decode(&GOLDEN), Ok(expected));
    }

    #[test]
    fn the_port_is_big_endian_on_the_wire() {
        const GOLDEN: [u8; 16] = [
            b'R', b'B', b'P', b'L', 1, 0b1000, 0, 0, // no roots
            1, 1, 10, 0, 2, 2, 0x12, 0x34, // TCP 10.0.2.2:0x1234
        ];
        let policy = Policy::decode(&GOLDEN).unwrap();
        assert_eq!(
            policy.check_connect(Protocol::Tcp, [10, 0, 2, 2], 0x1234),
            Verdict::Allow
        );
        assert_eq!(
            policy.check_connect(Protocol::Tcp, [10, 0, 2, 2], 0x3412),
            Verdict::Deny(Reason::EndpointNotListed)
        );
    }

    /// Every path made of up to four of these components, with and without a
    /// leading slash, and a check that none with a `..` is ever allowed.
    #[test]
    fn no_path_with_a_dot_dot_component_is_ever_allowed() {
        const COMPONENTS: [&str; 5] = ["", ".", "..", "a", "data"];
        let roots = [policy(&["/"], &["/"]), policy(&["/data"], &["/data"])];

        let mut paths = vec![Vec::<&str>::new()];
        let mut start = 0;
        for _ in 0..4 {
            let end = paths.len();
            for index in start..end {
                for component in COMPONENTS {
                    let mut longer = paths[index].clone();
                    longer.push(component);
                    paths.push(longer);
                }
            }
            start = end;
        }
        assert_eq!(paths.len(), 1 + 5 + 25 + 125 + 625);

        let mut allowed = 0;
        for components in &paths {
            let joined = components.join("/");
            for path in [joined.clone(), format!("/{joined}")] {
                let has_dot_dot = components.contains(&"..");
                for policy in &roots {
                    for write in [false, true] {
                        let verdict = open(policy, &path, write);
                        if has_dot_dot {
                            assert!(verdict != Verdict::Allow, "{path}");
                        }
                        if verdict == Verdict::Allow {
                            allowed += 1;
                        }
                    }
                }
                if path.starts_with('/') && !has_dot_dot {
                    assert_eq!(open(&roots[0], &path, false), Verdict::Allow, "{path}");
                }
            }
        }
        assert!(allowed > 0);
    }
}
