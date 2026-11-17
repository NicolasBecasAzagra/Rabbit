// SPDX-License-Identifier: GPL-2.0-only
//! The capability manifest of the tools.
//!
//! Each tool of [`tools::NAMES`] needs at most one capability, and the agent
//! declares the tools it uses before it runs. [`manifest`] turns that
//! declaration into the [`Policy`] of the run: it grants exactly the
//! capabilities of the declared tools, and it installs only the rules of the
//! [`Scope`] whose capability was granted, so a run that declares no network
//! tool carries no endpoint even when the scope lists one.

use rabbit_policy::{Capabilities, Capability, Endpoint, Policy, Reason};

use crate::tools;

/// The capability each tool needs, in the order of [`tools::NAMES`]. A tool
/// that touches neither the file system nor the network needs none.
///
/// The length of the array is that of the registry, so a tool added to
/// [`tools::NAMES`] does not compile until it has an entry here.
pub const CAPABILITIES: [(&str, Option<Capability>); tools::NAMES.len()] = [
    ("read_file", Some(Capability::FsRead)),
    ("list_dir", Some(Capability::FsRead)),
    ("write_file", Some(Capability::FsWrite)),
    ("compute", None),
    ("net_send", Some(Capability::NetUdp)),
    ("http_get", Some(Capability::NetTcp)),
    ("done", None),
];

/// What the task lets the tools reach, as plain data.
pub struct Scope<'a> {
    /// Roots the file tools may read under.
    pub read_roots: &'a [&'a str],
    /// Roots `write_file` may write under.
    pub write_roots: &'a [&'a str],
    /// Servers the network tools may talk to.
    pub endpoints: &'a [Endpoint],
}

/// The policy of a run that declares `tools` and has the reach of `scope`.
///
/// A name that is not in [`CAPABILITIES`] is [`Reason::Malformed`]: a tool the
/// manifest does not know cannot be granted anything, and dropping it
/// silently would hide a typo behind a policy that is too narrow. A root of
/// `/` is [`Reason::BadPath`]: the engine accepts it, but a scope that names
/// the whole tree grants every path, which is no scope at all.
pub fn manifest(tools: &[&str], scope: &Scope) -> Result<Policy, Reason> {
    let mut granted = Capabilities::EMPTY;
    for tool in tools {
        let (_, capability) = CAPABILITIES
            .iter()
            .find(|(name, _)| name == tool)
            .ok_or(Reason::Malformed)?;
        if let Some(capability) = capability {
            granted.insert(*capability);
        }
    }
    let mut policy = Policy::new(granted);
    if granted.contains(Capability::FsRead) {
        for root in scope.read_roots {
            refuse_whole_tree(root)?;
            policy.add_read_root(root.as_bytes())?;
        }
    }
    if granted.contains(Capability::FsWrite) {
        for root in scope.write_roots {
            refuse_whole_tree(root)?;
            policy.add_write_root(root.as_bytes())?;
        }
    }
    for endpoint in scope.endpoints {
        if granted.contains(endpoint.protocol.capability()) {
            policy.add_endpoint(*endpoint)?;
        }
    }
    Ok(policy)
}

fn refuse_whole_tree(root: &str) -> Result<(), Reason> {
    if root == "/" {
        Err(Reason::BadPath)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rabbit_policy::{Protocol, Verdict};

    use super::*;

    const SERVER: [u8; 4] = [10, 0, 2, 2];

    fn scope(endpoints: &[Endpoint]) -> Scope<'_> {
        Scope {
            read_roots: &["/in"],
            write_roots: &["/out"],
            endpoints,
        }
    }

    fn endpoints() -> [Endpoint; 2] {
        [
            Endpoint {
                protocol: Protocol::Udp,
                ip: SERVER,
                port: 9999,
            },
            Endpoint {
                protocol: Protocol::Tcp,
                ip: SERVER,
                port: 80,
            },
        ]
    }

    #[test]
    fn every_tool_has_exactly_one_entry_and_no_entry_is_extra() {
        for name in tools::NAMES {
            let entries = CAPABILITIES.iter().filter(|(entry, _)| *entry == name);
            assert_eq!(entries.count(), 1, "{name}");
        }
        for (entry, _) in CAPABILITIES {
            assert!(tools::NAMES.contains(&entry), "{entry}");
        }
        let names: Vec<&str> = CAPABILITIES.iter().map(|(name, _)| *name).collect();
        assert_eq!(names, tools::NAMES);
    }

    #[test]
    fn compute_and_done_grant_nothing() {
        let endpoints = endpoints();
        let policy = manifest(&["compute", "done"], &scope(&endpoints)).unwrap();
        assert_eq!(policy, Policy::new(Capabilities::EMPTY));
        assert_eq!(
            policy.check_open(b"/in/a", false),
            Verdict::Deny(Reason::NotGranted)
        );
        assert_eq!(
            policy.check_connect(Protocol::Tcp, SERVER, 80),
            Verdict::Deny(Reason::NotGranted)
        );
    }

    #[test]
    fn a_manifest_without_http_get_cannot_connect_over_tcp() {
        let endpoints = endpoints();
        let declared = ["read_file", "net_send", "compute", "done"];
        let policy = manifest(&declared, &scope(&endpoints)).unwrap();
        assert_eq!(
            policy.check_connect(Protocol::Tcp, SERVER, 80),
            Verdict::Deny(Reason::NotGranted)
        );
        assert_eq!(
            policy.check_connect(Protocol::Udp, SERVER, 9999),
            Verdict::Allow
        );
    }

    #[test]
    fn a_rule_is_installed_only_when_its_capability_is_granted() {
        let endpoints = endpoints();
        let policy = manifest(&["read_file"], &scope(&endpoints)).unwrap();
        assert_eq!(policy.check_open(b"/in/a", false), Verdict::Allow);
        assert_eq!(
            policy.check_open(b"/out/a", true),
            Verdict::Deny(Reason::NotGranted)
        );

        let policy = manifest(&["write_file"], &scope(&endpoints)).unwrap();
        assert_eq!(policy.check_open(b"/out/a", true), Verdict::Allow);
        assert_eq!(
            policy.check_open(b"/in/a", false),
            Verdict::Deny(Reason::NotGranted)
        );
    }

    #[test]
    fn all_seven_tools_allow_the_endpoint_of_the_scope_and_no_other_ip() {
        let endpoints = endpoints();
        let policy = manifest(&tools::NAMES, &scope(&endpoints)).unwrap();
        assert_eq!(
            policy.check_connect(Protocol::Tcp, SERVER, 80),
            Verdict::Allow
        );
        assert_eq!(
            policy.check_connect(Protocol::Udp, SERVER, 9999),
            Verdict::Allow
        );
        assert_eq!(
            policy.check_connect(Protocol::Tcp, [10, 0, 2, 3], 80),
            Verdict::Deny(Reason::EndpointNotListed)
        );
        assert_eq!(
            policy.check_connect(Protocol::Udp, [10, 0, 2, 3], 9999),
            Verdict::Deny(Reason::EndpointNotListed)
        );
        assert_eq!(policy.check_open(b"/in/a", false), Verdict::Allow);
        assert_eq!(policy.check_open(b"/out/a", true), Verdict::Allow);
    }

    #[test]
    fn an_unknown_tool_is_refused() {
        let endpoints = endpoints();
        assert_eq!(
            manifest(&["read_file", "rm_rf"], &scope(&endpoints)),
            Err(Reason::Malformed)
        );
    }

    #[test]
    fn a_rule_the_policy_cannot_hold_is_an_error() {
        let endpoints = endpoints();
        let scope = Scope {
            read_roots: &["/a", "/b", "/c", "/d", "/e"],
            write_roots: &[],
            endpoints: &endpoints,
        };
        assert_eq!(manifest(&["read_file"], &scope), Err(Reason::Capacity));
    }

    #[test]
    fn a_root_of_the_whole_tree_is_refused() {
        let endpoints = endpoints();
        let read = Scope {
            read_roots: &["/in", "/"],
            write_roots: &[],
            endpoints: &endpoints,
        };
        assert_eq!(manifest(&["read_file"], &read), Err(Reason::BadPath));
        let write = Scope {
            read_roots: &[],
            write_roots: &["/"],
            endpoints: &endpoints,
        };
        assert_eq!(manifest(&["write_file"], &write), Err(Reason::BadPath));
    }
}
