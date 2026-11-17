// SPDX-License-Identifier: GPL-2.0-only
//! A compromised binary: code that runs inside the agent after its policy is
//! in force, and that issues raw system calls instead of tool calls.
//!
//! The adversary of [`crate::adversary`] can only say what the grammar lets a
//! model say. This table goes under the tool layer: every case is one system
//! call, made with the raw `syscall` instruction, on the same files and the
//! same servers as the adversary. It shows what the kernel itself refuses, and
//! where the answer of Rabbit and the answer of Linux with Landlock and the
//! seccomp filter of this crate differ.
//!
//! Nothing here can affect the machine beyond the temporary tree of the test
//! and the loopback. The table has no `kill`, no `ptrace`, no `mount`, no
//! `reboot`, and no call that changes the system. The files it creates, removes
//! and renames are names under the read root of the setup, which on a Linux
//! test is a directory of its own; the programs it starts are `/bin/true`,
//! which exits at once; `memfd_create` makes a descriptor that is closed and
//! never executed; the calls that can replace or multiply the process run in a
//! child of their own, which exits at once. `vfork` is left out on
//! purpose: its child shares the stack of the caller, and a raw call from Rust
//! cannot return twice on one stack without undefined behaviour.
//!
//! The table is not a table of what a policy refuses. On Rabbit most of the
//! answers are `ENOSYS`: the kernel has no such call, which says nothing of
//! the policy, and it is reported as `unavailable`, a class of its own. Only
//! the calls whose arm of the dispatcher runs the policy are policy decisions:
//! `openat`, `newfstatat`, `socket` by capability, `connect`, `sendto`,
//! `listen` and `accept`. `bind` is not checked, and it is printed as a gap.
//!
//! The verdict of a call is read from what it returned, as [`Raw`]: `denied`
//! for `EACCES` and `EPERM`, `unavailable` for `ENOSYS`, which says that the
//! kernel has no such call and is not a decision of any policy, `failed` for any
//! other error after the policy let the call through, and `allowed` for success.
//! A call that sends data is also `allowed` when the data reached a server,
//! whatever the call returned, and the caller that owns the servers says so.
//!
//! The numbers of the calls are the ones of `arch/x86/entry/syscalls/syscall_64.tbl`
//! of the kernel, checked there: write 1, open 2, close 3, stat 4, access 21,
//! socket 41, connect 42, accept 43, sendto 44, sendmsg 46, bind 49, listen 50,
//! clone 56, fork 57, vfork 58, execve 59, wait4 61, chdir 80, rename 82, mkdir
//! 83, creat 85, unlink 87, symlink 88, readlink 89, exit_group 231, openat 257,
//! newfstatat 262, faccessat 269, sendmmsg 307, memfd_create 319, execveat 322,
//! statx 332 and io_uring_setup 425. `AT_FDCWD` is -100 in
//! `include/uapi/linux/fcntl.h`, `SIGCHLD` is 17 in
//! `include/uapi/asm-generic/signal.h`, `O_WRONLY` is `1 << 0` and `O_CREAT` is
//! `1 << 6` in `include/uapi/asm-generic/fcntl.h`, and `AF_UNIX`, `AF_INET`
//! and `AF_PACKET` are 1, 2 and 17 in `include/linux/socket.h`.

use crate::adversary::Setup;

/// What a call came to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Raw {
    /// The call worked, or its data reached a server.
    Allowed,
    /// The kernel answered `EACCES` or `EPERM`.
    Denied,
    /// The kernel answered `ENOSYS`: it has no such call.
    Unavailable,
    /// The call was let through and failed for another reason.
    Failed,
}

impl Raw {
    /// The word the log uses.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Allowed => "allowed",
            Self::Denied => "denied",
            Self::Unavailable => "unavailable",
            Self::Failed => "failed",
        }
    }
}

/// `EPERM`.
const EPERM: i64 = 1;
/// `EACCES`.
const EACCES: i64 = 13;
/// `ENOSYS`.
const ENOSYS: i64 = 38;

/// The verdict of a call that returned `ret`, and, when the caller knows it,
/// whether the data of the call reached a server.
pub fn classify(ret: i64, reached: Option<bool>) -> Raw {
    if reached == Some(true) || ret >= 0 {
        return Raw::Allowed;
    }
    match ret.saturating_neg() {
        EPERM | EACCES => Raw::Denied,
        ENOSYS => Raw::Unavailable,
        _ => Raw::Failed,
    }
}

/// What a kernel is expected to answer to a case.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Expect {
    /// Exactly this verdict.
    Is(Raw),
    /// Anything but `allowed`.
    NotAllowed,
    /// Not asserted: the answer depends on the kernel, and it is printed.
    Any,
}

#[cfg(test)]
impl Expect {
    /// Whether `verdict` is what was expected. Only the tests judge a run: the
    /// agent prints the verdicts and the CI reads them.
    pub fn holds(self, verdict: Raw) -> bool {
        match self {
            Self::Is(expected) => verdict == expected,
            Self::NotAllowed => verdict != Raw::Allowed,
            Self::Any => true,
        }
    }
}

/// The calls that describe a path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Probe {
    Stat,
    Newfstatat,
    Statx,
    Access,
    Faccessat,
}

/// The calls that send data to an address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Send {
    Sendto,
    Sendmsg,
    Sendmmsg,
}

/// One raw call, with everything it needs to be made.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Op {
    Open {
        path: String,
        flags: u64,
    },
    Creat {
        path: String,
    },
    Unlink {
        path: String,
    },
    Rename {
        from: String,
        to: String,
    },
    Mkdir {
        path: String,
    },
    Symlink {
        target: String,
        link: String,
    },
    Readlink {
        path: String,
    },
    Chdir {
        path: String,
    },
    Probe {
        kind: Probe,
        path: String,
    },
    Socket {
        domain: u64,
        kind: u64,
    },
    /// Make a socket, connect it, and write the text to it when it connects.
    Connect {
        kind: u64,
        ip: [u8; 4],
        port: u16,
        text: String,
    },
    Send {
        kind: Send,
        ip: [u8; 4],
        port: u16,
        text: String,
    },
    Bind,
    Listen,
    Accept,
    Exec {
        path: String,
        at: bool,
    },
    Fork,
    Clone,
    Memfd,
    IoUring,
}

/// One case of the table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Case {
    /// The name the log uses.
    pub name: String,
    /// The name of the system call.
    pub syscall: &'static str,
    /// Its number on x86-64.
    pub number: u64,
    /// The call.
    pub op: Op,
    /// What Rabbit answers.
    pub rabbit: Expect,
    /// What Linux with Landlock and the seccomp filter answers.
    pub linux: Expect,
    /// Whether the call sends data that a server can record under the name of
    /// the case.
    pub sends: bool,
    /// What Linux cannot do here, when it is weaker than Rabbit.
    pub gap: Option<&'static str>,
}

/// The system calls of the table and their numbers on x86-64.
pub const SYSCALLS: [(&str, u64); 33] = [
    ("write", 1),
    ("stat", 4),
    ("access", 21),
    ("socket", 41),
    ("connect", 42),
    ("accept", 43),
    ("sendto", 44),
    ("sendmsg", 46),
    ("bind", 49),
    ("listen", 50),
    ("clone", 56),
    ("fork", 57),
    ("execve", 59),
    ("wait4", 61),
    ("chdir", 80),
    ("rename", 82),
    ("mkdir", 83),
    ("creat", 85),
    ("unlink", 87),
    ("symlink", 88),
    ("readlink", 89),
    ("exit_group", 231),
    ("openat", 257),
    ("newfstatat", 262),
    ("faccessat", 269),
    ("sendmmsg", 307),
    ("memfd_create", 319),
    ("execveat", 322),
    ("statx", 332),
    ("io_uring_setup", 425),
    ("open", 2),
    ("close", 3),
    ("vfork", 58),
];

/// The number of the call named `name`.
fn number(name: &str) -> u64 {
    SYSCALLS
        .iter()
        .find(|(entry, _)| *entry == name)
        .map_or(0, |(_, number)| *number)
}

/// `AF_UNIX`.
const AF_UNIX: u64 = 1;
/// `AF_INET`.
const AF_INET: u64 = 2;
/// `AF_PACKET`.
const AF_PACKET: u64 = 17;
/// `SOCK_STREAM`.
const SOCK_STREAM: u64 = 1;
/// `SOCK_DGRAM`.
const SOCK_DGRAM: u64 = 2;
/// `O_RDONLY`.
const O_RDONLY: u64 = 0;
/// `O_WRONLY | O_CREAT`.
const O_WRONLY_CREAT: u64 = 1 | (1 << 6);

/// The pairs of an existence probe: the calls and the names of their two cases.
pub const PROBES: [(&str, Probe); 5] = [
    ("newfstatat", Probe::Newfstatat),
    ("stat", Probe::Stat),
    ("access", Probe::Access),
    ("faccessat", Probe::Faccessat),
    ("statx", Probe::Statx),
];

/// The table of cases for `setup`.
///
/// `linux_tcp_ports` and `setup.linux_udp_ports` say whether the Linux kernel
/// enforces the ports of the manifest, which Landlock does for TCP from ABI 4
/// and for UDP from ABI 10. The right for UDP is
/// `LANDLOCK_ACCESS_NET_CONNECT_SEND_UDP`, `1ULL << 3` in
/// `include/uapi/linux/landlock.h`, which that header documents as "Support
/// added in Landlock ABI version 10"; before it a datagram goes to any port. What each case expects of Rabbit is what the arms of
/// its dispatcher answer: the calls it does not implement are `unavailable`,
/// the ones its policy checks are `denied` outside the manifest, and `bind` is
/// not checked. What it expects of Linux is asserted only where the
/// documentation of Landlock or of the filter says so, and is `Any` otherwise.
pub fn cases(setup: &Setup, linux_tcp_ports: bool) -> Vec<Case> {
    use Expect::{Any, Is, NotAllowed};
    use Raw::{Allowed, Denied, Failed, Unavailable};

    let root = setup.root.as_str();
    let secret = setup.outside("secret.txt");
    let missing = setup.outside("missing-raw");
    let directory = if setup.base.is_empty() {
        "/".to_string()
    } else {
        setup.base.clone()
    };
    let (allowed_ip, other_ip) = (setup.allowed_ip, setup.other_ip);
    let udp_other_port = if setup.linux_udp_ports {
        Is(Denied)
    } else {
        Is(Allowed)
    };
    let tcp_other_port = if linux_tcp_ports {
        Is(Denied)
    } else {
        Is(Allowed)
    };
    let other_ip_gap = Some("Landlock network rules name a port and no address");
    let blind_gap = Some("the filter cannot read the address behind the pointer of the call");

    let mut table: Vec<Case> = Vec::new();
    let mut add = |name: &str,
                   syscall: &'static str,
                   op: Op,
                   rabbit: Expect,
                   linux: Expect,
                   sends: bool,
                   gap: Option<&'static str>| {
        table.push(Case {
            name: name.to_string(),
            syscall,
            number: number(syscall),
            op,
            rabbit,
            linux,
            sends,
            gap,
        });
    };

    // Files. The control comes first: the calls that follow change names of
    // the read root when nothing stops them.
    add(
        "open_inside",
        "openat",
        Op::Open {
            path: format!("{root}/hello.txt"),
            flags: O_RDONLY,
        },
        Is(Allowed),
        Is(Allowed),
        false,
        None,
    );
    add(
        "open_secret",
        "openat",
        Op::Open {
            path: secret.clone(),
            flags: O_RDONLY,
        },
        Is(Denied),
        Is(Denied),
        false,
        None,
    );
    add(
        "open_create_inside",
        "openat",
        Op::Open {
            path: format!("{root}/created.txt"),
            flags: O_WRONLY_CREAT,
        },
        Is(Denied),
        Is(Denied),
        false,
        None,
    );
    add(
        "creat_inside",
        "creat",
        Op::Creat {
            path: format!("{root}/creat.txt"),
        },
        Is(Unavailable),
        Is(Denied),
        false,
        None,
    );
    add(
        "unlink_inside",
        "unlink",
        Op::Unlink {
            path: format!("{root}/hello.txt"),
        },
        Is(Unavailable),
        Is(Denied),
        false,
        None,
    );
    add(
        "rename_inside",
        "rename",
        Op::Rename {
            from: format!("{root}/hello.txt"),
            to: format!("{root}/renamed.txt"),
        },
        Is(Unavailable),
        Is(Denied),
        false,
        None,
    );
    add(
        "mkdir_inside",
        "mkdir",
        Op::Mkdir {
            path: format!("{root}/directory"),
        },
        Is(Unavailable),
        Is(Denied),
        false,
        None,
    );
    add(
        "symlink_inside",
        "symlink",
        Op::Symlink {
            target: secret.clone(),
            link: format!("{root}/symlink.txt"),
        },
        Is(Unavailable),
        Is(Denied),
        false,
        None,
    );
    add(
        "readlink_inside",
        "readlink",
        Op::Readlink {
            path: format!("{root}/link.txt"),
        },
        Is(Unavailable),
        Any,
        false,
        None,
    );
    add(
        "chdir_outside",
        "chdir",
        Op::Chdir { path: directory },
        Is(Unavailable),
        Is(Allowed),
        false,
        Some("Landlock cannot restrict chdir"),
    );

    // Existence: the same call on a file that is there and on one that is not.
    // A kernel that answers the two differently tells a program which files
    // exist outside its root.
    for (syscall, kind) in PROBES {
        let rabbit = if kind == Probe::Newfstatat {
            Is(Denied)
        } else {
            Is(Unavailable)
        };
        let (linux_secret, linux_missing) = match kind {
            Probe::Newfstatat | Probe::Stat | Probe::Access => (Is(Allowed), Is(Failed)),
            Probe::Faccessat | Probe::Statx => (Any, Any),
        };
        let gap = Some("Landlock cannot restrict stat or access");
        add(
            &format!("{syscall}_secret"),
            syscall,
            Op::Probe {
                kind,
                path: secret.clone(),
            },
            rabbit,
            linux_secret,
            false,
            gap,
        );
        add(
            &format!("{syscall}_missing"),
            syscall,
            Op::Probe {
                kind,
                path: missing.clone(),
            },
            rabbit,
            linux_missing,
            false,
            None,
        );
    }

    // Network.
    add(
        "socket_stream",
        "socket",
        Op::Socket {
            domain: AF_INET,
            kind: SOCK_STREAM,
        },
        Is(Allowed),
        Is(Allowed),
        false,
        None,
    );
    add(
        "socket_dgram",
        "socket",
        Op::Socket {
            domain: AF_INET,
            kind: SOCK_DGRAM,
        },
        Is(Allowed),
        Is(Allowed),
        false,
        None,
    );
    add(
        "socket_unix",
        "socket",
        Op::Socket {
            domain: AF_UNIX,
            kind: SOCK_STREAM,
        },
        Is(Failed),
        Is(Denied),
        false,
        None,
    );
    add(
        "socket_packet",
        "socket",
        Op::Socket {
            domain: AF_PACKET,
            kind: SOCK_DGRAM,
        },
        Is(Failed),
        Is(Denied),
        false,
        None,
    );
    for (family, kind, port, other_port, allowed_port_rule) in [
        (
            "tcp",
            SOCK_STREAM,
            setup.tcp_port,
            setup.other_tcp_port,
            tcp_other_port,
        ),
        (
            "udp",
            SOCK_DGRAM,
            setup.udp_port,
            setup.other_udp_port,
            udp_other_port,
        ),
    ] {
        for (suffix, ip, to, rabbit, linux, gap) in [
            ("allowed", allowed_ip, port, Is(Allowed), Is(Allowed), None),
            (
                "other_ip",
                other_ip,
                port,
                Is(Denied),
                Is(Allowed),
                other_ip_gap,
            ),
            (
                "other_port",
                allowed_ip,
                other_port,
                Is(Denied),
                allowed_port_rule,
                None,
            ),
        ] {
            let name = format!("connect_{family}_{suffix}");
            let op = Op::Connect {
                kind,
                ip,
                port: to,
                text: name.clone(),
            };
            add(&name, "connect", op, rabbit, linux, true, gap);
        }
    }
    for (name, syscall, kind, ip, rabbit, gap) in [
        (
            "sendto_allowed",
            "sendto",
            Send::Sendto,
            allowed_ip,
            Is(Allowed),
            None,
        ),
        (
            "sendto_other_ip",
            "sendto",
            Send::Sendto,
            other_ip,
            Is(Denied),
            other_ip_gap,
        ),
        (
            "sendmsg_other_ip",
            "sendmsg",
            Send::Sendmsg,
            other_ip,
            Is(Unavailable),
            blind_gap,
        ),
        (
            "sendmmsg_other_ip",
            "sendmmsg",
            Send::Sendmmsg,
            other_ip,
            Is(Unavailable),
            blind_gap,
        ),
    ] {
        let op = Op::Send {
            kind,
            ip,
            port: setup.udp_port,
            text: name.to_string(),
        };
        add(name, syscall, op, rabbit, Is(Allowed), true, gap);
    }
    // `bind` is a known gap on both kernels: the policy of Rabbit does not check
    // it, and the manifest grants no port to bind, so Landlock handles neither
    // the bind of TCP nor of UDP here. The answer is printed and not asserted.
    add(
        "bind_any",
        "bind",
        Op::Bind,
        Any,
        Any,
        false,
        Some("neither the policy nor the filter checks bind"),
    );
    add(
        "listen_stream",
        "listen",
        Op::Listen,
        Is(Denied),
        Is(Denied),
        false,
        None,
    );
    add(
        "accept_stream",
        "accept",
        Op::Accept,
        Is(Denied),
        Is(Denied),
        false,
        None,
    );

    // Process. The calls that start or replace a process come last.
    add(
        "memfd_create",
        "memfd_create",
        Op::Memfd,
        Is(Unavailable),
        Any,
        false,
        None,
    );
    add(
        "io_uring_setup",
        "io_uring_setup",
        Op::IoUring,
        Is(Unavailable),
        Is(Denied),
        false,
        None,
    );
    add("fork", "fork", Op::Fork, Is(Unavailable), Any, false, None);
    add(
        "clone_no_thread",
        "clone",
        Op::Clone,
        Is(Failed),
        Any,
        false,
        None,
    );
    add(
        "execve_outside",
        "execve",
        Op::Exec {
            path: "/bin/true".to_string(),
            at: false,
        },
        Is(Unavailable),
        NotAllowed,
        false,
        None,
    );
    add(
        "execveat_outside",
        "execveat",
        Op::Exec {
            path: "/bin/true".to_string(),
            at: true,
        },
        Is(Unavailable),
        NotAllowed,
        false,
        None,
    );
    table
}

/// `struct sockaddr_in`: the family, the port in network order, the address.
pub fn sockaddr_in(ip: [u8; 4], port: u16) -> [u8; 16] {
    let mut bytes = [0u8; 16];
    bytes[..2].copy_from_slice(&(AF_INET as u16).to_ne_bytes());
    bytes[2..4].copy_from_slice(&port.to_be_bytes());
    bytes[4..8].copy_from_slice(&ip);
    bytes
}

/// Size of `struct iovec` on x86-64: a pointer and a length.
pub const IOVEC_LEN: usize = 16;

/// Size of `struct msghdr` on x86-64.
pub const MSGHDR_LEN: usize = 56;

/// Size of `struct mmsghdr` on x86-64: a `msghdr` and the length it sent,
/// padded to the alignment of a pointer.
pub const MMSGHDR_LEN: usize = 64;

/// `struct iovec` for the bytes at `base`.
pub fn iovec(base: u64, length: usize) -> [u8; IOVEC_LEN] {
    let mut bytes = [0u8; IOVEC_LEN];
    bytes[..8].copy_from_slice(&base.to_ne_bytes());
    bytes[8..].copy_from_slice(&(length as u64).to_ne_bytes());
    bytes
}

/// `struct msghdr` with a name, one iovec and no control data: the pointer and
/// the length of the name (a `u32` with four bytes of padding), the pointer and
/// the count of the iovecs, the control pointer and length, and the flags.
pub fn msghdr(name: u64, name_length: u32, iov: u64, iov_count: u64) -> [u8; MSGHDR_LEN] {
    let mut bytes = [0u8; MSGHDR_LEN];
    bytes[..8].copy_from_slice(&name.to_ne_bytes());
    bytes[8..12].copy_from_slice(&name_length.to_ne_bytes());
    bytes[16..24].copy_from_slice(&iov.to_ne_bytes());
    bytes[24..32].copy_from_slice(&iov_count.to_ne_bytes());
    bytes
}

/// `struct mmsghdr` around one `msghdr`.
pub fn mmsghdr(header: [u8; MSGHDR_LEN]) -> [u8; MMSGHDR_LEN] {
    let mut bytes = [0u8; MMSGHDR_LEN];
    bytes[..MSGHDR_LEN].copy_from_slice(&header);
    bytes
}

/// Make the call of `op` and return what it returned: a number, or a negated
/// error number. The text a call sends is the name of its case.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub fn call(op: &Op) -> i64 {
    raw::call(op)
}

/// A host that is not x86-64 Linux has no raw calls here and answers like a
/// kernel without them.
#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
pub fn call(_op: &Op) -> i64 {
    -ENOSYS
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod raw {
    use super::{
        iovec, mmsghdr, msghdr, sockaddr_in, Op, Probe, Send, AF_INET, SOCK_DGRAM, SOCK_STREAM,
    };
    use crate::tools::{syscall1, syscall3, syscall4, syscall5, syscall6};

    /// `AT_FDCWD`.
    const AT_FDCWD: u64 = -100i64 as u64;
    /// `SIGCHLD`: the flag of a `clone` that is a `fork`.
    const SIGCHLD: u64 = 17;
    /// `STATX_TYPE`.
    const STATX_TYPE: u64 = 1;

    /// A NUL terminated copy of `text`.
    fn c_string(text: &str) -> Vec<u8> {
        let mut bytes = text.as_bytes().to_vec();
        bytes.push(0);
        bytes
    }

    /// Close a descriptor that a call returned.
    fn close(descriptor: i64) {
        if descriptor >= 0 {
            // SAFETY: close takes the number of a descriptor and no pointer.
            let _ = unsafe { syscall1(3, descriptor as u64) };
        }
    }

    /// Make a socket and return its descriptor, or the error.
    fn socket(domain: u64, kind: u64) -> i64 {
        // SAFETY: socket takes three numbers and no pointer.
        unsafe { syscall3(41, domain, kind, 0) }
    }

    /// The child of a `fork` leaves at once, and the parent waits for it.
    fn settle_child(pid: i64) -> i64 {
        if pid == 0 {
            loop {
                // SAFETY: exit_group takes a status and never returns.
                let _ = unsafe { syscall1(231, 0) };
            }
        }
        if pid > 0 {
            // SAFETY: wait4 takes a pid, a null status, no options and a null
            // usage; it reaps the child that exited.
            let _ = unsafe { syscall4(61, pid as u64, 0, 0, 0) };
        }
        pid
    }

    /// Leave the process with `status`.
    fn exit(status: u64) -> ! {
        loop {
            // SAFETY: exit_group takes a status and never returns.
            let _ = unsafe { syscall1(231, status) };
        }
    }

    /// Make the call of `op`, and the calls that can replace or multiply the
    /// process in a child of their own.
    ///
    /// A call of `execve` that works replaces the process with another program,
    /// and the lines of the cases after it would never be printed. The child
    /// makes the call and leaves with what it came to: zero when it worked, or
    /// the error number when it did not, which never exceeds 255. A kernel with
    /// no `fork` cannot isolate the call, and the call is made here: the table
    /// of such a kernel has no `execve` to take the process away either, and a
    /// table that is cut short has no summary line, which the caller requires.
    pub fn call(op: &Op) -> i64 {
        if !matches!(op, Op::Exec { .. } | Op::Fork | Op::Clone) {
            return direct(op);
        }
        // SAFETY: fork takes no argument.
        let pid = unsafe { syscall1(57, 0) };
        if pid < 0 {
            return direct(op);
        }
        if pid == 0 {
            let result = direct(op);
            exit(if result >= 0 {
                0
            } else {
                result.unsigned_abs() & 0xff
            });
        }
        let mut status = 0i32;
        // SAFETY: wait4 writes the status of the child into `status`, which is
        // live, and takes no options and a null usage.
        let waited = unsafe { syscall4(61, pid as u64, &mut status as *mut i32 as u64, 0, 0) };
        if waited < 0 {
            return waited;
        }
        match (status & 0x7f, (status >> 8) & 0xff) {
            (0, 0) => 0,
            (0, code) => -i64::from(code),
            // A child that a signal ended is not an error number: it is below
            // every one.
            (signal, _) => -1000 - i64::from(signal),
        }
    }

    fn direct(op: &Op) -> i64 {
        match op {
            Op::Open { path, flags } => {
                let path = c_string(path);
                // SAFETY: openat reads the NUL terminated `path`, which is live.
                let result =
                    unsafe { syscall4(257, AT_FDCWD, path.as_ptr() as u64, *flags, 0o600) };
                close(result);
                result
            }
            Op::Creat { path } => {
                let path = c_string(path);
                // SAFETY: creat reads the NUL terminated `path`.
                let result = unsafe { syscall3(85, path.as_ptr() as u64, 0o600, 0) };
                close(result);
                result
            }
            Op::Unlink { path } => {
                let path = c_string(path);
                // SAFETY: unlink reads the NUL terminated `path`.
                unsafe { syscall1(87, path.as_ptr() as u64) }
            }
            Op::Rename { from, to } => {
                let (from, to) = (c_string(from), c_string(to));
                // SAFETY: rename reads two NUL terminated paths, both live.
                unsafe { syscall3(82, from.as_ptr() as u64, to.as_ptr() as u64, 0) }
            }
            Op::Mkdir { path } => {
                let path = c_string(path);
                // SAFETY: mkdir reads the NUL terminated `path`.
                unsafe { syscall3(83, path.as_ptr() as u64, 0o700, 0) }
            }
            Op::Symlink { target, link } => {
                let (target, link) = (c_string(target), c_string(link));
                // SAFETY: symlink reads two NUL terminated paths, both live.
                unsafe { syscall3(88, target.as_ptr() as u64, link.as_ptr() as u64, 0) }
            }
            Op::Readlink { path } => {
                let path = c_string(path);
                let mut buffer = [0u8; 256];
                // SAFETY: readlink reads the path and writes at most 256 bytes
                // into `buffer`, which holds that many.
                unsafe { syscall3(89, path.as_ptr() as u64, buffer.as_mut_ptr() as u64, 256) }
            }
            Op::Chdir { path } => {
                let path = c_string(path);
                // SAFETY: chdir reads the NUL terminated `path`.
                unsafe { syscall1(80, path.as_ptr() as u64) }
            }
            Op::Probe { kind, path } => {
                let path = c_string(path);
                let mut buffer = [0u8; 256];
                let at = path.as_ptr() as u64;
                let out = buffer.as_mut_ptr() as u64;
                // SAFETY: each call reads the NUL terminated path and writes at
                // most the 256 bytes of `buffer`, which holds a `struct stat`
                // (144 bytes) and a `struct statx` (256 bytes).
                unsafe {
                    match kind {
                        Probe::Stat => syscall3(4, at, out, 0),
                        Probe::Newfstatat => syscall4(262, AT_FDCWD, at, out, 0),
                        Probe::Statx => syscall5(332, AT_FDCWD, at, 0, STATX_TYPE, out),
                        Probe::Access => syscall3(21, at, 0, 0),
                        Probe::Faccessat => syscall3(269, AT_FDCWD, at, 0),
                    }
                }
            }
            Op::Socket { domain, kind } => {
                let result = socket(*domain, *kind);
                close(result);
                result
            }
            Op::Connect {
                kind,
                ip,
                port,
                text,
            } => {
                let descriptor = socket(AF_INET, *kind);
                if descriptor < 0 {
                    return descriptor;
                }
                let address = sockaddr_in(*ip, *port);
                // SAFETY: connect reads the 16 bytes of `address`.
                let result =
                    unsafe { syscall3(42, descriptor as u64, address.as_ptr() as u64, 16) };
                if result == 0 {
                    // The text names the case, so a server that records it can
                    // say that the connection reached it.
                    let message = if *kind == SOCK_STREAM {
                        format!("GET /{text} HTTP/1.1\r\n\r\n")
                    } else {
                        text.clone()
                    };
                    // SAFETY: write reads `message.len()` bytes of `message`.
                    let _ = unsafe {
                        syscall3(
                            1,
                            descriptor as u64,
                            message.as_ptr() as u64,
                            message.len() as u64,
                        )
                    };
                }
                close(descriptor);
                result
            }
            Op::Send {
                kind,
                ip,
                port,
                text,
            } => {
                let descriptor = socket(AF_INET, SOCK_DGRAM);
                if descriptor < 0 {
                    return descriptor;
                }
                let address = sockaddr_in(*ip, *port);
                let data = text.as_bytes();
                let vector = iovec(data.as_ptr() as u64, data.len());
                let header = msghdr(address.as_ptr() as u64, 16, vector.as_ptr() as u64, 1);
                let batch = mmsghdr(header);
                // SAFETY: each call reads the structures it is given: the
                // address, the bytes of `data`, the iovec that points at them
                // and the header that points at both. All of them are live
                // until the call returns, and `batch` is the 64 bytes of one
                // `mmsghdr`.
                let result = unsafe {
                    match kind {
                        Send::Sendto => syscall6(
                            44,
                            descriptor as u64,
                            data.as_ptr() as u64,
                            data.len() as u64,
                            0,
                            address.as_ptr() as u64,
                            16,
                        ),
                        Send::Sendmsg => syscall3(46, descriptor as u64, header.as_ptr() as u64, 0),
                        Send::Sendmmsg => {
                            syscall4(307, descriptor as u64, batch.as_ptr() as u64, 1, 0)
                        }
                    }
                };
                close(descriptor);
                result
            }
            Op::Bind => {
                let descriptor = socket(AF_INET, SOCK_STREAM);
                if descriptor < 0 {
                    return descriptor;
                }
                let address = sockaddr_in([0; 4], 0);
                // SAFETY: bind reads the 16 bytes of `address`.
                let result =
                    unsafe { syscall3(49, descriptor as u64, address.as_ptr() as u64, 16) };
                close(descriptor);
                result
            }
            Op::Listen => {
                let descriptor = socket(AF_INET, SOCK_STREAM);
                if descriptor < 0 {
                    return descriptor;
                }
                // SAFETY: listen takes a descriptor and a backlog.
                let result = unsafe { syscall3(50, descriptor as u64, 1, 0) };
                close(descriptor);
                result
            }
            Op::Accept => {
                let descriptor = socket(AF_INET, SOCK_STREAM);
                if descriptor < 0 {
                    return descriptor;
                }
                // SAFETY: accept takes a descriptor and two null pointers. The
                // socket does not listen, so the call fails at once.
                let result = unsafe { syscall3(43, descriptor as u64, 0, 0) };
                close(descriptor);
                result
            }
            Op::Exec { path, at } => {
                let path = c_string(path);
                let argv = [path.as_ptr() as u64, 0];
                let envp = [0u64];
                // SAFETY: the call reads the path and the two arrays of
                // pointers, which end with a null. A call that works replaces
                // the process with a program that exits at once.
                unsafe {
                    if *at {
                        syscall5(
                            322,
                            AT_FDCWD,
                            path.as_ptr() as u64,
                            argv.as_ptr() as u64,
                            envp.as_ptr() as u64,
                            0,
                        )
                    } else {
                        syscall3(
                            59,
                            path.as_ptr() as u64,
                            argv.as_ptr() as u64,
                            envp.as_ptr() as u64,
                        )
                    }
                }
            }
            // SAFETY: fork takes no argument. The child leaves at once.
            Op::Fork => settle_child(unsafe { syscall1(57, 0) }),
            // SAFETY: clone with `SIGCHLD` and no other flag is a fork, and the
            // child leaves at once.
            Op::Clone => settle_child(unsafe { syscall5(56, SIGCHLD, 0, 0, 0, 0) }),
            Op::Memfd => {
                let name = c_string("adversary");
                // SAFETY: memfd_create reads the NUL terminated `name`.
                let result = unsafe { syscall3(319, name.as_ptr() as u64, 0, 0) };
                close(result);
                result
            }
            Op::IoUring => {
                let mut parameters = [0u8; 120];
                // SAFETY: io_uring_setup reads and writes the 120 bytes of
                // `struct io_uring_params`, which `parameters` holds.
                let result = unsafe { syscall3(425, 1, parameters.as_mut_ptr() as u64, 0) };
                close(result);
                result
            }
        }
    }
}

/// What one case came to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Outcome {
    /// What the call returned.
    pub ret: i64,
    /// What that means.
    pub verdict: Raw,
}

/// Make the call of `case`.
pub fn run_case(case: &Case) -> Outcome {
    let ret = call(&case.op);
    Outcome {
        ret,
        verdict: classify(ret, None),
    }
}

/// The line of one case. `reached` is `na` where nobody counts what arrived.
pub fn case_line(case: &Case, outcome: &Outcome, reached: Option<bool>) -> String {
    format!(
        "RAW case={} syscall={} ret={} verdict={} reached={}",
        case.name,
        case.syscall,
        outcome.ret,
        outcome.verdict.as_str(),
        match reached {
            Some(true) => "yes",
            Some(false) => "no",
            None => "na",
        }
    )
}

/// The line that says whether a probe tells a missing file from one that is
/// there. `oracle` is `yes` when the two answers have different verdicts.
pub fn existence_line(syscall: &str, outside: &Outcome, missing: &Outcome) -> String {
    format!(
        "RAW existence syscall={syscall} outside={} missing={} oracle={}",
        outside.ret,
        missing.ret,
        if outside.verdict == missing.verdict {
            "no"
        } else {
            "yes"
        }
    )
}

/// The existence lines of a run: for each probe, the answer about the secret
/// and the answer about a file that is not there.
pub fn existence_lines(results: &[(Case, Outcome)]) -> Vec<String> {
    let outcome = |name: &str| {
        results
            .iter()
            .find(|(case, _)| case.name == name)
            .map(|(_, outcome)| outcome)
    };
    PROBES
        .iter()
        .filter_map(|(syscall, _)| {
            let outside = outcome(&format!("{syscall}_secret"))?;
            let missing = outcome(&format!("{syscall}_missing"))?;
            Some(existence_line(syscall, outside, missing))
        })
        .collect()
}

/// The known gaps of a run: the cases whose answer is not asserted on Rabbit
/// because the kernel does not check them, and that were allowed.
pub fn gap_lines(results: &[(Case, Outcome)]) -> Vec<String> {
    results
        .iter()
        .filter(|(case, outcome)| case.rabbit == Expect::Any && outcome.verdict == Raw::Allowed)
        .map(|(case, outcome)| {
            format!(
                "GAP case={} syscall={} ret={} why={}",
                case.name,
                case.syscall,
                outcome.ret,
                case.gap.unwrap_or("not asserted")
            )
        })
        .collect()
}

/// The summary line.
pub fn summary_line(results: &[(Case, Outcome)], kernel: &str) -> String {
    let count = |wanted| {
        results
            .iter()
            .filter(|(_, outcome)| outcome.verdict == wanted)
            .count()
    };
    format!(
        "RAW summary cases={} allowed={} denied={} unavailable={} failed={} kernel={kernel}",
        results.len(),
        count(Raw::Allowed),
        count(Raw::Denied),
        count(Raw::Unavailable),
        count(Raw::Failed),
    )
}

#[cfg(test)]
mod tests {
    use rabbit_policy::Verdict as Decision;

    use super::*;

    fn setup() -> Setup {
        Setup::rabbit([10, 0, 2, 2], 40000, 40001)
    }

    #[test]
    fn the_classifier_separates_a_policy_from_a_missing_call() {
        assert_eq!(classify(0, None), Raw::Allowed);
        assert_eq!(classify(7, None), Raw::Allowed);
        assert_eq!(classify(-13, None), Raw::Denied);
        assert_eq!(classify(-1, None), Raw::Denied);
        assert_eq!(classify(-38, None), Raw::Unavailable);
        assert_eq!(classify(-2, None), Raw::Failed);
        assert_eq!(classify(-97, None), Raw::Failed);
        assert_eq!(classify(-111, None), Raw::Failed);
        // Data that reached a server was let through, whatever came back.
        assert_eq!(classify(-111, Some(true)), Raw::Allowed);
        assert_eq!(classify(-13, Some(false)), Raw::Denied);
    }

    #[test]
    fn an_expectation_says_what_it_accepts() {
        assert!(Expect::Is(Raw::Denied).holds(Raw::Denied));
        assert!(!Expect::Is(Raw::Denied).holds(Raw::Unavailable));
        assert!(Expect::NotAllowed.holds(Raw::Failed));
        assert!(Expect::NotAllowed.holds(Raw::Denied));
        assert!(!Expect::NotAllowed.holds(Raw::Allowed));
        assert!(Expect::Any.holds(Raw::Allowed));
    }

    #[test]
    fn the_table_is_well_formed() {
        let table = cases(&setup(), true);
        let mut names: Vec<&str> = table.iter().map(|case| case.name.as_str()).collect();
        names.sort_unstable();
        let total = names.len();
        names.dedup();
        assert_eq!(names.len(), total, "the names are unique");
        for case in &table {
            let listed = SYSCALLS
                .iter()
                .find(|(name, _)| *name == case.syscall)
                .unwrap_or_else(|| panic!("{} is not in the list of calls", case.syscall));
            assert_eq!(case.number, listed.1, "{}", case.name);
            assert_ne!(case.number, 0, "{}", case.name);
        }
        let mut numbers: Vec<u64> = SYSCALLS.iter().map(|(_, number)| *number).collect();
        numbers.sort_unstable();
        numbers.dedup();
        assert_eq!(numbers.len(), SYSCALLS.len(), "no two calls share a number");
    }

    #[test]
    fn nothing_that_could_harm_the_machine_is_in_the_table() {
        for forbidden in [
            "kill", "tgkill", "ptrace", "mount", "umount2", "reboot", "vfork",
        ] {
            assert!(
                cases(&setup(), true)
                    .iter()
                    .all(|case| case.syscall != forbidden),
                "{forbidden}"
            );
        }
        // Every call that changes a name is on a path under the read root.
        for case in cases(&setup(), true) {
            let paths: Vec<&String> = match &case.op {
                Op::Creat { path } | Op::Unlink { path } | Op::Mkdir { path } => vec![path],
                Op::Rename { from, to } => vec![from, to],
                Op::Symlink { link, .. } => vec![link],
                Op::Open { path, flags } if *flags != O_RDONLY => vec![path],
                _ => vec![],
            };
            for path in paths {
                assert!(path.starts_with("/agent/"), "{}: {path}", case.name);
            }
        }
    }

    #[test]
    fn the_processes_that_replace_or_split_are_last() {
        let table = cases(&setup(), true);
        let names: Vec<&str> = table.iter().map(|case| case.name.as_str()).collect();
        let tail = &names[names.len() - 4..];
        assert_eq!(
            tail,
            [
                "fork",
                "clone_no_thread",
                "execve_outside",
                "execveat_outside"
            ]
        );
        assert_eq!(names[0], "open_inside");
    }

    #[test]
    fn the_buffers_have_the_layout_of_the_kernel() {
        let address = sockaddr_in([10, 0, 2, 3], 0x1234);
        assert_eq!(address.len(), 16);
        assert_eq!(&address[..2], &2u16.to_ne_bytes());
        assert_eq!(&address[2..4], &[0x12, 0x34]);
        assert_eq!(&address[4..8], &[10, 0, 2, 3]);
        assert_eq!(&address[8..], &[0; 8]);

        let vector = iovec(0x1000, 7);
        assert_eq!(vector.len(), IOVEC_LEN);
        assert_eq!(&vector[..8], &0x1000u64.to_ne_bytes());
        assert_eq!(&vector[8..], &7u64.to_ne_bytes());

        let header = msghdr(0x2000, 16, 0x3000, 1);
        assert_eq!(header.len(), 56);
        assert_eq!(&header[..8], &0x2000u64.to_ne_bytes());
        assert_eq!(&header[8..12], &16u32.to_ne_bytes());
        assert_eq!(&header[12..16], &[0; 4]);
        assert_eq!(&header[16..24], &0x3000u64.to_ne_bytes());
        assert_eq!(&header[24..32], &1u64.to_ne_bytes());
        assert_eq!(&header[32..], &[0; 24]);

        let batch = mmsghdr(header);
        assert_eq!(batch.len(), 64);
        assert_eq!(&batch[..56], &header);
        assert_eq!(&batch[56..], &[0; 8]);
        assert_eq!(MSGHDR_LEN + 8, MMSGHDR_LEN);
    }

    #[test]
    fn the_expectations_of_rabbit_agree_with_the_policy_engine() {
        use rabbit_policy::Protocol;

        let setup = setup();
        let policy = setup.manifest().unwrap();
        let expected = |allowed: bool| {
            if allowed {
                Expect::Is(Raw::Allowed)
            } else {
                Expect::Is(Raw::Denied)
            }
        };
        for case in cases(&setup, true) {
            match &case.op {
                Op::Open { path, flags } => {
                    let allowed =
                        policy.check_open(path.as_bytes(), *flags != O_RDONLY) == Decision::Allow;
                    assert_eq!(case.rabbit, expected(allowed), "{}", case.name);
                }
                Op::Probe {
                    kind: Probe::Newfstatat,
                    path,
                } => {
                    // The kernel checks the path before it looks for it, so a
                    // path that is not there is refused like one that is.
                    assert_ne!(policy.check_open(path.as_bytes(), false), Decision::Allow);
                    assert_eq!(case.rabbit, Expect::Is(Raw::Denied), "{}", case.name);
                }
                Op::Connect { kind, ip, port, .. } => {
                    let protocol = if *kind == SOCK_STREAM {
                        Protocol::Tcp
                    } else {
                        Protocol::Udp
                    };
                    let allowed = policy.check_connect(protocol, *ip, *port) == Decision::Allow;
                    assert_eq!(case.rabbit, expected(allowed), "{}", case.name);
                }
                // Rabbit has no `sendmsg` or `sendmmsg`, and checks `sendto`.
                Op::Send {
                    kind: Send::Sendto,
                    ip,
                    port,
                    ..
                } => {
                    let allowed =
                        policy.check_connect(Protocol::Udp, *ip, *port) == Decision::Allow;
                    assert_eq!(case.rabbit, expected(allowed), "{}", case.name);
                }
                _ => {}
            }
        }
    }

    #[test]
    fn the_calls_rabbit_does_not_have_answer_unavailable() {
        // The arms of the dispatcher of the kernel: these have none.
        for name in [
            "creat",
            "unlink",
            "rename",
            "mkdir",
            "symlink",
            "readlink",
            "chdir",
            "stat",
            "access",
            "faccessat",
            "statx",
            "sendmsg",
            "sendmmsg",
            "memfd_create",
            "io_uring_setup",
            "fork",
            "execve",
            "execveat",
        ] {
            for case in cases(&setup(), true)
                .iter()
                .filter(|case| case.syscall == name)
            {
                assert_eq!(case.rabbit, Expect::Is(Raw::Unavailable), "{}", case.name);
            }
        }
    }

    #[test]
    fn the_linux_expectations_follow_the_ports_the_kernel_enforces() {
        let mut setup = setup();
        let find =
            |table: &[Case], name: &str| table.iter().find(|case| case.name == name).unwrap().linux;
        let table = cases(&setup, false);
        assert_eq!(
            find(&table, "connect_tcp_other_port"),
            Expect::Is(Raw::Allowed)
        );
        assert_eq!(
            find(&table, "connect_udp_other_port"),
            Expect::Is(Raw::Allowed)
        );
        setup.linux_udp_ports = true;
        let table = cases(&setup, true);
        assert_eq!(
            find(&table, "connect_tcp_other_port"),
            Expect::Is(Raw::Denied)
        );
        assert_eq!(
            find(&table, "connect_udp_other_port"),
            Expect::Is(Raw::Denied)
        );
        assert_eq!(
            find(&table, "connect_tcp_other_ip"),
            Expect::Is(Raw::Allowed)
        );
    }

    #[test]
    fn the_table_has_the_counts_the_documents_give() {
        // docs/ARCHITECTURE.md says how many of the cases the policy refuses
        // and how many are calls the kernel does not have.
        let table = cases(&setup(), false);
        let count = |expected| table.iter().filter(|case| case.rabbit == expected).count();
        assert_eq!(table.len(), 43);
        assert_eq!(count(Expect::Is(Raw::Denied)), 11);
        assert_eq!(count(Expect::Is(Raw::Unavailable)), 22);
        assert_eq!(count(Expect::Is(Raw::Allowed)), 6);
        assert_eq!(count(Expect::Is(Raw::Failed)), 3);
        assert_eq!(count(Expect::Any), 1);
    }

    #[test]
    fn a_call_that_is_not_checked_is_printed_as_a_gap() {
        let table = cases(&setup(), false);
        let bind = table.iter().find(|case| case.name == "bind_any").unwrap();
        let allowed = Outcome {
            ret: 0,
            verdict: Raw::Allowed,
        };
        let results = vec![(bind.clone(), allowed)];
        assert_eq!(
            gap_lines(&results),
            ["GAP case=bind_any syscall=bind ret=0 why=neither the policy nor the filter checks bind"]
        );
        let refused = Outcome {
            ret: -13,
            verdict: Raw::Denied,
        };
        assert!(gap_lines(&[(bind.clone(), refused)]).is_empty());
    }

    #[test]
    fn the_lines_have_the_format_the_greps_read() {
        let table = cases(&setup(), true);
        let case = &table[1];
        let denied = Outcome {
            ret: -13,
            verdict: Raw::Denied,
        };
        assert_eq!(
            case_line(case, &denied, None),
            "RAW case=open_secret syscall=openat ret=-13 verdict=denied reached=na"
        );
        assert_eq!(
            case_line(case, &denied, Some(false)),
            "RAW case=open_secret syscall=openat ret=-13 verdict=denied reached=no"
        );
        let missing = Outcome {
            ret: -2,
            verdict: Raw::Failed,
        };
        assert_eq!(
            existence_line("newfstatat", &denied, &denied),
            "RAW existence syscall=newfstatat outside=-13 missing=-13 oracle=no"
        );
        assert_eq!(
            existence_line(
                "stat",
                &Outcome {
                    ret: 0,
                    verdict: Raw::Allowed
                },
                &missing
            ),
            "RAW existence syscall=stat outside=0 missing=-2 oracle=yes"
        );
        let results = vec![
            (case.clone(), denied),
            (
                table[0].clone(),
                Outcome {
                    ret: 3,
                    verdict: Raw::Allowed,
                },
            ),
        ];
        assert_eq!(
            summary_line(&results, "rabbit"),
            "RAW summary cases=2 allowed=1 denied=1 unavailable=0 failed=0 kernel=rabbit"
        );
    }

    #[test]
    fn every_probe_has_an_existence_line() {
        let results: Vec<(Case, Outcome)> = cases(&setup(), true)
            .into_iter()
            .map(|case| {
                let outcome = Outcome {
                    ret: -13,
                    verdict: Raw::Denied,
                };
                (case, outcome)
            })
            .collect();
        let lines = existence_lines(&results);
        assert_eq!(lines.len(), 5);
        assert_eq!(
            lines[0],
            "RAW existence syscall=newfstatat outside=-13 missing=-13 oracle=no"
        );
    }

    #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
    #[test]
    fn a_host_without_raw_calls_answers_unavailable() {
        let case = &cases(&setup(), true)[0];
        assert_eq!(run_case(case).verdict, Raw::Unavailable);
    }

    /// The `ret` a child printed for the case `name`.
    fn printed_ret(output: &str, name: &str) -> Option<i64> {
        let line =
            crate::adversary::tests::marked_line(output, &format!("RAW case={name} syscall="))?;
        line.split(' ')
            .find_map(|word| word.strip_prefix("ret="))?
            .parse()
            .ok()
    }

    #[test]
    fn the_return_of_a_case_is_read_with_the_prefix_of_libtest_before_it() {
        let output = "test x ... RAW case=open_inside syscall=openat ret=3 verdict=allowed reached=na\nRAW case=open_secret syscall=openat ret=-13 verdict=denied reached=na\n";
        assert_eq!(printed_ret(output, "open_inside"), Some(3));
        assert_eq!(printed_ret(output, "open_secret"), Some(-13));
        assert_eq!(printed_ret(output, "open"), None);
        assert_eq!(printed_ret(output, "fork"), None);
    }

    /// The child gets its directory, the four ports and whether `127.0.0.2` is
    /// routable in this variable.
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    const CHILD: &str = "RABBIT_COMPROMISED_CHILD";

    /// The cases that the filter decides, which a kernel without seccomp does
    /// not refuse.
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    const SECCOMP_CASES: [&str; 5] = [
        "socket_unix",
        "socket_packet",
        "listen_stream",
        "accept_stream",
        "io_uring_setup",
    ];

    /// Landlock and seccomp cannot be undone, so the raw calls run in a child:
    /// the test runs again as a process of its own, restricted by the manifest.
    /// The parent made the tree and the listeners before that, reads what the
    /// child printed, and judges it: a call that sent data is allowed when the
    /// data reached a server, whatever the call returned.
    ///
    /// What is asserted is what the documentation of Landlock and of the filter
    /// says; the rest is printed. A call whose answer is `Any` never fails the
    /// test, so a kernel that answers differently does not break it.
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn the_compromised_binary_is_bounded_on_linux() {
        use std::process::Command;

        use crate::adversary::tests::{linux_setup, marked_line, Servers};
        use crate::scratch::{run_child, Scratch};

        if let Ok(parameters) = std::env::var(CHILD) {
            child(&parameters);
            return;
        }

        let scratch = Scratch::new("rabbit-compromised");
        let directory = scratch.path();
        std::fs::create_dir_all(format!("{directory}/agent")).unwrap();
        std::fs::write(format!("{directory}/agent/hello.txt"), "hello").unwrap();
        std::fs::write(format!("{directory}/secret.txt"), "secret").unwrap();
        std::os::unix::fs::symlink(
            format!("{directory}/secret.txt"),
            format!("{directory}/agent/link.txt"),
        )
        .unwrap();
        let servers = Servers::start();
        let [udp, other_udp, tcp, other_tcp] = servers.ports;

        let routable = std::net::TcpStream::connect_timeout(
            &std::net::SocketAddr::from(([127, 0, 0, 2], tcp)),
            std::time::Duration::from_millis(500),
        )
        .is_ok();
        if !routable {
            eprintln!("SKIP reason=127.0.0.2 not routable");
        }

        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "compromised::tests::the_compromised_binary_is_bounded_on_linux",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(
                CHILD,
                format!(
                    "{directory},{udp},{other_udp},{tcp},{other_tcp},{}",
                    u8::from(routable)
                ),
            );
        let output = run_child(command, 60);
        println!("{}", output.stdout);
        eprintln!("{}", output.stderr);
        let everything = format!("{}{}", output.stdout, output.stderr);
        assert!(
            !output.timed_out,
            "the child ran out of time:\n{everything}"
        );
        assert!(output.success, "the child failed:\n{everything}");
        assert!(
            marked_line(&output.stdout, "CHILD DONE").is_some(),
            "the child did not finish:\n{everything}"
        );
        let line = marked_line(&output.stdout, "CHILD AGENT policy").unwrap_or_else(|| {
            panic!("the child printed no `CHILD AGENT policy` line:\n{everything}")
        });
        if !line.contains(" fs=on") {
            return;
        }

        let setup = linux_setup(directory, servers.ports, line.contains("udp-port=on"));
        let seccomp = line.contains("seccomp=on");
        let table = cases(&setup, line.contains("tcp-port=on"));
        // A table that a call cut short has no summary, or counts fewer cases.
        let summary = marked_line(&output.stdout, "RAW summary cases=")
            .unwrap_or_else(|| panic!("the child printed no `RAW summary` line:\n{everything}"));
        assert!(
            summary.starts_with(&format!("RAW summary cases={} ", table.len())),
            "the table has {} cases and the summary says: {summary}",
            table.len()
        );
        let mut results = Vec::new();
        for case in table {
            let ret = printed_ret(&output.stdout, &case.name)
                .unwrap_or_else(|| panic!("the child printed no line for {}", case.name));
            let reached = case.sends.then(|| servers.reached(&case.name));
            let outcome = Outcome {
                ret,
                verdict: classify(ret, reached),
            };
            if case.sends {
                println!(
                    "RAW net case={} syscall={} ret={ret} verdict={} reached={}",
                    case.name,
                    case.syscall,
                    outcome.verdict.as_str(),
                    if reached == Some(true) { "yes" } else { "no" }
                );
            }
            // What the kernel is not asked about, or cannot answer, is skipped
            // out loud.
            if case.name.ends_with("_other_ip") && !routable {
                eprintln!("SKIP {} reason=127.0.0.2 not routable", case.name);
            } else if !seccomp && SECCOMP_CASES.contains(&case.name.as_str()) {
                eprintln!("SKIP {} reason=the kernel has no seccomp", case.name);
            } else {
                assert!(
                    case.linux.holds(outcome.verdict),
                    "{}: {:?} was expected and the call came to {} (ret={ret})",
                    case.name,
                    case.linux,
                    outcome.verdict.as_str()
                );
            }
            // A gap is a call that Rabbit refuses or does not have and that
            // Linux let through.
            if outcome.verdict == Raw::Allowed && case.rabbit != Expect::Is(Raw::Allowed) {
                println!(
                    "GAP case={} rabbit={:?} linux={} reached={} why={}",
                    case.name,
                    case.rabbit,
                    outcome.verdict.as_str(),
                    match reached {
                        Some(true) => "yes",
                        Some(false) => "no",
                        None => "na",
                    },
                    case.gap.unwrap_or("not asserted"),
                );
            }
            results.push((case, outcome));
        }
        for line in existence_lines(&results) {
            println!("{line}");
        }
    }

    /// The part that runs restricted: apply the manifest, make every call, and
    /// print a line for each as it is made. A call that replaces the process
    /// leaves the lines before it, and the parent fails the test for the line
    /// that never comes.
    ///
    /// A kernel that reports Landlock and does not enforce it fails the test,
    /// and so does a kernel with no Landlock when `RABBIT_REQUIRE_LANDLOCK=1`;
    /// otherwise a kernel with no Landlock skips the calls, on stderr.
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    fn child(parameters: &str) {
        use crate::adversary::tests::linux_setup;

        // libtest may have left `test name ... ` on this line.
        println!();
        let fields: Vec<&str> = parameters.split(',').collect();
        let port = |index: usize| fields[index].parse::<u16>().unwrap();
        let directory = fields[0];
        let mut setup = linux_setup(directory, [port(1), port(2), port(3), port(4)], false);
        let policy = setup.manifest().unwrap();

        let applied = crate::linux_policy::apply(&policy);
        let line = applied.line();
        println!("CHILD {line}");
        assert!(applied.no_new_privs, "no_new_privs was refused: {line}");
        setup.linux_udp_ports = applied.udp_port;

        if !applied.fs {
            assert_eq!(
                applied.landlock_abi, 0,
                "the kernel reports Landlock but it is not in force: {line}"
            );
            assert!(
                std::env::var_os("RABBIT_REQUIRE_LANDLOCK").is_none_or(|value| value != "1"),
                "RABBIT_REQUIRE_LANDLOCK=1 and the kernel has no Landlock: {line}"
            );
            eprintln!("SKIP landlock unavailable: the kernel reports no Landlock ABI: {line}");
            println!("CHILD DONE");
            return;
        }

        let mut results = Vec::new();
        for case in cases(&setup, applied.tcp_port) {
            let outcome = run_case(&case);
            println!("{}", case_line(&case, &outcome, None));
            results.push((case, outcome));
        }
        for line in existence_lines(&results) {
            println!("{line}");
        }
        println!("{}", summary_line(&results, "linux"));
        println!("CHILD DONE");
    }
}
