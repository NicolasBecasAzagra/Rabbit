// SPDX-License-Identifier: GPL-2.0-only
//! The manifest on a Linux kernel: Landlock and seccomp, as far as they go.
//!
//! Rabbit enforces the manifest in the kernel with a call of its own. A Linux
//! kernel answers `ENOSYS` to that call, and [`apply`] then restricts the
//! process with what Linux has, from inside the agent: `no_new_privs`, a
//! Landlock ruleset for the files and the ports, and a seccomp filter for the
//! types of socket and for the inbound calls. The rules are those of
//! [`crate::landlock`] and [`crate::seccomp`].
//!
//! Linux cannot express all of the manifest, and the [`Report`] says so. The
//! address of an endpoint is not enforced: Landlock keys its network rules by
//! port, and seccomp does not follow the pointer to the address that `connect`
//! and `sendto` take. A port of the manifest is therefore open to every
//! address. Every step that the kernel refuses is reported and is not fatal:
//! the agent runs with whatever protection the kernel could give it.
//!
//! UDP ports are restricted only from Landlock ABI 10: the right is
//! `LANDLOCK_ACCESS_NET_CONNECT_SEND_UDP`, `1ULL << 3` in
//! `include/uapi/linux/landlock.h`, documented there as "Support added in
//! Landlock ABI version 10". On an older kernel the report says `udp-port=off`.
//!
//! The values come from the public headers of the kernel: the calls from
//! `arch/x86/entry/syscalls/syscall_64.tbl`, `PR_SET_NO_NEW_PRIVS` from
//! `include/uapi/linux/prctl.h`, `O_PATH` from `include/uapi/asm-generic/fcntl.h`,
//! and the seccomp and Landlock constants from their own headers.

/// A step that the kernel refused: which one, and the error number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Failure {
    /// The step that failed.
    pub step: &'static str,
    /// The error number the kernel answered, positive.
    pub errno: i64,
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}(errno={})", self.step, self.errno)
    }
}

/// What [`apply`] managed to put in place.
///
/// A protection is reported only once the kernel has accepted the call that
/// puts it in force: a ruleset that was built and never restricted the
/// process is not a protection.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Report {
    /// The Landlock ABI version the kernel reports, or zero when it has none.
    pub landlock_abi: u32,
    /// Whether the files are restricted to the roots of the manifest.
    pub fs: bool,
    /// Whether TCP connections are restricted to the ports of the manifest.
    pub tcp_port: bool,
    /// Whether UDP datagrams are restricted to the ports of the manifest.
    pub udp_port: bool,
    /// Whether the Landlock ruleset covers every thread, not only the caller.
    pub landlock_all_threads: bool,
    /// The step that stopped Landlock, if one did.
    pub landlock_error: Option<Failure>,
    /// Whether the seccomp filter is in place.
    pub seccomp: bool,
    /// Whether the seccomp filter covers every thread, not only the caller.
    pub seccomp_all_threads: bool,
    /// The step that stopped seccomp, if one did.
    pub seccomp_error: Option<Failure>,
    /// Whether `no_new_privs` was set, which both of the others need.
    pub no_new_privs: bool,
    /// Roots of the manifest that could not be opened and got no rule.
    pub roots_skipped: usize,
}

fn on_off(active: bool) -> &'static str {
    if active {
        "on"
    } else {
        "off"
    }
}

impl Report {
    /// The line for the log. It starts the way the line of a kernel without
    /// the private call always did, and it says what is active and, always,
    /// that the address of an endpoint is not enforced.
    ///
    /// `threads` is `none` when nothing was applied, `all` when everything
    /// that was applied covers every thread, and `caller-only` otherwise.
    pub fn line(&self) -> String {
        let landlock = if self.landlock_abi == 0 {
            "unavailable".to_string()
        } else if self.fs {
            format!("abi{}", self.landlock_abi)
        } else {
            format!("abi{}-not-applied", self.landlock_abi)
        };
        let threads = if !self.fs && !self.seccomp {
            "none"
        } else if (!self.fs || self.landlock_all_threads)
            && (!self.seccomp || self.seccomp_all_threads)
        {
            "all"
        } else {
            "caller-only"
        };
        let mut line = format!(
            "AGENT policy kernel=without private syscall landlock={landlock} fs={} tcp-port={} udp-port={} seccomp={} threads={threads} ip=not-expressible",
            on_off(self.fs),
            on_off(self.tcp_port),
            on_off(self.udp_port),
            on_off(self.seccomp),
        );
        if !self.no_new_privs {
            line.push_str(" no-new-privs=off");
        }
        if self.roots_skipped != 0 {
            line.push_str(&format!(" roots-skipped={}", self.roots_skipped));
        }
        if let Some(error) = self.landlock_error {
            line.push_str(&format!(" landlock-error={error}"));
        }
        if let Some(error) = self.seccomp_error {
            line.push_str(&format!(" seccomp-error={error}"));
        }
        line
    }
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod raw {
    use std::fs::{File, OpenOptions};
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::OpenOptionsExt;
    use std::path::Path;

    use rabbit_policy::{Capabilities, Capability, Policy, Protocol};

    use super::{Failure, Report};
    use crate::landlock::{net_port_attr, path_beneath_attr, Support};
    use crate::seccomp::{self, PROGRAM_LEN};
    use crate::tools::{syscall3, syscall4, syscall5};

    /// `prctl` in the x86-64 table.
    const SYS_PRCTL: u64 = 157;
    /// `seccomp` in the x86-64 table.
    const SYS_SECCOMP: u64 = 317;
    /// `landlock_create_ruleset` in the x86-64 table.
    const SYS_LANDLOCK_CREATE_RULESET: u64 = 444;
    /// `landlock_add_rule` in the x86-64 table.
    const SYS_LANDLOCK_ADD_RULE: u64 = 445;
    /// `landlock_restrict_self` in the x86-64 table.
    const SYS_LANDLOCK_RESTRICT_SELF: u64 = 446;

    /// `PR_SET_NO_NEW_PRIVS`.
    const PR_SET_NO_NEW_PRIVS: u64 = 38;
    /// `SECCOMP_SET_MODE_FILTER`.
    const SECCOMP_SET_MODE_FILTER: u64 = 1;
    /// `SECCOMP_FILTER_FLAG_TSYNC`: install the filter on every thread.
    const SECCOMP_FILTER_FLAG_TSYNC: u64 = 1 << 0;
    /// `LANDLOCK_CREATE_RULESET_VERSION`: ask for the ABI version.
    const LANDLOCK_CREATE_RULESET_VERSION: u64 = 1 << 0;
    /// `LANDLOCK_RULE_PATH_BENEATH`.
    const LANDLOCK_RULE_PATH_BENEATH: u64 = 1;
    /// `LANDLOCK_RULE_NET_PORT`.
    const LANDLOCK_RULE_NET_PORT: u64 = 2;
    /// `O_PATH`, which is `1 << 21`: a descriptor that only names the file.
    const O_PATH: i32 = 1 << 21;

    /// Restrict the process to the manifest in `policy`, as far as the kernel
    /// lets it.
    pub fn apply(policy: &Policy) -> Report {
        let mut report = Report::default();
        // SAFETY: the call takes numbers only and no pointer.
        let no_new_privs = unsafe { syscall5(SYS_PRCTL, PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) } == 0;
        report.no_new_privs = no_new_privs;
        if no_new_privs {
            landlock(policy, &mut report);
            seccomp_filter(policy.granted(), &mut report);
        }
        report
    }

    /// The error number of a failed call.
    fn errno(result: i64) -> i64 {
        result.saturating_neg()
    }

    /// Put a Landlock ruleset in place, or say in the report which step
    /// stopped it.
    fn landlock(policy: &Policy, report: &mut Report) {
        // SAFETY: with a null attribute and the version flag the call reads no
        // memory.
        let abi = unsafe {
            syscall3(
                SYS_LANDLOCK_CREATE_RULESET,
                0,
                0,
                LANDLOCK_CREATE_RULESET_VERSION,
            )
        };
        let Ok(abi) = u32::try_from(abi) else {
            report.landlock_error = Some(Failure {
                step: "version",
                errno: errno(abi),
            });
            return;
        };
        report.landlock_abi = abi;
        if abi == 0 {
            return;
        }
        let support = Support::for_abi(abi);
        match build_ruleset(policy, &support, report) {
            Ok(ruleset) => restrict(&ruleset, &support, report),
            Err(failure) => report.landlock_error = Some(failure),
        }
    }

    /// Create the ruleset and add the rules of the manifest to it.
    fn build_ruleset(
        policy: &Policy,
        support: &Support,
        report: &mut Report,
    ) -> Result<OwnedFd, Failure> {
        let (attribute, size) = support.ruleset_attr();
        // SAFETY: the kernel reads `size` bytes of `attribute`, which is a
        // valid array of at least that many.
        let created = unsafe {
            syscall3(
                SYS_LANDLOCK_CREATE_RULESET,
                attribute.as_ptr() as u64,
                size as u64,
                0,
            )
        };
        let descriptor = i32::try_from(created)
            .ok()
            .filter(|descriptor| *descriptor >= 0)
            .ok_or(Failure {
                step: "create_ruleset",
                errno: errno(created),
            })?;
        // SAFETY: the call returned this descriptor and nothing else owns it.
        let ruleset = unsafe { OwnedFd::from_raw_fd(descriptor) };

        let granted = policy.granted();
        let roots = [
            (
                Capability::FsRead,
                policy.read_roots(),
                support.read_access(),
            ),
            (
                Capability::FsWrite,
                policy.write_roots(),
                support.write_access(),
            ),
        ];
        for (capability, list, access) in roots {
            if !granted.contains(capability) {
                continue;
            }
            for root in list {
                let Ok(directory) = open_path(root.as_bytes()) else {
                    report.roots_skipped += 1;
                    continue;
                };
                let attribute = path_beneath_attr(access, directory.as_raw_fd());
                add_rule(&ruleset, LANDLOCK_RULE_PATH_BENEATH, &attribute).map_err(|errno| {
                    Failure {
                        step: "add_rule_path",
                        errno,
                    }
                })?;
            }
        }
        for endpoint in policy.endpoints() {
            let (capability, access) = match endpoint.protocol {
                Protocol::Tcp => (Capability::NetTcp, support.tcp_access()),
                Protocol::Udp => (Capability::NetUdp, support.udp_access()),
            };
            if access == 0 || !granted.contains(capability) {
                continue;
            }
            let attribute = net_port_attr(access, endpoint.port);
            add_rule(&ruleset, LANDLOCK_RULE_NET_PORT, &attribute).map_err(|errno| Failure {
                step: "add_rule_port",
                errno,
            })?;
        }
        Ok(ruleset)
    }

    /// Enforce the ruleset, on every thread when the kernel can and on the
    /// caller alone when it cannot, and report what the kernel accepted.
    fn restrict(ruleset: &OwnedFd, support: &Support, report: &mut Report) {
        let all_threads = support.restrict_flags();
        let attempts: &[u64] = if all_threads != 0 {
            &[all_threads, 0]
        } else {
            &[0]
        };
        let mut failure = None;
        for &flags in attempts {
            // SAFETY: the call takes the descriptor of the ruleset and flags,
            // and no pointer.
            let restricted = unsafe {
                syscall3(
                    SYS_LANDLOCK_RESTRICT_SELF,
                    ruleset.as_raw_fd() as u64,
                    flags,
                    0,
                )
            };
            if restricted == 0 {
                report.fs = support.fs();
                report.tcp_port = support.tcp_port();
                report.udp_port = support.udp_port();
                report.landlock_all_threads = flags != 0;
                return;
            }
            failure = Some(Failure {
                step: "restrict_self",
                errno: errno(restricted),
            });
        }
        report.landlock_error = failure;
    }

    /// Open `path` as a descriptor that only names it, for a Landlock rule.
    fn open_path(path: &[u8]) -> std::io::Result<File> {
        OpenOptions::new()
            .read(true)
            .custom_flags(O_PATH)
            .open(Path::new(std::ffi::OsStr::from_bytes(path)))
    }

    /// Add one rule to the ruleset, or return the error number.
    fn add_rule(ruleset: &OwnedFd, rule_type: u64, attribute: &[u8]) -> Result<(), i64> {
        // SAFETY: the kernel reads the structure of the rule type from
        // `attribute`, which holds exactly the bytes of that structure.
        let added = unsafe {
            syscall4(
                SYS_LANDLOCK_ADD_RULE,
                ruleset.as_raw_fd() as u64,
                rule_type,
                attribute.as_ptr() as u64,
                0,
            )
        };
        if added == 0 {
            Ok(())
        } else {
            Err(errno(added))
        }
    }

    /// Put the seccomp filter in place, or say in the report why not. The
    /// filter goes to every thread when the kernel can do that, and to the
    /// caller alone when it cannot.
    fn seccomp_filter(granted: Capabilities, report: &mut Report) {
        let program: [[u8; 8]; PROGRAM_LEN] =
            seccomp::build(granted).map(|instruction| instruction.to_bytes());
        // `struct sock_fprog`: the count of instructions, padding up to the
        // alignment of a pointer, and the pointer.
        let mut fprog = [0u8; 16];
        fprog[..2].copy_from_slice(&(PROGRAM_LEN as u16).to_le_bytes());
        fprog[8..].copy_from_slice(&(program.as_ptr() as u64).to_le_bytes());

        for flags in [SECCOMP_FILTER_FLAG_TSYNC, 0] {
            // SAFETY: the kernel reads the `sock_fprog` in `fprog`, and the
            // `PROGRAM_LEN` instructions in `program` that it points to; both
            // live until the call returns.
            let installed = unsafe {
                syscall3(
                    SYS_SECCOMP,
                    SECCOMP_SET_MODE_FILTER,
                    flags,
                    fprog.as_ptr() as u64,
                )
            };
            if installed == 0 {
                report.seccomp = true;
                report.seccomp_all_threads = flags != 0;
                report.seccomp_error = None;
                return;
            }
            report.seccomp_error = Some(Failure {
                step: "seccomp",
                errno: errno(installed),
            });
        }
    }
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub use raw::apply;

#[cfg(test)]
mod tests {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    use crate::scratch::{run_child, Scratch};

    use super::{Failure, Report};

    fn full() -> Report {
        Report {
            landlock_abi: 4,
            fs: true,
            tcp_port: true,
            landlock_all_threads: true,
            seccomp: true,
            seccomp_all_threads: true,
            no_new_privs: true,
            ..Report::default()
        }
    }

    #[test]
    fn the_line_says_what_is_active_and_that_the_address_is_not() {
        assert_eq!(
            full().line(),
            "AGENT policy kernel=without private syscall landlock=abi4 fs=on tcp-port=on udp-port=off seccomp=on threads=all ip=not-expressible"
        );
    }

    #[test]
    fn nothing_applied_is_not_reported_as_every_thread() {
        let report = Report {
            no_new_privs: true,
            landlock_error: Some(Failure {
                step: "version",
                errno: 38,
            }),
            seccomp_error: Some(Failure {
                step: "seccomp",
                errno: 38,
            }),
            ..Report::default()
        };
        assert_eq!(
            report.line(),
            "AGENT policy kernel=without private syscall landlock=unavailable fs=off tcp-port=off udp-port=off seccomp=off threads=none ip=not-expressible landlock-error=version(errno=38) seccomp-error=seccomp(errno=38)"
        );
    }

    #[test]
    fn landlock_without_tsync_covers_the_caller_only() {
        let report = Report {
            landlock_abi: 4,
            fs: true,
            tcp_port: true,
            landlock_all_threads: false,
            no_new_privs: true,
            ..Report::default()
        };
        assert_eq!(
            report.line(),
            "AGENT policy kernel=without private syscall landlock=abi4 fs=on tcp-port=on udp-port=off seccomp=off threads=caller-only ip=not-expressible"
        );
        let seccomp_only = Report {
            seccomp_all_threads: false,
            ..full()
        };
        assert!(seccomp_only.line().contains("threads=caller-only"));
    }

    #[test]
    fn a_landlock_that_failed_at_add_rule_is_not_applied_and_says_why() {
        let report = Report {
            landlock_abi: 4,
            landlock_error: Some(Failure {
                step: "add_rule_path",
                errno: 13,
            }),
            seccomp: true,
            seccomp_all_threads: true,
            no_new_privs: true,
            ..Report::default()
        };
        assert_eq!(
            report.line(),
            "AGENT policy kernel=without private syscall landlock=abi4-not-applied fs=off tcp-port=off udp-port=off seccomp=on threads=all ip=not-expressible landlock-error=add_rule_path(errno=13)"
        );
    }

    #[test]
    fn a_refusal_of_no_new_privs_and_the_skipped_roots_are_in_the_line() {
        let report = Report {
            no_new_privs: false,
            roots_skipped: 2,
            ..Report::default()
        };
        let line = report.line();
        assert!(line.contains(" no-new-privs=off roots-skipped=2"), "{line}");
        assert!(line.contains("ip=not-expressible"));
    }

    /// Landlock and seccomp cannot be undone, so the enforcement is checked in
    /// a child: the test runs again as a process of its own, with this
    /// variable naming the directory it works in, and the parent reads what
    /// it printed.
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    const CHILD: &str = "RABBIT_LINUX_POLICY_CHILD";

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn the_manifest_is_enforced_in_a_child_process() {
        use std::process::Command;

        if let Ok(directory) = std::env::var(CHILD) {
            child(&directory);
            return;
        }

        let scratch = Scratch::new("rabbit-linux-policy");
        let directory = scratch.path();
        std::fs::create_dir_all(format!("{directory}/in")).unwrap();
        std::fs::create_dir_all(format!("{directory}/out")).unwrap();
        std::fs::write(format!("{directory}/in/inside.txt"), "inside").unwrap();
        std::fs::write(format!("{directory}/out/outside.txt"), "outside").unwrap();
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "linux_policy::tests::the_manifest_is_enforced_in_a_child_process",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(CHILD, directory);
        let output = run_child(command, 60);
        println!("{}", output.stdout);
        eprintln!("{}", output.stderr);
        assert!(
            !output.timed_out,
            "the child ran out of time:\n{}{}",
            output.stdout, output.stderr
        );
        assert!(
            output.success,
            "the child failed:\n{}{}",
            output.stdout, output.stderr
        );
        assert!(
            output.stdout.contains("CHILD DONE"),
            "the child did not finish:\n{}{}",
            output.stdout,
            output.stderr
        );
    }

    /// The part of the test that runs restricted: the manifest grants reads
    /// under `in` and datagrams, and no streams.
    ///
    /// A protection that the kernel reports and that is not in force is a
    /// failure of the test. Only a kernel that has no Landlock, or no seccomp,
    /// skips what it cannot check, and the skip goes to stderr with its reason.
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    fn child(directory: &str) {
        // libtest may have left `test name ... ` on this line.
        println!();
        use std::net::{TcpListener, TcpStream, UdpSocket};

        use rabbit_policy::{Capabilities, Capability, Endpoint, Policy, Protocol};

        const EACCES: i32 = 13;
        const ENOSYS: i64 = 38;
        const SERVER: [u8; 4] = [10, 0, 2, 2];

        let mut granted = Capabilities::EMPTY;
        granted.insert(Capability::FsRead);
        granted.insert(Capability::NetUdp);
        let mut policy = Policy::new(granted);
        policy
            .add_read_root(format!("{directory}/in").as_bytes())
            .unwrap();
        policy
            .add_endpoint(Endpoint {
                protocol: Protocol::Udp,
                ip: SERVER,
                port: 9999,
            })
            .unwrap();

        let report = super::apply(&policy);
        let line = report.line();
        println!("CHILD {line}");
        assert!(report.no_new_privs, "no_new_privs was refused: {line}");

        if report.fs {
            let inside = std::fs::File::open(format!("{directory}/in/inside.txt"));
            assert!(inside.is_ok(), "open inside the root: {inside:?}");
            let outside = std::fs::File::open(format!("{directory}/out/outside.txt"));
            assert_eq!(
                outside.err().and_then(|error| error.raw_os_error()),
                Some(EACCES),
                "open outside the root"
            );
            println!("CHILD fs checked");
        } else {
            assert_eq!(
                report.landlock_abi, 0,
                "the kernel reports Landlock but it is not in force: {line}"
            );
            assert!(
                std::env::var_os("RABBIT_REQUIRE_LANDLOCK").is_none_or(|value| value != "1"),
                "RABBIT_REQUIRE_LANDLOCK=1 and the kernel has no Landlock: {line}"
            );
            eprintln!("SKIP landlock unavailable: the kernel reports no Landlock ABI: {line}");
        }

        if report.seccomp {
            let stream = TcpStream::connect(("127.0.0.1", 9));
            assert_eq!(
                stream.err().and_then(|error| error.raw_os_error()),
                Some(EACCES),
                "a stream socket without NetTcp"
            );
            let listener = TcpListener::bind(("127.0.0.1", 0));
            assert_eq!(
                listener.err().and_then(|error| error.raw_os_error()),
                Some(EACCES),
                "a listening socket"
            );
            let datagram = UdpSocket::bind(("127.0.0.1", 0));
            assert!(
                datagram.is_ok(),
                "a datagram socket with NetUdp: {datagram:?}"
            );
            println!("CHILD seccomp checked");
        } else {
            assert_eq!(
                report.seccomp_error.map(|failure| failure.errno),
                Some(ENOSYS),
                "the kernel has seccomp but the filter is not in force: {line}"
            );
            eprintln!("SKIP seccomp unavailable: the kernel has no seccomp call: {line}");
        }

        if report.udp_port {
            let socket = UdpSocket::bind(("127.0.0.1", 0)).unwrap();
            let refused = socket.send_to(b"x", ("127.0.0.1", 9));
            assert_eq!(
                refused.err().and_then(|error| error.raw_os_error()),
                Some(EACCES),
                "a datagram to a port that is not listed"
            );
            let listed = socket.send_to(b"x", ("127.0.0.1", 9999));
            assert!(
                listed.as_ref().err().and_then(std::io::Error::raw_os_error) != Some(EACCES),
                "a datagram to a listed port: {listed:?}"
            );
            println!("CHILD udp port checked");
        } else if report.fs {
            eprintln!(
                "SKIP landlock udp port unavailable: UDP ports need ABI 10, the kernel has ABI {}",
                report.landlock_abi
            );
        }
        println!("CHILD DONE");
    }
}
