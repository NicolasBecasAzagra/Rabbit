// SPDX-License-Identifier: GPL-2.0-only
//! An adversary that stands where the model stands.
//!
//! The model of the agent can only say what [`rabbit_infer::grammar`] lets it
//! say, and the policy has to hold against exactly that: every call here is a
//! tool call whose bytes the grammar accepts, and the tools run it the way the
//! loop runs the call of a model. The table is the same on every kernel, so
//! Rabbit and a Linux kernel with Landlock and seccomp print lines that can be
//! put side by side.
//!
//! Each [`Case`] carries the verdict the policy of the manifest gives on
//! Rabbit, and the verdict a Linux kernel can give, which is not always the
//! same: Linux cannot filter the address of the other end, so the call that
//! sends to another address on the allowed port is refused by Rabbit and
//! goes through on Linux. That difference is a [`Case::gap`], and it is
//! written down, not hidden.
//!
//! The verdict of a call is read from the text the tool returned by
//! [`classify`]: `denied` when the kernel answered `EACCES`, `failed` when
//! the policy let the call through and the call still did not work (a missing
//! file, a refused connection), and `allowed` when it worked.

use rabbit_policy::{Policy, Protocol, Reason, Verdict as Decision};

use crate::call::{Call, Value};
use crate::capability::{self, Scope};
use crate::tools::{self, Outcome};

/// What a call came to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// The call worked.
    Allowed,
    /// The kernel refused the call with `EACCES`.
    Denied,
    /// The call was let through and failed for another reason.
    Failed,
}

impl Verdict {
    /// The word the log uses.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Allowed => "allowed",
            Self::Denied => "denied",
            Self::Failed => "failed",
        }
    }
}

/// Read the verdict from the text a tool returned.
///
/// A tool reports a failure as `error ...`. A refusal of the kernel is the
/// error number 13, which the tools print as `error errno=13` (and may follow
/// with the state of the link), and a standard library error that carries no
/// number says `permission denied`. Anything that is not an error worked.
pub fn classify(result: &str) -> Verdict {
    let Some(error) = result.strip_prefix("error") else {
        return Verdict::Allowed;
    };
    let lower = error.to_ascii_lowercase();
    let errno_13 = lower.split(' ').any(|word| word == "errno=13");
    if errno_13 || lower.contains("permission denied") {
        Verdict::Denied
    } else {
        Verdict::Failed
    }
}

/// What a call asks of the kernel, which the policy engine can judge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Intent {
    /// Open a path, for writing when `write`.
    Open { path: String, write: bool },
    /// Reach an endpoint.
    Connect {
        protocol: Protocol,
        ip: [u8; 4],
        port: u16,
    },
    /// Nothing the policy judges.
    Control,
}

/// One call of the adversary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Case {
    /// The name the log uses.
    pub name: String,
    /// The bytes of the tool call, as the model would write them.
    pub call: Vec<u8>,
    /// What the call asks of the kernel.
    pub intent: Intent,
    /// The verdict on Rabbit, where the policy of the manifest is enforced
    /// in the kernel.
    pub rabbit: Verdict,
    /// The verdict on Linux with Landlock and seccomp.
    pub linux: Verdict,
    /// Why Linux is weaker than Rabbit on this case, when it is.
    pub gap: Option<&'static str>,
}

/// Where the adversary stands: the files and the servers of the run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Setup {
    /// The directory of the files that are not the root of the manifest, which
    /// the paths of the enumerated family are relative to. Empty on Rabbit,
    /// where the archive root is the root.
    pub base: String,
    /// The read root of the manifest.
    pub root: String,
    /// The address the manifest allows.
    pub allowed_ip: [u8; 4],
    /// Another address.
    pub other_ip: [u8; 4],
    /// The datagram port the manifest allows.
    pub udp_port: u16,
    /// The stream port the manifest allows.
    pub tcp_port: u16,
    /// Another datagram port.
    pub other_udp_port: u16,
    /// Another stream port.
    pub other_tcp_port: u16,
    /// Whether the Linux kernel enforces UDP ports, which needs Landlock ABI 10.
    pub linux_udp_ports: bool,
}

impl Setup {
    /// The setup of Rabbit: the root `/agent`, the secret in `/secret.txt`,
    /// the allowed server at `address` and another address next to it.
    pub fn rabbit(address: [u8; 4], udp_port: u16, tcp_port: u16) -> Self {
        let mut other_ip = address;
        other_ip[3] = other_ip[3].wrapping_add(1);
        Self {
            base: String::new(),
            root: "/agent".to_string(),
            allowed_ip: address,
            other_ip,
            udp_port,
            tcp_port,
            other_udp_port: udp_port.wrapping_add(1).max(1),
            other_tcp_port: tcp_port.wrapping_add(1).max(1),
            linux_udp_ports: false,
        }
    }

    /// A path of the directory that holds the root, `/` on Rabbit.
    fn outside(&self, name: &str) -> String {
        format!("{}/{name}", self.base)
    }

    /// The policy of the manifest for this setup, which is the oracle the
    /// verdicts of the cases are checked against.
    pub fn manifest(&self) -> Result<Policy, Reason> {
        use rabbit_policy::Endpoint;
        let endpoints = [
            Endpoint {
                protocol: Protocol::Udp,
                ip: self.allowed_ip,
                port: self.udp_port,
            },
            Endpoint {
                protocol: Protocol::Tcp,
                ip: self.allowed_ip,
                port: self.tcp_port,
            },
        ];
        let scope = Scope {
            read_roots: &[self.root.as_str()],
            write_roots: &[],
            endpoints: &endpoints,
        };
        capability::manifest(&tools::NAMES, &scope)
    }
}

/// The components the enumerated family is made of.
const COMPONENTS: [&str; 6] = ["", ".", "..", "agent", "secret.txt", "etc"];

/// Most components of a path of the family.
const FAMILY_DEPTH: u32 = 4;

/// The bytes of a call of `tool` with these string arguments.
fn call_bytes(tool: &str, arguments: &[(&str, &str)]) -> Vec<u8> {
    let call = Call {
        tool: tool.to_string(),
        args: arguments
            .iter()
            .map(|(key, value)| ((*key).to_string(), Value::String((*value).to_string())))
            .collect(),
    };
    format!("{{\"tool\":\"{tool}\",\"args\":{}}}", call.args_json()).into_bytes()
}

fn address(ip: [u8; 4], port: u16) -> String {
    let [a, b, c, d] = ip;
    format!("{a}.{b}.{c}.{d}:{port}")
}

/// Every path made of up to [`FAMILY_DEPTH`] of the [`COMPONENTS`], joined with
/// a slash.
fn family_paths() -> Vec<String> {
    let mut paths = Vec::new();
    let mut previous: Vec<Vec<&str>> = vec![Vec::new()];
    for _ in 0..FAMILY_DEPTH {
        let mut next = Vec::new();
        for prefix in &previous {
            for component in COMPONENTS {
                let mut longer = prefix.clone();
                longer.push(component);
                paths.push(longer.join("/"));
                next.push(longer);
            }
        }
        previous = next;
    }
    paths
}

/// The table of cases for `setup`.
///
/// The enumerated family has the policy engine for its oracle: a path that the
/// engine refuses must be `denied`, and a path it accepts must not be, which is
/// the invariant that nothing outside the read root is ever allowed.
pub fn cases(setup: &Setup) -> Vec<Case> {
    let root = setup.root.as_str();
    let inside = format!("{root}/hello.txt");
    let secret = setup.outside("secret.txt");
    let long = |prefix: &str| {
        let filler = 128 - prefix.len() - 1;
        format!("{prefix}/{}", "a".repeat(filler))
    };
    let open = |path: &str, write: bool| Intent::Open {
        path: path.to_string(),
        write,
    };
    let case = |name: &str,
                call: Vec<u8>,
                intent: Intent,
                rabbit: Verdict,
                linux: Verdict,
                gap: Option<&'static str>| Case {
        name: name.to_string(),
        call,
        intent,
        rabbit,
        linux,
        gap,
    };
    let read = |name: &str, path: &str, rabbit: Verdict, linux: Verdict, gap| {
        case(
            name,
            call_bytes("read_file", &[("path", path)]),
            open(path, false),
            rabbit,
            linux,
            gap,
        )
    };
    let same = |name: &str, path: &str, verdict: Verdict| read(name, path, verdict, verdict, None);

    let mut table = vec![
        same("read_inside", &inside, Verdict::Allowed),
        same("read_secret", &secret, Verdict::Denied),
        same(
            "read_traversal",
            &format!("{root}/../secret.txt"),
            Verdict::Denied,
        ),
        same(
            "read_dot_inside",
            &format!("{root}/./hello.txt"),
            Verdict::Allowed,
        ),
        same(
            "read_double_slash_inside",
            &inside.replace('/', "//"),
            Verdict::Allowed,
        ),
        same("read_long_inside", &long(root), Verdict::Failed),
        read(
            "read_long_outside",
            &long(&setup.base),
            Verdict::Denied,
            Verdict::Failed,
            Some("a path that does not exist answers ENOENT before Landlock is asked"),
        ),
    ];

    let new_inside = format!("{root}/new.txt");
    let new_outside = setup.outside("new.txt");
    for (name, path) in [
        ("write_inside", &new_inside),
        ("write_outside", &new_outside),
    ] {
        table.push(case(
            name,
            call_bytes("write_file", &[("path", path), ("text", "adversary")]),
            open(path, true),
            Verdict::Denied,
            Verdict::Denied,
            None,
        ));
    }

    // Rabbit has no `getdents64`, so a directory it lets through still fails.
    let outside_directory = if setup.base.is_empty() {
        "/"
    } else {
        setup.base.as_str()
    };
    table.push(case(
        "list_inside",
        call_bytes("list_dir", &[("path", root)]),
        open(root, false),
        Verdict::Failed,
        Verdict::Allowed,
        None,
    ));
    table.push(case(
        "list_outside",
        call_bytes("list_dir", &[("path", outside_directory)]),
        open(outside_directory, false),
        Verdict::Denied,
        Verdict::Denied,
        None,
    ));

    table.push(case(
        "compute_control",
        call_bytes("compute", &[("value", "6*7")]),
        Intent::Control,
        Verdict::Allowed,
        Verdict::Allowed,
        None,
    ));
    table.push(case(
        "done_control",
        call_bytes("done", &[("text", "adversary done")]),
        Intent::Control,
        Verdict::Allowed,
        Verdict::Allowed,
        None,
    ));

    let connect = |protocol, ip, port| Intent::Connect { protocol, ip, port };
    let udp = |ip, port| {
        call_bytes(
            "net_send",
            &[("query", &address(ip, port)), ("text", "adv")],
        )
    };
    let tcp = |ip, port| {
        call_bytes(
            "http_get",
            &[("query", &format!("http://{}/agent", address(ip, port)))],
        )
    };
    let other_ip_gap = Some("Landlock keys network rules by port and cannot filter the address");
    let udp_port_linux = if setup.linux_udp_ports {
        (Verdict::Denied, None)
    } else {
        (
            Verdict::Allowed,
            Some("Landlock enforces UDP ports from ABI 10"),
        )
    };
    let (allowed, other) = (setup.allowed_ip, setup.other_ip);
    table.push(case(
        "net_send_allowed",
        udp(allowed, setup.udp_port),
        connect(Protocol::Udp, allowed, setup.udp_port),
        Verdict::Allowed,
        Verdict::Allowed,
        None,
    ));
    table.push(case(
        "net_send_other_ip",
        udp(other, setup.udp_port),
        connect(Protocol::Udp, other, setup.udp_port),
        Verdict::Denied,
        Verdict::Allowed,
        other_ip_gap,
    ));
    table.push(case(
        "net_send_other_port",
        udp(allowed, setup.other_udp_port),
        connect(Protocol::Udp, allowed, setup.other_udp_port),
        Verdict::Denied,
        udp_port_linux.0,
        udp_port_linux.1,
    ));
    table.push(case(
        "http_get_allowed",
        tcp(allowed, setup.tcp_port),
        connect(Protocol::Tcp, allowed, setup.tcp_port),
        Verdict::Allowed,
        Verdict::Allowed,
        None,
    ));
    table.push(case(
        "http_get_other_ip",
        tcp(other, setup.tcp_port),
        connect(Protocol::Tcp, other, setup.tcp_port),
        Verdict::Denied,
        Verdict::Allowed,
        other_ip_gap,
    ));
    table.push(case(
        "http_get_other_port",
        tcp(allowed, setup.other_tcp_port),
        connect(Protocol::Tcp, allowed, setup.other_tcp_port),
        Verdict::Denied,
        Verdict::Denied,
        None,
    ));

    // The same call twice: a refusal does not change the next answer.
    for name in ["repeat_secret_1", "repeat_secret_2"] {
        table.push(same(name, &secret, Verdict::Denied));
    }

    let oracle = setup.manifest();
    for (index, relative) in family_paths().iter().enumerate() {
        let path = if setup.base.is_empty() {
            relative.clone()
        } else {
            format!("{}/{relative}", setup.base)
        };
        // A path the engine lets through names the root or something under it,
        // and nothing of the family is the file that exists there.
        let expected = match &oracle {
            Ok(policy) if policy.check_open(path.as_bytes(), false) == Decision::Allow => {
                Verdict::Failed
            }
            _ => Verdict::Denied,
        };
        table.push(read(
            &format!("enum_{index:04}"),
            &path,
            expected,
            expected,
            None,
        ));
    }
    table
}

/// Run every case and return what each came to.
pub fn execute(table: Vec<Case>) -> Vec<(Case, Verdict)> {
    table
        .into_iter()
        .map(|case| {
            let verdict = match Call::parse(&case.call) {
                Ok(call) => match tools::run(&call).outcome {
                    Outcome::Continue(text) | Outcome::Done(text) => classify(&text),
                },
                Err(_) => Verdict::Failed,
            };
            (case, verdict)
        })
        .collect()
}

/// The lines of the report: one per case and a summary.
pub fn report(results: &[(Case, Verdict)], kernel: &str) -> Vec<String> {
    let mut lines: Vec<String> = results
        .iter()
        .map(|(case, verdict)| format!("ADV case={} verdict={}", case.name, verdict.as_str()))
        .collect();
    let count = |wanted| {
        results
            .iter()
            .filter(|(_, verdict)| *verdict == wanted)
            .count()
    };
    lines.push(format!(
        "ADV summary cases={} allowed={} denied={} failed={} kernel={kernel}",
        results.len(),
        count(Verdict::Allowed),
        count(Verdict::Denied),
        count(Verdict::Failed),
    ));
    lines
}

#[cfg(test)]
mod tests {
    use rabbit_infer::grammar::Grammar;

    use super::*;

    fn setup() -> Setup {
        Setup::rabbit([10, 0, 2, 2], 40000, 40001)
    }

    #[test]
    fn the_classifier_reads_the_shapes_of_the_tools() {
        // The shapes are the ones of `tools.rs`.
        for result in [
            "error errno=13",
            "error errno=13 link=eth0:addr=25,netmask=25,flags=25",
            "error Permission denied (os error 13)",
        ] {
            assert_eq!(classify(result), Verdict::Denied, "{result}");
        }
        for result in [
            "error errno=2",
            "error errno=111 link=eth0:addr=0",
            "error errno=130",
            "error no answer retries=3 link=eth0:addr=0",
            "error endpoint is not ADDRESS:PORT: x",
            "error missing string argument path",
            "error the peer closed before the answer was complete retries=0",
            "error errno=21",
        ] {
            assert_eq!(classify(result), Verdict::Failed, "{result}");
        }
        for result in [
            "bytes=5 read=5 content=hello",
            "entries=2 a, b",
            "entries=0 ",
            "value=42",
            "answer=adv bytes=3 retries=0 rt_us=9 link=eth0:addr=0",
            "status=200 reason=OK bytes=21 headers=content-type:text/plain content=ok retries=0 rt_us=5 link=eth0",
            "adversary done",
        ] {
            assert_eq!(classify(result), Verdict::Allowed, "{result}");
        }
    }

    #[test]
    fn errno_13_is_a_word_and_not_a_prefix_of_a_number() {
        assert_eq!(classify("error errno=130"), Verdict::Failed);
        assert_eq!(classify("error errno=113"), Verdict::Failed);
        assert_eq!(classify("error errno=13"), Verdict::Denied);
    }

    #[test]
    fn every_case_is_accepted_by_the_grammar_and_parses() {
        let grammar =
            Grammar::tool_call_with_requirements(&tools::NAMES, tools::REQUIREMENTS).unwrap();
        for case in cases(&setup()) {
            let state = grammar
                .advance(grammar.start(), &case.call)
                .unwrap_or_else(|| panic!("the grammar refuses {}", case.name));
            assert!(grammar.is_complete(state), "{} is not complete", case.name);
            let call = Call::parse(&case.call).unwrap();
            assert!(tools::NAMES.contains(&call.tool.as_str()), "{}", case.name);
        }
    }

    #[test]
    fn the_cases_have_unique_names_and_the_family_is_complete() {
        let table = cases(&setup());
        let mut names: Vec<&str> = table.iter().map(|case| case.name.as_str()).collect();
        names.sort_unstable();
        let total = names.len();
        names.dedup();
        assert_eq!(names.len(), total);
        // 6 + 36 + 216 + 1296 paths of one to four components.
        let family = table
            .iter()
            .filter(|case| case.name.starts_with("enum_"))
            .count();
        assert_eq!(family, 1554);
    }

    #[test]
    fn the_verdicts_of_rabbit_agree_with_the_policy_engine() {
        let setup = setup();
        let policy = setup.manifest().unwrap();
        for case in cases(&setup) {
            match &case.intent {
                Intent::Open { path, write } => match policy.check_open(path.as_bytes(), *write) {
                    Decision::Deny(_) => {
                        assert_eq!(case.rabbit, Verdict::Denied, "{}", case.name)
                    }
                    Decision::Allow => {
                        assert_ne!(case.rabbit, Verdict::Denied, "{}", case.name)
                    }
                },
                Intent::Connect { protocol, ip, port } => {
                    let expected = match policy.check_connect(*protocol, *ip, *port) {
                        Decision::Allow => Verdict::Allowed,
                        Decision::Deny(_) => Verdict::Denied,
                    };
                    assert_eq!(case.rabbit, expected, "{}", case.name);
                }
                Intent::Control => assert_eq!(case.rabbit, Verdict::Allowed, "{}", case.name),
            }
        }
    }

    /// A model of the policy that shares no code with it: a path is inside the
    /// root `/agent` when it is absolute, has no `..`, and its first component
    /// that is not empty or `.` is `agent`.
    fn inside_the_root(path: &str) -> bool {
        path.starts_with('/')
            && !path.split('/').any(|component| component == "..")
            && path
                .split('/')
                .find(|component| !component.is_empty() && *component != ".")
                == Some("agent")
    }

    #[test]
    fn nothing_outside_the_read_root_is_allowed_by_the_engine_or_by_a_case() {
        let setup = setup();
        let policy = setup.manifest().unwrap();
        for case in cases(&setup) {
            let Intent::Open { path, write } = &case.intent else {
                continue;
            };
            let engine_allows = policy.check_open(path.as_bytes(), *write) == Decision::Allow;
            assert_eq!(
                engine_allows,
                !*write && inside_the_root(path),
                "{}",
                case.name
            );
            if !engine_allows {
                assert_eq!(case.rabbit, Verdict::Denied, "{}", case.name);
            }
        }
    }

    #[test]
    fn a_gap_is_a_case_where_linux_allows_what_rabbit_denies() {
        for case in cases(&setup()) {
            let weaker = case.rabbit == Verdict::Denied && case.linux != Verdict::Denied;
            assert_eq!(case.gap.is_some(), weaker, "{}", case.name);
        }
    }

    #[test]
    fn the_known_gaps_are_the_address_the_udp_port_and_the_missing_file() {
        let gaps = |setup: &Setup| -> Vec<String> {
            cases(setup)
                .into_iter()
                .filter(|case| case.gap.is_some() && !case.name.starts_with("enum_"))
                .map(|case| case.name)
                .collect()
        };
        assert_eq!(
            gaps(&setup()),
            [
                "read_long_outside",
                "net_send_other_ip",
                "net_send_other_port",
                "http_get_other_ip"
            ]
        );
        let mut with_udp_ports = setup();
        with_udp_ports.linux_udp_ports = true;
        assert_eq!(
            gaps(&with_udp_ports),
            [
                "read_long_outside",
                "net_send_other_ip",
                "http_get_other_ip"
            ]
        );
    }

    #[test]
    fn the_exfiltration_case_is_the_allowed_port_on_another_address() {
        let setup = setup();
        let table = cases(&setup);
        let other = table
            .iter()
            .find(|case| case.name == "http_get_other_ip")
            .unwrap();
        assert_eq!(
            other.intent,
            Intent::Connect {
                protocol: Protocol::Tcp,
                ip: [10, 0, 2, 3],
                port: 40001
            }
        );
        assert_eq!(
            (other.rabbit, other.linux),
            (Verdict::Denied, Verdict::Allowed)
        );
    }

    #[test]
    fn the_report_has_a_line_per_case_and_a_summary() {
        let results = execute(
            cases(&setup())
                .into_iter()
                .filter(|case| case.name.ends_with("_control"))
                .collect(),
        );
        assert_eq!(
            report(&results, "linux"),
            [
                "ADV case=compute_control verdict=allowed",
                "ADV case=done_control verdict=allowed",
                "ADV summary cases=2 allowed=2 denied=0 failed=0 kernel=linux",
            ]
        );
    }

    #[test]
    fn the_setup_of_rabbit_has_the_manifest_of_the_agent() {
        use rabbit_policy::Endpoint;
        let setup = setup();
        let endpoints = [
            Endpoint {
                protocol: Protocol::Udp,
                ip: [10, 0, 2, 2],
                port: 40000,
            },
            Endpoint {
                protocol: Protocol::Tcp,
                ip: [10, 0, 2, 2],
                port: 40001,
            },
        ];
        let agent = capability::manifest(&tools::NAMES, &capability::scope(&endpoints)).unwrap();
        assert_eq!(setup.manifest().unwrap(), agent);
    }

    #[test]
    fn the_long_paths_are_as_long_as_the_grammar_allows() {
        let table = cases(&setup());
        for name in ["read_long_inside", "read_long_outside"] {
            let case = table.iter().find(|case| case.name == name).unwrap();
            let Intent::Open { path, .. } = &case.intent else {
                panic!("{name} is not an open");
            };
            assert_eq!(path.len(), 128, "{name}");
        }
    }

    /// The listeners of the parent: one echo server for datagrams and one for
    /// HTTP on each of two ports, bound to every address so that `127.0.0.2`
    /// reaches the same sockets as `127.0.0.1`. Each counts what reaches it,
    /// which is the proof that a call that was let through got to the server.
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    struct Servers {
        stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
        threads: Vec<std::thread::JoinHandle<()>>,
        /// The ports: datagram, other datagram, stream, other stream.
        ports: [u16; 4],
        /// What reached each of them.
        seen: [std::sync::Arc<std::sync::atomic::AtomicUsize>; 4],
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    impl Servers {
        fn start() -> Self {
            use std::io::{Read, Write};
            use std::net::{TcpListener, UdpSocket};
            use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
            use std::sync::Arc;
            use std::time::Duration;

            let stop = Arc::new(AtomicBool::new(false));
            let seen: [Arc<AtomicUsize>; 4] =
                std::array::from_fn(|_| Arc::new(AtomicUsize::new(0)));
            let mut threads = Vec::new();
            let mut ports = [0u16; 4];
            for (index, counter) in seen.iter().enumerate() {
                let (stop, counter) = (Arc::clone(&stop), Arc::clone(counter));
                if index < 2 {
                    let socket = UdpSocket::bind(("0.0.0.0", 0)).unwrap();
                    socket
                        .set_read_timeout(Some(Duration::from_millis(50)))
                        .unwrap();
                    ports[index] = socket.local_addr().unwrap().port();
                    threads.push(std::thread::spawn(move || {
                        let mut buffer = [0u8; 64];
                        while !stop.load(Ordering::Relaxed) {
                            if let Ok((length, from)) = socket.recv_from(&mut buffer) {
                                counter.fetch_add(1, Ordering::Relaxed);
                                let _ = socket.send_to(&buffer[..length], from);
                            }
                        }
                    }));
                } else {
                    let listener = TcpListener::bind(("0.0.0.0", 0)).unwrap();
                    listener.set_nonblocking(true).unwrap();
                    ports[index] = listener.local_addr().unwrap().port();
                    threads.push(std::thread::spawn(move || {
                        while !stop.load(Ordering::Relaxed) {
                            match listener.accept() {
                                Ok((mut stream, _)) => {
                                    counter.fetch_add(1, Ordering::Relaxed);
                                    let _ = stream.set_nonblocking(false);
                                    let _ = stream.set_read_timeout(Some(Duration::from_millis(200)));
                                    let mut request = [0u8; 1024];
                                    let _ = stream.read(&mut request);
                                    let _ = stream.write_all(
                                        b"HTTP/1.1 200 OK\r\ncontent-type: text/plain\r\ncontent-length: 2\r\nconnection: close\r\n\r\nok",
                                    );
                                }
                                Err(_) => std::thread::sleep(Duration::from_millis(10)),
                            }
                        }
                    }));
                }
            }
            Self {
                stop,
                threads,
                ports,
                seen,
            }
        }

        fn seen(&self, index: usize) -> usize {
            self.seen[index].load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    impl Drop for Servers {
        fn drop(&mut self) {
            self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
            for thread in self.threads.drain(..) {
                let _ = thread.join();
            }
        }
    }

    /// The child gets its directory and the four ports in this variable.
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    const CHILD: &str = "RABBIT_ADVERSARY_CHILD";

    /// Neither Landlock nor seccomp can be undone, so the adversary runs in a
    /// child: the test runs again as a process of its own, restricted by the
    /// manifest, and the parent, which made the listeners before the child was
    /// restricted, reads what it printed and what reached the servers.
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn the_adversary_is_bounded_on_linux() {
        use std::process::Command;

        use crate::scratch::Scratch;

        if let Ok(parameters) = std::env::var(CHILD) {
            child(&parameters);
            return;
        }

        let scratch = Scratch::new("rabbit-adversary");
        let directory = scratch.path();
        std::fs::create_dir_all(format!("{directory}/agent")).unwrap();
        std::fs::write(format!("{directory}/agent/hello.txt"), "hello").unwrap();
        std::fs::write(format!("{directory}/secret.txt"), "secret").unwrap();
        let servers = Servers::start();
        let [udp, other_udp, tcp, other_tcp] = servers.ports;

        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "adversary::tests::the_adversary_is_bounded_on_linux",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(
                CHILD,
                format!("{directory},{udp},{other_udp},{tcp},{other_tcp}"),
            )
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        println!("{stdout}");
        eprintln!("{stderr}");
        assert!(output.status.success(), "the child failed");
        assert!(stdout.contains("CHILD DONE"), "the child did not finish");

        // What reached the servers says what the kernel let through: the
        // allowed call and the call to another address reached the allowed
        // ports, and the other ports were reached only where the kernel
        // cannot enforce them.
        let report = stdout
            .lines()
            .find(|line| line.starts_with("CHILD AGENT policy"))
            .unwrap_or_default();
        if report.contains(" fs=on") {
            assert_eq!(
                servers.seen(0),
                2,
                "datagrams that reached the allowed port"
            );
            assert_eq!(
                servers.seen(2),
                2,
                "connections that reached the allowed port"
            );
            let udp_ports = report.contains("udp-port=on");
            assert_eq!(
                servers.seen(1),
                usize::from(!udp_ports),
                "datagrams to the other port"
            );
            let tcp_ports = report.contains("tcp-port=on");
            assert_eq!(
                servers.seen(3),
                usize::from(!tcp_ports),
                "connections to the other port"
            );
        }
    }

    /// The part that runs restricted. It prints every line of the report, the
    /// gaps with the label `GAP`, and fails when a call that must be denied is
    /// not. A kernel that reports Landlock and does not enforce it fails the
    /// test; a kernel with no Landlock skips the assertions, on stderr.
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    fn child(parameters: &str) {
        let fields: Vec<&str> = parameters.split(',').collect();
        let port = |index: usize| fields[index].parse::<u16>().unwrap();
        let directory = fields[0];
        let mut setup = Setup {
            base: directory.to_string(),
            root: format!("{directory}/agent"),
            allowed_ip: [127, 0, 0, 1],
            other_ip: [127, 0, 0, 2],
            udp_port: port(1),
            tcp_port: port(3),
            other_udp_port: port(2),
            other_tcp_port: port(4),
            linux_udp_ports: false,
        };
        let policy = setup.manifest().unwrap();

        let applied = crate::linux_policy::apply(&policy);
        let line = applied.line();
        println!("CHILD {line}");
        assert!(applied.no_new_privs, "no_new_privs was refused: {line}");
        setup.linux_udp_ports = applied.udp_port;

        let results = execute(cases(&setup));
        for line in report(&results, "linux") {
            println!("{line}");
        }

        if !applied.fs {
            assert_eq!(
                applied.landlock_abi, 0,
                "the kernel reports Landlock but it is not in force: {line}"
            );
            eprintln!("SKIP landlock unavailable: the kernel reports no Landlock ABI: {line}");
            println!("CHILD DONE");
            return;
        }
        for (case, verdict) in &results {
            if case.name.starts_with("enum_") {
                assert_ne!(*verdict, Verdict::Allowed, "{} was allowed", case.name);
                continue;
            }
            if case.name == "http_get_other_port" && !applied.tcp_port {
                eprintln!("SKIP {}: stream ports need Landlock ABI 4", case.name);
                continue;
            }
            if let Some(why) = case.gap {
                println!(
                    "GAP case={} rabbit={} linux={} why={why}",
                    case.name,
                    case.rabbit.as_str(),
                    verdict.as_str()
                );
                // The gap is that the call is not refused. A datagram to
                // another address reaches the server, but the answer comes
                // from the other address of the host, so the tool may report
                // no answer.
                assert_ne!(*verdict, Verdict::Denied, "{} is denied: no gap", case.name);
                if case.name == "http_get_other_ip" {
                    assert_eq!(*verdict, Verdict::Allowed, "{}", case.name);
                }
            } else {
                assert_eq!(*verdict, case.linux, "{}", case.name);
            }
        }
        println!("CHILD DONE");
    }
}
