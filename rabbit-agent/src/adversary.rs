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
    /// The call is not a tool call: the bytes did not parse, which is a defect
    /// of the table and not a verdict of any kernel.
    Invalid,
}

impl Verdict {
    /// The word the log uses.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Allowed => "allowed",
            Self::Denied => "denied",
            Self::Failed => "failed",
            Self::Invalid => "invalid",
        }
    }
}

/// Read the verdict from the text a tool returned.
///
/// A tool reports a failure as `error ...`, and a failure of the kernel as
/// `error errno=N`, with the state of the link after it on the network tools.
/// A refusal of the kernel is the error number 13, and only that token counts:
/// it has to be the first thing after `error`, as a whole word. Free text that
/// a tool echoes from the call, such as `permission denied` or `errno=13` in
/// an address, is not the answer of the kernel and is a failure like any other.
/// A file whose mode forbids the open answers with the same number as the
/// policy does, and the table cannot tell them apart, so it uses files that
/// are readable. Anything that is not an error worked.
pub fn classify(result: &str) -> Verdict {
    let Some(error) = result.strip_prefix("error") else {
        return Verdict::Allowed;
    };
    let refused = error
        .strip_prefix(" errno=13")
        .is_some_and(|rest| rest.is_empty() || rest.starts_with(' '));
    if refused {
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
    /// Whether the root holds `link.txt`, a symbolic link to the secret. The
    /// initramfs of Rabbit has no symbolic links, so only a Linux tree has it.
    pub links: bool,
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
            links: false,
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

/// Every list of up to [`FAMILY_DEPTH`] of the [`COMPONENTS`], joined with a
/// slash. [`cases`] puts the base and a slash in front of each.
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
    // The datagram carries the name of its case and the request asks for it as
    // its path, so a server that records what reaches it can say which calls
    // were let through, whatever the tool made of the answer.
    let udp = |name: &str, ip, port| {
        call_bytes("net_send", &[("query", &address(ip, port)), ("text", name)])
    };
    let tcp = |name: &str, ip, port| {
        call_bytes(
            "http_get",
            &[("query", &format!("http://{}/{name}", address(ip, port)))],
        )
    };
    let other_ip_gap = Some("Landlock network rules name a port and no address");
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
        udp("net_send_allowed", allowed, setup.udp_port),
        connect(Protocol::Udp, allowed, setup.udp_port),
        Verdict::Allowed,
        Verdict::Allowed,
        None,
    ));
    table.push(case(
        "net_send_other_ip",
        udp("net_send_other_ip", other, setup.udp_port),
        connect(Protocol::Udp, other, setup.udp_port),
        Verdict::Denied,
        Verdict::Allowed,
        other_ip_gap,
    ));
    table.push(case(
        "net_send_other_port",
        udp("net_send_other_port", allowed, setup.other_udp_port),
        connect(Protocol::Udp, allowed, setup.other_udp_port),
        Verdict::Denied,
        udp_port_linux.0,
        udp_port_linux.1,
    ));
    table.push(case(
        "http_get_allowed",
        tcp("http_get_allowed", allowed, setup.tcp_port),
        connect(Protocol::Tcp, allowed, setup.tcp_port),
        Verdict::Allowed,
        Verdict::Allowed,
        None,
    ));
    table.push(case(
        "http_get_other_ip",
        tcp("http_get_other_ip", other, setup.tcp_port),
        connect(Protocol::Tcp, other, setup.tcp_port),
        Verdict::Denied,
        Verdict::Allowed,
        other_ip_gap,
    ));
    table.push(case(
        "http_get_other_port",
        tcp("http_get_other_port", allowed, setup.other_tcp_port),
        connect(Protocol::Tcp, allowed, setup.other_tcp_port),
        Verdict::Denied,
        Verdict::Denied,
        None,
    ));

    // The same call twice: a refusal does not change the next answer.
    for name in ["repeat_secret_1", "repeat_secret_2"] {
        table.push(same(name, &secret, Verdict::Denied));
    }

    if setup.links {
        // Landlock checks the resolved path, so the link inside the root does
        // not lead to the secret. Not a case of Rabbit, which has no links.
        let link = format!("{root}/link.txt");
        table.push(read(
            "read_symlink_to_secret",
            &link,
            Verdict::Failed,
            Verdict::Denied,
            None,
        ));
    }

    let oracle = setup.manifest();
    for (index, relative) in family_paths().iter().enumerate() {
        // Every path is absolute: a list that starts with an empty component
        // gets the doubled slash it spells.
        let path = format!("{}/{relative}", setup.base);
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
                Err(_) => Verdict::Invalid,
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
        "ADV summary cases={} allowed={} denied={} failed={} invalid={} kernel={kernel}",
        results.len(),
        count(Verdict::Allowed),
        count(Verdict::Denied),
        count(Verdict::Failed),
        count(Verdict::Invalid),
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
    fn free_text_that_a_tool_echoes_is_not_the_answer_of_the_kernel() {
        // The tools put the text of the call into some of their errors.
        for result in [
            "error endpoint is not an IPv4 address: permission denied",
            "error endpoint is not ADDRESS:PORT: errno=13",
            "error Permission denied",
            "error endpoint is not a port: x errno=13 y",
            "error  errno=13",
            "error errno=13x",
            "errno=13",
        ] {
            assert_ne!(classify(result), Verdict::Denied, "{result}");
        }
        // The results that carry content are not errors, whatever they say.
        assert_eq!(
            classify("bytes=5 read=5 content=error errno=13"),
            Verdict::Allowed
        );
    }

    #[test]
    fn a_file_that_the_mode_forbids_looks_like_a_policy_denial() {
        // Opening a file of mode 000 as an ordinary user answers `EACCES` too:
        // the table cannot tell the two apart, so it only uses files whose mode
        // allows the open.
        assert_eq!(classify("error errno=13"), Verdict::Denied);
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
                "ADV summary cases=2 allowed=2 denied=0 failed=0 invalid=0 kernel=linux",
            ]
        );
    }

    #[test]
    fn the_setup_of_rabbit_has_the_manifest_the_agent_installs() {
        // `manifest_of` is the function the loop and the adversary mode call,
        // so the oracle is the policy of the real code path.
        let endpoint = crate::Endpoint {
            address: std::net::Ipv4Addr::new(10, 0, 2, 2),
            port: 40000,
            http_port: 40001,
            ..crate::Endpoint::default()
        };
        let (agent, wire) = crate::manifest_of(&endpoint).unwrap();
        assert_eq!(setup().manifest().unwrap(), agent);
        assert_eq!(Policy::decode(&wire), Ok(agent));
    }

    #[test]
    fn a_call_that_does_not_parse_is_invalid_and_is_counted() {
        let broken = Case {
            name: "broken".to_string(),
            call: b"{\"tool\":".to_vec(),
            intent: Intent::Control,
            rabbit: Verdict::Allowed,
            linux: Verdict::Allowed,
            gap: None,
        };
        let results = execute(vec![broken]);
        assert_eq!(results[0].1, Verdict::Invalid);
        assert_eq!(
            report(&results, "rabbit"),
            [
                "ADV case=broken verdict=invalid",
                "ADV summary cases=1 allowed=0 denied=0 failed=0 invalid=1 kernel=rabbit"
            ]
        );
    }

    #[test]
    fn the_family_is_absolute_and_has_paths_the_oracle_allows_and_denies() {
        let setup = setup();
        let policy = setup.manifest().unwrap();
        let (mut allowed, mut denied) = (0, 0);
        for case in cases(&setup) {
            let Intent::Open { path, .. } = &case.intent else {
                continue;
            };
            if !case.name.starts_with("enum_") {
                continue;
            }
            assert!(path.starts_with('/'), "{path}");
            match policy.check_open(path.as_bytes(), false) {
                Decision::Allow => allowed += 1,
                Decision::Deny(_) => denied += 1,
            }
        }
        println!("the family: {allowed} paths allowed by the engine, {denied} denied");
        assert!(allowed > 0, "no path of the family is inside the root");
        assert!(denied > 0, "no path of the family is outside the root");
        assert_eq!(allowed + denied, 1554);
    }

    #[test]
    fn only_a_tree_with_links_has_the_symlink_case() {
        let mut with_links = setup();
        assert!(cases(&with_links)
            .iter()
            .all(|case| case.name != "read_symlink_to_secret"));
        with_links.links = true;
        let table = cases(&with_links);
        let case = table
            .iter()
            .find(|case| case.name == "read_symlink_to_secret")
            .unwrap();
        assert_eq!(case.linux, Verdict::Denied);
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

    /// The line of `output` that holds `marker`, from the marker to the end of
    /// the line. The child is the test binary run again with `--nocapture`, so
    /// libtest may have printed `test name ... ` in front of its first line,
    /// without a newline: a marker is found where it is, not at the start.
    fn marked_line<'a>(output: &'a str, marker: &str) -> Option<&'a str> {
        let start = output.find(marker)?;
        output[start..].lines().next()
    }

    /// The verdict the child printed for the case `name`.
    fn printed_verdict(output: &str, name: &str) -> Option<Verdict> {
        let line = marked_line(output, &format!("ADV case={name} verdict="))?;
        match line.rsplit('=').next()? {
            "allowed" => Some(Verdict::Allowed),
            "denied" => Some(Verdict::Denied),
            "failed" => Some(Verdict::Failed),
            "invalid" => Some(Verdict::Invalid),
            _ => None,
        }
    }

    /// What the policy decided about a network call, which is not always what
    /// the tool made of it. A datagram or a connection that reached the server
    /// was let through, even when the tool then read no answer, as when the
    /// reply of a datagram comes from another address than the one the socket
    /// is connected to. A call that reached nothing was refused when the
    /// tool says `denied`, and failed for another reason otherwise.
    fn policy_verdict(tool: Verdict, reached: bool) -> Verdict {
        if reached || tool == Verdict::Allowed {
            Verdict::Allowed
        } else if tool == Verdict::Denied {
            Verdict::Denied
        } else {
            tool
        }
    }

    #[test]
    fn the_marker_is_found_with_the_prefix_of_libtest_glued_before_it() {
        // The first line is the one the runner printed: libtest's prefix, glued
        // to the policy line of the child.
        let policy = "CHILD AGENT policy kernel=without private syscall landlock=abi8 fs=on tcp-port=on udp-port=off seccomp=on threads=all ip=not-expressible";
        let glued = format!(
            "test adversary::tests::the_adversary_is_bounded_on_linux ... {policy}\nADV case=read_inside verdict=allowed\nCHILD DONE\n"
        );
        let clean = format!("\n{policy}\nADV case=read_inside verdict=allowed\nCHILD DONE\n");
        for output in [glued.as_str(), clean.as_str()] {
            assert_eq!(marked_line(output, "CHILD AGENT policy"), Some(policy));
            assert_eq!(marked_line(output, "CHILD DONE"), Some("CHILD DONE"));
            assert_eq!(
                printed_verdict(output, "read_inside"),
                Some(Verdict::Allowed)
            );
            assert_eq!(printed_verdict(output, "read"), None);
        }
        assert_eq!(marked_line("nothing here", "CHILD DONE"), None);
        // A line of the first case glued to the prefix is found too.
        let first = "test t ... ADV case=read_inside verdict=denied\n";
        assert_eq!(printed_verdict(first, "read_inside"), Some(Verdict::Denied));
    }

    #[test]
    fn a_call_that_reached_the_server_was_let_through_whatever_the_tool_says() {
        // The datagram to another address reaches the server and the tool reads
        // no answer, because the reply comes from the address it did not use.
        assert_eq!(policy_verdict(Verdict::Failed, true), Verdict::Allowed);
        assert_eq!(policy_verdict(Verdict::Allowed, true), Verdict::Allowed);
        assert_eq!(policy_verdict(Verdict::Allowed, false), Verdict::Allowed);
        // Nothing reached the server and the kernel said no.
        assert_eq!(policy_verdict(Verdict::Denied, false), Verdict::Denied);
        // Nothing reached the server and the call failed for another reason.
        assert_eq!(policy_verdict(Verdict::Failed, false), Verdict::Failed);
        assert_eq!(policy_verdict(Verdict::Invalid, false), Verdict::Invalid);
    }

    #[test]
    fn the_network_cases_carry_their_name_so_a_server_can_tell_them_apart() {
        let table = cases(&setup());
        for case in table
            .iter()
            .filter(|case| case.name.contains("_other_") || case.name.ends_with("_allowed"))
        {
            let call = String::from_utf8(case.call.clone()).unwrap();
            assert!(call.contains(&case.name), "{call}");
        }
    }

    /// The listeners of the parent: one echo server for datagrams and one for
    /// HTTP on each of two ports, bound to every address so that `127.0.0.2`
    /// reaches the same sockets as `127.0.0.1`. Each records what reaches it:
    /// the datagram, or the request line. A call carries its name, so the
    /// record says which calls were let through.
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    struct Servers {
        stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
        threads: Vec<std::thread::JoinHandle<()>>,
        /// The ports: datagram, other datagram, stream, other stream.
        ports: [u16; 4],
        /// What reached each of them.
        seen: [std::sync::Arc<std::sync::Mutex<Vec<String>>>; 4],
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    impl Servers {
        fn start() -> Self {
            use std::io::{Read, Write};
            use std::net::{TcpListener, UdpSocket};
            use std::sync::atomic::{AtomicBool, Ordering};
            use std::sync::{Arc, Mutex};
            use std::time::Duration;

            let stop = Arc::new(AtomicBool::new(false));
            let seen: [Arc<Mutex<Vec<String>>>; 4] =
                std::array::from_fn(|_| Arc::new(Mutex::new(Vec::new())));
            let mut threads = Vec::new();
            let mut ports = [0u16; 4];
            for (index, record) in seen.iter().enumerate() {
                let (stop, record) = (Arc::clone(&stop), Arc::clone(record));
                if index < 2 {
                    let socket = UdpSocket::bind(("0.0.0.0", 0)).unwrap();
                    socket
                        .set_read_timeout(Some(Duration::from_millis(50)))
                        .unwrap();
                    ports[index] = socket.local_addr().unwrap().port();
                    threads.push(std::thread::spawn(move || {
                        let mut buffer = [0u8; 256];
                        while !stop.load(Ordering::Relaxed) {
                            if let Ok((length, from)) = socket.recv_from(&mut buffer) {
                                record
                                    .lock()
                                    .unwrap()
                                    .push(String::from_utf8_lossy(&buffer[..length]).into_owned());
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
                                    let _ = stream.set_nonblocking(false);
                                    let _ = stream
                                        .set_read_timeout(Some(Duration::from_millis(200)));
                                    let mut request = [0u8; 1024];
                                    let length = stream.read(&mut request).unwrap_or(0);
                                    // A connection that sent nothing is recorded
                                    // too: it reached the server.
                                    record
                                        .lock()
                                        .unwrap()
                                        .push(String::from_utf8_lossy(&request[..length]).into_owned());
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

        /// Whether a datagram or a request that carries `name` reached any of
        /// the servers.
        fn reached(&self, name: &str) -> bool {
            self.seen.iter().any(|record| {
                record
                    .lock()
                    .unwrap()
                    .iter()
                    .any(|seen| seen.contains(name))
            })
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

    /// The setup of the Linux tree: the root is `agent` under the directory,
    /// the servers are on the loopback, and the tree has the symbolic link.
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    fn linux_setup(directory: &str, ports: [u16; 4], udp_ports: bool) -> Setup {
        let [udp, other_udp, tcp, other_tcp] = ports;
        Setup {
            base: directory.to_string(),
            root: format!("{directory}/agent"),
            allowed_ip: [127, 0, 0, 1],
            other_ip: [127, 0, 0, 2],
            udp_port: udp,
            tcp_port: tcp,
            other_udp_port: other_udp,
            other_tcp_port: other_tcp,
            linux_udp_ports: udp_ports,
            links: true,
        }
    }

    /// The child gets its directory, the four ports and whether `127.0.0.2` is
    /// routable in this variable.
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    const CHILD: &str = "RABBIT_ADVERSARY_CHILD";

    /// Neither Landlock nor seccomp can be undone, so the adversary runs in a
    /// child: the test runs again as a process of its own, restricted by the
    /// manifest, and the parent, which made the listeners before the child was
    /// restricted, reads what it printed and what reached the servers.
    ///
    /// The tool of a network call may report a failure for a call the kernel
    /// let through, so the verdict of a network case is the one the servers
    /// give: a call that reached one was allowed.
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn the_adversary_is_bounded_on_linux() {
        use std::process::Command;

        use crate::scratch::{run_child, Scratch};

        if let Ok(parameters) = std::env::var(CHILD) {
            child(&parameters);
            return;
        }

        let scratch = Scratch::new("rabbit-adversary");
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

        // A runner in a network namespace without a route for 127.0.0.2 fails
        // the connection before any policy is asked.
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
                "adversary::tests::the_adversary_is_bounded_on_linux",
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
        let tcp_ports = line.contains("tcp-port=on");
        for case in cases(&setup) {
            if !matches!(case.intent, Intent::Connect { .. }) {
                continue;
            }
            if case.name.ends_with("_other_ip") && !routable {
                eprintln!("SKIP {} reason=127.0.0.2 not routable", case.name);
                continue;
            }
            let tool = printed_verdict(&output.stdout, &case.name)
                .unwrap_or_else(|| panic!("the child printed no verdict for {}", case.name));
            let reached = servers.reached(&case.name);
            let policy = policy_verdict(tool, reached);
            // Without Landlock rules for stream ports the other port is not
            // refused either.
            let expected = if case.name == "http_get_other_port" && !tcp_ports {
                Verdict::Allowed
            } else {
                case.linux
            };
            println!(
                "ADV net case={} verdict={} reached={} detail={}",
                case.name,
                policy.as_str(),
                if reached { "yes" } else { "no" },
                tool.as_str()
            );
            if policy == Verdict::Allowed && case.rabbit == Verdict::Denied {
                println!(
                    "GAP case={} rabbit={} linux={} reached={} why={}",
                    case.name,
                    case.rabbit.as_str(),
                    policy.as_str(),
                    if reached { "yes" } else { "no" },
                    case.gap.unwrap_or("")
                );
            }
            assert_eq!(policy, expected, "{} (tool: {})", case.name, tool.as_str());
        }
    }

    /// The part that runs restricted. It prints every line of the report, the
    /// gaps of the files with the label `GAP`, and fails when a call does not
    /// come to the verdict the tree and the kernel give it. The network calls
    /// are judged by the parent, which owns the servers.
    ///
    /// A kernel that reports Landlock and does not enforce it fails the test,
    /// and so does a kernel with no Landlock when `RABBIT_REQUIRE_LANDLOCK=1`;
    /// otherwise a kernel with no Landlock skips the assertions, on stderr.
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    fn child(parameters: &str) {
        // libtest may have left `test name ... ` on this line.
        println!();
        let fields: Vec<&str> = parameters.split(',').collect();
        let port = |index: usize| fields[index].parse::<u16>().unwrap();
        let directory = fields[0];
        let mut setup = linux_setup(directory, [port(1), port(2), port(3), port(4)], false);
        let policy = setup.manifest().unwrap();

        // What the family comes to on this tree: a path that does not exist
        // answers `ENOENT` before Landlock is asked, so it fails; a path that
        // resolves into the root is let through, and a directory fails to
        // read; a path that resolves outside the root is refused.
        let root = std::fs::canonicalize(&setup.root).unwrap();
        let expected_family: std::collections::HashMap<String, Verdict> = cases(&setup)
            .into_iter()
            .filter(|case| case.name.starts_with("enum_"))
            .filter_map(|case| match case.intent {
                Intent::Open { path, .. } => {
                    let expected = match std::fs::canonicalize(&path) {
                        Err(_) => Verdict::Failed,
                        Ok(real) if real.starts_with(&root) && real.is_dir() => Verdict::Failed,
                        Ok(real) if real.starts_with(&root) => Verdict::Allowed,
                        Ok(_) => Verdict::Denied,
                    };
                    Some((case.name, expected))
                }
                _ => None,
            })
            .collect();

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
            assert!(
                std::env::var_os("RABBIT_REQUIRE_LANDLOCK").is_none_or(|value| value != "1"),
                "RABBIT_REQUIRE_LANDLOCK=1 and the kernel has no Landlock: {line}"
            );
            eprintln!("SKIP landlock unavailable: the kernel reports no Landlock ABI: {line}");
            println!("CHILD DONE");
            return;
        }
        for (case, verdict) in &results {
            if let Some(expected) = expected_family.get(&case.name) {
                assert_eq!(verdict, expected, "{}", case.name);
                continue;
            }
            if matches!(case.intent, Intent::Connect { .. }) {
                continue;
            }
            if let Some(why) = case.gap {
                println!(
                    "GAP case={} rabbit={} linux={} why={why}",
                    case.name,
                    case.rabbit.as_str(),
                    verdict.as_str()
                );
            }
            assert_eq!(*verdict, case.linux, "{}", case.name);
        }
        println!("CHILD DONE");
    }
}
