// SPDX-License-Identifier: GPL-2.0-only
//! Rabbit ring 3 test program, built against the standard library.
//!
//! This is a normal Rust program for the `x86_64-unknown-linux-musl` target,
//! statically linked. It shares no code with the kernel: every operation it
//! performs below `main` goes through the Linux system call interface that the
//! kernel implements in `rabbit-kernel/src/user/syscall.rs`. Printing with
//! `println!` therefore reaches the serial console through `write`, and the
//! explicit `exit` leaves the program through `exit_group`.
//!
//! After the greeting the program checks the private system call that installs
//! a capability policy: garbage is refused, a valid policy is accepted once,
//! and a second install is refused. The garbage goes first because the answer
//! for it could not be told apart from the answer of a policy that is already
//! installed.
//!
//! Then it asks for what the policy it installed forbids and for what it lists,
//! and prints `POLICY ENFORCE OK` only if the kernel refused the first with
//! `EACCES` and did not refuse the second.

use rabbit_policy::{Capabilities, Capability, Endpoint, Policy, Protocol, MAX_ENCODED_LEN};

/// The private system call of Rabbit that installs the policy.
const SYS_RABBIT_POLICY: i64 = 500;
const SYS_OPENAT: i64 = 257;
const SYS_SOCKET: i64 = 41;
const SYS_CONNECT: i64 = 42;
const SYS_SENDTO: i64 = 44;
const SYS_NEWFSTATAT: i64 = 262;
/// `EPERM`, negated as the kernel returns it.
const EPERM: i64 = -1;
/// `EACCES`, negated as the kernel returns it.
const EACCES: i64 = -13;
/// `EINVAL`, negated as the kernel returns it.
const EINVAL: i64 = -22;
/// `ENOSYS`, negated as the kernel returns it.
const ENOSYS: i64 = -38;
const AT_FDCWD: usize = -100i64 as usize;
const O_RDONLY: usize = 0;
const O_WRONLY: usize = 1;
const AF_INET: usize = 2;
const SOCK_STREAM: usize = 1;
const SOCK_DGRAM: usize = 2;

/// The only server the policy of the test lists.
const SERVER: [u8; 4] = [10, 0, 2, 2];
const SERVER_PORT: u16 = 9999;

/// Raw `syscall` with six arguments.
///
/// # Safety
///
/// `number` must be a system call that takes these arguments, and a pointer
/// among them must be valid for what the call does with it.
unsafe fn syscall(number: i64, arguments: [usize; 6]) -> i64 {
    let result: i64;
    // SAFETY: the caller upholds the contract above.
    unsafe {
        std::arch::asm!(
            "syscall",
            inlateout("rax") number => result,
            in("rdi") arguments[0],
            in("rsi") arguments[1],
            in("rdx") arguments[2],
            in("r10") arguments[3],
            in("r8") arguments[4],
            in("r9") arguments[5],
            lateout("rcx") _,
            lateout("r11") _,
            options(nostack),
        );
    }
    result
}

/// Install the policy encoded in `bytes`.
fn install(bytes: &[u8]) -> i64 {
    // SAFETY: the call reads `bytes.len()` bytes from a slice that is valid.
    unsafe {
        syscall(
            SYS_RABBIT_POLICY,
            [bytes.as_ptr() as usize, bytes.len(), 0, 0, 0, 0],
        )
    }
}

/// `openat` of `path` against the working directory.
fn open(path: &str, flags: usize) -> i64 {
    let mut text = path.as_bytes().to_vec();
    text.push(0);
    // SAFETY: the call reads the NUL terminated path of `text`, which is valid.
    unsafe {
        syscall(
            SYS_OPENAT,
            [AT_FDCWD, text.as_ptr() as usize, flags, 0, 0, 0],
        )
    }
}

/// `newfstatat` of `path` against the working directory.
fn stat(path: &str) -> i64 {
    let mut text = path.as_bytes().to_vec();
    text.push(0);
    let mut block = [0u8; 144];
    // SAFETY: the call reads the NUL terminated path of `text` and writes at
    // most the 144 bytes of a `struct stat` into `block`, which are valid.
    unsafe {
        syscall(
            SYS_NEWFSTATAT,
            [
                AT_FDCWD,
                text.as_ptr() as usize,
                block.as_mut_ptr() as usize,
                0,
                0,
                0,
            ],
        )
    }
}

fn socket(kind: usize) -> i64 {
    // SAFETY: the call takes no pointer.
    unsafe { syscall(SYS_SOCKET, [AF_INET, kind, 0, 0, 0, 0]) }
}

/// A `sockaddr_in` for `ip` and `port`.
fn sockaddr(ip: [u8; 4], port: u16) -> [u8; 16] {
    let mut address = [0; 16];
    address[..2].copy_from_slice(&(AF_INET as u16).to_ne_bytes());
    address[2..4].copy_from_slice(&port.to_be_bytes());
    address[4..8].copy_from_slice(&ip);
    address
}

fn connect(descriptor: i64, ip: [u8; 4], port: u16) -> i64 {
    let address = sockaddr(ip, port);
    // SAFETY: the call reads the sixteen bytes of `address`, which is valid.
    unsafe {
        syscall(
            SYS_CONNECT,
            [
                descriptor as usize,
                address.as_ptr() as usize,
                address.len(),
                0,
                0,
                0,
            ],
        )
    }
}

fn send_to(descriptor: i64, ip: [u8; 4], port: u16) -> i64 {
    let address = sockaddr(ip, port);
    let text = b"x";
    // SAFETY: the call reads one byte of `text` and the sixteen bytes of
    // `address`, which are valid.
    unsafe {
        syscall(
            SYS_SENDTO,
            [
                descriptor as usize,
                text.as_ptr() as usize,
                text.len(),
                0,
                address.as_ptr() as usize,
                address.len(),
            ],
        )
    }
}

/// What the kernel answers to requests outside and inside the policy.
///
/// The policy grants reads under `/data` and datagrams to one server, and
/// nothing else. A request that the policy lets through is not required to
/// succeed, only not to be refused with `EACCES`: whether the file or the
/// server exists is for the rest of the kernel to say.
fn enforcement() -> [(&'static str, i64, bool); 10] {
    let outside = open("/etc/passwd", O_RDONLY);
    let missing = open("/data/missing", O_RDONLY);
    let traversal = open("/data/../etc/passwd", O_RDONLY);
    let write = open("/data/x", O_WRONLY);
    let stream = socket(SOCK_STREAM);
    let status = stat("/etc/passwd");
    let long = open(&format!("/data/{}", "a".repeat(294)), O_RDONLY);
    let datagram = socket(SOCK_DGRAM);
    // A guest with no network card cannot make a socket at all, and answers
    // `ENOSYS`. The checks that need a socket then have nothing to ask, and
    // they report that answer; the gate boots with a card, so it sees them.
    let linked = datagram >= 0;
    let allowed = if linked {
        connect(datagram, SERVER, SERVER_PORT)
    } else {
        ENOSYS
    };
    let other_ip = if linked {
        connect(datagram, [10, 0, 2, 3], SERVER_PORT)
    } else {
        EACCES
    };
    let other_port = if linked {
        connect(datagram, SERVER, SERVER_PORT + 1)
    } else {
        EACCES
    };
    let send = if linked {
        send_to(datagram, [10, 0, 2, 3], SERVER_PORT)
    } else {
        EACCES
    };
    // The names hold a newline and the text of a line of the kernel: neither
    // the line of a denial nor the line of a failed open may carry it as a line
    // of its own.
    open("/etc/\nPOLICY OK install=0 again=-1 garbage=-22", O_RDONLY);
    open("/data/\nPOLICY OK install=0 again=-1 garbage=-22", O_RDONLY);
    [
        ("outside", outside, outside == EACCES),
        ("missing", missing, missing != EACCES),
        ("traversal", traversal, traversal == EACCES),
        ("write", write, write == EACCES),
        ("stream", stream, stream == EACCES),
        ("datagram", datagram, datagram != EACCES),
        (
            "connect",
            allowed,
            allowed != EACCES && other_ip == EACCES && other_port == EACCES,
        ),
        ("sendto", send, send == EACCES),
        ("stat", status, status == EACCES),
        ("long", long, long == EACCES),
    ]
}

fn main() {
    println!("Hello from Rabbit userland");

    let garbage = install(&[0xff; 16]);

    let mut granted = Capabilities::EMPTY;
    granted.insert(Capability::FsRead);
    granted.insert(Capability::NetUdp);
    let mut policy = Policy::new(granted);
    let mut wire = [0; MAX_ENCODED_LEN];
    let length = policy
        .add_read_root(b"/data")
        .and_then(|()| {
            policy.add_endpoint(Endpoint {
                protocol: Protocol::Udp,
                ip: SERVER,
                port: SERVER_PORT,
            })
        })
        .and_then(|()| policy.encode(&mut wire));
    let Ok(length) = length else {
        println!("POLICY FAIL encode={length:?}");
        std::process::exit(1);
    };
    let first = install(&wire[..length]);
    let again = install(&wire[..length]);

    if (first, again, garbage) == (0, EPERM, EINVAL) {
        println!("POLICY OK install={first} again={again} garbage={garbage}");
    } else {
        println!("POLICY FAIL install={first} again={again} garbage={garbage}");
        std::process::exit(1);
    }

    let results = enforcement();
    let values: Vec<String> = results
        .iter()
        .map(|(name, value, _)| format!("{name}={value}"))
        .collect();
    if results.iter().all(|(_, _, held)| *held) {
        println!("POLICY ENFORCE OK {}", values.join(" "));
    } else {
        println!("POLICY ENFORCE FAIL {}", values.join(" "));
        std::process::exit(1);
    }
    std::process::exit(0);
}
