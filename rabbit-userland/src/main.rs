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

use rabbit_policy::{Capabilities, Capability, Policy, MAX_ENCODED_LEN};

/// The private system call of Rabbit that installs the policy.
const SYS_RABBIT_POLICY: i64 = 500;
/// `EPERM`, negated as the kernel returns it.
const EPERM: i64 = -1;
/// `EINVAL`, negated as the kernel returns it.
const EINVAL: i64 = -22;

/// Raw `syscall` with two arguments.
///
/// # Safety
///
/// `number` must be a system call whose first two arguments are the values
/// given, and a pointer among them must be valid for what the call does with it.
unsafe fn syscall2(number: i64, argument1: usize, argument2: usize) -> i64 {
    let result: i64;
    // SAFETY: the caller upholds the contract above.
    unsafe {
        std::arch::asm!(
            "syscall",
            inlateout("rax") number => result,
            in("rdi") argument1,
            in("rsi") argument2,
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
    unsafe { syscall2(SYS_RABBIT_POLICY, bytes.as_ptr() as usize, bytes.len()) }
}

fn main() {
    println!("Hello from Rabbit userland");

    let garbage = install(&[0xff; 16]);

    let mut granted = Capabilities::EMPTY;
    granted.insert(Capability::FsRead);
    let mut policy = Policy::new(granted);
    let mut wire = [0; MAX_ENCODED_LEN];
    let length = policy
        .add_read_root(b"/data")
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
    std::process::exit(0);
}
