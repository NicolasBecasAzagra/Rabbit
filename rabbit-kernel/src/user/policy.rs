// SPDX-License-Identifier: GPL-2.0-only
//! The capability policy of the running program, installed once and enforced.
//!
//! The program hands the kernel an encoded [`rabbit_policy::Policy`] through
//! the private system call `SYS_RABBIT_POLICY` of [`super::syscall`]. The
//! kernel decodes it, which validates every byte, and keeps it for the rest of
//! the boot. A second install is refused and leaves the first policy in place:
//! a program that could replace its own policy could widen it, so the policy
//! can only be set while it is still absent.
//!
//! Until a policy is installed every request is allowed, which is what the
//! kernel did before the policy existed. Once it is installed the rule is deny
//! by default: the checks below answer `EACCES` unless the policy lists the
//! request. Each denial prints one line, with a limit, and the text that comes
//! from the program is sanitized first so a program cannot forge a line of the
//! log.

use core::cell::UnsafeCell;
use core::fmt;
use core::sync::atomic::{AtomicU64, Ordering};

use rabbit_policy::{Policy, Protocol, Reason, Verdict, MAX_ENCODED_LEN};

use crate::net::SocketType;
use crate::sync::IrqSpinLock;

use super::errno::{EACCES, EFAULT, EINVAL, EPERM};
use super::syscall::user_range_faulted;

/// The slot of the policy.
///
/// There is one policy per boot: the kernel starts one program per boot and
/// has no `execve`, so a second install answers `EPERM` by design.
///
/// Known limitation: the helpers that read user memory have no fault fixup, so
/// a sibling thread that unmaps the range between the check and the copy can
/// fault the kernel. Every system call that reads user memory has this
/// property; it is not specific to this one.
struct Installed(UnsafeCell<Option<Policy>>);

// SAFETY: the cell is only reached while `LOCK` is held, which also keeps
// interrupts disabled, so no other thread or handler borrows it meanwhile.
unsafe impl Sync for Installed {}

static INSTALLED: Installed = Installed(UnsafeCell::new(None));

/// Guards [`INSTALLED`]. The recursion guard of the lock rejects a nested take.
static LOCK: IrqSpinLock = IrqSpinLock::new("policy");

/// Denial lines the log prints per boot. A program that is refused in a loop
/// cannot flood the serial console with them.
const MAX_DENIAL_LINES: u64 = 16;

/// Bytes of a path that a denial line prints.
const MAX_LOGGED_PATH: usize = 48;

/// Denials seen so far, printed or not.
static DENIALS: AtomicU64 = AtomicU64::new(0);

/// A path from user memory, as text that is safe to print.
///
/// Only the first [`MAX_LOGGED_PATH`] bytes are shown and every byte that is
/// not printable ASCII is shown as `?`, so a path that holds a newline and the
/// text of another line cannot make a line of the log that is not the
/// kernel's.
pub(super) struct Sanitized<'a>(pub(super) &'a [u8]);

impl fmt::Display for Sanitized<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for &byte in self.0.iter().take(MAX_LOGGED_PATH) {
            let shown = if (0x20..=0x7e).contains(&byte) {
                char::from(byte)
            } else {
                '?'
            };
            fmt::Write::write_char(f, shown)?;
        }
        Ok(())
    }
}

/// The address of an endpoint as `a.b.c.d:port`.
struct Address([u8; 4], u16);

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [a, b, c, d] = self.0;
        write!(f, "{a}.{b}.{c}.{d}:{}", self.1)
    }
}

fn protocol_name(protocol: Protocol) -> &'static str {
    match protocol {
        Protocol::Udp => "udp",
        Protocol::Tcp => "tcp",
    }
}

fn protocol_of(socket_type: SocketType) -> Protocol {
    match socket_type {
        SocketType::Datagram => Protocol::Udp,
        SocketType::Stream => Protocol::Tcp,
    }
}

/// Run `check` on the installed policy, or return `None` when there is none.
///
/// The slot is read inside one critical section and through a reference: the
/// policy is large for a kernel stack and is not copied. `check` must not
/// print or call anything that takes the lock.
fn with_installed<R>(check: impl FnOnce(&Policy) -> R) -> Option<R> {
    let _guard = LOCK.lock();
    // SAFETY: `LOCK` is held, so nothing writes the cell while this borrow
    // lives.
    let installed = unsafe { &*INSTALLED.0.get() };
    installed.as_ref().map(check)
}

/// Print one denial line, within the limit, and return the error to answer.
fn deny(line: fmt::Arguments<'_>) -> i64 {
    let seen = DENIALS.fetch_add(1, Ordering::Relaxed);
    if seen < MAX_DENIAL_LINES {
        crate::kprintln!("POLICY deny {line}");
    } else if seen == MAX_DENIAL_LINES {
        crate::kprintln!("POLICY deny log limit reached");
    }
    EACCES
}

/// Ask whether the program may open `path`, for writing when `write`.
///
/// The caller passes a path it has already copied into the kernel, so the
/// bytes that are checked are the bytes that are opened.
pub fn check_open(path: &[u8], write: bool) -> Result<(), i64> {
    match with_installed(|policy| policy.check_open(path, write)) {
        None | Some(Verdict::Allow) => Ok(()),
        Some(Verdict::Deny(reason)) => Err(deny(format_args!(
            "open reason={reason:?} path={}",
            Sanitized(path)
        ))),
    }
}

/// Ask whether the program may create a socket of `socket_type`.
pub fn check_socket(socket_type: SocketType) -> Result<(), i64> {
    let protocol = protocol_of(socket_type);
    match with_installed(|policy| policy.granted().contains(protocol.capability())) {
        None | Some(true) => Ok(()),
        Some(false) => Err(deny(format_args!(
            "socket reason={:?} proto={}",
            Reason::NotGranted,
            protocol_name(protocol)
        ))),
    }
}

/// Ask whether the program may reach `address` on `port` with a socket of
/// `socket_type`.
pub fn check_connect(socket_type: SocketType, address: [u8; 4], port: u16) -> Result<(), i64> {
    let protocol = protocol_of(socket_type);
    match with_installed(|policy| policy.check_connect(protocol, address, port)) {
        None | Some(Verdict::Allow) => Ok(()),
        Some(Verdict::Deny(reason)) => Err(deny(format_args!(
            "connect reason={reason:?} proto={} addr={}",
            protocol_name(protocol),
            Address(address, port)
        ))),
    }
}

/// Refuse the inbound call `call` once a policy is installed.
///
/// The policy has no capability for accepting connections, so with a policy
/// installed there is nothing that could allow it.
pub fn check_inbound(call: &str) -> Result<(), i64> {
    match with_installed(|_| ()) {
        None => Ok(()),
        Some(()) => Err(deny(format_args!("{call}"))),
    }
}

/// Install the policy encoded in `length` bytes at `address` of user memory.
///
/// `length` must be between one byte and [`MAX_ENCODED_LEN`], else `EINVAL`.
/// The bytes are copied into the kernel before they are decoded: the program
/// owns the memory and could change it while the decoder reads it, so the
/// bytes that were validated are the bytes that are kept. Bytes that do not
/// decode answer `EINVAL`, memory that is not mapped answers `EFAULT`, and a
/// policy that is already installed answers `EPERM`.
pub fn install(address: u64, length: u64) -> i64 {
    let length = match usize::try_from(length) {
        Ok(length) if (1..=MAX_ENCODED_LEN).contains(&length) => length,
        _ => return EINVAL,
    };
    let Some(length) = user_range_faulted(address, length as u64) else {
        return EFAULT;
    };
    let mut bytes = [0u8; MAX_ENCODED_LEN];
    for (offset, byte) in bytes[..length].iter_mut().enumerate() {
        // SAFETY: the range was checked as mapped user memory, and `offset` is
        // below `length`. The read is volatile because the program owns the
        // memory and may rewrite it while the kernel reads it: a plain read of
        // a slice would be a data race, and the compiler may read a byte more
        // than once. No reference to user memory is kept.
        *byte = unsafe { core::ptr::read_volatile((address as *const u8).add(offset)) };
    }
    let Ok(policy) = Policy::decode(&bytes[..length]) else {
        return EINVAL;
    };

    {
        let _guard = LOCK.lock();
        // SAFETY: `LOCK` is held, so this is the only borrow of the cell.
        let installed = unsafe { &mut *INSTALLED.0.get() };
        if installed.is_some() {
            return EPERM;
        }
        *installed = Some(policy);
    }
    crate::kprintln!("POLICY installed bytes={length}");
    0
}
