// SPDX-License-Identifier: GPL-2.0-only
//! The capability policy of the running program, installed once.
//!
//! The program hands the kernel an encoded [`rabbit_policy::Policy`] through
//! the private system call `SYS_RABBIT_POLICY` of [`super::syscall`]. The
//! kernel decodes it, which validates every byte, and keeps it for the rest of
//! the boot. A
//! second install is refused and leaves the first policy in place: a program
//! that could replace its own policy could widen it, so the policy can only
//! be set while it is still absent.

use core::cell::UnsafeCell;

use rabbit_policy::{Policy, MAX_ENCODED_LEN};

use crate::sync::IrqSpinLock;

use super::errno::{EFAULT, EINVAL, EPERM};
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
