// SPDX-License-Identifier: GPL-2.0-only
//! The vector register state of a thread: which instruction saves it, and how.
//!
//! The kernel is built without SSE, so it never touches the vector registers
//! itself. The registers of the thread that was running therefore stay in the
//! CPU while the kernel switches to another thread, and the second thread would
//! read the values of the first unless the switch saves and restores them.
//!
//! The decision about what to enable is a pure function of two CPUID results
//! ([`plan`]), so it is tested on the host. The instructions that act on it
//! are only built for x86_64.
//!
//! AVX is enabled when the processor offers it: a program that detects AVX2 at
//! run time, as the standard library does, only sees it when `CR4.OSXSAVE` is
//! set and `XCR0` names the AVX state. AVX-512 is not enabled.

use core::sync::atomic::{AtomicU64, AtomicU8, Ordering};

/// Bytes reserved for the saved state of one thread.
///
/// The standard format with x87, SSE and AVX needs 832 bytes, see
/// [`required_size`]. The rest is room that a wider state does not use yet.
pub const AREA_SIZE: usize = 1024;

/// Alignment `XSAVE` requires of its area.
pub const AREA_ALIGN: usize = 64;

/// The state components of `XCR0`.
const XCR0_X87: u64 = 1 << 0;
const XCR0_SSE: u64 = 1 << 1;
const XCR0_AVX: u64 = 1 << 2;

/// `CPUID.1:ECX` bits.
const ECX_XSAVE: u32 = 1 << 26;
const ECX_AVX: u32 = 1 << 28;

/// Power on value of the x87 control word, and of `MXCSR`: every exception
/// masked, round to nearest.
const FCW_DEFAULT: u16 = 0x037F;
const MXCSR_DEFAULT: u32 = 0x1F80;

/// How the state is saved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// `FXSAVE`: x87 and SSE, 512 bytes, no AVX.
    Fxsave,
    /// `XSAVE` with the components of [`Plan::xcr0`].
    Xsave,
}

/// What to enable on every processor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub mode: Mode,
    /// Value for `XCR0`; zero when the mode is [`Mode::Fxsave`].
    pub xcr0: u64,
}

/// Bytes the standard `XSAVE` format uses for the components of `xcr0`.
///
/// The legacy area is 512 bytes and the header 64. The AVX component, the upper
/// halves of the sixteen `ymm` registers, is 256 bytes at offset 576.
pub const fn required_size(xcr0: u64) -> usize {
    if xcr0 & XCR0_AVX != 0 {
        576 + 256
    } else {
        576
    }
}

const _: () = assert!(required_size(XCR0_X87 | XCR0_SSE | XCR0_AVX) <= AREA_SIZE);

/// Decide what to enable from `CPUID.1:ECX` and from the components the
/// processor can put in `XCR0` (`CPUID.0xD.0:EDX:EAX`, zero without `XSAVE`).
pub const fn plan(leaf1_ecx: u32, supported_xcr0: u64) -> Plan {
    let legacy = XCR0_X87 | XCR0_SSE;
    if leaf1_ecx & ECX_XSAVE == 0 || supported_xcr0 & legacy != legacy {
        return Plan {
            mode: Mode::Fxsave,
            xcr0: 0,
        };
    }
    let mut xcr0 = legacy;
    if leaf1_ecx & ECX_AVX != 0 && supported_xcr0 & XCR0_AVX != 0 {
        xcr0 |= XCR0_AVX;
    }
    Plan {
        mode: Mode::Xsave,
        xcr0,
    }
}

/// The saved vector state of one thread.
#[repr(C, align(64))]
pub struct Area([u8; AREA_SIZE]);

impl Area {
    /// The state of a thread that has not run yet: the power on values.
    ///
    /// The control word and `MXCSR` are written because an all zero area would
    /// load `MXCSR = 0`, which leaves every floating point exception unmasked.
    /// The header of the `XSAVE` format is zero, so the components are
    /// initialised by the restore.
    pub const fn clean() -> Self {
        let mut bytes = [0u8; AREA_SIZE];
        let fcw = FCW_DEFAULT.to_le_bytes();
        bytes[0] = fcw[0];
        bytes[1] = fcw[1];
        let mxcsr = MXCSR_DEFAULT.to_le_bytes();
        bytes[24] = mxcsr[0];
        bytes[25] = mxcsr[1];
        bytes[26] = mxcsr[2];
        bytes[27] = mxcsr[3];
        Area(bytes)
    }

    /// The bytes of the area, for tests and diagnostics.
    pub const fn bytes(&self) -> &[u8; AREA_SIZE] {
        &self.0
    }
}

/// Mode chosen at boot: 0 for `FXSAVE`, 1 for `XSAVE`.
static MODE: AtomicU8 = AtomicU8::new(0);

/// `XCR0` chosen at boot, used as the requested feature bitmap of `XSAVE`.
static XCR0: AtomicU64 = AtomicU64::new(0);

/// The mode in use.
pub fn mode() -> Mode {
    if MODE.load(Ordering::Acquire) == 1 {
        Mode::Xsave
    } else {
        Mode::Fxsave
    }
}

/// The value `XCR0` holds on every processor.
pub fn xcr0() -> u64 {
    XCR0.load(Ordering::Acquire)
}

#[cfg(target_arch = "x86_64")]
mod imp {
    use core::arch::asm;
    use core::sync::atomic::Ordering;

    use super::{plan, Area, Mode, Plan, MODE, XCR0};
    use crate::cpu;

    const CR4_OSXSAVE: u64 = 1 << 18;

    /// Set `CR4.OSXSAVE` and write `XCR0` on the processor that calls it.
    fn enable_xsave(xcr0: u64) {
        let cr4: u64;
        // SAFETY: reading CR4 has no side effect.
        unsafe {
            asm!("mov {}, cr4", out(reg) cr4, options(nomem, nostack, preserves_flags));
        }
        // SAFETY: `OSXSAVE` is only set when CPUID reported `XSAVE`, which the
        // caller checked through the plan, and the other bits keep their value.
        unsafe {
            asm!(
                "mov cr4, {}",
                in(reg) cr4 | CR4_OSXSAVE,
                options(nostack, preserves_flags),
            );
        }
        // SAFETY: `OSXSAVE` is set, and `xcr0` is built from components that
        // CPUID reported as supported, with the legacy ones included.
        unsafe {
            asm!(
                "xsetbv",
                in("ecx") 0u32,
                in("eax") xcr0 as u32,
                in("edx") (xcr0 >> 32) as u32,
                options(nomem, nostack, preserves_flags),
            );
        }
    }

    /// Choose and enable the mode on the boot processor.
    pub fn init_boot() -> Plan {
        let leaf1 = cpu::cpuid(1);
        let supported = if leaf1.ecx & super::ECX_XSAVE != 0 && cpu::cpuid(0).eax >= 0xD {
            let state = cpu::cpuid_subleaf(0xD, 0);
            (u64::from(state.edx) << 32) | u64::from(state.eax)
        } else {
            0
        };
        let chosen = plan(leaf1.ecx, supported);
        if chosen.mode == Mode::Xsave {
            enable_xsave(chosen.xcr0);
            MODE.store(1, Ordering::Release);
        } else {
            MODE.store(0, Ordering::Release);
        }
        XCR0.store(chosen.xcr0, Ordering::Release);
        chosen
    }

    /// Enable on another processor what the boot processor chose.
    pub fn init_secondary() {
        if super::mode() == Mode::Xsave {
            enable_xsave(super::xcr0());
        }
    }

    /// Save the vector state of the running thread into `area`.
    ///
    /// # Safety
    ///
    /// `area` must point to a valid, writable [`Area`], and the mode must have
    /// been enabled on this processor.
    pub unsafe fn save(area: *mut Area) {
        let mask = super::xcr0();
        if super::mode() == Mode::Xsave {
            // SAFETY: the caller guarantees the area, which is 64 byte aligned
            // and larger than the state of the enabled components.
            unsafe {
                asm!(
                    "xsave64 [{area}]",
                    area = in(reg) area,
                    in("eax") mask as u32,
                    in("edx") (mask >> 32) as u32,
                    options(nostack, preserves_flags),
                );
            }
        } else {
            // SAFETY: as above; `FXSAVE` needs 16 byte alignment and 512 bytes.
            unsafe {
                asm!("fxsave64 [{area}]", area = in(reg) area, options(nostack, preserves_flags));
            }
        }
    }

    /// Load the vector state of the thread about to run from `area`.
    ///
    /// # Safety
    ///
    /// `area` must hold a state written by [`save`] or [`Area::clean`], and the
    /// mode must have been enabled on this processor.
    pub unsafe fn restore(area: *const Area) {
        let mask = super::xcr0();
        if super::mode() == Mode::Xsave {
            // SAFETY: the caller guarantees the area holds a valid state.
            unsafe {
                asm!(
                    "xrstor64 [{area}]",
                    area = in(reg) area,
                    in("eax") mask as u32,
                    in("edx") (mask >> 32) as u32,
                    options(nostack, preserves_flags),
                );
            }
        } else {
            // SAFETY: as above.
            unsafe {
                asm!("fxrstor64 [{area}]", area = in(reg) area, options(nostack, preserves_flags));
            }
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
mod imp {
    use super::{Area, Mode, Plan};

    /// Host stub: nothing to enable.
    pub fn init_boot() -> Plan {
        Plan {
            mode: Mode::Fxsave,
            xcr0: 0,
        }
    }

    /// Host stub: nothing to enable.
    pub fn init_secondary() {}

    /// Host stub: nothing to save.
    ///
    /// # Safety
    ///
    /// The host has no vector state to corrupt.
    pub unsafe fn save(_area: *mut Area) {}

    /// Host stub: nothing to restore.
    ///
    /// # Safety
    ///
    /// The host has no vector state to corrupt.
    pub unsafe fn restore(_area: *const Area) {}
}

pub use imp::{init_boot, init_secondary, restore, save};

#[cfg(test)]
mod tests {
    use super::*;

    const LEGACY: u64 = XCR0_X87 | XCR0_SSE;
    const WITH_AVX: u64 = LEGACY | XCR0_AVX;

    #[test]
    fn a_processor_without_xsave_uses_fxsave_and_no_avx() {
        let chosen = plan(ECX_AVX, 0);
        assert_eq!(chosen.mode, Mode::Fxsave);
        assert_eq!(chosen.xcr0, 0);
    }

    #[test]
    fn xsave_with_avx_enables_x87_sse_and_avx() {
        let chosen = plan(ECX_XSAVE | ECX_AVX, WITH_AVX);
        assert_eq!(chosen.mode, Mode::Xsave);
        assert_eq!(chosen.xcr0, WITH_AVX);
    }

    #[test]
    fn xsave_without_avx_stays_with_the_legacy_components() {
        let chosen = plan(ECX_XSAVE, LEGACY);
        assert_eq!(chosen.mode, Mode::Xsave);
        assert_eq!(chosen.xcr0, LEGACY);
    }

    #[test]
    fn the_avx_bit_is_not_enough_without_the_avx_state_component() {
        let chosen = plan(ECX_XSAVE | ECX_AVX, LEGACY);
        assert_eq!(chosen.xcr0, LEGACY);
    }

    #[test]
    fn avx512_components_are_never_enabled() {
        let everything = u64::MAX;
        let chosen = plan(ECX_XSAVE | ECX_AVX, everything);
        assert_eq!(chosen.xcr0, WITH_AVX);
    }

    #[test]
    fn a_processor_that_cannot_hold_the_legacy_state_falls_back() {
        // Bit 0 and bit 1 are both required in XCR0.
        assert_eq!(plan(ECX_XSAVE, XCR0_X87).mode, Mode::Fxsave);
    }

    #[test]
    fn the_area_covers_what_xsave_writes() {
        assert_eq!(required_size(LEGACY), 576);
        assert_eq!(required_size(WITH_AVX), 832);
        assert!(required_size(WITH_AVX) <= AREA_SIZE);
    }

    #[test]
    fn a_clean_area_has_the_power_on_control_words() {
        let area = Area::clean();
        let bytes = area.bytes();
        assert_eq!(u16::from_le_bytes([bytes[0], bytes[1]]), FCW_DEFAULT);
        assert_eq!(
            u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]),
            MXCSR_DEFAULT
        );
        // The XSAVE header is zero: nothing is marked as modified.
        assert!(bytes[512..576].iter().all(|byte| *byte == 0));
    }

    #[test]
    fn the_area_is_aligned_for_xsave() {
        assert_eq!(core::mem::align_of::<Area>(), AREA_ALIGN);
        assert_eq!(core::mem::size_of::<Area>(), AREA_SIZE);
    }
}
