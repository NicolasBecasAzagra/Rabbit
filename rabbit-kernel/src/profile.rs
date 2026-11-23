// SPDX-License-Identifier: GPL-2.0-only
//! Cycle counters for the page fault path.
//!
//! Compiled in only with the `fault-profile` feature, which no published build
//! enables: a timestamp read on every fault would change what is measured. The
//! build without the feature has a `Lap` that holds nothing and a `mark` that
//! does nothing, so the call sites need no `cfg`.
//!
//! The counters answer one question, where the cycles of a fault go, so a change
//! to the path starts from a measurement and not from a guess.

#[cfg(feature = "fault-profile")]
mod enabled {
    use core::sync::atomic::{AtomicU64, Ordering};

    use rabbit_arch::cpu;

    /// A part of the fault path.
    #[derive(Clone, Copy)]
    pub enum Phase {
        /// The whole handler, from the Rust entry to its return.
        Total,
        /// The mapping lookup and the checks that precede the allocation.
        Lookup,
        /// The lock, the buddy allocation and the zero fill of the frame.
        Alloc,
        /// The zero fill alone, which `Alloc` includes.
        Zero,
        /// The page table update.
        Map,
        /// The TLB invalidation of the page.
        Invalidate,
    }

    const PHASES: usize = 6;
    const NAMES: [&str; PHASES] = ["total", "lookup", "alloc", "zero", "map", "invlpg"];

    static CYCLES: [AtomicU64; PHASES] = [const { AtomicU64::new(0) }; PHASES];
    static FAULTS: AtomicU64 = AtomicU64::new(0);
    static ZEROED: AtomicU64 = AtomicU64::new(0);

    /// A running timestamp: `mark` charges the cycles since the last mark.
    pub struct Lap(u64);

    impl Lap {
        #[inline(always)]
        pub fn start() -> Self {
            Lap(cpu::read_tsc())
        }

        #[inline(always)]
        pub fn mark(&mut self, phase: Phase) {
            let now = cpu::read_tsc();
            CYCLES[phase as usize].fetch_add(now.wrapping_sub(self.0), Ordering::Relaxed);
            self.0 = now;
            if matches!(phase, Phase::Zero) {
                ZEROED.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Count one fault that went through the whole handler.
    #[inline(always)]
    pub fn count_fault() {
        FAULTS.fetch_add(1, Ordering::Relaxed);
    }

    /// Print the mean cycles of every phase.
    pub fn report() {
        let faults = FAULTS.load(Ordering::Relaxed);
        let zeroed = ZEROED.load(Ordering::Relaxed);
        crate::kprintln!("[profile] faults={} zeroed={}", faults, zeroed);
        if faults == 0 {
            return;
        }
        for (index, name) in NAMES.iter().enumerate() {
            let cycles = CYCLES[index].load(Ordering::Relaxed);
            let count = if index == Phase::Zero as usize {
                zeroed
            } else {
                faults
            };
            crate::kprintln!(
                "[profile] {} cycles_per_event={}",
                name,
                cycles.checked_div(count).unwrap_or(0)
            );
        }
    }
}

#[cfg(feature = "fault-profile")]
pub use enabled::{count_fault, report, Lap, Phase};

#[cfg(not(feature = "fault-profile"))]
mod disabled {
    /// The phases exist so the call sites compile; nothing is recorded.
    #[derive(Clone, Copy)]
    pub enum Phase {
        Total,
        Lookup,
        Alloc,
        Zero,
        Map,
        Invalidate,
    }

    /// Holds nothing.
    pub struct Lap;

    impl Lap {
        #[inline(always)]
        pub fn start() -> Self {
            Lap
        }

        #[inline(always)]
        pub fn mark(&mut self, _phase: Phase) {}
    }

    #[inline(always)]
    pub fn count_fault() {}

    pub fn report() {}
}

#[cfg(not(feature = "fault-profile"))]
pub use disabled::{count_fault, report, Lap, Phase};
