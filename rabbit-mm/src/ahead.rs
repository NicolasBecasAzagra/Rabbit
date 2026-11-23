// SPDX-License-Identifier: GPL-2.0-only
//! How many pages one anonymous page fault maps: the sequential detector.
//!
//! A page fault costs a hardware exception, and a program that streams through
//! fresh memory, such as one that loads the weights of a model into a buffer,
//! takes one for every page. When a fault lands exactly where the previous one
//! left off, the program is walking forward, and the kernel can map the next
//! pages in the same call and pay for the exception once.
//!
//! The detector never reads ahead of a program that is not sequential. A fault
//! that is not at the expected address maps one page, so a sparse access
//! pattern, such as a hash table or a guard page, reserves exactly what it
//! touches. A sequential run grows its window by doubling, up to
//! [`MAX_WINDOW`], so the memory mapped and never used is bounded by one
//! window.
//!
//! The module is pure: it holds the policy and no page tables, so the host
//! tests cover what the kernel decides.

use crate::paging::PAGE_SIZE;

/// Most pages one fault maps: 64 KiB.
pub const MAX_WINDOW: u32 = 16;

/// What one processor remembers about its last anonymous fault.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stream {
    /// Address a sequential fault would land on, or zero before the first.
    next: u64,
    /// Pages the last fault mapped.
    window: u32,
}

impl Default for Stream {
    fn default() -> Self {
        Self::new()
    }
}

impl Stream {
    /// A detector that has seen no fault.
    pub const fn new() -> Self {
        Self { next: 0, window: 1 }
    }

    /// Pages to map for a fault at `address`, which is page aligned, when the
    /// mapping ends at `limit`. At least one, at most the pages that fit.
    pub fn pages_for(&self, address: u64, limit: u64) -> u32 {
        let wanted = if self.next != 0 && address == self.next {
            self.window.saturating_mul(2).min(MAX_WINDOW)
        } else {
            1
        };
        let room = limit.saturating_sub(address) / PAGE_SIZE;
        wanted.min(u32::try_from(room).unwrap_or(u32::MAX)).max(1)
    }

    /// Record that the fault at `address` mapped `mapped` pages.
    ///
    /// `mapped` can be smaller than what [`Stream::pages_for`] asked for: the
    /// window stops at a page that is already present or when memory runs low.
    pub fn record(&mut self, address: u64, mapped: u32) {
        let mapped = mapped.max(1);
        self.next = address.saturating_add(u64::from(mapped) * PAGE_SIZE);
        self.window = mapped;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: u64 = 0x4000_0000;
    const LIMIT: u64 = BASE + 1024 * PAGE_SIZE;

    /// Run `faults` sequential faults and return the window of each.
    fn sequential_windows(faults: usize) -> Vec<u32> {
        let mut stream = Stream::new();
        let mut address = BASE;
        let mut windows = Vec::new();
        for _ in 0..faults {
            let pages = stream.pages_for(address, LIMIT);
            windows.push(pages);
            stream.record(address, pages);
            address += u64::from(pages) * PAGE_SIZE;
        }
        windows
    }

    #[test]
    fn the_first_fault_maps_one_page() {
        assert_eq!(Stream::new().pages_for(BASE, LIMIT), 1);
    }

    #[test]
    fn a_sequential_run_doubles_its_window_up_to_the_cap() {
        assert_eq!(sequential_windows(7), vec![1, 2, 4, 8, 16, 16, 16]);
    }

    #[test]
    fn a_fault_that_is_not_the_next_page_maps_one_page() {
        let mut stream = Stream::new();
        stream.record(BASE, 1);
        // The expected address is BASE + 1 page; this one jumps ahead.
        assert_eq!(stream.pages_for(BASE + 10 * PAGE_SIZE, LIMIT), 1);
        // Going back is not sequential either.
        assert_eq!(stream.pages_for(BASE - PAGE_SIZE, LIMIT), 1);
    }

    #[test]
    fn a_sparse_access_pattern_never_reads_ahead() {
        let mut stream = Stream::new();
        for step in 0..64u64 {
            // Every fault is a few pages past the previous one.
            let address = BASE + step * 7 * PAGE_SIZE;
            let pages = stream.pages_for(address, LIMIT);
            assert_eq!(pages, 1, "fault {step} must map one page");
            stream.record(address, pages);
        }
    }

    #[test]
    fn a_run_that_breaks_starts_again_from_one_page() {
        let mut stream = Stream::new();
        let mut address = BASE;
        for _ in 0..4 {
            let pages = stream.pages_for(address, LIMIT);
            stream.record(address, pages);
            address += u64::from(pages) * PAGE_SIZE;
        }
        assert_eq!(stream.pages_for(address, LIMIT), 16);
        // A fault elsewhere ends the run, and the next sequential pair starts
        // small again.
        let elsewhere = BASE + 500 * PAGE_SIZE;
        assert_eq!(stream.pages_for(elsewhere, LIMIT), 1);
        stream.record(elsewhere, 1);
        assert_eq!(stream.pages_for(elsewhere + PAGE_SIZE, LIMIT), 2);
    }

    #[test]
    fn the_window_never_leaves_the_mapping() {
        let mut stream = Stream::new();
        stream.record(LIMIT - 9 * PAGE_SIZE, 8);
        // 8 mapped, so 16 are wanted, but only 1 page is left.
        assert_eq!(stream.pages_for(LIMIT - PAGE_SIZE, LIMIT), 1);
        // With 5 pages left a doubled window is clamped to the room.
        let mut stream = Stream::new();
        stream.record(LIMIT - 10 * PAGE_SIZE, 5);
        assert_eq!(stream.pages_for(LIMIT - 5 * PAGE_SIZE, LIMIT), 5);
    }

    #[test]
    fn a_window_cut_short_grows_from_what_was_mapped() {
        let mut stream = Stream::new();
        stream.record(BASE, 16);
        // The run stopped at 3 pages because the fourth was already present.
        stream.record(BASE + 16 * PAGE_SIZE, 3);
        assert_eq!(stream.pages_for(BASE + 19 * PAGE_SIZE, LIMIT), 6);
    }

    #[test]
    fn recording_zero_pages_still_counts_as_one() {
        let mut stream = Stream::new();
        stream.record(BASE, 0);
        assert_eq!(stream.pages_for(BASE + PAGE_SIZE, LIMIT), 2);
    }

    #[test]
    fn a_mapping_that_ends_at_the_fault_still_maps_the_page() {
        // The caller found the mapping, so the page is inside it: one page.
        let stream = Stream::new();
        assert_eq!(stream.pages_for(LIMIT - PAGE_SIZE, LIMIT), 1);
        assert_eq!(stream.pages_for(LIMIT, LIMIT), 1);
    }

    #[test]
    fn the_cap_bounds_the_memory_a_stopped_stream_wastes() {
        // A stream that stops right after a fault wasted at most the window
        // minus one page.
        let windows = sequential_windows(20);
        assert!(windows.iter().all(|w| *w <= MAX_WINDOW));
        assert_eq!(u64::from(MAX_WINDOW) * PAGE_SIZE, 64 * 1024);
    }
}
