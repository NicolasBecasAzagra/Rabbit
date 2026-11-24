// SPDX-License-Identifier: GPL-2.0-only
//! Pure helpers to turn a UEFI memory map into a PVH start info block.
//!
//! Nothing here touches hardware: the caller reads the UEFI memory map and
//! provides the buffers the start info, the memory map and the module list are
//! written into, so the whole path is unit-testable on the host.

use crate::{
    HvmMemmapEntry, HvmModule, HvmStartInfo, MemmapKind, HVM_START_INFO_VERSION, HVM_START_MAGIC,
};

/// A physical range `(start, length, kind)`.
pub type Range = (u64, u64, MemmapKind);

/// Why a helper refused its input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// The caller buffer is too small for the output.
    BufferFull,
    /// A range wraps past the end of the address space, or the input is not
    /// sorted by start.
    InvalidRange,
}

/// Map a UEFI `EFI_MEMORY_TYPE` to the PVH memory map type.
///
/// Loader and boot services memory is free once the kernel has called
/// `ExitBootServices`, so it counts as RAM. Runtime services memory and every
/// type not listed stay reserved.
pub const fn memmap_kind_from_uefi(uefi_type: u32) -> MemmapKind {
    match uefi_type {
        1..=4 | 7 => MemmapKind::Ram,
        8 => MemmapKind::Unusable,
        9 => MemmapKind::Acpi,
        10 => MemmapKind::Nvs,
        14 => MemmapKind::Pmem,
        _ => MemmapKind::Reserved,
    }
}

/// Merge touching or overlapping ranges of the same kind.
///
/// `input` must be sorted by start; a range that wraps past the end of the
/// address space, or one that starts before the previous one, is an
/// [`Error::InvalidRange`]. Empty ranges are dropped. The merged ranges are
/// written to the front of `out` and their count is returned.
pub fn merge_adjacent(input: &[Range], out: &mut [Range]) -> Result<usize, Error> {
    let mut count = 0;
    let mut previous_start = 0;
    for &(start, len, kind) in input {
        let end = start.checked_add(len).ok_or(Error::InvalidRange)?;
        if start < previous_start {
            return Err(Error::InvalidRange);
        }
        previous_start = start;
        if len == 0 {
            continue;
        }
        if count > 0 {
            let last = &mut out[count - 1];
            let last_end = last.0 + last.1;
            if last.2 == kind && start <= last_end {
                last.1 = end.max(last_end) - last.0;
                continue;
            }
        }
        *out.get_mut(count).ok_or(Error::BufferFull)? = (start, len, kind);
        count += 1;
    }
    Ok(count)
}

/// Fills an [`HvmStartInfo`], its memory map and its module list.
///
/// The writer does not know where its buffers live. The caller must keep the
/// start info, the memory map and the module list out of every entry it pushes
/// as RAM, or the kernel will hand them out as free memory.
pub struct StartInfoWriter<'a> {
    start: HvmStartInfo,
    memmap: &'a mut [HvmMemmapEntry],
    memmap_len: usize,
    modules: &'a mut [HvmModule],
    modules_len: usize,
}

impl<'a> StartInfoWriter<'a> {
    /// Start an empty version 1 start info backed by the given buffers.
    pub fn new(memmap: &'a mut [HvmMemmapEntry], modules: &'a mut [HvmModule]) -> Self {
        Self {
            start: HvmStartInfo {
                magic: HVM_START_MAGIC,
                version: HVM_START_INFO_VERSION,
                flags: 0,
                nr_modules: 0,
                modlist_paddr: 0,
                cmdline_paddr: 0,
                rsdp_paddr: 0,
                memmap_paddr: 0,
                memmap_entries: 0,
                reserved: 0,
            },
            memmap,
            memmap_len: 0,
            modules,
            modules_len: 0,
        }
    }

    /// Physical address of the ACPI RSDP, or 0 when there is none.
    pub fn set_rsdp(&mut self, rsdp_paddr: u64) {
        self.start.rsdp_paddr = rsdp_paddr;
    }

    /// Physical addresses the memory map and module list buffers live at.
    pub fn set_table_paddrs(&mut self, memmap_paddr: u64, modlist_paddr: u64) {
        self.start.memmap_paddr = memmap_paddr;
        self.start.modlist_paddr = modlist_paddr;
    }

    /// Append one memory map entry.
    pub fn push_memmap(&mut self, addr: u64, size: u64, kind: MemmapKind) -> Result<(), Error> {
        addr.checked_add(size).ok_or(Error::InvalidRange)?;
        *self
            .memmap
            .get_mut(self.memmap_len)
            .ok_or(Error::BufferFull)? = HvmMemmapEntry {
            addr,
            size,
            kind: kind as u32,
            reserved: 0,
        };
        self.memmap_len += 1;
        Ok(())
    }

    /// Append one module, such as the initramfs.
    pub fn push_module(&mut self, paddr: u64, size: u64, cmdline_paddr: u64) -> Result<(), Error> {
        *self
            .modules
            .get_mut(self.modules_len)
            .ok_or(Error::BufferFull)? = HvmModule {
            paddr,
            size,
            cmdline_paddr,
            reserved: 0,
        };
        self.modules_len += 1;
        Ok(())
    }

    /// Finish the header counts and return the start info with the written
    /// part of the memory map and module list.
    pub fn finish(mut self) -> (HvmStartInfo, &'a [HvmMemmapEntry], &'a [HvmModule]) {
        self.start.memmap_entries = self.memmap_len as u32;
        self.start.nr_modules = self.modules_len as u32;
        let memmap: &'a [HvmMemmapEntry] = &self.memmap[..self.memmap_len];
        let modules: &'a [HvmModule] = &self.modules[..self.modules_len];
        (self.start, memmap, modules)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BootInfo;

    const EMPTY_ENTRY: HvmMemmapEntry = HvmMemmapEntry {
        addr: 0,
        size: 0,
        kind: 0,
        reserved: 0,
    };
    const EMPTY_MODULE: HvmModule = HvmModule {
        paddr: 0,
        size: 0,
        cmdline_paddr: 0,
        reserved: 0,
    };

    #[test]
    fn maps_uefi_types() {
        let table = [
            (0, MemmapKind::Reserved),
            (1, MemmapKind::Ram),
            (2, MemmapKind::Ram),
            (3, MemmapKind::Ram),
            (4, MemmapKind::Ram),
            (5, MemmapKind::Reserved),
            (6, MemmapKind::Reserved),
            (7, MemmapKind::Ram),
            (8, MemmapKind::Unusable),
            (9, MemmapKind::Acpi),
            (10, MemmapKind::Nvs),
            (11, MemmapKind::Reserved),
            (12, MemmapKind::Reserved),
            (13, MemmapKind::Reserved),
            (14, MemmapKind::Pmem),
            (15, MemmapKind::Reserved),
            (0x8000_0000, MemmapKind::Reserved),
        ];
        for (uefi, kind) in table {
            assert_eq!(memmap_kind_from_uefi(uefi), kind, "uefi type {uefi}");
        }
    }

    #[test]
    fn merges_adjacent_ranges_of_the_same_kind() {
        let input = [
            (0x1000, 0x1000, MemmapKind::Ram),
            (0x2000, 0x2000, MemmapKind::Ram),
            (0x4000, 0x1000, MemmapKind::Ram),
        ];
        let mut out = [(0, 0, MemmapKind::Reserved); 3];
        let n = merge_adjacent(&input, &mut out).unwrap();
        assert_eq!(&out[..n], &[(0x1000, 0x4000, MemmapKind::Ram)]);
    }

    #[test]
    fn keeps_ranges_apart_when_there_is_a_gap() {
        let input = [
            (0x1000, 0x1000, MemmapKind::Ram),
            (0x3000, 0x1000, MemmapKind::Ram),
        ];
        let mut out = [(0, 0, MemmapKind::Reserved); 2];
        let n = merge_adjacent(&input, &mut out).unwrap();
        assert_eq!(&out[..n], &input);
    }

    #[test]
    fn keeps_ranges_apart_when_the_kinds_differ() {
        let input = [
            (0x1000, 0x1000, MemmapKind::Ram),
            (0x2000, 0x1000, MemmapKind::Acpi),
            (0x3000, 0x1000, MemmapKind::Ram),
        ];
        let mut out = [(0, 0, MemmapKind::Reserved); 3];
        let n = merge_adjacent(&input, &mut out).unwrap();
        assert_eq!(&out[..n], &input);
    }

    #[test]
    fn merge_drops_empty_ranges_and_reports_a_full_buffer() {
        let input = [
            (0x1000, 0, MemmapKind::Ram),
            (0x2000, 0x1000, MemmapKind::Ram),
            (0x4000, 0x1000, MemmapKind::Nvs),
        ];
        let mut big = [(0, 0, MemmapKind::Reserved); 3];
        assert_eq!(merge_adjacent(&input, &mut big), Ok(2));
        let mut small = [(0, 0, MemmapKind::Reserved); 1];
        assert_eq!(merge_adjacent(&input, &mut small), Err(Error::BufferFull));
    }

    #[test]
    fn round_trips_through_boot_info_parse() {
        let mut memmap = [EMPTY_ENTRY; 4];
        let mut modules = [EMPTY_MODULE; 2];
        let mut writer = StartInfoWriter::new(&mut memmap, &mut modules);
        writer.set_rsdp(0xf_6a00);
        writer.set_table_paddrs(0x7000, 0x8000);
        writer.push_memmap(0x0, 0x9_f000, MemmapKind::Ram).unwrap();
        writer
            .push_memmap(0x10_0000, 0x1f0_0000, MemmapKind::Ram)
            .unwrap();
        writer
            .push_memmap(0x200_0000, 0x1_0000, MemmapKind::Acpi)
            .unwrap();
        writer.push_module(0x100_0000, 0x4000, 0x7f00).unwrap();
        let (start, memmap, modules) = writer.finish();

        assert_eq!(start.magic, HVM_START_MAGIC);
        assert_eq!(start.version, 1);
        let info = BootInfo::parse(&start, memmap, modules).expect("writer output parses");
        assert_eq!(info.start_info().rsdp_paddr, 0xf_6a00);
        assert_eq!(info.start_info().memmap_paddr, 0x7000);
        assert_eq!(info.start_info().modlist_paddr, 0x8000);
        assert_eq!(info.memmap().len(), 3);
        assert_eq!(info.memmap()[1].addr, 0x10_0000);
        assert_eq!(info.memmap()[1].size, 0x1f0_0000);
        assert_eq!(info.memmap()[1].kind(), Some(MemmapKind::Ram));
        assert_eq!(info.memmap()[2].kind(), Some(MemmapKind::Acpi));
        assert_eq!(info.total_ram(), 0x9_f000 + 0x1f0_0000);
        assert_eq!(info.modules().len(), 1);
        assert_eq!(info.modules()[0].paddr, 0x100_0000);
        assert_eq!(info.modules()[0].size, 0x4000);
        assert_eq!(info.modules()[0].cmdline_paddr, 0x7f00);
    }

    #[test]
    fn writer_without_modules_or_rsdp_parses() {
        let mut memmap = [EMPTY_ENTRY; 1];
        let mut writer = StartInfoWriter::new(&mut memmap, &mut []);
        writer.push_memmap(0, 0x1000, MemmapKind::Ram).unwrap();
        let (start, memmap, modules) = writer.finish();
        let info = BootInfo::parse(&start, memmap, modules).expect("valid");
        assert_eq!(info.start_info().rsdp_paddr, 0);
        assert!(info.modules().is_empty());
    }

    #[test]
    fn writer_reports_full_buffers() {
        let mut memmap = [EMPTY_ENTRY; 1];
        let mut modules = [EMPTY_MODULE; 0];
        let mut writer = StartInfoWriter::new(&mut memmap, &mut modules);
        writer.push_memmap(0, 0x1000, MemmapKind::Ram).unwrap();
        assert_eq!(
            writer.push_memmap(0x1000, 0x1000, MemmapKind::Ram),
            Err(Error::BufferFull)
        );
        assert_eq!(writer.push_module(0, 1, 0), Err(Error::BufferFull));
    }

    #[test]
    fn merge_rejects_wrapping_and_unsorted_input() {
        let mut out = [(0, 0, MemmapKind::Reserved); 2];
        let wrap = [(u64::MAX - 0xfff, 0x2000, MemmapKind::Ram)];
        assert_eq!(merge_adjacent(&wrap, &mut out), Err(Error::InvalidRange));
        let unsorted = [
            (0x2000, 0x1000, MemmapKind::Ram),
            (0x1000, 0x1000, MemmapKind::Ram),
        ];
        assert_eq!(
            merge_adjacent(&unsorted, &mut out),
            Err(Error::InvalidRange)
        );
    }

    #[test]
    fn merge_reaches_the_last_byte_of_the_address_space() {
        let input = [
            (u64::MAX - 0x1fff, 0x1000, MemmapKind::Reserved),
            (u64::MAX - 0xfff, 0xfff, MemmapKind::Reserved),
        ];
        let mut out = [(0, 0, MemmapKind::Ram); 2];
        let n = merge_adjacent(&input, &mut out).unwrap();
        assert_eq!(
            &out[..n],
            &[(u64::MAX - 0x1fff, 0x1fff, MemmapKind::Reserved)]
        );
    }

    #[test]
    fn merge_joins_across_an_empty_range() {
        let input = [
            (0x1000, 0x1000, MemmapKind::Ram),
            (0x2000, 0, MemmapKind::Acpi),
            (0x2000, 0x1000, MemmapKind::Ram),
        ];
        let mut out = [(0, 0, MemmapKind::Reserved); 2];
        let n = merge_adjacent(&input, &mut out).unwrap();
        assert_eq!(&out[..n], &[(0x1000, 0x2000, MemmapKind::Ram)]);
    }

    #[test]
    fn writer_rejects_an_entry_that_wraps() {
        let mut memmap = [EMPTY_ENTRY; 1];
        let mut writer = StartInfoWriter::new(&mut memmap, &mut []);
        assert_eq!(
            writer.push_memmap(u64::MAX, 2, MemmapKind::Ram),
            Err(Error::InvalidRange)
        );
    }

    #[test]
    fn finish_with_nothing_pushed_reports_zero_counts() {
        let writer = StartInfoWriter::new(&mut [], &mut []);
        let (start, memmap, modules) = writer.finish();
        assert_eq!(start.memmap_entries, 0);
        assert_eq!(start.nr_modules, 0);
        assert!(memmap.is_empty() && modules.is_empty());
    }
}
