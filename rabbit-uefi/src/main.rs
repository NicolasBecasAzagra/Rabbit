// SPDX-License-Identifier: GPL-2.0-only
//! Rabbit UEFI loader (first stage).
//!
//! A `no_std`, `no_main` PE32+ EFI application. It embeds the kernel ELF and
//! loads it at its physical addresses, finds the ACPI RSDP in the
//! configuration table, builds a PVH `hvm_start_info` with a memory map below
//! 1 MiB, leaves boot services and checks its own output with the same parser
//! the kernel uses, reporting on COM1. The jump to the kernel comes in a later
//! change.
//!
//! The firmware structures are declared by hand with the layout of the UEFI
//! specification, and every offset this program relies on is asserted at
//! compile time. The console protocols are never used: after boot services
//! are gone only the serial port is touched, with direct port access.

#![no_std]
#![no_main]
#![deny(unsafe_op_in_unsafe_fn)]

use core::arch::asm;
use core::ffi::c_void;
use core::mem::{align_of, offset_of, size_of};
use core::ptr;

use rabbit_bootinfo::uefi::{memmap_kind_from_uefi, merge_adjacent, Range, StartInfoWriter};
use rabbit_bootinfo::{BootInfo, HvmMemmapEntry, HvmModule, HvmStartInfo, MemmapKind};

mod elf;

/// The kernel ELF, embedded at build time. `RABBIT_KERNEL_ELF` is the absolute
/// path of the kernel the `uefi` make target builds first.
static KERNEL_ELF: &[u8] = include_bytes!(env!("RABBIT_KERNEL_ELF"));

/// An opaque firmware handle.
type Handle = *mut c_void;
/// An `EFI_STATUS`: zero is success, errors have the top bit set.
type Status = usize;

const EFI_SUCCESS: Status = 0;
const EFI_BUFFER_TOO_SMALL: Status = 0x8000_0000_0000_0005;

/// `EFI_ALLOCATE_TYPE`.
#[allow(dead_code)]
mod alloc_type {
    pub const ANY_PAGES: u32 = 0;
    pub const MAX_ADDRESS: u32 = 1;
    pub const ADDRESS: u32 = 2;
}

/// `EFI_MEMORY_TYPE` of the buffers this program allocates.
const LOADER_DATA: u32 = 2;

/// Vendor GUID of the ACPI 2.0 table in the configuration table.
const ACPI_20_GUID: [u8; 16] = [
    0x71, 0xE8, 0x68, 0x88, 0xF1, 0xE4, 0xD3, 0x11, 0xBC, 0x22, 0x00, 0x80, 0xC7, 0x3C, 0x88, 0x81,
];
/// Vendor GUID of the ACPI 1.0 table in the configuration table.
const ACPI_10_GUID: [u8; 16] = [
    0x30, 0x2D, 0x9D, 0xEB, 0x88, 0x2D, 0xD3, 0x11, 0x9A, 0x16, 0x00, 0x90, 0x27, 0x3F, 0xC1, 0x4D,
];

/// Slack, in descriptors, added to the size the firmware reports: the call
/// that allocates the buffer can itself add entries to the map.
const MAP_SLACK_DESCRIPTORS: usize = 16;
/// Descriptor stride assumed if the firmware reports none in the size query.
const DEFAULT_DESCRIPTOR_SIZE: usize = 48;
/// Attempts to obtain a buffer large enough for the memory map.
const MAP_SIZE_ATTEMPTS: usize = 4;
/// Attempts at `ExitBootServices` before the loader gives up.
const EXIT_ATTEMPTS: usize = 8;

const PAGE_SIZE: usize = elf::PAGE_SIZE as usize;
/// Physical address of the AP trampoline page (`rabbit-kernel/src/smp.rs`).
const AP_TRAMPOLINE_PHYS: u64 = 0x8000;
/// Highest byte the start info area may use: it lives below 1 MiB, where a
/// kernel that follows the PVH contract expects to find it.
const START_INFO_LIMIT: u64 = 0xF_FFFF;
/// Descriptors of headroom over the map size seen before the buffers below are
/// allocated: allocating and merging both change the count.
const MEMMAP_SLACK_ENTRIES: usize = 64;
/// Module slots reserved in the start info area; none is used yet.
const MODULE_SLOTS: usize = 1;
/// Offset of the module list in the start info area, right after the header.
const MODLIST_OFFSET: usize = size_of::<HvmStartInfo>();
/// Offset of the memory map array, right after the module slots.
const MEMMAP_OFFSET: usize = MODLIST_OFFSET + MODULE_SLOTS * size_of::<HvmModule>();

/// `EFI_TABLE_HEADER`.
#[repr(C)]
struct TableHeader {
    signature: u64,
    revision: u32,
    header_size: u32,
    crc32: u32,
    reserved: u32,
}

/// `EFI_SYSTEM_TABLE`, up to the configuration table.
#[repr(C)]
struct SystemTable {
    header: TableHeader,
    firmware_vendor: *const u16,
    firmware_revision: u32,
    console_in_handle: Handle,
    con_in: *mut c_void,
    console_out_handle: Handle,
    con_out: *mut c_void,
    standard_error_handle: Handle,
    std_err: *mut c_void,
    runtime_services: *mut c_void,
    boot_services: *const BootServices,
    number_of_table_entries: usize,
    configuration_table: *const ConfigurationTable,
}

type AllocatePages = unsafe extern "efiapi" fn(u32, u32, usize, *mut u64) -> Status;
type FreePages = unsafe extern "efiapi" fn(u64, usize) -> Status;
type GetMemoryMap =
    unsafe extern "efiapi" fn(*mut usize, *mut u8, *mut usize, *mut usize, *mut u32) -> Status;
type AllocatePool = unsafe extern "efiapi" fn(u32, usize, *mut *mut u8) -> Status;
type FreePool = unsafe extern "efiapi" fn(*mut u8) -> Status;
type ExitBootServices = unsafe extern "efiapi" fn(Handle, usize) -> Status;

/// `EFI_BOOT_SERVICES`, up to `ExitBootServices`.
#[repr(C)]
#[allow(dead_code)]
struct BootServices {
    header: TableHeader,
    raise_tpl: usize,
    restore_tpl: usize,
    allocate_pages: AllocatePages,
    free_pages: FreePages,
    get_memory_map: GetMemoryMap,
    allocate_pool: AllocatePool,
    free_pool: FreePool,
    /// Entries between `FreePool` and `ExitBootServices`, unused here.
    unused: [usize; 19],
    exit_boot_services: ExitBootServices,
}

/// `EFI_MEMORY_DESCRIPTOR`. The firmware may use a larger stride, so a map
/// is always walked with the descriptor size that `GetMemoryMap` returns.
#[repr(C)]
#[allow(dead_code)]
struct MemoryDescriptor {
    kind: u32,
    physical_start: u64,
    virtual_start: u64,
    number_of_pages: u64,
    attribute: u64,
}

/// `EFI_CONFIGURATION_TABLE`.
#[repr(C)]
struct ConfigurationTable {
    vendor_guid: [u8; 16],
    vendor_table: *const c_void,
}

const _: () = {
    assert!(size_of::<TableHeader>() == 24);
    assert!(offset_of!(SystemTable, boot_services) == 96);
    assert!(offset_of!(SystemTable, number_of_table_entries) == 104);
    assert!(offset_of!(SystemTable, configuration_table) == 112);
    assert!(offset_of!(BootServices, allocate_pages) == 40);
    assert!(offset_of!(BootServices, free_pages) == 48);
    assert!(offset_of!(BootServices, get_memory_map) == 56);
    assert!(offset_of!(BootServices, allocate_pool) == 64);
    assert!(offset_of!(BootServices, free_pool) == 72);
    assert!(offset_of!(BootServices, exit_boot_services) == 232);
    assert!(offset_of!(MemoryDescriptor, kind) == 0);
    assert!(offset_of!(MemoryDescriptor, physical_start) == 8);
    assert!(offset_of!(MemoryDescriptor, virtual_start) == 16);
    assert!(offset_of!(MemoryDescriptor, number_of_pages) == 24);
    assert!(offset_of!(MemoryDescriptor, attribute) == 32);
    assert!(size_of::<MemoryDescriptor>() == 40);
    assert!(offset_of!(ConfigurationTable, vendor_guid) == 0);
    assert!(offset_of!(ConfigurationTable, vendor_table) == 16);
    assert!(size_of::<ConfigurationTable>() == 24);
    // The start info area is page aligned, so 8-byte offsets keep every
    // structure in it aligned.
    assert!(MODLIST_OFFSET.is_multiple_of(8) && MEMMAP_OFFSET.is_multiple_of(8));
    assert!(align_of::<HvmStartInfo>() == 8 && align_of::<HvmModule>() == 8);
    assert!(align_of::<HvmMemmapEntry>() == 8 && align_of::<Range>() == 8);
};

/// A memory map buffer and what the firmware reported about it.
struct MemoryMap {
    buffer: *mut u8,
    capacity: usize,
    key: usize,
    /// Bytes of the buffer the last successful `GetMemoryMap` filled.
    size: usize,
    /// Stride of the descriptors, as the last successful `GetMemoryMap`
    /// returned it. A map is always walked with this and never with the size
    /// of `MemoryDescriptor`.
    descriptor_size: usize,
}

/// Stop the processor for good.
fn halt() -> ! {
    // SAFETY: `cli` and `hlt` touch no memory. Masking interrupts keeps the
    // maskable ones from waking the loop; an NMI or SMI still can, and then
    // the loop halts again.
    unsafe { asm!("cli", options(nomem, nostack)) };
    loop {
        // SAFETY: see above.
        unsafe { asm!("hlt", options(nomem, nostack)) };
    }
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    halt()
}

/// Find the RSDP in the configuration table: ACPI 2.0 first, then 1.0.
/// Returns zero if the firmware publishes none.
fn find_rsdp(st: &SystemTable) -> u64 {
    if st.configuration_table.is_null() || st.number_of_table_entries == 0 {
        return 0;
    }
    // SAFETY: the pointer is not null, and the firmware guarantees it points to
    // `number_of_table_entries` valid entries while boot services run.
    let tables =
        unsafe { core::slice::from_raw_parts(st.configuration_table, st.number_of_table_entries) };
    for guid in [&ACPI_20_GUID, &ACPI_10_GUID] {
        if let Some(entry) = tables.iter().find(|entry| &entry.vendor_guid == guid) {
            return entry.vendor_table as u64;
        }
    }
    0
}

/// Ask for the size of the memory map and the descriptor stride.
fn memory_map_size(bs: &BootServices) -> Option<(usize, usize)> {
    let mut size = 0usize;
    let mut key = 0usize;
    let mut descriptor_size = 0usize;
    let mut version = 0u32;
    // SAFETY: every pointer refers to a live local; a null map with size zero
    // is the documented way to query the size.
    let status = unsafe {
        (bs.get_memory_map)(
            &mut size,
            ptr::null_mut(),
            &mut key,
            &mut descriptor_size,
            &mut version,
        )
    };
    if status != EFI_BUFFER_TOO_SMALL {
        return None;
    }
    if descriptor_size == 0 {
        descriptor_size = DEFAULT_DESCRIPTOR_SIZE;
    }
    Some((size, descriptor_size))
}

/// Allocate `size` bytes of loader data.
fn allocate_pool(bs: &BootServices, size: usize) -> Option<*mut u8> {
    let mut buffer: *mut u8 = ptr::null_mut();
    // SAFETY: `buffer` is a live local that receives the allocation.
    let status = unsafe { (bs.allocate_pool)(LOADER_DATA, size, &mut buffer) };
    (status == EFI_SUCCESS && !buffer.is_null()).then_some(buffer)
}

/// Why a `GetMemoryMap` call did not give a usable map.
enum MapError {
    /// The firmware returned this status; with `EFI_BUFFER_TOO_SMALL` the size
    /// is what it wants.
    Firmware(Status, usize),
    /// The firmware reported a descriptor size below that of `MemoryDescriptor`.
    Stride,
}

const STRIDE_ERROR: &str = "firmware reports a memory map descriptor size below 40";

/// Read the memory map into the buffer. A map whose descriptor size is below
/// that of `MemoryDescriptor` is rejected, so no key from such a call is used.
fn read_memory_map(bs: &BootServices, map: &mut MemoryMap) -> Result<(), MapError> {
    let mut size = map.capacity;
    let mut descriptor_size = 0usize;
    let mut version = 0u32;
    // SAFETY: `buffer` was returned by `AllocatePool` with `capacity` bytes,
    // and the other pointers refer to live locals or fields.
    let status = unsafe {
        (bs.get_memory_map)(
            &mut size,
            map.buffer,
            &mut map.key,
            &mut descriptor_size,
            &mut version,
        )
    };
    if status != EFI_SUCCESS {
        return Err(MapError::Firmware(status, size));
    }
    if descriptor_size < size_of::<MemoryDescriptor>() {
        return Err(MapError::Stride);
    }
    map.size = size;
    map.descriptor_size = descriptor_size;
    Ok(())
}

/// Allocate a buffer for the memory map and read it, growing the buffer when
/// the firmware asks for more.
fn first_memory_map(bs: &BootServices) -> Result<MemoryMap, &'static str> {
    let (mut size, descriptor_size) = memory_map_size(bs).ok_or("cannot size the memory map")?;
    for _ in 0..MAP_SIZE_ATTEMPTS {
        let capacity = size + MAP_SLACK_DESCRIPTORS * descriptor_size;
        let buffer = allocate_pool(bs, capacity).ok_or("cannot allocate the memory map buffer")?;
        let mut map = MemoryMap {
            buffer,
            capacity,
            key: 0,
            size: 0,
            descriptor_size,
        };
        match read_memory_map(bs, &mut map) {
            Ok(()) => return Ok(map),
            Err(MapError::Firmware(EFI_BUFFER_TOO_SMALL, wanted)) => size = wanted.max(capacity),
            Err(MapError::Firmware(..)) => return Err("cannot read the memory map"),
            Err(MapError::Stride) => return Err(STRIDE_ERROR),
        }
    }
    Err("the memory map keeps growing")
}

/// Allocate a larger buffer for a map that no longer fits and read it again,
/// as many times as the firmware keeps asking for more.
fn regrow_memory_map(
    bs: &BootServices,
    map: &mut MemoryMap,
    mut wanted: usize,
) -> Result<(), &'static str> {
    for _ in 0..MAP_SIZE_ATTEMPTS {
        // Allocating changes the map once more, so the slack covers that.
        let capacity = wanted + MAP_SLACK_DESCRIPTORS * map.descriptor_size;
        map.buffer =
            allocate_pool(bs, capacity).ok_or("cannot allocate a larger memory map buffer")?;
        map.capacity = capacity;
        match read_memory_map(bs, map) {
            Ok(()) => return Ok(()),
            Err(MapError::Firmware(EFI_BUFFER_TOO_SMALL, more)) => wanted = more.max(capacity),
            Err(MapError::Firmware(..)) => return Err("cannot read the grown memory map"),
            Err(MapError::Stride) => return Err(STRIDE_ERROR),
        }
    }
    Err("the memory map keeps growing")
}

/// Leave boot services. A failure means the map changed since the key was
/// taken, so the map is read again; nothing else is called in between.
fn exit_boot_services(
    image: Handle,
    bs: &BootServices,
    map: &mut MemoryMap,
) -> Result<(), &'static str> {
    for _ in 0..EXIT_ATTEMPTS {
        // SAFETY: `image` is the handle the firmware passed to `efi_main` and
        // `key` comes from the latest `GetMemoryMap`.
        let status = unsafe { (bs.exit_boot_services)(image, map.key) };
        if status == EFI_SUCCESS {
            return Ok(());
        }
        match read_memory_map(bs, map) {
            Ok(()) => {}
            Err(MapError::Firmware(EFI_BUFFER_TOO_SMALL, wanted)) => {
                regrow_memory_map(bs, map, wanted)?
            }
            Err(MapError::Firmware(..)) => return Err("cannot read the memory map again"),
            Err(MapError::Stride) => return Err(STRIDE_ERROR),
        }
    }
    Err("boot services did not exit after repeated attempts")
}

const COM1: u16 = 0x3F8;
const REG_DATA: u16 = 0;
const REG_IER: u16 = 1;
const REG_FCR: u16 = 2;
const REG_LCR: u16 = 3;
const REG_MCR: u16 = 4;
const REG_LSR: u16 = 5;
const LSR_THRE: u8 = 1 << 5;
const LCR_DLAB: u8 = 1 << 7;
const LCR_8N1: u8 = 0x03;
const FCR_ENABLE_CLEAR_14: u8 = 0xC7;
const MCR_DTR_RTS: u8 = 0x03;
/// Divisor latch value for 115200 baud.
const DIVISOR_115200: u16 = 1;

fn outb(port: u16, value: u8) {
    // SAFETY: a byte write to an I/O port of the legacy UART; it touches no
    // memory.
    unsafe {
        asm!("out dx, al", in("dx") port, in("al") value, options(nomem, nostack, preserves_flags))
    };
}

fn inb(port: u16) -> u8 {
    let value: u8;
    // SAFETY: a byte read from an I/O port of the legacy UART; it touches no
    // memory.
    unsafe {
        asm!("in al, dx", in("dx") port, out("al") value, options(nomem, nostack, preserves_flags))
    };
    value
}

/// Configure COM1 like `rabbit-drivers`: 115200 baud, 8N1, FIFOs on,
/// interrupts disabled.
fn serial_init() {
    outb(COM1 + REG_IER, 0x00);
    outb(COM1 + REG_LCR, LCR_DLAB);
    outb(COM1 + REG_DATA, DIVISOR_115200 as u8);
    outb(COM1 + REG_IER, (DIVISOR_115200 >> 8) as u8);
    outb(COM1 + REG_LCR, LCR_8N1);
    outb(COM1 + REG_FCR, FCR_ENABLE_CLEAR_14);
    outb(COM1 + REG_MCR, MCR_DTR_RTS);
}

/// Send one byte, polling the line status register while the FIFO is full.
fn serial_byte(byte: u8) {
    while inb(COM1 + REG_LSR) & LSR_THRE == 0 {
        core::hint::spin_loop();
    }
    outb(COM1 + REG_DATA, byte);
}

fn serial_str(text: &str) {
    text.bytes().for_each(serial_byte);
}

/// Send `value` in hexadecimal without leading zeros.
fn serial_hex(value: u64) {
    let digits = (64 - value.leading_zeros()).div_ceil(4).max(1);
    for index in (0..digits).rev() {
        let nibble = ((value >> (index * 4)) & 0xF) as u8;
        serial_byte(b"0123456789abcdef"[nibble as usize]);
    }
}

/// Send `value` in decimal.
fn serial_dec(value: u64) {
    let mut digits = [0u8; 20];
    let mut count = 0;
    let mut rest = value;
    loop {
        digits[count] = b'0' + (rest % 10) as u8;
        count += 1;
        rest /= 10;
        if rest == 0 {
            break;
        }
    }
    digits[..count]
        .iter()
        .rev()
        .for_each(|&digit| serial_byte(digit));
}

/// Report why the loader stops and halt.
fn fail(reason: &str) -> ! {
    serial_str("rabbit-uefi: error: ");
    serial_str(reason);
    serial_str("\n");
    halt()
}

/// Like [`fail`], for a firmware call that returned `status`.
fn fail_status(reason: &str, status: Status) -> ! {
    serial_str("rabbit-uefi: error: ");
    serial_str(reason);
    serial_str(" (status 0x");
    serial_hex(status as u64);
    serial_str(")\n");
    halt()
}

/// Ask the firmware for `pages` pages of loader data. `address` is the exact
/// address for `alloc_type::ADDRESS` and the highest byte for
/// `alloc_type::MAX_ADDRESS`. Returns the address of the pages.
fn allocate_pages(bs: &BootServices, kind: u32, address: u64, pages: usize) -> Result<u64, Status> {
    let mut memory = address;
    // SAFETY: `memory` is a live local that holds the request and receives the
    // address of the allocation.
    let status = unsafe { (bs.allocate_pages)(kind, LOADER_DATA, pages, &mut memory) };
    if status == EFI_SUCCESS {
        Ok(memory)
    } else {
        Err(status)
    }
}

/// Copy the segments of the kernel to their physical addresses and zero the
/// rest of the range, so `.bss` and the pages between segments are clean.
///
/// # Safety
///
/// `[LOAD_MIN, kernel.physical_end)` must be owned by the caller: allocated
/// with `AllocatePages` and not used by anything else.
unsafe fn load_kernel(kernel: &elf::Image) {
    let length = (kernel.physical_end - elf::LOAD_MIN) as usize;
    // SAFETY: the caller owns the range, and boot services identity map it.
    unsafe { ptr::write_bytes(elf::LOAD_MIN as *mut u8, 0, length) };
    for segment in kernel.segments() {
        // `parse` checked that the file bytes lie inside the image and that the
        // segment lies inside the range, so both slices are in bounds.
        let source = &KERNEL_ELF[segment.offset..segment.offset + segment.file_size];
        // SAFETY: the destination is inside the owned range, which no source
        // byte of the embedded image overlaps.
        unsafe {
            ptr::copy_nonoverlapping(source.as_ptr(), segment.paddr as *mut u8, source.len())
        };
    }
}

/// Zero `pages` pages at `paddr`.
///
/// # Safety
///
/// The pages must be owned by the caller.
unsafe fn zero_pages(paddr: u64, pages: usize) {
    // SAFETY: the caller owns the pages, which boot services identity map.
    unsafe { ptr::write_bytes(paddr as *mut u8, 0, pages * PAGE_SIZE) };
}

/// The memory map and module list inside the start info area at `paddr`.
///
/// # Safety
///
/// `paddr` must be the start of a zeroed area of at least
/// `MEMMAP_OFFSET + entries * size_of::<HvmMemmapEntry>()` bytes that nothing
/// else references, and it must stay allocated for the rest of the program.
unsafe fn area_tables(
    paddr: u64,
    entries: usize,
) -> (&'static mut [HvmMemmapEntry], &'static mut [HvmModule]) {
    let base = paddr as *mut u8;
    // SAFETY: both ranges are inside the area and disjoint, they are 8-byte
    // aligned (asserted at compile time), and zero is a valid value of both
    // plain integer structures.
    unsafe {
        (
            core::slice::from_raw_parts_mut(
                base.add(MEMMAP_OFFSET) as *mut HvmMemmapEntry,
                entries,
            ),
            core::slice::from_raw_parts_mut(
                base.add(MODLIST_OFFSET) as *mut HvmModule,
                MODULE_SLOTS,
            ),
        )
    }
}

/// Store the finished header at the start of the area and borrow it back, so
/// the check reads the bytes the kernel would read.
///
/// # Safety
///
/// `paddr` must satisfy the contract of [`area_tables`].
unsafe fn store_header(paddr: u64, start: HvmStartInfo) -> &'static HvmStartInfo {
    let header = paddr as *mut HvmStartInfo;
    // SAFETY: the header slot is aligned, inside the area, and disjoint from
    // the tables, which are the only other references into the area.
    unsafe {
        header.write(start);
        &*header
    }
}

/// Carve two arrays of `entries` ranges out of one pool allocation.
fn allocate_ranges(
    bs: &BootServices,
    entries: usize,
) -> Option<(&'static mut [Range], &'static mut [Range])> {
    let bytes = entries.checked_mul(2 * size_of::<Range>())?;
    let buffer = allocate_pool(bs, bytes)?;
    if !(buffer as usize).is_multiple_of(align_of::<Range>()) {
        return None;
    }
    let start = buffer as *mut Range;
    for index in 0..entries * 2 {
        // SAFETY: `index` is inside the allocation, which is aligned for `Range`.
        unsafe { start.add(index).write((0, 0, MemmapKind::Reserved)) };
    }
    // SAFETY: the allocation holds `2 * entries` initialised ranges, nothing
    // else references it and it is never freed.
    let all = unsafe { core::slice::from_raw_parts_mut(start, entries * 2) };
    Some(all.split_at_mut(entries))
}

/// Convert the final firmware map into `(start, length, kind)` ranges sorted
/// by start, which the firmware does not promise. Returns how many there are.
fn collect_ranges(map: &MemoryMap, ranges: &mut [Range]) -> Result<usize, &'static str> {
    let stride = map.descriptor_size;
    if stride < size_of::<MemoryDescriptor>() {
        return Err("memory map descriptors are too small");
    }
    let count = map.size / stride;
    if count > ranges.len() {
        return Err("memory map grew past the room reserved for it");
    }
    for index in 0..count {
        // SAFETY: `index < size / stride` and `stride >= 40`, so the descriptor
        // lies inside the `size` bytes the firmware wrote. The stride need not
        // keep the structure aligned, hence the unaligned read.
        let descriptor = unsafe {
            ptr::read_unaligned(map.buffer.add(index * stride) as *const MemoryDescriptor)
        };
        let length = descriptor
            .number_of_pages
            .checked_mul(PAGE_SIZE as u64)
            .ok_or("memory descriptor length overflows")?;
        let entry = (
            descriptor.physical_start,
            length,
            memmap_kind_from_uefi(descriptor.kind),
        );
        // Insertion into the sorted prefix: linear for a map that is sorted.
        let mut slot = index;
        while slot > 0 && ranges[slot - 1].0 > entry.0 {
            ranges[slot] = ranges[slot - 1];
            slot -= 1;
        }
        ranges[slot] = entry;
    }
    Ok(count)
}

/// The start info area below 1 MiB and the buffers the final map is built in.
struct Buffers {
    /// Physical address of the start info area.
    area: u64,
    /// Capacity, in entries, of the memory map in the area and of both range
    /// arrays.
    entries: usize,
    sorted: &'static mut [Range],
    merged: &'static mut [Range],
}

/// Allocate everything the start info needs. The firmware map changes with
/// each allocation, so this runs before the map that feeds `ExitBootServices`.
fn allocate_buffers(bs: &BootServices) -> Buffers {
    let Some((size, stride)) = memory_map_size(bs) else {
        fail("cannot size the memory map");
    };
    let entries = size / stride + MEMMAP_SLACK_ENTRIES;
    let pages = (MEMMAP_OFFSET + entries * size_of::<HvmMemmapEntry>()).div_ceil(PAGE_SIZE);
    let area = match allocate_pages(bs, alloc_type::MAX_ADDRESS, START_INFO_LIMIT, pages) {
        Ok(area) => area,
        Err(status) => fail_status("cannot allocate the start info area below 1 MiB", status),
    };
    // SAFETY: the pages were just allocated and nothing else uses them.
    unsafe { zero_pages(area, pages) };
    let Some((sorted, merged)) = allocate_ranges(bs, entries) else {
        fail("cannot allocate the memory map ranges");
    };
    Buffers {
        area,
        entries,
        sorted,
        merged,
    }
}

/// Build the start info from the final memory map and check it with the parser
/// the kernel uses.
///
/// It takes the buffers by value, so the area behind them is turned into
/// `&'static mut` tables once: `Buffers` is only made by `allocate_buffers`.
///
/// The map is written as the firmware reports it. The areas the loader itself
/// holds (the start info area below 1 MiB, the page at 0x8000 and the kernel
/// image) are loader data, so they come out as RAM. That is deliberate: the
/// kernel reserves `[0, 1 MiB)` and its whole image before it frees any frame,
/// so nothing it hands out can overlap them, and the map does not need to hide
/// them. The start info area never relies on the map for that: it lives below
/// 1 MiB, which the kernel reserves.
fn build_start_info(
    map: &MemoryMap,
    rsdp: u64,
    buffers: Buffers,
) -> Result<BootInfo<'static>, &'static str> {
    let count = collect_ranges(map, buffers.sorted)?;
    let merged = merge_adjacent(&buffers.sorted[..count], buffers.merged)
        .map_err(|_| "memory map cannot be merged")?;
    // SAFETY: `allocate_buffers` allocated and zeroed the area with room for
    // `entries` memory map entries, `buffers` is consumed so this runs once per
    // area, nothing else references it, and it is never freed.
    let (memmap, modules) = unsafe { area_tables(buffers.area, buffers.entries) };
    let mut writer = StartInfoWriter::new(memmap, modules);
    writer.set_rsdp(rsdp);
    writer.set_table_paddrs(
        buffers.area + MEMMAP_OFFSET as u64,
        buffers.area + MODLIST_OFFSET as u64,
    );
    for &(start, length, kind) in &buffers.merged[..merged] {
        writer
            .push_memmap(start, length, kind)
            .map_err(|_| "memory map does not fit the start info area")?;
    }
    let (start, memmap, modules) = writer.finish();
    // SAFETY: same area, and the tables above are the only other references.
    let start = unsafe { store_header(buffers.area, start) };
    BootInfo::parse(start, memmap, modules).map_err(|_| "start info rejected")
}

#[no_mangle]
extern "efiapi" fn efi_main(image: Handle, st: *const SystemTable) -> usize {
    serial_init();
    // SAFETY: the firmware passes a valid system table that stays readable
    // until the end of the program.
    let st = unsafe { &*st };
    // SAFETY: a valid system table points to a valid boot services table.
    let bs = unsafe { &*st.boot_services };

    let rsdp = find_rsdp(st);
    let kernel = elf::parse(KERNEL_ELF).unwrap_or_else(|reason| fail(reason));

    // Reserve the kernel and the AP trampoline page before anything else asks
    // for memory, then load. If the firmware keeps either, nothing is touched.
    let kernel_pages = ((kernel.physical_end - elf::LOAD_MIN) / elf::PAGE_SIZE) as usize;
    if let Err(status) = allocate_pages(bs, alloc_type::ADDRESS, elf::LOAD_MIN, kernel_pages) {
        fail_status("cannot reserve the kernel range at 0x100000", status);
    }
    if let Err(status) = allocate_pages(bs, alloc_type::ADDRESS, AP_TRAMPOLINE_PHYS, 1) {
        fail_status("cannot reserve the trampoline page at 0x8000", status);
    }
    // SAFETY: the kernel range was just allocated and nothing else uses it.
    unsafe { load_kernel(&kernel) };

    let buffers = allocate_buffers(bs);
    let mut map = first_memory_map(bs).unwrap_or_else(|reason| fail(reason));
    exit_boot_services(image, bs, &mut map).unwrap_or_else(|reason| fail(reason));

    serial_str("rabbit-uefi: boot services exited, rsdp=0x");
    serial_hex(rsdp);
    serial_str("\n");

    let info = build_start_info(&map, rsdp, buffers).unwrap_or_else(|reason| fail(reason));
    serial_str("rabbit-uefi: start info ok, memmap=");
    serial_dec(info.memmap().len() as u64);
    serial_str(" entries, ram=");
    serial_dec(info.total_ram() >> 20);
    serial_str(" MiB, rsdp=0x");
    serial_hex(info.start_info().rsdp_paddr);
    serial_str(", kernel=0x");
    serial_hex(elf::LOAD_MIN);
    serial_str("..0x");
    serial_hex(kernel.physical_end);
    serial_str("\n");
    halt()
}
