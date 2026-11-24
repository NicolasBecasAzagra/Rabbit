// SPDX-License-Identifier: GPL-2.0-only
//! Rabbit UEFI loader (first stage).
//!
//! A `no_std`, `no_main` PE32+ EFI application. It finds the ACPI RSDP in the
//! configuration table, reads the memory map, leaves boot services and reports
//! the result on COM1. The handover to the kernel comes in a later change.
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
use core::mem::{offset_of, size_of};
use core::ptr;

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
};

/// A memory map buffer and what the firmware reported about it.
struct MemoryMap {
    buffer: *mut u8,
    capacity: usize,
    key: usize,
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

/// Read the memory map into the buffer. On `EFI_BUFFER_TOO_SMALL` the error
/// carries the size the firmware wants.
fn read_memory_map(bs: &BootServices, map: &mut MemoryMap) -> Result<(), (Status, usize)> {
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
    if status == EFI_SUCCESS {
        map.descriptor_size = descriptor_size;
        Ok(())
    } else {
        Err((status, size))
    }
}

/// Allocate a buffer for the memory map and read it, growing the buffer when
/// the firmware asks for more.
fn first_memory_map(bs: &BootServices) -> Option<MemoryMap> {
    let (mut size, descriptor_size) = memory_map_size(bs)?;
    for _ in 0..MAP_SIZE_ATTEMPTS {
        let capacity = size + MAP_SLACK_DESCRIPTORS * descriptor_size;
        let buffer = allocate_pool(bs, capacity)?;
        let mut map = MemoryMap {
            buffer,
            capacity,
            key: 0,
            descriptor_size,
        };
        match read_memory_map(bs, &mut map) {
            Ok(()) => return Some(map),
            Err((EFI_BUFFER_TOO_SMALL, wanted)) => size = wanted.max(capacity),
            Err(_) => return None,
        }
    }
    None
}

/// Leave boot services. A failure means the map changed since the key was
/// taken, so the map is read again; nothing else is called in between.
fn exit_boot_services(image: Handle, bs: &BootServices, map: &mut MemoryMap) -> bool {
    for _ in 0..EXIT_ATTEMPTS {
        // SAFETY: `image` is the handle the firmware passed to `efi_main` and
        // `key` comes from the latest `GetMemoryMap`.
        let status = unsafe { (bs.exit_boot_services)(image, map.key) };
        if status == EFI_SUCCESS {
            return true;
        }
        match read_memory_map(bs, map) {
            Ok(()) => {}
            Err((EFI_BUFFER_TOO_SMALL, wanted)) => {
                // The map grew. Allocating changes it once more, so the slack
                // covers that, and the map is read again before the next exit.
                let capacity = wanted + MAP_SLACK_DESCRIPTORS * map.descriptor_size;
                let Some(buffer) = allocate_pool(bs, capacity) else {
                    return false;
                };
                map.buffer = buffer;
                map.capacity = capacity;
                if read_memory_map(bs, map).is_err() {
                    return false;
                }
            }
            Err(_) => return false,
        }
    }
    false
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

#[no_mangle]
extern "efiapi" fn efi_main(image: Handle, st: *const SystemTable) -> usize {
    // SAFETY: the firmware passes a valid system table that stays readable
    // until the end of the program.
    let st = unsafe { &*st };
    // SAFETY: a valid system table points to a valid boot services table.
    let bs = unsafe { &*st.boot_services };

    let rsdp = find_rsdp(st);
    let Some(mut map) = first_memory_map(bs) else {
        halt()
    };
    if !exit_boot_services(image, bs, &mut map) {
        halt();
    }

    serial_init();
    serial_str("rabbit-uefi: boot services exited, rsdp=0x");
    serial_hex(rsdp);
    serial_str("\n");
    halt()
}
