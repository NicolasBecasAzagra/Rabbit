// SPDX-License-Identifier: GPL-2.0-only
//! Validation of the kernel ELF the loader embeds.
//!
//! The kernel is a 64-bit executable whose `PT_LOAD` segments carry physical
//! load addresses in `p_paddr`. This module checks the header and every
//! segment against the window the kernel may occupy and returns where each
//! segment goes. It touches no memory but the image it is given, and every
//! read is bounds checked: a malformed image is an error, never a panic.

/// Lowest physical address a segment may load at: the kernel starts at 1 MiB.
pub const LOAD_MIN: u64 = 0x10_0000;
/// End of the window a segment must stay in: the image fits in the 2 MiB the
/// kernel maps with 4 KiB pages, so it ends before 3 MiB.
pub const LOAD_MAX: u64 = 0x30_0000;
/// Size of a page, to which the end of the image is rounded.
pub const PAGE_SIZE: u64 = 4096;
/// Most `PT_LOAD` segments accepted; the kernel has two.
pub const MAX_SEGMENTS: usize = 8;

const PT_LOAD: u32 = 1;
const ET_EXEC: u16 = 2;
const EM_X86_64: u16 = 0x3E;
const ELF_HEADER_SIZE: usize = 64;
const PROGRAM_HEADER_SIZE: usize = 56;

/// A `PT_LOAD` segment that passed validation.
#[derive(Clone, Copy)]
pub struct Segment {
    /// Offset of the file bytes in the image.
    pub offset: usize,
    /// Number of bytes to copy from the image.
    pub file_size: usize,
    /// Physical address the segment loads at.
    pub paddr: u64,
    /// Number of bytes the segment occupies in memory.
    pub mem_size: u64,
}

/// The loadable part of a validated kernel image.
pub struct Image {
    segments: [Segment; MAX_SEGMENTS],
    count: usize,
    /// End of the highest segment, rounded up to a page.
    pub physical_end: u64,
}

impl Image {
    /// The segments, in program header order.
    pub fn segments(&self) -> &[Segment] {
        &self.segments[..self.count]
    }
}

/// `N` little endian bytes at `offset`, if the image holds them.
fn bytes_at<const N: usize>(image: &[u8], offset: usize) -> Option<[u8; N]> {
    image.get(offset..offset.checked_add(N)?)?.try_into().ok()
}

fn u16_at(image: &[u8], offset: usize) -> Option<u16> {
    bytes_at(image, offset).map(u16::from_le_bytes)
}

fn u32_at(image: &[u8], offset: usize) -> Option<u32> {
    bytes_at(image, offset).map(u32::from_le_bytes)
}

fn u64_at(image: &[u8], offset: usize) -> Option<u64> {
    bytes_at(image, offset).map(u64::from_le_bytes)
}

/// Validate `image` and return where its segments load.
///
/// `e_entry` and `PT_NOTE` are not used yet: the handover (next commit)
/// validates the PVH entry (`XEN_ELFNOTE_PHYS32_ENTRY`) against the loaded
/// segments.
pub fn parse(image: &[u8]) -> Result<Image, &'static str> {
    const TRUNCATED: &str = "kernel ELF is truncated";
    if image.len() < ELF_HEADER_SIZE || image[..4] != [0x7F, b'E', b'L', b'F'] {
        return Err("kernel image is not an ELF file");
    }
    // Class 64 and little endian.
    if image[4] != 2 || image[5] != 1 {
        return Err("kernel ELF is not 64-bit little endian");
    }
    if u16_at(image, 16) != Some(ET_EXEC) || u16_at(image, 18) != Some(EM_X86_64) {
        return Err("kernel ELF is not an x86_64 executable");
    }
    let phoff = u64_at(image, 32).ok_or(TRUNCATED)?;
    let phentsize = u16_at(image, 54).ok_or(TRUNCATED)?;
    let phnum = u16_at(image, 56).ok_or(TRUNCATED)?;
    if usize::from(phentsize) != PROGRAM_HEADER_SIZE {
        return Err("kernel ELF has an unexpected program header size");
    }
    let phoff = usize::try_from(phoff).map_err(|_| TRUNCATED)?;

    let empty = Segment {
        offset: 0,
        file_size: 0,
        paddr: 0,
        mem_size: 0,
    };
    let mut out = Image {
        segments: [empty; MAX_SEGMENTS],
        count: 0,
        physical_end: 0,
    };
    let mut highest_end = 0u64;
    for index in 0..usize::from(phnum) {
        let base = index
            .checked_mul(PROGRAM_HEADER_SIZE)
            .and_then(|delta| phoff.checked_add(delta))
            .ok_or(TRUNCATED)?;
        if u32_at(image, base).ok_or(TRUNCATED)? != PT_LOAD {
            continue;
        }
        let field = |delta: usize| {
            base.checked_add(delta)
                .and_then(|at| u64_at(image, at))
                .ok_or(TRUNCATED)
        };
        let offset = field(8)?;
        let paddr = field(24)?;
        let file_size = field(32)?;
        let mem_size = field(40)?;
        if mem_size == 0 {
            continue;
        }
        if file_size > mem_size {
            return Err("kernel segment is larger in the file than in memory");
        }
        let file_end = offset.checked_add(file_size).ok_or(TRUNCATED)?;
        if file_end > image.len() as u64 {
            return Err(TRUNCATED);
        }
        let end = paddr
            .checked_add(mem_size)
            .ok_or("kernel segment wraps the address space")?;
        if paddr < LOAD_MIN || end > LOAD_MAX {
            return Err("kernel segment is outside 0x100000..0x300000");
        }
        let overlaps = out.segments().iter().any(|accepted| {
            // Accepted segments were bounds checked, so this cannot overflow.
            let accepted_end = accepted.paddr + accepted.mem_size;
            paddr < accepted_end && accepted.paddr < end
        });
        if overlaps {
            return Err("kernel ELF has overlapping loadable segments");
        }
        let slot = out
            .segments
            .get_mut(out.count)
            .ok_or("kernel ELF has too many loadable segments")?;
        *slot = Segment {
            // Both fit: they are bounded by the image length checked above.
            offset: offset as usize,
            file_size: file_size as usize,
            paddr,
            mem_size,
        };
        out.count += 1;
        highest_end = highest_end.max(end);
    }
    if out.count == 0 {
        return Err("kernel ELF has no loadable segment");
    }
    out.physical_end = highest_end.next_multiple_of(PAGE_SIZE);
    Ok(out)
}
