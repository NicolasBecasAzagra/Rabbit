// SPDX-License-Identifier: GPL-2.0-only
//! Validation of the kernel ELF the loader embeds.
//!
//! The kernel is a 64-bit executable whose `PT_LOAD` segments carry physical
//! load addresses in `p_paddr`. This module checks the header and every
//! segment against the window the kernel may occupy and returns where each
//! segment goes, together with the PVH entry the `Xen` note names. It touches no memory but the image it is given, and every
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
const PT_NOTE: u32 = 4;
/// `XEN_ELFNOTE_PHYS32_ENTRY`: the physical address of the 32-bit PVH entry.
const XEN_ELFNOTE_PHYS32_ENTRY: u32 = 0x12;
/// Name of the note, with its terminating zero.
const XEN_NOTE_NAME: &[u8] = b"Xen\0";
/// `n_namesz`, `n_descsz` and `n_type`.
const NOTE_HEADER_SIZE: usize = 12;
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
    /// Physical address of the 32-bit PVH entry, inside a loaded segment and
    /// below 4 GiB.
    pub entry: u64,
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

/// Round `value` up to a multiple of four, the alignment of note fields.
fn align4(value: usize) -> Option<usize> {
    Some(value.checked_add(3)? & !3)
}

/// The entry address of the first `XEN_ELFNOTE_PHYS32_ENTRY` note in `notes`,
/// the bytes of one `PT_NOTE` segment. Every field is checked against the
/// length of `notes`.
fn xen_entry(notes: &[u8]) -> Result<Option<u64>, &'static str> {
    const MALFORMED: &str = "kernel ELF has a malformed note";
    let mut at = 0usize;
    while at < notes.len() {
        let name_size = u32_at(notes, at).ok_or(MALFORMED)?;
        let desc_size = u32_at(notes, at + 4).ok_or(MALFORMED)?;
        let kind = u32_at(notes, at + 8).ok_or(MALFORMED)?;
        let name_start = at + NOTE_HEADER_SIZE;
        let name_end = name_start
            .checked_add(usize::try_from(name_size).map_err(|_| MALFORMED)?)
            .ok_or(MALFORMED)?;
        let desc_start = align4(name_end).ok_or(MALFORMED)?;
        let desc_end = desc_start
            .checked_add(usize::try_from(desc_size).map_err(|_| MALFORMED)?)
            .ok_or(MALFORMED)?;
        let name = notes.get(name_start..name_end).ok_or(MALFORMED)?;
        let desc = notes.get(desc_start..desc_end).ok_or(MALFORMED)?;
        if kind == XEN_ELFNOTE_PHYS32_ENTRY && name == XEN_NOTE_NAME {
            return match desc.len() {
                4 => Ok(u32_at(desc, 0).map(u64::from)),
                8 => Ok(u64_at(desc, 0)),
                _ => Err("PVH entry note has an unexpected size"),
            };
        }
        at = align4(desc_end).ok_or(MALFORMED)?;
    }
    Ok(None)
}

/// Validate `image` and return where its segments load and where the PVH
/// entry is.
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
        entry: 0,
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

    let mut entry = None;
    for index in 0..usize::from(phnum) {
        // The same arithmetic succeeded in the loop above.
        let base = phoff + index * PROGRAM_HEADER_SIZE;
        if u32_at(image, base).ok_or(TRUNCATED)? != PT_NOTE {
            continue;
        }
        let offset = u64_at(image, base + 8).ok_or(TRUNCATED)?;
        let file_size = u64_at(image, base + 32).ok_or(TRUNCATED)?;
        // The note fields are aligned relative to the start of the segment, so
        // that is the file alignment only if the segment itself starts aligned.
        if offset % 4 != 0 {
            return Err("kernel ELF has a misaligned note segment");
        }
        let notes = usize::try_from(offset)
            .ok()
            .zip(usize::try_from(file_size).ok())
            .and_then(|(start, size)| image.get(start..start.checked_add(size)?))
            .ok_or(TRUNCATED)?;
        if let Some(found) = xen_entry(notes)? {
            entry = Some(found);
            break;
        }
    }
    let entry = entry.ok_or("kernel ELF has no PVH entry note")?;
    let loaded = out.segments().iter().any(|segment| {
        // Accepted segments were bounds checked, so this cannot overflow.
        entry >= segment.paddr && entry - segment.paddr < segment.file_size as u64
    });
    if entry >= 1 << 32 || !loaded {
        return Err("PVH entry is not inside a loaded segment below 4 GiB");
    }
    out.entry = entry;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec;
    use std::vec::Vec;

    const BASE: u64 = LOAD_MIN;
    const XEN: u32 = XEN_ELFNOTE_PHYS32_ENTRY;

    fn put(image: &mut [u8], at: usize, bytes: &[u8]) {
        image[at..at + bytes.len()].copy_from_slice(bytes);
    }

    /// One `PT_LOAD` segment to build: where it loads, its size in memory and
    /// its file bytes.
    struct Load {
        paddr: u64,
        mem_size: u64,
        data: Vec<u8>,
    }

    fn load(paddr: u64, mem_size: u64, data: &[u8]) -> Load {
        Load {
            paddr,
            mem_size,
            data: data.to_vec(),
        }
    }

    /// A note with explicit `namesz` and `descsz` fields, which a test can make
    /// disagree with the bytes that follow. Name and descriptor are padded to
    /// four bytes.
    fn raw_note(name_size: u32, desc_size: u32, kind: u32, name: &[u8], desc: &[u8]) -> Vec<u8> {
        let mut note = Vec::new();
        for field in [name_size, desc_size, kind] {
            note.extend_from_slice(&field.to_le_bytes());
        }
        note.extend_from_slice(name);
        note.resize(note.len().next_multiple_of(4), 0);
        note.extend_from_slice(desc);
        note.resize(note.len().next_multiple_of(4), 0);
        note
    }

    fn note(name: &[u8], kind: u32, desc: &[u8]) -> Vec<u8> {
        raw_note(name.len() as u32, desc.len() as u32, kind, name, desc)
    }

    fn xen_note(desc: &[u8]) -> Vec<u8> {
        note(XEN_NOTE_NAME, XEN, desc)
    }

    /// An ELF header, the program headers (the loads, then the note segment if
    /// there is one) and the bytes they point to, in that order. The note
    /// segment starts `note_skew` bytes after a four byte boundary.
    fn build(loads: &[Load], notes: Option<&[u8]>, note_skew: usize) -> Vec<u8> {
        let count = loads.len() + usize::from(notes.is_some());
        let mut image = vec![0u8; ELF_HEADER_SIZE + count * PROGRAM_HEADER_SIZE];
        put(&mut image, 0, &[0x7F, b'E', b'L', b'F', 2, 1, 1]);
        put(&mut image, 16, &ET_EXEC.to_le_bytes());
        put(&mut image, 18, &EM_X86_64.to_le_bytes());
        put(&mut image, 32, &(ELF_HEADER_SIZE as u64).to_le_bytes());
        put(&mut image, 54, &(PROGRAM_HEADER_SIZE as u16).to_le_bytes());
        put(&mut image, 56, &(count as u16).to_le_bytes());
        let header = |image: &mut Vec<u8>, index: usize, kind: u32, fields: [u64; 4]| {
            let at = ELF_HEADER_SIZE + index * PROGRAM_HEADER_SIZE;
            put(image, at, &kind.to_le_bytes());
            let [offset, paddr, file_size, mem_size] = fields;
            put(image, at + 8, &offset.to_le_bytes());
            put(image, at + 16, &paddr.to_le_bytes());
            put(image, at + 24, &paddr.to_le_bytes());
            put(image, at + 32, &file_size.to_le_bytes());
            put(image, at + 40, &mem_size.to_le_bytes());
        };
        for (index, segment) in loads.iter().enumerate() {
            image.resize(image.len().next_multiple_of(4), 0);
            let offset = image.len() as u64;
            image.extend_from_slice(&segment.data);
            let fields = [
                offset,
                segment.paddr,
                segment.data.len() as u64,
                segment.mem_size,
            ];
            header(&mut image, index, PT_LOAD, fields);
        }
        if let Some(notes) = notes {
            image.resize(image.len().next_multiple_of(4) + note_skew, 0);
            let offset = image.len() as u64;
            image.extend_from_slice(notes);
            let size = notes.len() as u64;
            header(&mut image, loads.len(), PT_NOTE, [offset, 0, size, size]);
        }
        image
    }

    /// A valid kernel: sixteen file bytes at 1 MiB in an 8 KiB segment.
    fn kernel(notes: &[u8]) -> Vec<u8> {
        build(&[load(BASE, 0x2000, &[0x90; 16])], Some(notes), 0)
    }

    fn entry_of(image: &[u8]) -> Result<u64, &'static str> {
        parse(image).map(|parsed| parsed.entry)
    }

    fn error_of(image: &[u8]) -> Option<&'static str> {
        parse(image).err()
    }

    #[test]
    fn valid_image_gives_segments_and_entry() {
        let image = kernel(&xen_note(&(BASE as u32).to_le_bytes()));
        let parsed = parse(&image).ok().unwrap();
        assert_eq!(parsed.entry, BASE);
        assert_eq!(parsed.physical_end, BASE + 0x2000);
        let [segment] = parsed.segments() else {
            panic!("expected one segment");
        };
        assert_eq!((segment.paddr, segment.file_size), (BASE, 16));
        assert_eq!(segment.mem_size, 0x2000);
    }

    #[test]
    fn truncated_header_is_rejected() {
        let image = kernel(&xen_note(&[0; 4]));
        for length in [0, 3, 4, ELF_HEADER_SIZE - 1] {
            assert_eq!(
                error_of(&image[..length]),
                Some("kernel image is not an ELF file")
            );
        }
        // A header that promises program headers the file does not have.
        assert_eq!(
            error_of(&image[..ELF_HEADER_SIZE]),
            Some("kernel ELF is truncated")
        );
        // A segment whose file bytes run past the end of the image.
        let mut cut = image.clone();
        let phdr = ELF_HEADER_SIZE;
        put(&mut cut, phdr + 32, &0x1000u64.to_le_bytes());
        assert_eq!(error_of(&cut), Some("kernel ELF is truncated"));
    }

    #[test]
    fn bad_magic_class_data_and_machine_are_rejected() {
        let image = kernel(&xen_note(&(BASE as u32).to_le_bytes()));
        assert!(parse(&image).is_ok());

        let mut bad = image.clone();
        bad[1] = b'X';
        assert_eq!(error_of(&bad), Some("kernel image is not an ELF file"));

        let mut bad = image.clone();
        bad[4] = 1;
        assert_eq!(
            error_of(&bad),
            Some("kernel ELF is not 64-bit little endian")
        );

        let mut bad = image.clone();
        bad[5] = 2;
        assert_eq!(
            error_of(&bad),
            Some("kernel ELF is not 64-bit little endian")
        );

        let mut bad = image.clone();
        put(&mut bad, 18, &0x28u16.to_le_bytes());
        assert_eq!(
            error_of(&bad),
            Some("kernel ELF is not an x86_64 executable")
        );

        let mut bad = image;
        put(&mut bad, 16, &3u16.to_le_bytes());
        assert_eq!(
            error_of(&bad),
            Some("kernel ELF is not an x86_64 executable")
        );
    }

    #[test]
    fn segments_outside_the_window_are_rejected() {
        let notes = xen_note(&(BASE as u32).to_le_bytes());
        let outside = "kernel segment is outside 0x100000..0x300000";
        for (paddr, mem_size) in [
            (BASE - 0x1000, 0x2000),
            (LOAD_MAX - 0x1000, 0x2000),
            (LOAD_MAX, 0x1000),
            (0, 0x1000),
        ] {
            let image = build(&[load(paddr, mem_size, &[0x90; 16])], Some(&notes), 0);
            assert_eq!(error_of(&image), Some(outside), "{paddr:#x} {mem_size:#x}");
        }
        let image = build(&[load(u64::MAX - 8, 0x1000, &[0x90; 16])], Some(&notes), 0);
        assert_eq!(
            error_of(&image),
            Some("kernel segment wraps the address space")
        );
        // The last page of the window is valid.
        let image = build(
            &[load(LOAD_MAX - 0x1000, 0x1000, &[0x90; 16])],
            Some(&notes),
            0,
        );
        assert!(error_of(&image).is_some_and(|reason| reason.starts_with("PVH entry")));
    }

    #[test]
    fn overlapping_segments_are_rejected() {
        let notes = xen_note(&(BASE as u32).to_le_bytes());
        let loads = [
            load(BASE, 0x2000, &[0x90; 16]),
            load(BASE + 0x1FFF, 0x1000, &[0; 4]),
        ];
        assert_eq!(
            error_of(&build(&loads, Some(&notes), 0)),
            Some("kernel ELF has overlapping loadable segments")
        );
        // Adjacent segments do not overlap.
        let loads = [
            load(BASE, 0x2000, &[0x90; 16]),
            load(BASE + 0x2000, 0x1000, &[0; 4]),
        ];
        assert_eq!(entry_of(&build(&loads, Some(&notes), 0)), Ok(BASE));
    }

    #[test]
    fn note_sizes_that_overflow_are_rejected() {
        let malformed = Some("kernel ELF has a malformed note");
        let desc = (BASE as u32).to_le_bytes();
        for (name_size, desc_size) in [
            (u32::MAX, 4),
            (u32::MAX - 2, 4),
            (4, u32::MAX),
            (4, u32::MAX - 2),
            (4, 0x1000),
            (0x1000, 4),
        ] {
            let notes = raw_note(name_size, desc_size, XEN, XEN_NOTE_NAME, &desc);
            assert_eq!(
                error_of(&kernel(&notes)),
                malformed,
                "{name_size} {desc_size}"
            );
        }
        // A header cut short.
        assert_eq!(error_of(&kernel(&[4, 0, 0, 0, 4, 0])), malformed);
        // A note whose descriptor is one byte short of what it says.
        let mut notes = xen_note(&desc);
        notes.truncate(notes.len() - 1);
        notes[4..8].copy_from_slice(&8u32.to_le_bytes());
        assert_eq!(error_of(&kernel(&notes)), malformed);
    }

    #[test]
    fn four_and_eight_byte_descriptors_give_the_entry() {
        let entry = BASE + 4;
        assert_eq!(
            entry_of(&kernel(&xen_note(&(entry as u32).to_le_bytes()))),
            Ok(entry)
        );
        assert_eq!(
            entry_of(&kernel(&xen_note(&entry.to_le_bytes()))),
            Ok(entry)
        );
    }

    #[test]
    fn other_descriptor_sizes_are_rejected() {
        for size in [0, 1, 2, 3, 5, 6, 7, 9, 12] {
            let notes = xen_note(&vec![0u8; size]);
            assert_eq!(
                error_of(&kernel(&notes)),
                Some("PVH entry note has an unexpected size"),
                "descsz {size}"
            );
        }
    }

    #[test]
    fn xen_note_after_other_notes_is_found() {
        let entry = (BASE + 8) as u32;
        let mut notes = note(b"GNU\0", 1, &[1, 2, 3, 4, 5, 6]);
        // Same type with another name, and the Xen name with another type.
        notes.extend(note(b"Foo\0", XEN, &[0xEE; 4]));
        notes.extend(note(XEN_NOTE_NAME, 0x11, &[0xEE; 8]));
        notes.extend(note(b"", 7, &[]));
        notes.extend(xen_note(&entry.to_le_bytes()));
        assert_eq!(entry_of(&kernel(&notes)), Ok(u64::from(entry)));
    }

    #[test]
    fn name_without_the_terminating_zero_is_not_the_xen_note() {
        let notes = note(b"Xen", XEN, &(BASE as u32).to_le_bytes());
        assert_eq!(
            error_of(&kernel(&notes)),
            Some("kernel ELF has no PVH entry note")
        );
    }

    #[test]
    fn entry_must_be_in_the_file_bytes_of_a_segment() {
        let last = BASE + 15;
        assert_eq!(
            entry_of(&kernel(&xen_note(&(last as u32).to_le_bytes()))),
            Ok(last)
        );
        let past = "PVH entry is not inside a loaded segment below 4 GiB";
        for entry in [last + 1, BASE - 1, BASE + 0x1FFF] {
            assert_eq!(
                error_of(&kernel(&xen_note(&(entry as u32).to_le_bytes()))),
                Some(past),
                "{entry:#x}"
            );
        }
    }

    #[test]
    fn entry_at_or_above_4_gib_is_rejected() {
        let past = Some("PVH entry is not inside a loaded segment below 4 GiB");
        for entry in [1u64 << 32, (1 << 32) + BASE, u64::MAX] {
            assert_eq!(error_of(&kernel(&xen_note(&entry.to_le_bytes()))), past);
        }
    }

    #[test]
    fn image_without_a_note_is_rejected() {
        let image = build(&[load(BASE, 0x2000, &[0x90; 16])], None, 0);
        assert_eq!(error_of(&image), Some("kernel ELF has no PVH entry note"));
        // A note segment with no Xen note in it.
        let image = kernel(&note(b"GNU\0", 1, &[0; 4]));
        assert_eq!(error_of(&image), Some("kernel ELF has no PVH entry note"));
        // An empty note segment.
        assert_eq!(
            error_of(&kernel(&[])),
            Some("kernel ELF has no PVH entry note")
        );
    }

    #[test]
    fn misaligned_note_segment_is_rejected() {
        let notes = xen_note(&(BASE as u32).to_le_bytes());
        let image = build(&[load(BASE, 0x2000, &[0x90; 16])], Some(&notes), 2);
        assert_eq!(
            error_of(&image),
            Some("kernel ELF has a misaligned note segment")
        );
    }

    #[test]
    fn image_with_no_loadable_segment_is_rejected() {
        let notes = xen_note(&(BASE as u32).to_le_bytes());
        let image = build(&[], Some(&notes), 0);
        assert_eq!(error_of(&image), Some("kernel ELF has no loadable segment"));
    }
}
