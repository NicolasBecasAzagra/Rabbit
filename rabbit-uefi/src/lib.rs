// SPDX-License-Identifier: GPL-2.0-only
//! The parts of the UEFI loader that touch no firmware state, so they can be
//! tested on the host. The loader itself is `main.rs`.

#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(test)]
extern crate std;

pub mod elf;
