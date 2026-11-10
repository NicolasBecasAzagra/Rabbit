// SPDX-License-Identifier: GPL-2.0-only
//! Controlled fault injection used to prove that exceptions are recoverable.

use core::arch::global_asm;

use crate::idt::InterruptFrame;

global_asm!(
    r#"
    .section .text.div0, "ax", @progbits
    .code64
    .global rabbit_trigger_div0
    .global rabbit_div0_instruction
    .global rabbit_div0_after
rabbit_trigger_div0:
    movl %edi, %ecx
    movl $1, %eax
    xorl %edx, %edx
rabbit_div0_instruction:
    divl %ecx
rabbit_div0_after:
    ret
"#,
    options(att_syntax)
);

extern "C" {
    fn rabbit_trigger_div0(divisor: u32);
    static rabbit_div0_instruction: u8;
    static rabbit_div0_after: u8;
}

/// Execute a divide by zero whose dividend and divisor cannot be folded away
/// by the compiler.
pub fn trigger_divide_by_zero() {
    // SAFETY: the IDT is installed before this is called, so the resulting
    // #DE is delivered to `rabbit_interrupt_handler`, which resumes at
    // `rabbit_div0_after`.
    unsafe {
        rabbit_trigger_div0(core::hint::black_box(0));
    }
}

/// Move the interrupted RIP past the injected divide instruction.
///
/// Any other divide error keeps its RIP and faults again; only the known
/// trigger can make progress, so a deliberate fault elsewhere stays visible.
///
/// Returns `true` when the fault was the registered trigger and execution may
/// continue.
pub fn resume_after_divide_by_zero(frame: &mut InterruptFrame) -> bool {
    let faulting = core::ptr::addr_of!(rabbit_div0_instruction) as u64;
    let resume = core::ptr::addr_of!(rabbit_div0_after) as u64;

    if frame.rip == faulting {
        frame.rip = resume;
        true
    } else {
        false
    }
}
