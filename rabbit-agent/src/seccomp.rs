// SPDX-License-Identifier: GPL-2.0-only
//! The seccomp filter of the manifest, for a Linux kernel.
//!
//! The filter is a classic BPF program over `struct seccomp_data`. It is pure:
//! it builds the instructions and does not load them, so the logic is tested on
//! any host with a small interpreter of the opcodes it emits.
//!
//! The filter enforces the points of the manifest that seccomp can express:
//!
//! * the architecture must be x86-64, and the x32 bit of the system call
//!   number is a violation, because a filter that names numbers is bypassed by
//!   the same call under another ABI;
//! * `socket` is refused with `EACCES` unless the domain is `AF_INET` and the
//!   type, once `SOCK_NONBLOCK` and `SOCK_CLOEXEC` are masked, is a stream
//!   with `NetTcp` or a datagram with `NetUdp`;
//! * the protocol of that socket must be the default or the one of its type,
//!   TCP for a stream and UDP for a datagram: SCTP, UDP-Lite and ICMP are other
//!   protocols behind the same two types, and the manifest names none of them;
//! * `listen`, `accept` and `accept4` are refused with `EACCES`, because the
//!   manifest has no inbound capability;
//! * `io_uring_setup`, `io_uring_enter` and `io_uring_register` are refused
//!   with `EACCES`, because a ring can open a socket (`IORING_OP_SOCKET`) and
//!   run `connect` and `accept` without going through the calls above.
//!
//! Every other system call is allowed. A full allowlist would be the stronger
//! filter, but the exact set of calls a static musl program with the standard
//! library makes was not determined, and a list that is missing one call kills
//! the agent in the middle of a task. What the filter cannot see is the
//! destination of `connect` and `sendto`: the address is behind a pointer in
//! the call, and seccomp does not follow pointers.
//!
//! The values come from the public headers of the kernel: the `seccomp_data`
//! layout and the return actions from `include/uapi/linux/seccomp.h`, the
//! instruction layout from `include/uapi/linux/filter.h`, the opcodes from
//! `include/uapi/linux/bpf_common.h`, `AUDIT_ARCH_X86_64` from
//! `include/uapi/linux/audit.h`, and the numbers of the calls from
//! `arch/x86/entry/syscalls/syscall_64.tbl`, and the numbers of the protocols
//! from `include/uapi/linux/in.h`.

use rabbit_policy::{Capabilities, Capability};

/// One instruction of a classic BPF program, `struct sock_filter`: a code, the
/// jump offsets of the true and the false branch, and the operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SockFilter {
    pub code: u16,
    pub jt: u8,
    pub jf: u8,
    pub k: u32,
}

impl SockFilter {
    /// The eight bytes the kernel reads, in the byte order of x86-64.
    pub fn to_bytes(self) -> [u8; 8] {
        let mut bytes = [0; 8];
        bytes[..2].copy_from_slice(&self.code.to_le_bytes());
        bytes[2] = self.jt;
        bytes[3] = self.jf;
        bytes[4..].copy_from_slice(&self.k.to_le_bytes());
        bytes
    }
}

/// `BPF_LD | BPF_W | BPF_ABS` (0x00, 0x00, 0x20): load a 32-bit word of
/// `seccomp_data`.
const LOAD_WORD: u16 = 0x20;
/// `BPF_ALU | BPF_AND | BPF_K` (0x04, 0x50, 0x00): mask the accumulator.
const AND_CONSTANT: u16 = 0x54;
/// `BPF_JMP | BPF_JEQ | BPF_K` (0x05, 0x10, 0x00): compare the accumulator with
/// a constant.
const JUMP_EQUAL: u16 = 0x15;
/// `BPF_JMP | BPF_JSET | BPF_K` (0x05, 0x40, 0x00): test bits of the
/// accumulator.
const JUMP_SET: u16 = 0x45;
/// `BPF_RET | BPF_K` (0x06, 0x00): end the program with a constant.
const RETURN: u16 = 0x06;

/// Offset of `nr` in `struct seccomp_data`.
const OFFSET_NUMBER: u32 = 0;
/// Offset of `arch` in `struct seccomp_data`.
const OFFSET_ARCH: u32 = 4;
/// Offset of `args[0]`, and of its low word on a little endian machine.
const OFFSET_ARGUMENT0: u32 = 16;
/// Offset of the low word of `args[1]`.
const OFFSET_ARGUMENT1: u32 = 24;
/// Offset of the low word of `args[2]`.
const OFFSET_ARGUMENT2: u32 = 32;

/// `AUDIT_ARCH_X86_64`: `EM_X86_64` (62) | `__AUDIT_ARCH_64BIT` (0x80000000)
/// | `__AUDIT_ARCH_LE` (0x40000000).
pub const AUDIT_ARCH_X86_64: u32 = 62 | 0x8000_0000 | 0x4000_0000;
/// The bit that marks a system call of the x32 ABI.
pub const X32_SYSCALL_BIT: u32 = 0x4000_0000;

/// `SECCOMP_RET_KILL_PROCESS`.
pub const RET_KILL_PROCESS: u32 = 0x8000_0000;
/// `SECCOMP_RET_ERRNO`: the low sixteen bits are the error number.
const RET_ERRNO: u32 = 0x0005_0000;
/// `SECCOMP_RET_ALLOW`.
pub const RET_ALLOW: u32 = 0x7fff_0000;
/// The `errno` of a denial, `EACCES`.
const EACCES: u32 = 13;
/// The answer of a refused call: `SECCOMP_RET_ERRNO` with `EACCES`.
pub const RET_DENY: u32 = RET_ERRNO | EACCES;

/// `socket` in the x86-64 table.
const SYS_SOCKET: u32 = 41;
/// `accept` in the x86-64 table.
const SYS_ACCEPT: u32 = 43;
/// `listen` in the x86-64 table.
const SYS_LISTEN: u32 = 50;
/// `accept4` in the x86-64 table.
const SYS_ACCEPT4: u32 = 288;
/// `io_uring_setup` in the x86-64 table.
const SYS_IO_URING_SETUP: u32 = 425;
/// `io_uring_enter` in the x86-64 table.
const SYS_IO_URING_ENTER: u32 = 426;
/// `io_uring_register` in the x86-64 table.
const SYS_IO_URING_REGISTER: u32 = 427;

/// `AF_INET`.
const AF_INET: u32 = 2;
/// `SOCK_STREAM`.
const SOCK_STREAM: u32 = 1;
/// `SOCK_DGRAM`.
const SOCK_DGRAM: u32 = 2;
/// `IPPROTO_IP`: the protocol of a type, as the kernel picks it.
const IPPROTO_DEFAULT: u32 = 0;
/// `IPPROTO_TCP`.
const IPPROTO_TCP: u32 = 6;
/// `IPPROTO_UDP`.
const IPPROTO_UDP: u32 = 17;
/// `SOCK_NONBLOCK`, which is `O_NONBLOCK`: `1 << 11`.
const SOCK_NONBLOCK: u32 = 1 << 11;
/// `SOCK_CLOEXEC`, which is `O_CLOEXEC`: `1 << 19`.
const SOCK_CLOEXEC: u32 = 1 << 19;

/// Instructions of the program.
pub const PROGRAM_LEN: usize = 29;

/// Positions in the program that a jump names.
const ALLOW: usize = 11;
const DENY: usize = 12;
const KILL: usize = 13;
const SOCKET_CHECK: usize = 14;
/// Where the protocol of a stream socket is checked.
const STREAM_PROTOCOL: usize = 20;
/// Where the protocol of a datagram socket is checked.
const DATAGRAM_PROTOCOL: usize = 24;
/// The last instruction: the refusal that the socket checks fall through to.
/// A jump only goes forward, so they cannot use [`DENY`].
const SOCKET_DENY: usize = PROGRAM_LEN - 1;

/// A jump offset from the instruction at `from` to the one at `to`, which has
/// to come later and within the reach of a byte.
const fn skip(from: usize, to: usize) -> u8 {
    let distance = to - from - 1;
    assert!(distance <= u8::MAX as usize);
    distance as u8
}

const fn load(offset: u32) -> SockFilter {
    SockFilter {
        code: LOAD_WORD,
        jt: 0,
        jf: 0,
        k: offset,
    }
}

const fn equal(at: usize, value: u32, if_true: usize, if_false: usize) -> SockFilter {
    SockFilter {
        code: JUMP_EQUAL,
        jt: skip(at, if_true),
        jf: skip(at, if_false),
        k: value,
    }
}

const fn ret(value: u32) -> SockFilter {
    SockFilter {
        code: RETURN,
        jt: 0,
        jf: 0,
        k: value,
    }
}

/// The program for the capabilities `granted`.
///
/// The result of a socket type that the manifest does not grant is the same
/// instruction as the result of a type that does not exist: it is refused.
pub fn build(granted: Capabilities) -> [SockFilter; PROGRAM_LEN] {
    let stream = if granted.contains(Capability::NetTcp) {
        RET_ALLOW
    } else {
        RET_DENY
    };
    let datagram = if granted.contains(Capability::NetUdp) {
        RET_ALLOW
    } else {
        RET_DENY
    };
    [
        load(OFFSET_ARCH),
        equal(1, AUDIT_ARCH_X86_64, 2, KILL),
        load(OFFSET_NUMBER),
        SockFilter {
            code: JUMP_SET,
            jt: skip(3, KILL),
            jf: 0,
            k: X32_SYSCALL_BIT,
        },
        equal(4, SYS_SOCKET, SOCKET_CHECK, 5),
        equal(5, SYS_LISTEN, DENY, 6),
        equal(6, SYS_ACCEPT, DENY, 7),
        equal(7, SYS_ACCEPT4, DENY, 8),
        equal(8, SYS_IO_URING_SETUP, DENY, 9),
        equal(9, SYS_IO_URING_ENTER, DENY, 10),
        equal(10, SYS_IO_URING_REGISTER, DENY, ALLOW),
        ret(RET_ALLOW),
        ret(RET_DENY),
        ret(RET_KILL_PROCESS),
        load(OFFSET_ARGUMENT0),
        equal(15, AF_INET, 16, SOCKET_DENY),
        load(OFFSET_ARGUMENT1),
        SockFilter {
            code: AND_CONSTANT,
            jt: 0,
            jf: 0,
            k: !(SOCK_NONBLOCK | SOCK_CLOEXEC),
        },
        equal(18, SOCK_STREAM, STREAM_PROTOCOL, 19),
        equal(19, SOCK_DGRAM, DATAGRAM_PROTOCOL, SOCKET_DENY),
        load(OFFSET_ARGUMENT2),
        equal(21, IPPROTO_DEFAULT, 23, 22),
        equal(22, IPPROTO_TCP, 23, SOCKET_DENY),
        ret(stream),
        load(OFFSET_ARGUMENT2),
        equal(25, IPPROTO_DEFAULT, 27, 26),
        equal(26, IPPROTO_UDP, 27, SOCKET_DENY),
        ret(datagram),
        ret(RET_DENY),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    const I386: u32 = 3 | 0x4000_0000;
    const NO_ARGUMENT: u64 = 0;

    /// Run `program` over a `seccomp_data` and return what it ends with. It
    /// knows the opcodes of this module and no others.
    fn run(program: &[SockFilter], arch: u32, number: u32, arguments: [u64; 3]) -> u32 {
        let mut data = [0u8; 64];
        data[..4].copy_from_slice(&number.to_le_bytes());
        data[4..8].copy_from_slice(&arch.to_le_bytes());
        data[16..24].copy_from_slice(&arguments[0].to_le_bytes());
        data[24..32].copy_from_slice(&arguments[1].to_le_bytes());
        data[32..40].copy_from_slice(&arguments[2].to_le_bytes());

        let mut accumulator = 0u32;
        let mut position = 0;
        loop {
            let instruction = program[position];
            position += 1;
            let k = instruction.k;
            match instruction.code {
                LOAD_WORD => {
                    let at = k as usize;
                    accumulator = u32::from_le_bytes(data[at..at + 4].try_into().unwrap());
                }
                AND_CONSTANT => accumulator &= k,
                JUMP_EQUAL => {
                    position += usize::from(if accumulator == k {
                        instruction.jt
                    } else {
                        instruction.jf
                    });
                }
                JUMP_SET => {
                    position += usize::from(if accumulator & k != 0 {
                        instruction.jt
                    } else {
                        instruction.jf
                    });
                }
                RETURN => return k,
                other => panic!("opcode {other:#x} is not one the builder emits"),
            }
        }
    }

    fn granted(capabilities: &[Capability]) -> Capabilities {
        let mut granted = Capabilities::EMPTY;
        for capability in capabilities {
            granted.insert(*capability);
        }
        granted
    }

    fn verdict(
        capabilities: &[Capability],
        arch: u32,
        number: u32,
        argument0: u64,
        argument1: u64,
    ) -> u32 {
        run(
            &build(granted(capabilities)),
            arch,
            number,
            [argument0, argument1, 0],
        )
    }

    /// The verdict for `socket(AF_INET, kind, protocol)`.
    fn socket(capabilities: &[Capability], kind: u32, protocol: u32) -> u32 {
        run(
            &build(granted(capabilities)),
            AUDIT_ARCH_X86_64,
            SYS_SOCKET,
            [u64::from(AF_INET), u64::from(kind), u64::from(protocol)],
        )
    }

    const TCP: &[Capability] = &[Capability::NetTcp];
    const UDP: &[Capability] = &[Capability::NetUdp];
    const BOTH: &[Capability] = &[Capability::NetTcp, Capability::NetUdp];
    const NONE: &[Capability] = &[];

    #[test]
    fn the_wrong_architecture_and_the_x32_bit_kill_the_process() {
        for capabilities in [NONE, BOTH] {
            assert_eq!(
                verdict(capabilities, I386, 0, NO_ARGUMENT, NO_ARGUMENT),
                RET_KILL_PROCESS
            );
            assert_eq!(
                verdict(
                    capabilities,
                    AUDIT_ARCH_X86_64,
                    X32_SYSCALL_BIT | SYS_SOCKET,
                    u64::from(AF_INET),
                    u64::from(SOCK_STREAM)
                ),
                RET_KILL_PROCESS
            );
            assert_eq!(
                verdict(
                    capabilities,
                    AUDIT_ARCH_X86_64,
                    X32_SYSCALL_BIT,
                    NO_ARGUMENT,
                    NO_ARGUMENT
                ),
                RET_KILL_PROCESS
            );
        }
    }

    #[test]
    fn a_stream_socket_needs_the_tcp_capability() {
        let stream = u64::from(SOCK_STREAM);
        let inet = u64::from(AF_INET);
        assert_eq!(
            verdict(TCP, AUDIT_ARCH_X86_64, SYS_SOCKET, inet, stream),
            RET_ALLOW
        );
        assert_eq!(
            verdict(BOTH, AUDIT_ARCH_X86_64, SYS_SOCKET, inet, stream),
            RET_ALLOW
        );
        assert_eq!(
            verdict(UDP, AUDIT_ARCH_X86_64, SYS_SOCKET, inet, stream),
            RET_DENY
        );
        assert_eq!(
            verdict(NONE, AUDIT_ARCH_X86_64, SYS_SOCKET, inet, stream),
            RET_DENY
        );
    }

    #[test]
    fn a_datagram_socket_needs_the_udp_capability_and_ignores_the_flags() {
        let inet = u64::from(AF_INET);
        let datagram = u64::from(SOCK_DGRAM);
        let flags = u64::from(SOCK_CLOEXEC | SOCK_NONBLOCK);
        assert_eq!(
            verdict(UDP, AUDIT_ARCH_X86_64, SYS_SOCKET, inet, datagram),
            RET_ALLOW
        );
        assert_eq!(
            verdict(
                UDP,
                AUDIT_ARCH_X86_64,
                SYS_SOCKET,
                inet,
                datagram | u64::from(SOCK_CLOEXEC)
            ),
            RET_ALLOW
        );
        assert_eq!(
            verdict(UDP, AUDIT_ARCH_X86_64, SYS_SOCKET, inet, datagram | flags),
            RET_ALLOW
        );
        assert_eq!(
            verdict(TCP, AUDIT_ARCH_X86_64, SYS_SOCKET, inet, datagram | flags),
            RET_DENY
        );
        assert_eq!(
            verdict(
                TCP,
                AUDIT_ARCH_X86_64,
                SYS_SOCKET,
                inet,
                u64::from(SOCK_STREAM | SOCK_CLOEXEC)
            ),
            RET_ALLOW
        );
    }

    #[test]
    fn only_inet_streams_and_datagrams_are_sockets() {
        // AF_UNIX is 1, AF_INET6 is 10, and SOCK_RAW is 3.
        for (domain, kind) in [(1, 1), (1, 2), (10, 1), (10, 2), (2, 3), (2, 5), (2, 0)] {
            assert_eq!(
                verdict(BOTH, AUDIT_ARCH_X86_64, SYS_SOCKET, domain, kind),
                RET_DENY,
                "domain {domain} type {kind}"
            );
        }
        // The kernel reads the domain as an int, so the high word is ignored.
        assert_eq!(
            verdict(
                TCP,
                AUDIT_ARCH_X86_64,
                SYS_SOCKET,
                0x1_0000_0000 | u64::from(AF_INET),
                u64::from(SOCK_STREAM)
            ),
            RET_ALLOW
        );
    }

    #[test]
    fn a_socket_type_with_another_protocol_is_refused() {
        // IPPROTO_ICMP is 1, IPPROTO_SCTP is 132 and IPPROTO_UDPLITE is 136.
        for protocol in [1, 132, 136, 255] {
            assert_eq!(socket(BOTH, SOCK_STREAM, protocol), RET_DENY, "{protocol}");
            assert_eq!(socket(BOTH, SOCK_DGRAM, protocol), RET_DENY, "{protocol}");
        }
        assert_eq!(socket(BOTH, SOCK_STREAM, IPPROTO_UDP), RET_DENY);
        assert_eq!(socket(BOTH, SOCK_DGRAM, IPPROTO_TCP), RET_DENY);
    }

    #[test]
    fn the_default_protocol_and_the_one_of_the_type_are_allowed() {
        assert_eq!(socket(TCP, SOCK_STREAM, IPPROTO_DEFAULT), RET_ALLOW);
        assert_eq!(socket(TCP, SOCK_STREAM, IPPROTO_TCP), RET_ALLOW);
        assert_eq!(socket(UDP, SOCK_DGRAM, IPPROTO_DEFAULT), RET_ALLOW);
        assert_eq!(socket(UDP, SOCK_DGRAM, IPPROTO_UDP), RET_ALLOW);
        assert_eq!(socket(UDP, SOCK_STREAM, IPPROTO_TCP), RET_DENY);
        assert_eq!(socket(TCP, SOCK_DGRAM, IPPROTO_UDP), RET_DENY);
    }

    #[test]
    fn io_uring_is_refused_because_a_ring_opens_sockets_on_its_own() {
        for number in [
            SYS_IO_URING_SETUP,
            SYS_IO_URING_ENTER,
            SYS_IO_URING_REGISTER,
        ] {
            for capabilities in [NONE, BOTH] {
                assert_eq!(
                    verdict(
                        capabilities,
                        AUDIT_ARCH_X86_64,
                        number,
                        NO_ARGUMENT,
                        NO_ARGUMENT
                    ),
                    RET_DENY,
                    "call {number}"
                );
            }
        }
    }

    #[test]
    fn nothing_is_accepted_inbound() {
        for number in [SYS_LISTEN, SYS_ACCEPT, SYS_ACCEPT4] {
            assert_eq!(
                verdict(BOTH, AUDIT_ARCH_X86_64, number, NO_ARGUMENT, NO_ARGUMENT),
                RET_DENY,
                "call {number}"
            );
        }
    }

    #[test]
    fn an_unrelated_call_is_allowed() {
        // read, openat, connect and exit_group.
        for number in [0, 257, 42, 231] {
            assert_eq!(
                verdict(NONE, AUDIT_ARCH_X86_64, number, NO_ARGUMENT, NO_ARGUMENT),
                RET_ALLOW,
                "call {number}"
            );
        }
    }

    #[test]
    fn the_program_is_small_and_every_jump_stays_inside_it() {
        for capabilities in [NONE, TCP, UDP, BOTH] {
            let program = build(granted(capabilities));
            assert!(program.len() < 4096);
            for (position, instruction) in program.iter().enumerate() {
                if instruction.code == JUMP_EQUAL || instruction.code == JUMP_SET {
                    assert!(position + 1 + usize::from(instruction.jt) < program.len());
                    assert!(position + 1 + usize::from(instruction.jf) < program.len());
                }
            }
            assert_eq!(program[program.len() - 1].code, RETURN);
        }
    }

    #[test]
    fn an_instruction_is_eight_bytes_in_the_order_of_the_kernel() {
        let instruction = SockFilter {
            code: 0x1234,
            jt: 5,
            jf: 6,
            k: 0x1122_3344,
        };
        assert_eq!(
            instruction.to_bytes(),
            [0x34, 0x12, 5, 6, 0x44, 0x33, 0x22, 0x11]
        );
    }
}
