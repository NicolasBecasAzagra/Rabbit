// SPDX-License-Identifier: GPL-2.0-only
//! The Landlock rules of the manifest, for a Linux kernel.
//!
//! Landlock restricts a process to the file hierarchies and the network ports
//! that a ruleset lists. This module is pure: it computes which rights a
//! kernel of a given ABI version understands and lays out the structures the
//! system calls read, and it makes no system call, so it is tested on any host.
//!
//! Compatibility follows the documentation of the kernel: the ABI version is
//! asked first, and the rights the running kernel does not know are masked out
//! of the ruleset instead of being sent, because a kernel refuses a ruleset
//! with a bit it does not know. A protection that needs a newer ABI is then
//! reported as unavailable by [`Support`], and the caller says so in its log.
//!
//! What the rules cover is the filesystem and the ports of TCP and UDP. They
//! are keyed by port: Landlock has no rule for the address of the other end,
//! so the address of an endpoint of the manifest cannot be enforced here.
//! `bind` is not handled: the agent only reaches out, `listen` and `accept`
//! are refused by the seccomp filter, and handling `bind` would need a rule for
//! the ephemeral port every client socket gets.
//!
//! The values come from `include/uapi/linux/landlock.h` of the kernel, and the
//! ABI version in which each right appeared from the comments of that header
//! and from <https://docs.kernel.org/userspace-api/landlock.html>.

/// `LANDLOCK_ACCESS_FS_EXECUTE` (ABI 1).
const FS_EXECUTE: u64 = 1 << 0;
/// `LANDLOCK_ACCESS_FS_WRITE_FILE` (ABI 1).
const FS_WRITE_FILE: u64 = 1 << 1;
/// `LANDLOCK_ACCESS_FS_READ_FILE` (ABI 1).
const FS_READ_FILE: u64 = 1 << 2;
/// `LANDLOCK_ACCESS_FS_READ_DIR` (ABI 1).
const FS_READ_DIR: u64 = 1 << 3;
/// `LANDLOCK_ACCESS_FS_REMOVE_DIR` (ABI 1).
const FS_REMOVE_DIR: u64 = 1 << 4;
/// `LANDLOCK_ACCESS_FS_REMOVE_FILE` (ABI 1).
const FS_REMOVE_FILE: u64 = 1 << 5;
/// `LANDLOCK_ACCESS_FS_MAKE_CHAR` (ABI 1).
const FS_MAKE_CHAR: u64 = 1 << 6;
/// `LANDLOCK_ACCESS_FS_MAKE_DIR` (ABI 1).
const FS_MAKE_DIR: u64 = 1 << 7;
/// `LANDLOCK_ACCESS_FS_MAKE_REG` (ABI 1).
const FS_MAKE_REG: u64 = 1 << 8;
/// `LANDLOCK_ACCESS_FS_MAKE_SOCK` (ABI 1).
const FS_MAKE_SOCK: u64 = 1 << 9;
/// `LANDLOCK_ACCESS_FS_MAKE_FIFO` (ABI 1).
const FS_MAKE_FIFO: u64 = 1 << 10;
/// `LANDLOCK_ACCESS_FS_MAKE_BLOCK` (ABI 1).
const FS_MAKE_BLOCK: u64 = 1 << 11;
/// `LANDLOCK_ACCESS_FS_MAKE_SYM` (ABI 1).
const FS_MAKE_SYM: u64 = 1 << 12;
/// `LANDLOCK_ACCESS_FS_REFER` (ABI 2).
const FS_REFER: u64 = 1 << 13;
/// `LANDLOCK_ACCESS_FS_TRUNCATE` (ABI 3).
const FS_TRUNCATE: u64 = 1 << 14;
/// `LANDLOCK_ACCESS_FS_IOCTL_DEV` (ABI 5).
const FS_IOCTL_DEV: u64 = 1 << 15;
/// `LANDLOCK_ACCESS_FS_RESOLVE_UNIX` (ABI 9).
const FS_RESOLVE_UNIX: u64 = 1 << 16;

/// `LANDLOCK_ACCESS_NET_CONNECT_TCP` (ABI 4).
const NET_CONNECT_TCP: u64 = 1 << 1;
/// `LANDLOCK_ACCESS_NET_CONNECT_SEND_UDP` (ABI 10).
const NET_CONNECT_SEND_UDP: u64 = 1 << 3;

/// `LANDLOCK_RESTRICT_SELF_TSYNC` (ABI 8): enforce the ruleset on every thread
/// of the process, not only on the one that calls.
const RESTRICT_SELF_TSYNC: u64 = 1 << 3;

/// What a kernel of one ABI version can enforce.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Support {
    /// The ABI version, zero when the kernel has no Landlock.
    pub abi: u32,
    /// The filesystem rights the ruleset handles: denied unless a rule allows.
    pub handled_fs: u64,
    /// The network rights the ruleset handles.
    pub handled_net: u64,
}

impl Support {
    /// The rights a kernel of ABI version `abi` knows. A version newer than
    /// this module knows is treated as the newest it knows.
    pub fn for_abi(abi: u32) -> Self {
        let mut handled_fs = 0;
        if abi >= 1 {
            handled_fs |= FS_EXECUTE
                | FS_WRITE_FILE
                | FS_READ_FILE
                | FS_READ_DIR
                | FS_REMOVE_DIR
                | FS_REMOVE_FILE
                | FS_MAKE_CHAR
                | FS_MAKE_DIR
                | FS_MAKE_REG
                | FS_MAKE_SOCK
                | FS_MAKE_FIFO
                | FS_MAKE_BLOCK
                | FS_MAKE_SYM;
        }
        if abi >= 2 {
            handled_fs |= FS_REFER;
        }
        if abi >= 3 {
            handled_fs |= FS_TRUNCATE;
        }
        if abi >= 5 {
            handled_fs |= FS_IOCTL_DEV;
        }
        if abi >= 9 {
            handled_fs |= FS_RESOLVE_UNIX;
        }
        let mut handled_net = 0;
        if abi >= 4 {
            handled_net |= NET_CONNECT_TCP;
        }
        if abi >= 10 {
            handled_net |= NET_CONNECT_SEND_UDP;
        }
        Self {
            abi,
            handled_fs,
            handled_net,
        }
    }

    /// Whether the filesystem is restricted.
    pub fn fs(&self) -> bool {
        self.handled_fs != 0
    }

    /// Whether the ports of TCP connections are restricted.
    pub fn tcp_port(&self) -> bool {
        self.handled_net & NET_CONNECT_TCP != 0
    }

    /// Whether the ports of UDP datagrams are restricted.
    pub fn udp_port(&self) -> bool {
        self.handled_net & NET_CONNECT_SEND_UDP != 0
    }

    /// The rights a read root gets: open a file for reading, list a directory.
    pub fn read_access(&self) -> u64 {
        (FS_READ_FILE | FS_READ_DIR) & self.handled_fs
    }

    /// The rights a write root gets: the read rights, and open a file for
    /// writing, truncate it, and create a file or a directory.
    pub fn write_access(&self) -> u64 {
        self.read_access()
            | ((FS_WRITE_FILE | FS_TRUNCATE | FS_MAKE_REG | FS_MAKE_DIR) & self.handled_fs)
    }

    /// The right a rule for a TCP port gets, or none when the kernel has no
    /// such right.
    pub fn tcp_access(&self) -> u64 {
        NET_CONNECT_TCP & self.handled_net
    }

    /// The right a rule for a UDP port gets, or none when the kernel has no
    /// such right.
    pub fn udp_access(&self) -> u64 {
        NET_CONNECT_SEND_UDP & self.handled_net
    }

    /// The flags of `landlock_restrict_self`: the ruleset goes to every thread
    /// of the process when the kernel can do that, and to the caller alone when
    /// it cannot (before ABI 8).
    pub fn restrict_flags(&self) -> u64 {
        if self.abi >= 8 {
            RESTRICT_SELF_TSYNC
        } else {
            0
        }
    }

    /// `struct landlock_ruleset_attr` for the system call, and the size to
    /// pass with it.
    ///
    /// Only the fields the kernel of this ABI has are sent: the kernel accepts
    /// a smaller structure than it knows, and refuses a larger one whose extra
    /// bytes are not zero. The first ABI has the filesystem field alone, and
    /// the network field came with ABI 4.
    pub fn ruleset_attr(&self) -> ([u8; 16], usize) {
        let mut bytes = [0; 16];
        bytes[..8].copy_from_slice(&self.handled_fs.to_le_bytes());
        bytes[8..].copy_from_slice(&self.handled_net.to_le_bytes());
        (bytes, if self.abi >= 4 { 16 } else { 8 })
    }
}

/// `struct landlock_path_beneath_attr`, which is packed: the allowed rights
/// and then the descriptor of the directory or file.
pub fn path_beneath_attr(allowed_access: u64, parent_fd: i32) -> [u8; 12] {
    let mut bytes = [0; 12];
    bytes[..8].copy_from_slice(&allowed_access.to_le_bytes());
    bytes[8..].copy_from_slice(&parent_fd.to_le_bytes());
    bytes
}

/// `struct landlock_net_port_attr`: the allowed rights and the port, which the
/// header declares as a `__u64` in host byte order.
pub fn net_port_attr(allowed_access: u64, port: u16) -> [u8; 16] {
    let mut bytes = [0; 16];
    bytes[..8].copy_from_slice(&allowed_access.to_le_bytes());
    bytes[8..].copy_from_slice(&u64::from(port).to_le_bytes());
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_kernel_without_landlock_enforces_nothing() {
        let support = Support::for_abi(0);
        assert_eq!(support.handled_fs, 0);
        assert_eq!(support.handled_net, 0);
        assert!(!support.fs() && !support.tcp_port() && !support.udp_port());
        assert_eq!(support.read_access(), 0);
        assert_eq!(support.write_access(), 0);
    }

    #[test]
    fn abi_1_has_the_thirteen_original_filesystem_rights_and_no_network() {
        let support = Support::for_abi(1);
        assert_eq!(support.handled_fs, 0x1fff);
        assert_eq!(support.handled_net, 0);
        assert!(support.fs());
        assert!(!support.tcp_port() && !support.udp_port());
        assert_eq!(support.read_access(), FS_READ_FILE | FS_READ_DIR);
        // TRUNCATE came with ABI 3, so a write root of ABI 1 cannot have it.
        assert_eq!(
            support.write_access(),
            FS_READ_FILE | FS_READ_DIR | FS_WRITE_FILE | FS_MAKE_REG | FS_MAKE_DIR
        );
        assert_eq!(support.ruleset_attr().1, 8);
    }

    #[test]
    fn abi_3_adds_the_reparent_and_truncate_rights() {
        let support = Support::for_abi(3);
        assert_eq!(support.handled_fs, 0x1fff | FS_REFER | FS_TRUNCATE);
        assert_eq!(support.handled_fs, 0x7fff);
        assert_eq!(support.handled_net, 0);
        assert_eq!(
            support.write_access(),
            FS_READ_FILE | FS_READ_DIR | FS_WRITE_FILE | FS_TRUNCATE | FS_MAKE_REG | FS_MAKE_DIR
        );
        assert_eq!(support.ruleset_attr().1, 8);
    }

    #[test]
    fn abi_4_adds_the_tcp_connect_right() {
        let support = Support::for_abi(4);
        assert_eq!(support.handled_fs, 0x7fff);
        assert_eq!(support.handled_net, NET_CONNECT_TCP);
        assert!(support.tcp_port() && !support.udp_port());
        assert_eq!(support.tcp_access(), NET_CONNECT_TCP);
        assert_eq!(support.udp_access(), 0);
        assert_eq!(support.ruleset_attr().1, 16);
    }

    #[test]
    fn abi_10_adds_the_udp_connect_and_send_right() {
        let support = Support::for_abi(10);
        assert_eq!(support.handled_fs, 0x1_ffff);
        assert_eq!(support.handled_net, NET_CONNECT_TCP | NET_CONNECT_SEND_UDP);
        assert!(support.fs() && support.tcp_port() && support.udp_port());
        assert_eq!(support.tcp_access(), NET_CONNECT_TCP);
        assert_eq!(support.udp_access(), NET_CONNECT_SEND_UDP);
        // A kernel newer than the module is treated as the newest it knows.
        assert_eq!(Support::for_abi(99).handled_fs, support.handled_fs);
        assert_eq!(Support::for_abi(99).handled_net, support.handled_net);
    }

    #[test]
    fn only_abi_8_and_later_restrict_every_thread() {
        assert_eq!(Support::for_abi(7).restrict_flags(), 0);
        assert_eq!(Support::for_abi(8).restrict_flags(), 1 << 3);
        assert_eq!(Support::for_abi(10).restrict_flags(), 1 << 3);
    }

    #[test]
    fn the_intermediate_versions_add_ioctl_and_unix_resolution() {
        assert_eq!(Support::for_abi(5).handled_fs, 0xffff);
        assert_eq!(Support::for_abi(8).handled_fs, 0xffff);
        assert_eq!(Support::for_abi(9).handled_fs, 0x1_ffff);
        assert_eq!(Support::for_abi(9).handled_net, NET_CONNECT_TCP);
    }

    #[test]
    fn a_rule_never_allows_a_right_the_ruleset_does_not_handle() {
        for abi in 0..=12 {
            let support = Support::for_abi(abi);
            assert_eq!(support.write_access() & !support.handled_fs, 0, "abi {abi}");
            assert_eq!(support.read_access() & !support.handled_fs, 0, "abi {abi}");
            assert_eq!(support.tcp_access() & !support.handled_net, 0, "abi {abi}");
            assert_eq!(support.udp_access() & !support.handled_net, 0, "abi {abi}");
        }
    }

    #[test]
    fn the_ruleset_attr_is_the_two_words_of_the_header() {
        let (bytes, size) = Support::for_abi(4).ruleset_attr();
        assert_eq!(size, 16);
        let mut expected = [0u8; 16];
        expected[..8].copy_from_slice(&[0xff, 0x7f, 0, 0, 0, 0, 0, 0]);
        expected[8..].copy_from_slice(&[0x02, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(bytes, expected);

        let (bytes, size) = Support::for_abi(1).ruleset_attr();
        assert_eq!(size, 8);
        assert_eq!(&bytes[..8], &[0xff, 0x1f, 0, 0, 0, 0, 0, 0]);
        assert_eq!(&bytes[8..], &[0; 8]);
    }

    #[test]
    fn the_path_beneath_attr_is_packed_to_twelve_bytes() {
        assert_eq!(
            path_beneath_attr(0x0c, 7),
            [0x0c, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0]
        );
        assert_eq!(
            path_beneath_attr(1 << 2, -1),
            [0x04, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff]
        );
    }

    #[test]
    fn the_net_port_attr_carries_the_port_in_a_word() {
        assert_eq!(
            net_port_attr(NET_CONNECT_TCP, 8080),
            [0x02, 0, 0, 0, 0, 0, 0, 0, 0x90, 0x1f, 0, 0, 0, 0, 0, 0]
        );
    }
}
