# SPDX-License-Identifier: GPL-2.0-only
#
# Rabbit build entry points. Works on macOS and Linux.

CARGO ?= cargo
TOOLCHAIN := $(shell sed -n 's/^channel = "\(.*\)"/\1/p' rust-toolchain.toml)
CARGO := $(CARGO) +$(TOOLCHAIN)

TARGET := x86_64-rabbit
KERNEL_TARGET := $(TARGET).json
# The build output lives outside the repository by default: the checkout can sit
# on an exFAT volume, where the linker writes through mmap and the file system
# can leave large artifacts zeroed. Override TARGET_DIR to put it elsewhere.
TARGET_DIR ?= $(HOME)/.cache/rabbit/target
KERNEL := $(TARGET_DIR)/$(TARGET)/release/rabbit-kernel
HOST_TARGET := $(shell rustc -vV | sed -n 's/^host: //p')

# The bare-metal target has no precompiled core, so it is built from source.
# These flags stay on the command line instead of .cargo/config.toml to keep
# the host toolchain usable for `cargo test` and `cargo clippy`.
KERNEL_FLAGS := --target $(KERNEL_TARGET) \
	--target-dir $(TARGET_DIR) \
	-Z build-std=core,alloc,compiler_builtins \
	-Z build-std-features=compiler-builtins-mem

# The kernel binary is `no_std` + `no_main`: it cannot be built as a host test
# target and is excluded from the host runs. The same goes for the loader
# binary, which embeds the kernel; its library part is tested on its own below.
HOST_ONLY := --exclude rabbit-kernel --exclude rabbit-uefi

# The UEFI loader is a PE32+ application for the firmware target, built from
# source like the kernel. The artifact is x86_64-unknown-uefi/release/rabbit-uefi.efi.
UEFI_TARGET := x86_64-unknown-uefi
UEFI_FLAGS := --target $(UEFI_TARGET) \
	--target-dir $(TARGET_DIR)/uefi \
	-Z build-std=core,compiler_builtins \
	-Z build-std-features=compiler-builtins-mem
UEFI_EFI := $(TARGET_DIR)/uefi/$(UEFI_TARGET)/release/rabbit-uefi.efi

# The user program is a separate workspace and a normal Rust program built
# against the standard library of the Linux musl target. That target is
# precompiled, so no `-Z build-std` is needed and no C toolchain is involved:
# rust-lld links the static position independent executable. The kernel embeds
# the copy in rabbit-userland/rabbit-userland.elf, so it must be built first.
USER_TARGET := x86_64-unknown-linux-musl
USERLAND_ELF := rabbit-userland/rabbit-userland.elf
USERLAND_RUSTFLAGS := -Z unstable-options -C linker=rust-lld -C linker-flavor=gnu-lld
USERLAND_FLAGS := --manifest-path rabbit-userland/Cargo.toml \
	--target $(USER_TARGET) \
	--target-dir $(TARGET_DIR)/userland

# The benchmark workload is a second standard library program, built exactly
# like the userland one. The rig puts the copy into an uncompressed cpio
# archive instead of linking it into the kernel.
BENCH_ELF := rabbit-bench/rabbit-bench.elf
BENCH_FLAGS := --manifest-path rabbit-bench/Cargo.toml \
	--target $(USER_TARGET) \
	--target-dir $(TARGET_DIR)/bench

# The network client is a third standard library program, built exactly like the
# workload. The rig packs it into its own initramfs next to the configuration
# that names the server, so the same bytes run on both kernels.
NETCLIENT_ELF := rabbit-netclient/rabbit-netclient.elf
NETCLIENT_FLAGS := --manifest-path rabbit-netclient/Cargo.toml \
	--target $(USER_TARGET) \
	--target-dir $(TARGET_DIR)/netclient

# The stream client is the fourth program of the network measurement: the same
# shape as the datagram client, over a stream socket instead. It has its own
# archive per run, because the server of the run answers on its own port.
NETSTREAM_ELF := rabbit-netstream/rabbit-netstream.elf
NETSTREAM_FLAGS := --manifest-path rabbit-netstream/Cargo.toml \
	--target $(USER_TARGET) \
	--target-dir $(TARGET_DIR)/netstream

# The inference runtime is a fourth standard library program, built exactly like
# the workload. Its synthetic model, vocabulary and config are packed into the
# same kind of initramfs, so the kernel runs it unmodified.
INFER_DIR := rabbit-infer
INFER_ELF := $(INFER_DIR)/rabbit-infer.elf
INFER_TARGET_DIR := $(TARGET_DIR)/infer
INFER_FLAGS := --manifest-path $(INFER_DIR)/Cargo.toml \
	--target $(USER_TARGET) \
	--target-dir $(INFER_TARGET_DIR)
INFER_MODEL_DIR := $(TARGET_DIR)/infer-model
INFER_INITRAMFS := $(INFER_DIR)/initramfs.cpio

# The agent is a fourth standard library program, built like the runtime and
# packed with the same model so the rig runs the same bytes on both kernels.
AGENT_DIR := rabbit-agent
AGENT_ELF := $(AGENT_DIR)/rabbit-agent.elf
AGENT_TARGET_DIR := $(TARGET_DIR)/agent
AGENT_FLAGS := --manifest-path $(AGENT_DIR)/Cargo.toml \
	--target $(USER_TARGET) \
	--target-dir $(AGENT_TARGET_DIR)
AGENT_INITRAMFS := $(AGENT_DIR)/initramfs.cpio
# The archive of the network gate carries the same model and the endpoint of the
# server of the run, which neither kernel passes to its init as an argument. The
# variables are `?=` so the rig overrides them from the environment.
AGENT_NET_INITRAMFS ?= $(AGENT_DIR)/initramfs-net.cpio

# The quantized checkpoint: the width, the directory it is read from and the
# directory it is written to.
QUANT_BITS ?= 8
QUANT_IN ?= $(INFER_MODEL_DIR)
QUANT_OUT ?= $(TARGET_DIR)/model-quant
AGENT_NET_STAGE ?= $(TARGET_DIR)/agent-net
AGENT_NET_ADDRESS ?= 10.0.2.2
AGENT_NET_PORT ?= 9999
AGENT_NET_HTTP_PORT ?= 9998
AGENT_NET_PATH ?= /agent
AGENT_NET_SELFTEST ?= 1

# The reader of the cpio archive of the initramfs is a crate of its own, like
# the other programs that are not the kernel.
INITRAMFS_DIR := rabbit-initramfs

# The protocols of the network path are a crate of their own too, but they are
# not a program: the kernel depends on them, so they are built for the kernel
# target, and their tests run on the host.
NET_DIR := rabbit-net

# The protocol of the agent's HTTP client is a crate of the same kind: pure, no
# dependency, built for the kernel target and tested on the host. The agent
# depends on it, so it is built by the agent target too.
HTTP_DIR := rabbit-http

QEMU ?= qemu-system-x86_64
# Number of processors for the multiprocessor boot.
SMP ?= 4
QEMU_FLAGS := -M q35 -m 512M \
	-kernel $(KERNEL) \
	-display none -monitor none -no-reboot \
	-serial stdio \
	-device isa-debug-exit,iobase=0xf4,iosize=0x04

.PHONY: all build uefi run run-uefi run-smp test fmt fmt-check clippy clippy-host check clean userland bench netclient netstream print-kernel-path infer infer-model infer-initramfs matvec-check-initramfs matvec-check matvec-check-run matvec-check-gguf-gate agent agent-initramfs agent-net-initramfs run-infer quantize

# The path of the built kernel, so the CI does not have to repeat it.
print-kernel-path:
	@echo $(KERNEL)

all: build

# The kernel embeds the user program, so it is always built first. The copy
# into the repository is what the kernel embeds: the linker writes through
# mmap, which is not reliable on every file system, while a plain copy is.
# The user programs are standard library programs for Linux musl, so the
# precompiled standard library of that target has to exist for the pinned
# toolchain. Installing it here means a fresh machine builds without a manual
# step.
musl-std:
	@rustup target list --installed --toolchain $(TOOLCHAIN) | grep -qx $(USER_TARGET) || \
		rustup target add --toolchain $(TOOLCHAIN) $(USER_TARGET)

userland: musl-std
	RUSTFLAGS="$(USERLAND_RUSTFLAGS)" $(CARGO) build --release $(USERLAND_FLAGS)
	cp $(TARGET_DIR)/userland/$(USER_TARGET)/release/rabbit-userland $(USERLAND_ELF)

# Static workload binary for the benchmark rig.
# The features of the workload. The CI builds it with "small" so that the boot
# with the initramfs finishes in time under emulation.
BENCH_FEATURES ?=

bench:
	RUSTFLAGS="$(USERLAND_RUSTFLAGS)" $(CARGO) build --release $(BENCH_FLAGS) $(if $(BENCH_FEATURES),--features $(BENCH_FEATURES),)
	cp $(TARGET_DIR)/bench/$(USER_TARGET)/release/rabbit-bench $(BENCH_ELF)

# Static network client for the rig and for the gate of the pipeline. The rig
# packs it with the configuration that names the server, so the archive is built
# per run and not here.
netclient: musl-std
	RUSTFLAGS="$(USERLAND_RUSTFLAGS)" $(CARGO) build --release $(NETCLIENT_FLAGS)
	cp $(TARGET_DIR)/netclient/$(USER_TARGET)/release/rabbit-netclient $(NETCLIENT_ELF)

# Static stream client for the rig and for the gate of the pipeline. The archive
# is built per run, like the one of the datagram client, because the port of the
# server belongs to the run.
netstream: musl-std
	RUSTFLAGS="$(USERLAND_RUSTFLAGS)" $(CARGO) build --release $(NETSTREAM_FLAGS)
	cp $(TARGET_DIR)/netstream/$(USER_TARGET)/release/rabbit-netstream $(NETSTREAM_ELF)

# The inference runtime, built with the same rust-lld recipe as the workload so
# the same bytes can run on Rabbit and on mainline Linux.
infer: musl-std
	RUSTFLAGS="$(USERLAND_RUSTFLAGS)" $(CARGO) build --release $(INFER_FLAGS)
	cp $(INFER_TARGET_DIR)/$(USER_TARGET)/release/rabbit-infer $(INFER_ELF)

# The synthetic model is a development tool, built and run on the host. Its
# output lives outside the repository with the rest of the build products.
# The tool that writes a quantized checkpoint. It reads the directory of a
# checkpoint and writes another one, and it prints the digest of what it
# wrote. The width is a variable, so the same command makes the four bit and
# the eight bit form.
quantize:
	$(CARGO) run --release --manifest-path $(INFER_DIR)/Cargo.toml \
		--target $(HOST_TARGET) --target-dir $(TARGET_DIR)/infer-host \
		--bin rabbit-infer-quantize -- --bits $(QUANT_BITS) \
		--in $(QUANT_IN) --out $(QUANT_OUT)

# The development model. It is written into the directory of the model, and it
# must never be written over a checkpoint that somebody downloaded: the
# directory is the only copy of that checkpoint. The target refuses to write
# into a directory that already holds a checkpoint with more than a megabyte,
# which is what a real one weighs and what the development one does not.
infer-model:
	@if [ -d "$(INFER_MODEL_DIR)" ] && \
		[ "$$(wc -c < "$(INFER_MODEL_DIR)/model.safetensors" 2>/dev/null || echo 0)" -gt 1048576 ]; then \
		echo "infer-model: $(INFER_MODEL_DIR) holds a real checkpoint, not writing over it"; \
	else \
		$(CARGO) run --release --manifest-path $(INFER_DIR)/Cargo.toml \
			--target $(HOST_TARGET) --target-dir $(TARGET_DIR)/infer-host \
			--bin rabbit-infer-gen -- --out $(INFER_MODEL_DIR); \
	fi

# One uncompressed cpio archive carrying the runtime at /bench and /init, plus
# the model, vocabulary and config the runtime opens by path.
infer-initramfs: infer infer-model
	EXTRA_DIR=$(INFER_MODEL_DIR) bench/mkinitramfs.sh $(INFER_INITRAMFS) $(INFER_ELF)

# The tool calling agent over the runtime, built with the same rust-lld recipe
# and packed with the same model. The default task reads /vocab.json from the
# initramfs, so the archive carries the runtime files unchanged.
agent: musl-std
	RUSTFLAGS="$(USERLAND_RUSTFLAGS)" $(CARGO) build --release $(AGENT_FLAGS)
	cp $(AGENT_TARGET_DIR)/$(USER_TARGET)/release/rabbit-agent $(AGENT_ELF)

agent-initramfs: agent infer-model
	EXTRA_DIR=$(INFER_MODEL_DIR) bench/mkinitramfs.sh $(AGENT_INITRAMFS) $(AGENT_ELF)

# The archive of the agent with the endpoint of the server of this run. The
# address and the port reach the program through a file because neither kernel
# passes arguments to the program it starts as init. `selftest=1` makes the
# agent prove the network tool before it asks the model, which is what the gate
# needs when the model is the synthetic one.
agent-net-initramfs: agent
	rm -rf $(AGENT_NET_STAGE)
	mkdir -p $(AGENT_NET_STAGE)
	cp -R $(INFER_MODEL_DIR)/. $(AGENT_NET_STAGE)/
	printf 'address=%s\nport=%s\nhttp_port=%s\nscheme=http\npath=%s\nselftest=%s\n' \
		'$(AGENT_NET_ADDRESS)' '$(AGENT_NET_PORT)' '$(AGENT_NET_HTTP_PORT)' \
		'$(AGENT_NET_PATH)' '$(AGENT_NET_SELFTEST)' \
		> $(AGENT_NET_STAGE)/net.conf
	EXTRA_DIR=$(AGENT_NET_STAGE) bench/mkinitramfs.sh $(AGENT_NET_INITRAMFS) $(AGENT_ELF)

run-infer: infer-initramfs build
	@$(QEMU) -M q35 -m 512M \
		-kernel $(KERNEL) -initrd $(INFER_INITRAMFS) \
		-append console=ttyS0 \
		-display none -monitor none -no-reboot \
		-serial stdio \
		-device isa-debug-exit,iobase=0xf4,iosize=0x04; status=$$?; \
	if [ $$status -eq 33 ]; then \
		echo "boot OK (QEMU exit status 33)"; \
	else \
		echo "unexpected QEMU exit status $$status"; exit $$status; \
	fi

# The self-check of the vector kernels and of the decoder on every path. It is
# a second program of the inference crate, packed alone into an initramfs of
# the same kind, so Rabbit runs it as init on the processor of the guest. It
# checks the eight and four bit products bit for bit, and then decodes whole
# prompts with the synthetic model, in bfloat16 and in the Q8 and Q4 copies
# the repository's own quantizer writes from it, and compares the greedy tokens
# and the logits of every path with the scalar one. It also proves the GGUF
# reader: the synthetic weights are written as a GGUF `llama` file with the rows
# of the query and key projections permuted as the llama.cpp converter does (and
# once more without the permutation), the checkpoint runs with the split half
# RoPE and the GGUF file with adjacent pairs, and the two must decode the same
# tokens while the two wrong readings of the GGUF file must not. The model is the synthetic
# development model, written under MATVEC_DIR and not taken from INFER_MODEL_DIR,
# which can hold a real checkpoint far too large for the emulated guest. The
# outputs live outside the repository. MATVEC_CPU picks the processor model, for
# example `Skylake-Client,-avx2,-fma` to leave the AVX2 path out; then pass
# MATVEC_REQUIRE="scalar sse2" too, or the gate fails on the missing avx2 lines.
MATVEC_CPU ?= Skylake-Client
# The paths that must appear as exact, so a path the guest does not offer is a
# failure and not a silent skip. Use `scalar sse2` with a CPU model without AVX2.
MATVEC_REQUIRE ?= scalar sse2 avx2
MATVEC_DIR := $(TARGET_DIR)/matvec-check
MATVEC_ELF := $(MATVEC_DIR)/rabbit-infer-matvec-check.elf
MATVEC_INITRAMFS := $(MATVEC_DIR)/initramfs.cpio
MATVEC_MODEL := $(MATVEC_DIR)/model
MATVEC_STAGE := $(MATVEC_DIR)/stage
MATVEC_LOG := $(MATVEC_DIR)/serial.log

matvec-check-initramfs: musl-std
	RUSTFLAGS="$(USERLAND_RUSTFLAGS)" $(CARGO) build --release $(INFER_FLAGS) \
		--features matvec-check --bin rabbit-infer-matvec-check
	mkdir -p $(MATVEC_DIR)
	cp $(INFER_TARGET_DIR)/$(USER_TARGET)/release/rabbit-infer-matvec-check $(MATVEC_ELF)
	rm -rf "$(MATVEC_MODEL)" "$(MATVEC_DIR)/q8" "$(MATVEC_DIR)/q4" "$(MATVEC_STAGE)"
	$(CARGO) run --release --manifest-path $(INFER_DIR)/Cargo.toml \
		--target $(HOST_TARGET) --target-dir $(TARGET_DIR)/infer-host \
		--bin rabbit-infer-gen -- --out $(MATVEC_MODEL) --gguf --gguf-unpermuted
	$(MAKE) quantize QUANT_BITS=8 QUANT_IN=$(MATVEC_MODEL) QUANT_OUT=$(MATVEC_DIR)/q8
	$(MAKE) quantize QUANT_BITS=4 QUANT_IN=$(MATVEC_MODEL) QUANT_OUT=$(MATVEC_DIR)/q4
	mkdir -p $(MATVEC_STAGE)
	cp $(MATVEC_MODEL)/config.json $(MATVEC_MODEL)/model.safetensors $(MATVEC_STAGE)/
	cp $(MATVEC_MODEL)/model.gguf $(MATVEC_MODEL)/model-nopermute.gguf $(MATVEC_STAGE)/
	cp $(MATVEC_DIR)/q8/model.safetensors $(MATVEC_STAGE)/model-q8.safetensors
	cp $(MATVEC_DIR)/q4/model.safetensors $(MATVEC_STAGE)/model-q4.safetensors
	EXTRA_DIR=$(MATVEC_STAGE) bench/mkinitramfs.sh $(MATVEC_INITRAMFS) $(MATVEC_ELF)

matvec-check: matvec-check-initramfs build
	@$(MAKE) --no-print-directory matvec-check-run

# The run and the gates alone, on what `matvec-check` built, so a second
# processor model does not build the model and the archive again.
matvec-check-run:
	@$(QEMU) -M q35 -cpu $(MATVEC_CPU) -m 512M \
		-kernel $(KERNEL) -initrd $(MATVEC_INITRAMFS) \
		-append console=ttyS0 \
		-display none -monitor none -no-reboot \
		-serial file:$(MATVEC_LOG) \
		-device isa-debug-exit,iobase=0xf4,iosize=0x04; status=$$?; \
	tr -d '\r' < $(MATVEC_LOG); \
	if [ $$status -ne 33 ]; then \
		echo "unexpected QEMU exit status $$status"; exit 1; \
	fi; \
	if ! tr -d '\r' < $(MATVEC_LOG) | grep -q '^MATVEC OK checked='; then \
		echo "matvec-check: no MATVEC OK line"; exit 1; \
	fi; \
	for path in $(MATVEC_REQUIRE); do \
		for format in q8 q4; do \
			if ! tr -d '\r' < $(MATVEC_LOG) | grep -q "^MATVEC $$format path=$$path shapes=[0-9]* bit_exact=true$$"; then \
				echo "matvec-check: path $$path was not checked as exact for $$format"; exit 1; \
			fi; \
		done; \
	done; \
	if ! tr -d '\r' < $(MATVEC_LOG) | grep -q '^MATVEC model OK$$'; then \
		echo "matvec-check: no MATVEC model OK line"; exit 1; \
	fi; \
	if tr -d '\r' < $(MATVEC_LOG) | grep -q 'tokens_equal=false\|^MATVEC model MISMATCH'; then \
		echo "matvec-check: a path generated other tokens than the scalar one"; exit 1; \
	fi; \
	for path in $(MATVEC_REQUIRE); do \
		for format in bf16 q8 q4; do \
			case $$format in \
				bf16) difference='[0-9.e+-]*' ;; \
				*) difference='0' ;; \
			esac; \
			for steps in 8 64; do \
				if ! tr -d '\r' < $(MATVEC_LOG) | grep -q "^MATVEC model format=$$format path=$$path prompts=[0-9]* steps=$$steps tokens_equal=true max_abs_logit_diff=$$difference undecided=0$$"; then \
					echo "matvec-check: path $$path did not decode the same $$steps tokens as the scalar one for $$format, or a step was undecided, or the logits of a quantized format differ"; exit 1; \
				fi; \
			done; \
		done; \
	done; \
	$(MAKE) --no-print-directory matvec-check-gguf-gate || exit 1; \
	echo "matvec-check OK (cpu $(MATVEC_CPU), QEMU exit status 33)"

# The gates on the GGUF lines of the serial log, on the host and apart from the
# Rust check that prints them, so the log is judged twice. `grep -E` and awk are
# the portable subset (no GNU `\|`), and the numbers are compared as numbers.
# Run it alone on a log with `make matvec-check-gguf-gate MATVEC_LOG=<file>`.
# Every twin line must hold its tokens, have no undecided step and stay below
# 1e-5 in the logits, and the smallest margin between the two best logits must
# be positive.
GGUF_NUMBER := ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$$
GGUF_TWIN_AWK := '/^GGUF twin /{ n++; te=""; un=""; df=""; \
	for (i = 1; i <= NF; i++) { split($$i, kv, "="); \
		if (kv[1] == "tokens_equal") te = kv[2]; \
		if (kv[1] == "undecided") un = kv[2]; \
		if (kv[1] == "max_abs_logit_diff") df = kv[2] } \
	if (te != "true" || un != "0" || df !~ /$(GGUF_NUMBER)/ || df + 0 >= 1e-5) { bad = 1; print "bad twin line: " $$0 } } \
	END { exit (n > 0 && !bad) ? 0 : 1 }'
GGUF_MARGIN_AWK := '/^GGUF margin /{ n++; mg=""; \
	for (i = 1; i <= NF; i++) { split($$i, kv, "="); if (kv[1] == "min_top2") mg = kv[2] } \
	if (mg !~ /$(GGUF_NUMBER)/ || mg + 0 <= 0) { bad = 1; print "bad margin line: " $$0 } } \
	END { exit (n > 0 && !bad) ? 0 : 1 }'

matvec-check-gguf-gate:
	@for steps in 8 64; do \
		if ! tr -d '\r' < $(MATVEC_LOG) | grep -Eq "^GGUF twin format=f32 prompts=[0-9]+ steps=$$steps tokens_equal=true max_abs_logit_diff=[0-9.e+-]+ undecided=0 bit_exact=(true|false)$$"; then \
			echo "matvec-check: the GGUF file did not decode the same $$steps tokens as its safetensors twin"; exit 1; \
		fi; \
	done; \
	if ! tr -d '\r' < $(MATVEC_LOG) | awk $(GGUF_TWIN_AWK); then \
		echo "matvec-check: a GGUF twin line has other tokens, an undecided step or a logit difference of 1e-5 or more"; exit 1; \
	fi; \
	if ! tr -d '\r' < $(MATVEC_LOG) | awk $(GGUF_MARGIN_AWK); then \
		echo "matvec-check: the smallest margin of the GGUF check is missing or not positive"; exit 1; \
	fi; \
	for control in split_half_rope no_permutation; do \
		if ! tr -d '\r' < $(MATVEC_LOG) | grep -Eq "^GGUF negative_control $$control detected=true "; then \
			echo "matvec-check: the wrong reading $$control of the GGUF file was not detected"; exit 1; \
		fi; \
	done; \
	if tr -d '\r' < $(MATVEC_LOG) | grep -Eq '^GGUF (FAIL|ERR)|detected=false'; then \
		echo "matvec-check: the GGUF proof failed"; exit 1; \
	fi; \
	if ! tr -d '\r' < $(MATVEC_LOG) | grep -Eq '^GGUF OK$$'; then \
		echo "matvec-check: no GGUF OK line"; exit 1; \
	fi

build: userland infer
	$(CARGO) build --release $(KERNEL_FLAGS)

# The loader embeds the PVH kernel, so the kernel is built first and its
# absolute path reaches the compiler through RABBIT_KERNEL_ELF.
uefi: build
	RABBIT_KERNEL_ELF=$(abspath $(KERNEL)) $(CARGO) build -p rabbit-uefi --release $(UEFI_FLAGS)

# The firmware images of the UEFI boot. They are looked for where Ubuntu (the
# `ovmf` package) and Homebrew (`qemu`) put them; pass OVMF_CODE and OVMF_VARS
# to use others. The variable store is copied, so a run never changes the
# original.
OVMF_CODE ?= $(firstword $(wildcard \
	/usr/share/OVMF/OVMF_CODE_4M.fd \
	/usr/share/OVMF/OVMF_CODE.fd \
	/opt/homebrew/share/qemu/edk2-x86_64-code.fd \
	/usr/local/share/qemu/edk2-x86_64-code.fd))
OVMF_VARS ?= $(firstword $(wildcard \
	/usr/share/OVMF/OVMF_VARS_4M.fd \
	/usr/share/OVMF/OVMF_VARS.fd \
	/opt/homebrew/share/qemu/edk2-i386-vars.fd \
	/usr/local/share/qemu/edk2-i386-vars.fd))
UEFI_ESP := $(TARGET_DIR)/uefi-esp
UEFI_VARS := $(TARGET_DIR)/uefi-vars.fd
UEFI_MEM ?= 512M

# The same kernel booted through the UEFI loader: QEMU starts the firmware,
# the firmware starts BOOTX64.EFI from a directory used as a FAT disk, and the
# loader hands over to the PVH entry. It ends like `run`, with status 33.
run-uefi: uefi
	@test -n "$(OVMF_CODE)" -a -n "$(OVMF_VARS)" || { \
		echo "no OVMF firmware found: install the ovmf package or pass OVMF_CODE and OVMF_VARS"; exit 1; }
	@rm -rf $(UEFI_ESP) && mkdir -p $(UEFI_ESP)/EFI/BOOT
	@cp $(UEFI_EFI) $(UEFI_ESP)/EFI/BOOT/BOOTX64.EFI
	@cp $(OVMF_VARS) $(UEFI_VARS)
	@$(QEMU) -M q35 -m $(UEFI_MEM) \
		-drive if=pflash,format=raw,readonly=on,file=$(OVMF_CODE) \
		-drive if=pflash,format=raw,file=$(UEFI_VARS) \
		-drive format=raw,file=fat:rw:$(UEFI_ESP) \
		-display none -monitor none -no-reboot \
		-serial stdio \
		-device isa-debug-exit,iobase=0xf4,iosize=0x04; status=$$?; \
	if [ $$status -eq 33 ]; then \
		echo "boot OK through UEFI (QEMU exit status 33)"; \
	else \
		echo "unexpected QEMU exit status $$status"; exit $$status; \
	fi

# The multiprocessor boot. The kernel finds the processors in the ACPI table
# and starts them, so this is the only way to exercise that path.
run-smp: build
	@$(QEMU) -M q35 -cpu Skylake-Client -smp $(SMP) -m 512M \
		-kernel $(KERNEL) \
		-display none -monitor none -no-reboot \
		-serial stdio \
		-device isa-debug-exit,iobase=0xf4,iosize=0x04; status=$$?; \
	if [ $$status -eq 33 ]; then \
		echo "boot OK with $(SMP) processors (QEMU exit status 33)"; \
	else \
		echo "unexpected QEMU exit status $$status"; exit $$status; \
	fi

run: build
	@$(QEMU) $(QEMU_FLAGS); status=$$?; \
	if [ $$status -eq 33 ]; then \
		echo "boot OK (QEMU exit status 33)"; \
	else \
		echo "unexpected QEMU exit status $$status"; exit $$status; \
	fi

test:
	$(CARGO) test --workspace --target $(HOST_TARGET) --target-dir $(TARGET_DIR) $(HOST_ONLY)
	$(CARGO) test -p rabbit-uefi --lib \
		--target $(HOST_TARGET) --target-dir $(TARGET_DIR)
	$(CARGO) test --manifest-path $(INFER_DIR)/Cargo.toml --lib \
		--target $(HOST_TARGET) --target-dir $(TARGET_DIR)/infer-host
	$(CARGO) test --manifest-path $(AGENT_DIR)/Cargo.toml \
		--target $(HOST_TARGET) --target-dir $(TARGET_DIR)/agent-host
	$(CARGO) test --manifest-path $(INITRAMFS_DIR)/Cargo.toml \
		--target $(HOST_TARGET) --target-dir $(TARGET_DIR)/initramfs-host
	$(CARGO) test --manifest-path $(NET_DIR)/Cargo.toml \
		--target $(HOST_TARGET) --target-dir $(TARGET_DIR)/net-host
	$(CARGO) test --manifest-path $(HTTP_DIR)/Cargo.toml \
		--target $(HOST_TARGET) --target-dir $(TARGET_DIR)/http-host
	$(CARGO) test --manifest-path rabbit-netclient/Cargo.toml \
		--target $(HOST_TARGET) --target-dir $(TARGET_DIR)/netclient-host
	$(CARGO) test --manifest-path rabbit-netstream/Cargo.toml \
		--target $(HOST_TARGET) --target-dir $(TARGET_DIR)/netstream-host


fmt:
	$(CARGO) fmt --all
	$(CARGO) fmt --manifest-path rabbit-userland/Cargo.toml
	$(CARGO) fmt --manifest-path rabbit-bench/Cargo.toml
	$(CARGO) fmt --manifest-path $(INFER_DIR)/Cargo.toml
	$(CARGO) fmt --manifest-path $(AGENT_DIR)/Cargo.toml
	$(CARGO) fmt --manifest-path $(INITRAMFS_DIR)/Cargo.toml
	$(CARGO) fmt --manifest-path $(NET_DIR)/Cargo.toml
	$(CARGO) fmt --manifest-path $(HTTP_DIR)/Cargo.toml
	$(CARGO) fmt --manifest-path rabbit-netclient/Cargo.toml
	$(CARGO) fmt --manifest-path rabbit-netstream/Cargo.toml

fmt-check:
	$(CARGO) fmt --all -- --check
	$(CARGO) fmt --manifest-path rabbit-userland/Cargo.toml -- --check
	$(CARGO) fmt --manifest-path rabbit-bench/Cargo.toml -- --check
	$(CARGO) fmt --manifest-path $(INFER_DIR)/Cargo.toml -- --check
	$(CARGO) fmt --manifest-path $(AGENT_DIR)/Cargo.toml -- --check
	$(CARGO) fmt --manifest-path $(INITRAMFS_DIR)/Cargo.toml -- --check
	$(CARGO) fmt --manifest-path $(NET_DIR)/Cargo.toml -- --check
	$(CARGO) fmt --manifest-path $(HTTP_DIR)/Cargo.toml -- --check
	$(CARGO) fmt --manifest-path rabbit-netclient/Cargo.toml -- --check
	$(CARGO) fmt --manifest-path rabbit-netstream/Cargo.toml -- --check

# The kernel build reads the embedded user program, so both lint targets need
# the userland binary to exist.
clippy: userland
	$(CARGO) clippy --workspace --exclude rabbit-uefi --bins --lib $(KERNEL_FLAGS) -- -D warnings
	$(CARGO) clippy --manifest-path $(NET_DIR)/Cargo.toml --lib $(KERNEL_FLAGS) -- -D warnings
	$(CARGO) clippy --manifest-path $(HTTP_DIR)/Cargo.toml --lib $(KERNEL_FLAGS) -- -D warnings
	RUSTFLAGS="$(USERLAND_RUSTFLAGS)" $(CARGO) clippy $(USERLAND_FLAGS) -- -D warnings
	RUSTFLAGS="$(USERLAND_RUSTFLAGS)" $(CARGO) clippy $(BENCH_FLAGS) -- -D warnings
	RUSTFLAGS="$(USERLAND_RUSTFLAGS)" $(CARGO) clippy $(INFER_FLAGS) --bins --lib -- -D warnings
	RUSTFLAGS="$(USERLAND_RUSTFLAGS)" $(CARGO) clippy $(INFER_FLAGS) --features matvec-check --bins -- -D warnings
	RUSTFLAGS="$(USERLAND_RUSTFLAGS)" $(CARGO) clippy $(AGENT_FLAGS) --bins -- -D warnings
	RUSTFLAGS="$(USERLAND_RUSTFLAGS)" $(CARGO) clippy $(NETCLIENT_FLAGS) --bins -- -D warnings
	RUSTFLAGS="$(USERLAND_RUSTFLAGS)" $(CARGO) clippy $(NETSTREAM_FLAGS) --bins -- -D warnings

clippy-host:
	$(CARGO) clippy --workspace --all-targets --target $(HOST_TARGET) \
		--target-dir $(TARGET_DIR) $(HOST_ONLY) -- -D warnings
	$(CARGO) clippy -p rabbit-uefi --lib --tests --target $(HOST_TARGET) \
		--target-dir $(TARGET_DIR) -- -D warnings
	$(CARGO) clippy --manifest-path $(INFER_DIR)/Cargo.toml --all-targets \
		--target $(HOST_TARGET) --target-dir $(TARGET_DIR)/infer-host -- -D warnings
	$(CARGO) clippy --manifest-path $(AGENT_DIR)/Cargo.toml --all-targets \
		--target $(HOST_TARGET) --target-dir $(TARGET_DIR)/agent-host -- -D warnings
	$(CARGO) clippy --manifest-path $(INITRAMFS_DIR)/Cargo.toml --all-targets \
		--target $(HOST_TARGET) --target-dir $(TARGET_DIR)/initramfs-host -- -D warnings
	$(CARGO) clippy --manifest-path $(NET_DIR)/Cargo.toml --all-targets \
		--target $(HOST_TARGET) --target-dir $(TARGET_DIR)/net-host -- -D warnings
	$(CARGO) clippy --manifest-path $(HTTP_DIR)/Cargo.toml --all-targets \
		--target $(HOST_TARGET) --target-dir $(TARGET_DIR)/http-host -- -D warnings
	$(CARGO) clippy --manifest-path rabbit-netclient/Cargo.toml --all-targets \
		--target $(HOST_TARGET) --target-dir $(TARGET_DIR)/netclient-host -- -D warnings
	$(CARGO) clippy --manifest-path rabbit-netstream/Cargo.toml --all-targets \
		--target $(HOST_TARGET) --target-dir $(TARGET_DIR)/netstream-host -- -D warnings


check: userland
	$(CARGO) check --workspace --exclude rabbit-uefi --bins --lib $(KERNEL_FLAGS)

clean:
	$(CARGO) clean --target-dir $(TARGET_DIR)
	rm -rf $(TARGET_DIR) $(USERLAND_ELF) $(BENCH_ELF) $(NETCLIENT_ELF) $(NETSTREAM_ELF) $(INFER_ELF) $(INFER_INITRAMFS) \
		$(AGENT_ELF) $(AGENT_INITRAMFS) $(AGENT_NET_INITRAMFS) $(AGENT_NET_STAGE)
