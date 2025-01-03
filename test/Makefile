# SPDX-License-Identifier: MPL-2.0

BINARY_CACHE_DIR := /opt/linux_binary_cache
VDSO_DIR := ../target
VDSO_LIB := $(VDSO_DIR)/vdso64.so
MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CUR_DIR := $(patsubst %/,%,$(dir $(MKFILE_PATH)))
ATOMIC_WGET := $(CUR_DIR)/../tools/atomic_wget.sh
BUILD_DIR := $(CUR_DIR)/build
INITRAMFS := $(BUILD_DIR)/initramfs
INITRAMFS_FILELIST := $(BUILD_DIR)/initramfs.filelist
INITRAMFS_IMAGE := $(BUILD_DIR)/initramfs.cpio.gz
EXT2_IMAGE := $(BUILD_DIR)/ext2.img
EXFAT_IMAGE := $(BUILD_DIR)/exfat.img
INITRAMFS_EMPTY_DIRS := \
	$(INITRAMFS)/root \
	$(INITRAMFS)/tmp \
	$(INITRAMFS)/opt \
	$(INITRAMFS)/proc \
	$(INITRAMFS)/dev \
	$(INITRAMFS)/ext2 \
	$(INITRAMFS)/exfat
INITRAMFS_ALL_DIRS := \
	$(INITRAMFS)/etc \
	$(INITRAMFS)/lib/x86_64-linux-gnu \
	$(INITRAMFS)/lib/python3.10 \
	$(INITRAMFS)/lib/python3 \
	$(INITRAMFS)/lib64 \
	$(INITRAMFS)/bin \
	$(INITRAMFS)/sbin \
	$(INITRAMFS)/usr/bin \
	$(INITRAMFS)/usr/python \
	$(INITRAMFS)/usr/lib/python3 \
	$(INITRAMFS)/usr/lib/python3.10 \
	$(INITRAMFS)/usr/local \
	$(INITRAMFS)/usr/local/bin \
	$(INITRAMFS)/usr/local/lib/python3.10/dist-packages \
	$(INITRAMFS)/test \
	$(INITRAMFS)/benchmark \
	$(INITRAMFS_EMPTY_DIRS)
SYSCALL_TEST_DIR := $(INITRAMFS)/opt/syscall_test

.PHONY: all
all: build

$(INITRAMFS)/lib/x86_64-linux-gnu: | $(VDSO_LIB)
	@mkdir -p $@
	@cp -L /lib/x86_64-linux-gnu/libc.so.6 $@
	@cp -L /lib/x86_64-linux-gnu/libstdc++.so.6 $@
	@cp -L /lib/x86_64-linux-gnu/libm.so.6 $@
	@cp -L /lib/x86_64-linux-gnu/libgcc_s.so.1 $@
	@cp -L /lib/x86_64-linux-gnu/libpthread.so.0 $@
	@cp -L /lib/x86_64-linux-gnu/libexpat.so.1 $@
	@cp -L /lib/x86_64-linux-gnu/libc.so.6 $@
	@cp -L /lib/x86_64-linux-gnu/libffi.so.8 $@
	@cp -L /lib/x86_64-linux-gnu/librt.so.1 $@
	@cp -L /lib/x86_64-linux-gnu/ld-linux-x86-64.so.2  $@
	@cp -L /lib/x86_64-linux-gnu/libbz2.so.1.0 $@
	@cp -L /lib/x86_64-linux-gnu/liblzma.so.5 $@
	@# required for benchmarks
	@cp -L /lib/x86_64-linux-gnu/libcrypto.so.3 $@
	@cp -L /lib/x86_64-linux-gnu/libcrypt.so.1 $@
	@cp -L /lib/x86_64-linux-gnu/libdl.so.2 $@
	@cp -L /lib/x86_64-linux-gnu/libz.so.1 $@
	@cp -L /lib/x86_64-linux-gnu/libmvec.so.1 $@
	@cp -L /usr/local/benchmark/iperf/lib/libiperf.so.0 $@
	@# required for LMbench-network
	@cp -L /lib/x86_64-linux-gnu/libtirpc.so.3 $@
	@cp -L /lib/x86_64-linux-gnu/libgssapi_krb5.so.2 $@
	@cp -L /lib/x86_64-linux-gnu/libkrb5.so.3 $@
	@cp -L /lib/x86_64-linux-gnu/libk5crypto.so.3 $@
	@cp -L /lib/x86_64-linux-gnu/libcom_err.so.2 $@
	@cp -L /lib/x86_64-linux-gnu/libkrb5support.so.0 $@
	@cp -L /lib/x86_64-linux-gnu/libkeyutils.so.1 $@
	@cp -L /lib/x86_64-linux-gnu/libresolv.so.2 $@
	@# required for LevelDB-db_bench_sqlite3
	@cp -L /lib/x86_64-linux-gnu/libsqlite3.so $@
	@# required for VDSO
	@cp -L $(VDSO_LIB) $@


$(VDSO_LIB): | $(VDSO_DIR) $(BINARY_CACHE_DIR)/vdso64.so
	@# TODO: use a custom compiled vdso.so file in the future.
	@cp $(BINARY_CACHE_DIR)/vdso64.so $@
	
$(VDSO_DIR):
	@mkdir -p $@
	
$(BINARY_CACHE_DIR)/vdso64.so:
	@mkdir -p $(BINARY_CACHE_DIR)
	@$(ATOMIC_WGET) $@ "https://raw.githubusercontent.com/asterinas/linux_vdso/2a6d2db/vdso64.so"

$(INITRAMFS)/lib/python3.10:
	@mkdir -p $@
	@cp -r /lib/python3.10/* $@

$(INITRAMFS)/lib/python3:
	@mkdir -p $@
	@cp -r /lib/python3/* $@

$(INITRAMFS)/lib64:
	@mkdir -p $@
	@cp -L /lib64/ld-linux-x86-64.so.2 $@

$(INITRAMFS)/etc:
	@mkdir -p $@
	@cp $(CUR_DIR)/etc/* $@

# Install busybox into /bin and /usr/bin.
$(INITRAMFS)/bin:
	@mkdir -p $@
	@/bin/busybox --install -s $@
	@cp /bin/python $@
	@cp /bin/python3 $@
	@cp /bin/python3.10 $@
	@cp /bin/py3clean $@
	@cp /bin/py3compile $@
	@cp /bin/py3versions $@
	@cp /bin/pydoc3 $@
	@cp /bin/pydoc3.10 $@
	@cp /bin/pygettext3 $@
	@cp /bin/pygettext3.10 $@
	@cp /bin/pip $@
	@cp /bin/pip3 $@
	@cp /bin/pip3.10 $@

$(INITRAMFS)/sbin:
	@mkdir -p $@
	@cp /sbin/ldconfig $@
	@cp /sbin/ldconfig.real $@

$(INITRAMFS)/usr/python:
	@mkdir -p $@
	@cp -r $(CUR_DIR)/python/* $@
	@cp -r /root/asterinas/test_benchmark  $@

$(INITRAMFS)/usr/bin: | $(INITRAMFS)/bin
	@mkdir -p $@
	@cp /usr/bin/busybox $@
	@cp /usr/bin/python $@
	@cp /usr/bin/python3 $@
	@cp /usr/bin/python3.10 $@
	@cp /usr/bin/py3clean $@
	@cp /usr/bin/py3compile $@
	@cp /usr/bin/py3versions $@
	@cp /usr/bin/pydoc3 $@
	@cp /usr/bin/pydoc3.10 $@
	@cp /usr/bin/pygettext3 $@
	@cp /usr/bin/pygettext3.10 $@
	@cp /bin/pip $@
	@cp /bin/pip3 $@
	@cp /bin/pip3.10 $@

$(INITRAMFS)/usr/lib/python3:
	@mkdir -p $@
	@cp -r /usr/lib/python3/* $@

$(INITRAMFS)/usr/lib/python3.10:
	@mkdir -p $@
	@cp -r /usr/lib/python3.10/* $@

$(INITRAMFS)/usr/local:
	@mkdir -p $@
	@cp -r /usr/local/nginx $@
	@cp -r /usr/local/redis $@

$(INITRAMFS)/usr/local/bin:
	@mkdir -p $@
	@cp -r /usr/local/bin/* $@

$(INITRAMFS)/usr/local/lib/python3.10/dist-packages:
	@mkdir -p $@
	@cp -r /usr/local/lib/python3.10/dist-packages/* $@

.PHONY: $(INITRAMFS)/test
$(INITRAMFS)/test:
	@make --no-print-directory -C apps

$(INITRAMFS)/benchmark: | $(INITRAMFS)/benchmark/bin
	@cp -rf $(CUR_DIR)/benchmark/* $@

$(INITRAMFS)/benchmark/bin:
	@mkdir -p $@
	@cp /usr/local/benchmark/sysbench/bin/sysbench $@
	@cp /usr/local/benchmark/iperf/bin/iperf3 $@
	@cp /usr/local/benchmark/membench/membench $@
	@cp /usr/local/benchmark/iozone $@
	@cp -r /usr/local/benchmark/unixbench $@
	@cp -r /usr/local/benchmark/lmbench $@
	@cp /usr/local/benchmark/fio/bin/fio $@
	@cp /usr/local/benchmark/sqlite-speedtest1 $@
	@cp /usr/local/leveldb/benchmark/db_bench $@
	@mv $@/db_bench $@/db_bench_leveldb
	@cp /usr/local/leveldb/benchmark/db_bench_sqlite3 $@
	@cp /usr/local/benchmark/hackbench $@
	@cp /usr/local/benchmark/schbench $@

# Make necessary directories.
$(INITRAMFS_EMPTY_DIRS):
	@mkdir -p $@

.PHONY: $(SYSCALL_TEST_DIR)
$(SYSCALL_TEST_DIR):
	@make --no-print-directory -C syscall_test

.PHONY: $(INITRAMFS_IMAGE)
$(INITRAMFS_IMAGE): $(INITRAMFS_FILELIST)
	@if ! cmp -s $(INITRAMFS_FILELIST) $(INITRAMFS_FILELIST).previous ; then \
		echo "Generating the initramfs image..."; \
		cp -f $(INITRAMFS_FILELIST) $(INITRAMFS_FILELIST).previous; \
		( \
			cd $(INITRAMFS); \
			# `$(INITRAMFS_FILELIST)` contains files' last modification \
			# time in the first column, and files' relative path to \
			# `$(INITRAMFS)` in the second column. This prunes the first \
			# column and passes the second column to `cpio`. \
			cut -d " " -f 2- $(INITRAMFS_FILELIST) | \
				cpio -o -H newc | gzip \
		) > $@; \
	fi

.PHONY: $(INITRAMFS_FILELIST)
# If the BUILD_SYSCALL_TEST variable is set, we should depend on the
# sub make output to do incremental building.
ifeq ($(BUILD_SYSCALL_TEST), 1)
$(INITRAMFS_FILELIST): | $(INITRAMFS_ALL_DIRS) $(SYSCALL_TEST_DIR)
else
$(INITRAMFS_FILELIST): | $(INITRAMFS_ALL_DIRS)
endif
	@(cd $(INITRAMFS); find . -printf "%T@ %p\n") > $(INITRAMFS_FILELIST)

$(EXT2_IMAGE):
	@dd if=/dev/zero of=$(EXT2_IMAGE) bs=2G count=1
	@mke2fs $(EXT2_IMAGE)

$(EXFAT_IMAGE):
	@fallocate -l 64M $(EXFAT_IMAGE)
	@mkfs.exfat $(EXFAT_IMAGE)

.PHONY: build
build: $(INITRAMFS_IMAGE) $(EXT2_IMAGE) $(EXFAT_IMAGE)

.PHONY: format
format:
	@make --no-print-directory -C apps format

.PHONY: check
check:
	@make --no-print-directory -C apps check

.PHONY: clean
clean:
	@rm -rf $(BUILD_DIR)