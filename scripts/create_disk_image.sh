#!/bin/bash
set -e

IMG="disk.img"
MNT="mnt"
SRC_DIR="../imgs"


dd if=/dev/zero of=$IMG bs=1M count=64
mkfs.vfat $IMG
mkdir -p $MNT
sudo mount -o loop $IMG $MNT

find "$SRC_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" \) -exec cp {} $MNT \;

sync
sudo umount $MNT

echo "Disk image prepared"