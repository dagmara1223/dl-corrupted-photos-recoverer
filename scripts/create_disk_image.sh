#!/bin/bash

dd if=/dev/zero of=disk.img bs=1M count=64
mkfs.vfat disk.img

mkdir mnt
sudo mount -o loop disk.img mnt

