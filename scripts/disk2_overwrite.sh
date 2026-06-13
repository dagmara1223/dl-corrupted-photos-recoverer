#!/bin/bash
set -e

IMG="disk2.img"
SRC_DIR="../imgs"
MNT="mnt2a"
COUNT=20

dd if=/dev/zero of=$IMG bs=1M count=64
mkfs.vfat $IMG

mkdir -p $MNT
sudo mount -o loop $IMG $MNT

TOTAL=$(find "$SRC_DIR" -type f -iname "image_*.jpg" | wc -l)

if (( TOTAL < COUNT )); then
    echo "Not enough images: $TOTAL"
    exit 1
fi

START=$(( (RANDOM % (TOTAL - COUNT)) + 1 ))

i=0
for ((idx=START; idx<START+COUNT; idx++)); do
    src="$SRC_DIR/image_${idx}.jpg"
    if [[ -f "$src" ]]; then
        printf -v name "img_%03d.jpg" "$i"
        cp "$src" "$MNT/$name"
        i=$((i + 1))
    fi
done

sync
sudo umount $MNT


FILE_SIZE=$(stat -c%s "$IMG")
BLOCKS=$((FILE_SIZE / 4096))

for ((i=0; i<20; i++)); do
    OFFSET=$(( (i * BLOCKS) / 20 ))
    dd if=/dev/urandom of=$IMG bs=32 count=1 seek=$OFFSET conv=notrunc status=none
done

echo "Done: controlled mild corruption"