#!/bin/bash
set -e

IMG="disk2.img"
MNT="mnt2a"
SRC_DIR="../imgs"
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

rm $MNT/*

sync
sudo umount $MNT

for i in {1..10}; do
    OFFSET=$((RANDOM % 60))
    dd if=/dev/urandom of=$IMG bs=1M count=1 seek=$OFFSET conv=notrunc
done

echo "Script 2 done: full deletion + random overwrites"