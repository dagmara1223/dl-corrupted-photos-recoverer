#!/bin/bash
set -e

trap 'echo "ERROR on line $LINENO: $BASH_COMMAND" >&2' ERR

IMG="disk1.img"
MNT="mnt9"
SRC_DIR="../imgs"
COUNT=10

dd if=/dev/zero of="$IMG" bs=1M count=256
mkfs.vfat "$IMG"

mkdir -p "$MNT"
sudo mount -o loop "$IMG" "$MNT"

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

rm $MNT/*

sync

sudo umount "$MNT"

echo "Copied $i images starting from $START"