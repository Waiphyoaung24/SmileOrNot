"""Remap a Roboflow can dataset's class ids to the 4 canspector classes.

Edit CLASS_MAP after inspecting data/can/raw/data.yaml. Keys are source class
names (as they appear in data.yaml), values are target class ids:

    0 = intact_labeled
    1 = intact_unlabeled
    2 = damaged_labeled
    3 = damaged_unlabeled

Images whose labels contain only unmapped classes are skipped.

Inputs:
    data/can/raw/{train,valid,test}/{images,labels}/...
    data/can/raw/data.yaml
Outputs:
    data/can/yolo/images/<basename>.jpg
    data/can/yolo/labels/<basename>.txt
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

RAW = Path(__file__).parent.parent / "data" / "can" / "raw"
DST = Path(__file__).parent.parent / "data" / "can" / "yolo"

# TODO: fill in after picking the Roboflow dataset and reading raw/data.yaml.
CLASS_MAP: dict[str, int] = {
    # "can_intact_with_label": 0,
    # "can_intact_no_label": 1,
    # "can_damaged_with_label": 2,
    # "can_damaged_no_label": 3,
}


def load_source_names() -> list[str]:
    cfg = yaml.safe_load((RAW / "data.yaml").read_text())
    return cfg["names"]


def main() -> None:
    if not CLASS_MAP:
        raise SystemExit("CLASS_MAP is empty; edit scripts/can_02_remap.py first.")
    src_names = load_source_names()
    src_id_to_target: dict[int, int] = {}
    for i, name in enumerate(src_names):
        if name in CLASS_MAP:
            src_id_to_target[i] = CLASS_MAP[name]

    (DST / "images").mkdir(parents=True, exist_ok=True)
    (DST / "labels").mkdir(parents=True, exist_ok=True)

    kept = 0
    dropped = 0
    for split in ("train", "valid", "test"):
        labels_dir = RAW / split / "labels"
        images_dir = RAW / split / "images"
        if not labels_dir.exists():
            continue
        for lp in labels_dir.glob("*.txt"):
            new_lines: list[str] = []
            for line in lp.read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                src_cls = int(parts[0])
                if src_cls not in src_id_to_target:
                    continue
                parts[0] = str(src_id_to_target[src_cls])
                new_lines.append(" ".join(parts))
            if not new_lines:
                dropped += 1
                continue
            stem = lp.stem
            ip = images_dir / f"{stem}.jpg"
            if not ip.exists():
                alt = next(images_dir.glob(f"{stem}.*"), None)
                if alt is None:
                    dropped += 1
                    continue
                ip = alt
            (DST / "labels" / f"{stem}.txt").write_text("\n".join(new_lines) + "\n")
            shutil.copy2(ip, DST / "images" / f"{stem}.jpg")
            kept += 1

    print(f"Kept {kept} images, dropped {dropped}")


if __name__ == "__main__":
    main()
