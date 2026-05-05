"""Prepare a 1-class can-detection dataset from local Roboflow downloads.

Reads two YOLOv8-format datasets the user already downloaded, remaps every
"can-like" class to a single target class id 0, drops irrelevant classes
(BIODEGRADABLE, CARDBOARD, color labels, the rare 'distorted' class, etc.),
merges them, and writes a stratified-by-source 80/10/10 split.

Run:
    python scripts/can_prep_local.py

Output:
    data/can/yolo/{images,labels}/{train,val,test}/
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import yaml

# Source datasets: each maps a directory to the set of source class names that
# should be remapped to target class id 0 ("can"). Names not in this set are
# dropped along with their boxes.
SOURCES: list[tuple[Path, set[str]]] = [
    (
        Path("/Users/waiphyoaung/Downloads/detect can.v1i.yolo26"),
        {"CAN", "Can", "Cans", "METAL", "Metal", "can"},
    ),
    (
        Path("/Users/waiphyoaung/Downloads/metal can.v1i.yolo26"),
        {"metal"},
    ),
]

DST = Path(__file__).parent.parent / "data" / "can" / "yolo"
TRAIN_FRAC, VAL_FRAC = 0.8, 0.1
SEED = 42


def load_class_names(yaml_path: Path) -> list[str]:
    return yaml.safe_load(yaml_path.read_text())["names"]


def collect(source: Path, keep: set[str]) -> list[tuple[Path, list[str]]]:
    """Return (image_path, remapped_label_lines) for every kept image."""
    names = load_class_names(source / "data.yaml")
    keep_ids = {i for i, n in enumerate(names) if n in keep}
    out: list[tuple[Path, list[str]]] = []
    for split in ("train", "valid", "test"):
        labels_dir = source / split / "labels"
        images_dir = source / split / "images"
        if not labels_dir.exists():
            continue
        for lp in labels_dir.glob("*.txt"):
            new_lines: list[str] = []
            for line in lp.read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                if int(parts[0]) not in keep_ids:
                    continue
                parts[0] = "0"
                new_lines.append(" ".join(parts))
            if not new_lines:
                continue
            ip = next(images_dir.glob(f"{lp.stem}.*"), None)
            if ip is None:
                continue
            out.append((ip, new_lines))
    return out


def main() -> None:
    samples: list[tuple[Path, list[str], str]] = []  # (img, label_lines, prefix)
    for i, (source, keep) in enumerate(SOURCES):
        prefix = f"src{i}_"
        items = collect(source, keep)
        print(f"{source.name}: kept {len(items)} images")
        for ip, lines in items:
            samples.append((ip, lines, prefix))

    rng = random.Random(SEED)
    rng.shuffle(samples)
    n = len(samples)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    splits = {
        "train": samples[:n_train],
        "val": samples[n_train : n_train + n_val],
        "test": samples[n_train + n_val :],
    }

    for split, items in splits.items():
        (DST / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST / "labels" / split).mkdir(parents=True, exist_ok=True)
        for ip, lines, prefix in items:
            stem = f"{prefix}{ip.stem}"
            shutil.copy2(ip, DST / "images" / split / f"{stem}{ip.suffix.lower()}")
            (DST / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        print(f"{split}: {len(items)}")

    print(f"\nTotal: {n} images at {DST}")


if __name__ == "__main__":
    main()
