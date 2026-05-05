"""Stratified 80/10/10 split for the canspector dataset (4 classes).

Reads:  data/can/yolo/images/*.jpg + data/can/yolo/labels/*.txt
Writes: data/can/yolo/{images,labels}/{train,val,test}/
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent / "data" / "can" / "yolo"
TRAIN_FRAC, VAL_FRAC = 0.8, 0.1
SEED = 42
NUM_CLASSES = 4


def primary_class(label_path: Path) -> int:
    return int(label_path.read_text().strip().splitlines()[0].split()[0])


def main() -> None:
    label_files = [p for p in (ROOT / "labels").glob("*.txt") if p.parent.name == "labels"]
    by_class: dict[int, list[Path]] = {i: [] for i in range(NUM_CLASSES)}
    for lp in label_files:
        c = primary_class(lp)
        if c in by_class:
            by_class[c].append(lp)

    rng = random.Random(SEED)
    splits: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    for paths in by_class.values():
        rng.shuffle(paths)
        n = len(paths)
        n_train = int(n * TRAIN_FRAC)
        n_val = int(n * VAL_FRAC)
        splits["train"].extend(paths[:n_train])
        splits["val"].extend(paths[n_train : n_train + n_val])
        splits["test"].extend(paths[n_train + n_val :])

    for split, paths in splits.items():
        (ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)
        for lp in paths:
            stem = lp.stem
            ip = ROOT / "images" / f"{stem}.jpg"
            shutil.move(str(ip), str(ROOT / "images" / split / f"{stem}.jpg"))
            shutil.move(str(lp), str(ROOT / "labels" / split / f"{stem}.txt"))
        print(f"{split}: {len(paths)} samples")


if __name__ == "__main__":
    main()
