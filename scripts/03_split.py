"""Stratified 80/10/10 split of the auto-labeled YOLO dataset.

Reads:  data/yolo/images/*.jpg + data/yolo/labels/*.txt
Writes: data/yolo/images/{train,val,test}/ and data/yolo/labels/{train,val,test}/
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent / "data" / "yolo"
TRAIN_FRAC, VAL_FRAC = 0.8, 0.1
SEED = 42


def read_class(label_path: Path) -> int:
    return int(label_path.read_text().strip().split()[0])


def main() -> None:
    label_files = sorted((ROOT / "labels").glob("*.txt"))
    by_class: dict[int, list[Path]] = {0: [], 1: []}
    for lp in label_files:
        if lp.parent.name != "labels":
            continue  # skip already-split children if rerun
        c = read_class(lp)
        by_class[c].append(lp)

    rng = random.Random(SEED)
    splits: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    for c, paths in by_class.items():
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
