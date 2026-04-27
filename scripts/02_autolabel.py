"""Auto-label CelebA images with bounding boxes from a pretrained face detector.

For each image:
  - Run yolo11n-face → take the highest-confidence face box
  - If no face is found, skip the image
  - Map CelebA's `Smiling` attribute (1 → 0=smiling, -1 → 1=neutral) onto the box
  - Write a YOLO-format label file: `<class> <x_center> <y_center> <w> <h>` (all normalized)

Outputs (under data/yolo/):
    images/<basename>.jpg
    labels/<basename>.txt

Subset: balanced 5000 smiling + 5000 neutral.
"""

from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

SRC_DIR = Path(__file__).parent.parent / "data" / "celeba"
IMG_DIR = SRC_DIR / "img_align_celeba"
ATTR_FILE_CANDIDATES = ["list_attr_celeba.csv", "list_attr_celeba.txt"]
DST_DIR = Path(__file__).parent.parent / "data" / "yolo"
N_PER_CLASS = 5000
SEED = 42


def find_attr_file() -> Path:
    for name in ATTR_FILE_CANDIDATES:
        p = SRC_DIR / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No attribute file found under {SRC_DIR}; looked for {ATTR_FILE_CANDIDATES}")


def parse_attrs(path: Path) -> dict[str, int]:
    """Return {filename: smiling_label} where smiling_label is +1 or -1."""
    attrs: dict[str, int] = {}
    text = path.read_text()
    lines = text.strip().splitlines()
    if path.suffix == ".csv":
        reader = csv.DictReader(lines)
        for row in reader:
            attrs[row["image_id"]] = int(row["Smiling"])
    else:
        # The classic CelebA format: line 1 = count, line 2 = headers (space-separated),
        # subsequent lines = "<filename> <attr1> <attr2> ..." with values in {-1, 1}.
        headers = lines[1].split()
        smile_idx = headers.index("Smiling")
        for line in lines[2:]:
            parts = line.split()
            attrs[parts[0]] = int(parts[1 + smile_idx])
    return attrs


def select_subset(attrs: dict[str, int], n_per_class: int) -> list[tuple[str, int]]:
    """Return [(filename, yolo_class)] balanced across smiling/neutral."""
    smiling = [f for f, s in attrs.items() if s == 1]
    neutral = [f for f, s in attrs.items() if s == -1]
    rng = random.Random(SEED)
    rng.shuffle(smiling)
    rng.shuffle(neutral)
    chosen = [(f, 0) for f in smiling[:n_per_class]] + [(f, 1) for f in neutral[:n_per_class]]
    rng.shuffle(chosen)
    return chosen


def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    (DST_DIR / "images").mkdir(exist_ok=True)
    (DST_DIR / "labels").mkdir(exist_ok=True)

    attrs = parse_attrs(find_attr_file())
    subset = select_subset(attrs, N_PER_CLASS)
    print(f"Selected {len(subset)} images ({N_PER_CLASS} per class)")

    detector = YOLO("yolo11n-face.pt")  # Ultralytics pulls this on first run
    skipped = 0
    written = 0

    for filename, yolo_class in subset:
        img_path = IMG_DIR / filename
        if not img_path.exists():
            skipped += 1
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            skipped += 1
            continue

        results = detector.predict(img, conf=0.35, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            skipped += 1
            continue

        # Highest-confidence box only
        confs = boxes.conf.cpu().numpy()
        best = int(confs.argmax())
        x1, y1, x2, y2 = boxes.xyxyn[best].cpu().numpy().tolist()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        basename = Path(filename).stem
        shutil.copyfile(img_path, DST_DIR / "images" / f"{basename}.jpg")
        (DST_DIR / "labels" / f"{basename}.txt").write_text(
            f"{yolo_class} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
        )
        written += 1
        if written % 500 == 0:
            print(f"  {written} written, {skipped} skipped")

    print(f"Done. Wrote {written} labeled images. Skipped {skipped} (no-face or unreadable).")


if __name__ == "__main__":
    main()
