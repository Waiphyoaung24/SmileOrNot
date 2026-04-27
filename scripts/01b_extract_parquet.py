"""Extract HF parquet files (tpremoli/CelebA-attrs format) into legacy CelebA layout.

The HF dataset ships as parquet rows containing {image: {bytes, path}, <attrs>}.
Script 02_autolabel.py expects the legacy layout:
    data/celeba/img_align_celeba/<file>.jpg
    data/celeba/list_attr_celeba.csv  (with image_id, Smiling columns)
"""

from __future__ import annotations

import csv
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).parent.parent / "data" / "celeba"
PARQUET_DIR = ROOT / "data"
IMG_DIR = ROOT / "img_align_celeba"
CSV_PATH = ROOT / "list_attr_celeba.csv"


def main() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    parquets = sorted(PARQUET_DIR.glob("*.parquet"))
    if not parquets:
        raise SystemExit(f"No parquets in {PARQUET_DIR}")

    rows_out: list[dict[str, str | int]] = []
    written = 0
    for pq_path in parquets:
        print(f"Reading {pq_path.name}")
        table = pq.read_table(pq_path)
        cols = [c for c in table.column_names if c != "image" and c != "prompt_string"]
        for batch in table.to_batches(max_chunksize=1000):
            d = batch.to_pydict()
            images = d["image"]
            for i, img in enumerate(images):
                fname = img["path"] if img.get("path") else f"{written:06d}.jpg"
                if not fname.lower().endswith(".jpg"):
                    fname = f"{Path(fname).stem}.jpg"
                (IMG_DIR / fname).write_bytes(img["bytes"])
                row: dict[str, str | int] = {"image_id": fname}
                for c in cols:
                    row[c] = d[c][i]
                rows_out.append(row)
                written += 1
                if written % 5000 == 0:
                    print(f"  wrote {written}")

    print(f"Total images written: {written}")
    print(f"Writing {CSV_PATH}")
    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print("Done.")


if __name__ == "__main__":
    main()
