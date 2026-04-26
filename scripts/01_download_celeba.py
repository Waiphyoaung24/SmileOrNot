"""Download CelebA images and the Smiling attribute table from the Hugging Face mirror.

Outputs:
    data/celeba/img_align_celeba/*.jpg
    data/celeba/list_attr_celeba.csv
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "tpremoli/CelebA-attrs"
DEST = Path(__file__).parent.parent / "data" / "celeba"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID} → {DEST}")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(DEST),
    )
    print("Done. Listing top-level entries:")
    for p in sorted(DEST.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
