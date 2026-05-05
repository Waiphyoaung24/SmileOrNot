"""Download a Roboflow can-damage dataset in YOLOv8 format.

Edit the three constants below after picking a dataset on Roboflow Universe.
The script is idempotent: it checks for an existing download before re-fetching.

Outputs:
    data/can/raw/{train,valid,test}/{images,labels}/...
    data/can/raw/data.yaml
"""

from __future__ import annotations

import os
from pathlib import Path

from roboflow import Roboflow

# TODO: replace with the chosen Roboflow workspace/project/version.
ROBOFLOW_WORKSPACE = "REPLACE_ME_WORKSPACE"
ROBOFLOW_PROJECT = "REPLACE_ME_PROJECT"
ROBOFLOW_VERSION = 1

DEST = Path(__file__).parent.parent / "data" / "can" / "raw"


def main() -> None:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Set ROBOFLOW_API_KEY in your environment.")
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if (DEST / "data.yaml").exists():
        print(f"Existing dataset at {DEST}; skipping download.")
        return
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    project.version(ROBOFLOW_VERSION).download("yolov8", location=str(DEST))
    print(f"Downloaded to {DEST}")


if __name__ == "__main__":
    main()
