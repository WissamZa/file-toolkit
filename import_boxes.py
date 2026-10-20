#!/usr/bin/env python3
"""Seed the cropper with ready-made example boxes (starter_boxes.json).

    python import_boxes.py                       # uses starter_boxes.json
    python import_boxes.py my_boxes.json

Looks for each file name inside the toolkit's working folder (the same folder
the main app uses), saves the box as a taught example, and trains the model.
The boxes are rough: fix any crop that looks wrong in the teaching GUI
(option 13) and the model improves with every correction.
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageOps

import file_toolkit as tk
import image_tools as it


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else
                Path(__file__).with_name("starter_boxes.json"))
    data = json.loads(path.read_text(encoding="utf-8"))
    ew, eh = data.get("image_size", [0, 0])
    kind = data.get("type", "invoice")
    index = {}
    for f in it.collect_image_files(tk.TARGET_DIR, recursive=True):
        index.setdefault(f.name, f)

    conn = it._open_db()
    try:
        added = missing = 0
        for name, box in data["boxes"].items():
            f = index.get(name)
            if f is None:
                missing += 1
                continue
            with Image.open(f) as img:
                img = ImageOps.exif_transpose(img)
                w, h = img.size
                sx, sy = (w / ew, h / eh) if ew and eh else (1, 1)
                scaled = (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)
                it.save_crop_example(conn, f, img, kind, scaled)
                added += 1
        print(f"Imported {added} example(s) from {path.name}; "
              f"{missing} file(s) not found in {tk.TARGET_DIR}.")
        if added:
            model = it.get_model(conn)
            print("Model trained." if model else
                  "Need at least 3 examples to train a model.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
