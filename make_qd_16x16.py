import argparse
import json
import random
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw
from tqdm import tqdm

QD_NDJSON_URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/{name}.ndjson"
RECORD_BYTES = 512  # 16x16 * uint16 (little-endian)


def draw_strokes_to_image(drawing, canvas=256, padding=8, stroke_width=6):
    img = Image.new("L", (canvas, canvas), 255)  # white background
    d = ImageDraw.Draw(img)

    xs, ys = [], []
    for stroke in drawing:
        xs.extend(stroke[0])
        ys.extend(stroke[1])
    if not xs or not ys:
        return img

    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    w = max(maxx - minx, 1)
    h = max(maxy - miny, 1)
    scale = (canvas - 2 * padding) / max(w, h)

    def tx(x):
        return int((x - minx) * scale + padding)

    def ty(y):
        return int((y - miny) * scale + padding)

    for stroke in drawing:
        x, y = stroke[0], stroke[1]
        pts = list(zip([tx(v) for v in x], [ty(v) for v in y]))
        if len(pts) == 1:
            px, py = pts[0]
            d.ellipse((px - stroke_width, py - stroke_width, px + stroke_width, py + stroke_width), fill=0)
        else:
            d.line(pts, fill=0, width=stroke_width, joint="curve")
    return img


def to_16x16_u16(img):
    """
    img: PIL L image
    returns uint16 array (16,16) values in [0..65535]
    """
    small = img.resize((16, 16), resample=Image.Resampling.NEAREST)
    arr8 = np.array(small, dtype=np.uint8)  # 0..255
    return arr8.astype(np.uint16) * 257  # 255 -> 65535


def pack_u16_16x16(u16_16):
    """
    u16_16: (16,16) uint16
    returns bytes length 512 in little-endian.
    """
    flat = u16_16.reshape(-1)
    if flat.size != 256:
        raise ValueError("Expected 256 pixels")
    return flat.astype("<u2", copy=False).tobytes()


def stream_category(name):
    url = QD_NDJSON_URL.format(name=name.replace(" ", "%20"))
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", required=True)
    ap.add_argument("--per_category", type=int, default=5000)
    ap.add_argument("--out_dir", type=str, default="qd_16x16_u16_out")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--canvas", type=int, default=32)
    ap.add_argument("--stroke_width", type=int, default=2)
    ap.add_argument("--padding", type=int, default=2)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    img_path = out / "images_u16.bin"  # 512 bytes per record
    prompts_path = out / "prompts.txt"
    meta_path = out / "meta.json"
    total = len(args.categories) * args.per_category

    with open(img_path, "wb") as fout, open(prompts_path, "w", encoding="utf-8") as ftxt:
        pbar = tqdm(total=total, desc="Building 16x16 u16 dataset")
        written = 0

        for cat in args.categories:
            count = 0
            for ex in stream_category(cat):
                drawing = ex.get("drawing")
                if not drawing:
                    continue

                img = draw_strokes_to_image(
                    drawing,
                    canvas=args.canvas,
                    padding=args.padding,
                    stroke_width=args.stroke_width,
                )

                u16 = to_16x16_u16(img)  # (16,16) uint16
                packed = pack_u16_16x16(u16)  # 512 bytes

                fout.write(packed)
                ftxt.write(cat + "\n")

                count += 1
                written += 1
                pbar.update(1)

                if count >= args.per_category:
                    break

            if count < args.per_category:
                print(f"[WARN] category '{cat}' produced only {count} samples")

        pbar.close()

    meta = {
        "format": "quickdraw_16x16_grayscale_u16_le",
        "record_bytes": RECORD_BYTES,
        "dtype": "uint16",
        "endianness": "little",
        "image_size": [16, 16],
        "pixel_range": [0, 65535],
        "canvas": args.canvas,
        "stroke_width": args.stroke_width,
        "padding": args.padding,
        "num_records": written,
        "categories": args.categories,
        "per_category_requested": args.per_category,
        "prompts_file": prompts_path.name,
        "images_file": img_path.name,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Done. Wrote {written} records to {out}")


if __name__ == "__main__":
    main()
