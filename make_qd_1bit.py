import argparse
import json
import random
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw
from tqdm import tqdm

QD_NDJSON_URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/{name}.ndjson"
RECORD_BYTES = 32  # 16x16 bits = 256 bits = 32 bytes


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


def to_16x16_1bit(img, threshold=128):
    """
    img: PIL L image (white bg=255, black strokes=0)
    returns bool array (16,16): True=stroke, False=background

    Uses min-pooling: each 16x16 output pixel covers a 16x16 input block.
    If any pixel in the block is a stroke (dark), the output pixel is stroke.
    This preserves thin strokes that BILINEAR would wash out.
    """
    arr = np.array(img, dtype=np.uint8)    # (canvas, canvas)
    h, w = arr.shape                       # both must be divisible by 16
    bh, bw = h // 16, w // 16             # block size (e.g. 2x2 for canvas=32)
    blocks = arr.reshape(16, bh, 16, bw)  # (out_h, block_h, out_w, block_w)
    min_pool = blocks.min(axis=(1, 3))    # (16, 16) — darkest pixel per block
    return min_pool < threshold            # True = stroke


def pack_1bit_16x16(bits_16x16):
    """
    bits_16x16: (16,16) bool array, True=stroke
    returns bytes of length 32 (256 bits packed, MSB first)
    """
    flat = bits_16x16.reshape(-1).astype(np.uint8)
    if flat.size != 256:
        raise ValueError("Expected 256 pixels")
    return np.packbits(flat, bitorder="big").tobytes()


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
    ap.add_argument("--out_dir", type=str, default="qd_16x16_1bit_out")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--canvas", type=int, default=32)
    ap.add_argument("--stroke_width", type=int, default=2)
    ap.add_argument("--padding", type=int, default=2)
    ap.add_argument("--threshold", type=int, default=128)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    img_path = out / "images_1bit.bin"  # 32 bytes per record
    prompts_path = out / "prompts.txt"
    meta_path = out / "meta.json"
    total = len(args.categories) * args.per_category

    with open(img_path, "wb") as fout, open(prompts_path, "w", encoding="utf-8") as ftxt:
        pbar = tqdm(total=total, desc="Building 16x16 1-bit dataset")
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

                bits = to_16x16_1bit(img, threshold=args.threshold)  # (16,16) bool
                packed = pack_1bit_16x16(bits)  # 32 bytes

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
        "format": "quickdraw_16x16_1bit",
        "record_bytes": RECORD_BYTES,
        "dtype": "uint1_packed",
        "bitorder": "big",
        "image_size": [16, 16],
        "pixel_meaning": {"0": "background (white)", "1": "stroke (black)"},
        "threshold": args.threshold,
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
