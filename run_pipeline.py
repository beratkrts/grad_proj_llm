"""
Unified pipeline runner.

Usage:
    python run_pipeline.py --categories cat dog airplane --per_category 1000

Generates both u16 and 1-bit datasets, cleans & re-exports PNGs,
then shows a combined matplotlib grid (u16 top row, 1-bit bottom row).
"""
import argparse
import json
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from make_qd_16x16 import (
    draw_strokes_to_image,
    pack_u16_16x16,
    stream_category,
    to_16x16_u16,
)
from make_qd_1bit import pack_1bit_16x16, to_16x16_1bit

# ── defaults ──────────────────────────────────────────────────────────────────
U16_OUT = "qd_u16_out"
BIT_OUT = "qd_1bit_out"
U16_EXPORT = "exported_png_u16"
BIT_EXPORT = "exported_png_1bit"

U16_RECORD = 512
BIT_RECORD = 32


# ── generation ────────────────────────────────────────────────────────────────

def build_dataset(categories, per_category, canvas, stroke_width, padding,
                  out_dir, img_filename, pack_fn, record_bytes, desc, meta_extra):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    img_path = out / img_filename
    prompts_path = out / "prompts.txt"
    meta_path = out / "meta.json"

    written = 0
    with open(img_path, "wb") as fout, open(prompts_path, "w", encoding="utf-8") as ftxt:
        pbar = tqdm(total=len(categories) * per_category, desc=desc)
        for cat in categories:
            count = 0
            for ex in stream_category(cat):
                drawing = ex.get("drawing")
                if not drawing:
                    continue
                img = draw_strokes_to_image(drawing, canvas=canvas,
                                            padding=padding, stroke_width=stroke_width)
                fout.write(pack_fn(img))
                ftxt.write(cat + "\n")
                count += 1
                written += 1
                pbar.update(1)
                if count >= per_category:
                    break
            if count < per_category:
                print(f"[WARN] '{cat}' had only {count} samples")
        pbar.close()

    meta = {
        "image_size": [16, 16],
        "record_bytes": record_bytes,
        "canvas": canvas,
        "stroke_width": stroke_width,
        "padding": padding,
        "num_records": written,
        "categories": categories,
        "per_category_requested": per_category,
        **meta_extra,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  → {written} records saved to {out_dir}/")
    return written


# ── export ────────────────────────────────────────────────────────────────────

def export_u16_pngs(data_dir, export_dir, scale=16):
    data_dir, export_dir = Path(data_dir), Path(export_dir)
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)

    bin_path = data_dir / "images_u16.bin"
    prompts_path = data_dir / "prompts.txt"
    prompts = prompts_path.read_text(encoding="utf-8").splitlines()
    total = bin_path.stat().st_size // U16_RECORD

    with bin_path.open("rb") as f:
        for idx in tqdm(range(total), desc="Exporting u16 PNGs"):
            rec = f.read(U16_RECORD)
            pixels = np.frombuffer(rec, dtype="<u2").reshape(16, 16)
            img = Image.fromarray(pixels, mode="I;16")
            if scale > 1:
                img = img.resize((16 * scale, 16 * scale), Image.Resampling.NEAREST)
            safe = prompts[idx].replace(" ", "_").replace("/", "_") if idx < len(prompts) else ""
            img.save(export_dir / f"{idx:08d}_{safe}.png")


def export_1bit_pngs(data_dir, export_dir, scale=16):
    data_dir, export_dir = Path(data_dir), Path(export_dir)
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)

    bin_path = data_dir / "images_1bit.bin"
    prompts_path = data_dir / "prompts.txt"
    prompts = prompts_path.read_text(encoding="utf-8").splitlines()
    total = bin_path.stat().st_size // BIT_RECORD

    with bin_path.open("rb") as f:
        for idx in tqdm(range(total), desc="Exporting 1-bit PNGs"):
            rec = f.read(BIT_RECORD)
            bits = np.unpackbits(np.frombuffer(rec, dtype=np.uint8), bitorder="big").reshape(16, 16)
            img_arr = ((1 - bits) * 255).astype(np.uint8)
            img = Image.fromarray(img_arr, mode="L")
            if scale > 1:
                img = img.resize((16 * scale, 16 * scale), Image.Resampling.NEAREST)
            safe = prompts[idx].replace(" ", "_").replace("/", "_") if idx < len(prompts) else ""
            img.save(export_dir / f"{idx:08d}_{safe}.png")


# ── visualise ─────────────────────────────────────────────────────────────────

def load_random_u16(data_dir, n, rng):
    data_dir = Path(data_dir)
    total = (data_dir / "images_u16.bin").stat().st_size // U16_RECORD
    prompts = (data_dir / "prompts.txt").read_text(encoding="utf-8").splitlines()
    idxs = rng.choice(total, size=min(n, total), replace=False)
    images, titles = [], []
    with (data_dir / "images_u16.bin").open("rb") as f:
        for idx in idxs:
            f.seek(int(idx) * U16_RECORD)
            pixels = np.frombuffer(f.read(U16_RECORD), dtype="<u2").reshape(16, 16)
            images.append(pixels)
            titles.append(prompts[idx] if idx < len(prompts) else "")
    return images, titles


def load_random_1bit(data_dir, n, rng):
    data_dir = Path(data_dir)
    total = (data_dir / "images_1bit.bin").stat().st_size // BIT_RECORD
    prompts = (data_dir / "prompts.txt").read_text(encoding="utf-8").splitlines()
    idxs = rng.choice(total, size=min(n, total), replace=False)
    images, titles = [], []
    with (data_dir / "images_1bit.bin").open("rb") as f:
        for idx in idxs:
            f.seek(int(idx) * BIT_RECORD)
            bits = np.unpackbits(np.frombuffer(f.read(BIT_RECORD), dtype=np.uint8), bitorder="big").reshape(16, 16)
            images.append(bits)
            titles.append(prompts[idx] if idx < len(prompts) else "")
    return images, titles


def show_combined(u16_images, u16_titles, bit_images, bit_titles, cols=8, scale=2.0):
    n = min(len(u16_images), len(bit_images), cols * 4)  # up to 4 rows per format
    cols = min(cols, n)
    rows_per = (n + cols - 1) // cols
    total_rows = rows_per * 2 + 1  # +1 for separator row

    fig = plt.figure(figsize=(cols * scale, total_rows * scale))
    fig.patch.set_facecolor("white")

    # u16 top half
    for i in range(n):
        ax = fig.add_subplot(total_rows, cols, i + 1)
        ax.imshow(u16_images[i], cmap="gray", vmin=0, vmax=65535)
        ax.axis("off")
        ax.set_title(u16_titles[i], fontsize=7)
    # row label
    fig.text(0.01, 1 - (rows_per / total_rows) / 2, "16-bit", va="center",
             fontsize=10, fontweight="bold", rotation=90)

    # separator
    sep_row = rows_per + 1
    ax_sep = fig.add_subplot(total_rows, 1, sep_row)
    ax_sep.axhline(0, color="#aaaaaa", linewidth=1)
    ax_sep.axis("off")

    # 1-bit bottom half
    offset = rows_per * cols + cols  # skip u16 rows + separator row
    for i in range(n):
        ax = fig.add_subplot(total_rows, cols, offset + i + 1)
        ax.imshow(bit_images[i], cmap="binary", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(bit_titles[i], fontsize=7)
    fig.text(0.01, (rows_per / total_rows) / 2, "1-bit", va="center",
             fontsize=10, fontweight="bold", rotation=90)

    plt.tight_layout(pad=0.3)
    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Build both QuickDraw pipelines end-to-end.")
    ap.add_argument("--categories", nargs="+", required=True)
    ap.add_argument("--per_category", type=int, default=5000)
    ap.add_argument("--canvas",       type=int, default=32)
    ap.add_argument("--stroke_width", type=int, default=2)
    ap.add_argument("--padding",      type=int, default=2)
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--export_scale", type=int, default=16)
    ap.add_argument("--n_show",       type=int, default=32, help="Samples to show per format")
    ap.add_argument("--cols",         type=int, default=8)
    ap.add_argument("--save_grid",    type=str, default=None, help="Save combined grid to file")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    kw = dict(canvas=args.canvas, stroke_width=args.stroke_width, padding=args.padding)

    # ── 1. generate ──────────────────────────────────────────────────────────
    print("\n=== Generating 16-bit dataset ===")
    build_dataset(
        categories=args.categories,
        per_category=args.per_category,
        out_dir=U16_OUT,
        img_filename="images_u16.bin",
        pack_fn=lambda img: pack_u16_16x16(to_16x16_u16(img)),
        record_bytes=U16_RECORD,
        desc="16-bit u16",
        meta_extra={"format": "quickdraw_16x16_grayscale_u16_le", "dtype": "uint16",
                    "pixel_range": [0, 65535]},
        **kw,
    )

    print("\n=== Generating 1-bit dataset ===")
    build_dataset(
        categories=args.categories,
        per_category=args.per_category,
        out_dir=BIT_OUT,
        img_filename="images_1bit.bin",
        pack_fn=lambda img: pack_1bit_16x16(to_16x16_1bit(img)),
        record_bytes=BIT_RECORD,
        desc="1-bit",
        meta_extra={"format": "quickdraw_16x16_1bit", "dtype": "uint1_packed",
                    "bitorder": "big"},
        **kw,
    )

    # ── 2. export PNGs ───────────────────────────────────────────────────────
    print(f"\n=== Exporting PNGs (scale={args.export_scale}) ===")
    export_u16_pngs(U16_OUT, U16_EXPORT, scale=args.export_scale)
    export_1bit_pngs(BIT_OUT, BIT_EXPORT, scale=args.export_scale)

    # ── 3. show combined grid ─────────────────────────────────────────────────
    print("\n=== Preparing visualisation ===")
    rng = np.random.default_rng(args.seed)
    u16_imgs, u16_titles = load_random_u16(U16_OUT, args.n_show, rng)
    rng = np.random.default_rng(args.seed)
    bit_imgs, bit_titles = load_random_1bit(BIT_OUT, args.n_show, rng)

    fig = show_combined(u16_imgs, u16_titles, bit_imgs, bit_titles,
                        cols=args.cols)

    if args.save_grid:
        fig.savefig(args.save_grid, dpi=200, bbox_inches="tight")
        print(f"Grid saved to {args.save_grid}")

    plt.show()


if __name__ == "__main__":
    main()
