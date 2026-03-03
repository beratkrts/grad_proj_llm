import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


RECORD_BYTES = 512  # 16x16 * uint16 = 512 bytes


def count_records(bin_path: Path) -> int:
    size = bin_path.stat().st_size
    if size % RECORD_BYTES != 0:
        raise ValueError(f"{bin_path} size ({size}) not multiple of {RECORD_BYTES}.")
    return size // RECORD_BYTES


def read_prompt_line(prompts_path: Path, idx: int) -> str:
    with prompts_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == idx:
                return line.rstrip("\n")
    return ""


def decode_u16_record(rec_bytes: bytes) -> np.ndarray:
    pixels = np.frombuffer(rec_bytes, dtype="<u2")
    if pixels.size != 256:
        raise ValueError("Expected 256 pixels per record")
    return pixels.reshape(16, 16)  # values 0..65535


def read_record_u16(bin_path: Path, idx: int) -> np.ndarray:
    with bin_path.open("rb") as f:
        f.seek(idx * RECORD_BYTES)
        rec = f.read(RECORD_BYTES)
        if len(rec) != RECORD_BYTES:
            raise IndexError(f"Index {idx} out of range.")
    return decode_u16_record(rec)


def show_single(img16_u16: np.ndarray, title: str = ""):
    plt.figure()
    plt.imshow(img16_u16, cmap="gray", vmin=0, vmax=65535)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()


def show_grid(images, titles=None, cols=8, scale=2.0, pad=0.2):
    n = len(images)
    cols = max(1, cols)
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(cols * scale, rows * scale))
    for i, img16_u16 in enumerate(images):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(img16_u16, cmap="gray", vmin=0, vmax=65535)
        ax.axis("off")
        if titles is not None:
            ax.set_title(titles[i], fontsize=8)
    plt.tight_layout(pad=pad)
    return plt.gcf()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--bin_name", type=str, default="images_u16.bin")
    ap.add_argument("--prompts_name", type=str, default="prompts.txt")

    ap.add_argument("--index", type=int, default=None, help="Show single sample")
    ap.add_argument("--n", type=int, default=64, help="How many random samples to show")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--no_titles", action="store_true", help="Hide category titles in random grid")
    ap.add_argument("--save_png", type=str, default=None, help="Save grid image to this file (png)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    bin_path = data_dir / args.bin_name
    prompts_path = data_dir / args.prompts_name

    if not bin_path.exists():
        raise FileNotFoundError(bin_path)
    if not prompts_path.exists():
        raise FileNotFoundError(prompts_path)

    total = count_records(bin_path)
    rng = np.random.default_rng(args.seed)

    if args.index is not None:
        if args.index < 0 or args.index >= total:
            raise IndexError(f"--index must be in [0, {total-1}]")
        img = read_record_u16(bin_path, args.index)
        prompt = read_prompt_line(prompts_path, args.index)
        title = f"#{args.index}  {prompt}" if prompt else f"#{args.index}"
        show_single(img, title=title)
        return

    n = min(max(args.n, 1), total)
    idxs = rng.choice(total, size=n, replace=False)

    images = []
    titles = [] if not args.no_titles else None
    for idx in idxs:
        images.append(read_record_u16(bin_path, int(idx)))
        if titles is not None:
            category = read_prompt_line(prompts_path, int(idx))
            titles.append(category if category else f"#{int(idx)}")

    fig = show_grid(images, titles=titles, cols=args.cols)

    if args.save_png:
        out_path = Path(args.save_png)
        fig.savefig(out_path, dpi=200)
        print(f"Saved: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
