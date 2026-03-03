import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


RECORD_BYTES = 512  # 16x16 * uint16 (little-endian)


def count_records(bin_path: Path) -> int:
    size = bin_path.stat().st_size
    if size % RECORD_BYTES != 0:
        raise ValueError(f"Binary file size is not multiple of {RECORD_BYTES} bytes.")
    return size // RECORD_BYTES


def read_prompt_line(prompts_path: Path, idx: int) -> str:
    with prompts_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == idx:
                return line.strip()
    return ""


def decode_u16_record(rec_bytes: bytes) -> np.ndarray:
    pixels = np.frombuffer(rec_bytes, dtype="<u2")
    if pixels.size != 256:
        raise ValueError("Expected 256 pixels per record")
    return pixels.reshape(16, 16)  # 0..65535


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="exported_png")
    ap.add_argument("--scale", type=int, default=16, help="16 -> 256x256 output")
    ap.add_argument("--bin_name", type=str, default="images_u16.bin")
    ap.add_argument("--prompts_name", type=str, default="prompts.txt")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    bin_path = data_dir / args.bin_name
    prompts_path = data_dir / args.prompts_name

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = count_records(bin_path)

    with bin_path.open("rb") as f:
        for idx in tqdm(range(total), desc="Exporting PNGs (u16)"):
            rec = f.read(RECORD_BYTES)
            if len(rec) != RECORD_BYTES:
                break

            img_u16 = decode_u16_record(rec)
            img = Image.fromarray(img_u16, mode="I;16")
            if args.scale > 1:
                img = img.resize((16 * args.scale, 16 * args.scale), Image.Resampling.NEAREST)

            prompt = read_prompt_line(prompts_path, idx)
            safe_prompt = prompt.replace(" ", "_").replace("/", "_").replace("\\", "_")

            filename = f"{idx:08d}_{safe_prompt}.png"
            img.save(out_dir / filename)

    print(f"\nDone. Exported {total} PNG files to {out_dir}")


if __name__ == "__main__":
    main()
