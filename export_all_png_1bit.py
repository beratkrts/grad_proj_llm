import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


RECORD_BYTES = 32  # 16x16 bits = 256 bits = 32 bytes


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


def decode_1bit_record(rec_bytes: bytes) -> np.ndarray:
    packed = np.frombuffer(rec_bytes, dtype=np.uint8)
    bits = np.unpackbits(packed, bitorder="big")  # 256 bits: 1=stroke, 0=background
    return bits.reshape(16, 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="exported_png_1bit")
    ap.add_argument("--scale", type=int, default=16, help="16 -> 256x256 output")
    ap.add_argument("--bin_name", type=str, default="images_1bit.bin")
    ap.add_argument("--prompts_name", type=str, default="prompts.txt")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    bin_path = data_dir / args.bin_name
    prompts_path = data_dir / args.prompts_name

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = count_records(bin_path)

    with bin_path.open("rb") as f:
        for idx in tqdm(range(total), desc="Exporting PNGs (1-bit)"):
            rec = f.read(RECORD_BYTES)
            if len(rec) != RECORD_BYTES:
                break

            bits = decode_1bit_record(rec)  # (16,16) uint8: 1=stroke, 0=background
            # Convert to uint8 image: stroke=0 (black), background=255 (white)
            img_arr = ((1 - bits) * 255).astype(np.uint8)
            img = Image.fromarray(img_arr, mode="L")
            if args.scale > 1:
                img = img.resize((16 * args.scale, 16 * args.scale), Image.Resampling.NEAREST)

            prompt = read_prompt_line(prompts_path, idx)
            safe_prompt = prompt.replace(" ", "_").replace("/", "_").replace("\\", "_")
            filename = f"{idx:08d}_{safe_prompt}.png"
            img.save(out_dir / filename)

    print(f"\nDone. Exported {total} PNG files to {out_dir}")


if __name__ == "__main__":
    main()
