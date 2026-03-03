# QuickDraw 16x16 U16 Pipeline

Bu repo, QuickDraw verisini `16x16` gri-seviye ve `16-bit (uint16)` olarak üretir, görüntüler ve PNG olarak dışa aktarır.

## Dosyalar

- `make_qd_16x16.py`: QuickDraw ndjson akışından dataset üretir.
- `view_qd_16x16.py`: `images_u16.bin` içinden örnek gösterir.
- `export_all_png.py`: Tüm kayıtları 16-bit PNG olarak dışa aktarır.

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy requests pillow matplotlib tqdm
```

## 1) Dataset üret

```bash
python make_qd_16x16.py \
  --categories cat airplane tree bicycle \
  --per_category 5000 \
  --out_dir qd_u16_test
```

Üretilen dosyalar:

- `qd_u16_test/images_u16.bin` (`512` byte/kayıt, `16x16 uint16 little-endian`)
- `qd_u16_test/prompts.txt`
- `qd_u16_test/meta.json`

## 2) Rastgele örnekleri görüntüle

```bash
python view_qd_16x16.py --data_dir qd_u16_test --n 64 --cols 8
```

Bu görünümde kategori adları varsayılan olarak her görselin üstünde yazılır. Gizlemek için:

```bash
python view_qd_16x16.py --data_dir qd_u16_test --n 64 --cols 8 --no_titles
```

Tek örnek görüntülemek için:

```bash
python view_qd_16x16.py --data_dir qd_u16_test --index 42
```

## 3) Tüm örnekleri PNG dışa aktar

```bash
python export_all_png.py \
  --data_dir qd_u16_test \
  --out_dir exported_png_u16 \
  --scale 16
```

Notlar:

- PNG dosyaları `16-bit grayscale` (`I;16`) olarak kaydedilir.
- Eski `u4` (4-bit) veri dosyaları (`images_u4.bin`) bu scriptlerle uyumlu değildir.
