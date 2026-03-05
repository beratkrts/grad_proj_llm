# QuickDraw 16x16 Pipeline

Bu repo, QuickDraw verisini `16x16` çözünürlükte iki farklı formatta üretir:
- **16-bit grayscale (uint16)**: Çizgi yoğunluğunu sürekli değerle encode eder.
- **1-bit siyah/beyaz**: Her piksel ya çizgi (1) ya arka plan (0); min-pooling ile ince çizgiler korunur.

Her iki formatta da arka plan **beyaz**, çizgiler **siyahtır**.

## Dosyalar

### 16-bit pipeline
- `make_qd_16x16.py`: QuickDraw ndjson akışından dataset üretir (`512` byte/kayıt).
- `view_qd_16x16.py`: `images_u16.bin` içinden örnek gösterir.
- `export_all_png.py`: Tüm kayıtları 16-bit PNG olarak dışa aktarır.

### 1-bit pipeline
- `make_qd_1bit.py`: QuickDraw ndjson akışından 1-bit dataset üretir (`32` byte/kayıt).
- `view_qd_1bit.py`: `images_1bit.bin` içinden örnek gösterir.
- `export_all_png_1bit.py`: Tüm kayıtları 8-bit PNG (beyaz bg, siyah çizgi) olarak dışa aktarır.

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

> **Not:** Varsayılan olarak `--canvas 32 --stroke_width 2 --padding 2` kullanılır.
> 256x256'dan 16x16'ya 16x küçültme yerine yalnızca **2x küçültme** yapılır; çizgiler çok daha keskin görünür.

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

PNG dosyaları `16-bit grayscale` (`I;16`) olarak kaydedilir.

---

## 1-bit Pipeline

### 1) Dataset üret

```bash
python make_qd_1bit.py \
  --categories cat airplane tree bicycle \
  --per_category 5000 \
  --out_dir qd_1bit_test
```

> **Not:** Varsayılan olarak `--canvas 32 --stroke_width 2 --padding 2` kullanılır.
> Min-pooling bloğu 2x2 piksel olur; 256x256'daki 16x16 bloktan çok daha temiz sonuç üretir.

Üretilen dosyalar:

- `qd_1bit_test/images_1bit.bin` (`32` byte/kayıt, `256` bit packed MSB-first)
- `qd_1bit_test/prompts.txt`
- `qd_1bit_test/meta.json`

Downsampling için **min-pooling** kullanılır: her 16x16 çıktı pikseli, 256x256 canvas'taki 16x16 bloğun en koyu pikselini alır. BILINEAR ile yok olacak ince çizgiler bu sayede korunur.

### 2) Rastgele örnekleri görüntüle

```bash
python view_qd_1bit.py --data_dir qd_1bit_test --n 64 --cols 8
```

Tek örnek:

```bash
python view_qd_1bit.py --data_dir qd_1bit_test --index 42
```

### 3) PNG dışa aktar

```bash
python export_all_png_1bit.py \
  --data_dir qd_1bit_test \
  --out_dir exported_png_1bit \
  --scale 16
```

---

## Format karşılaştırması

| Özellik | 16-bit (u16) | 1-bit |
|---|---|---|
| Byte/kayıt | 512 | 32 |
| Piksel değerleri | 0–65535 | 0 veya 1 |
| Downsampling | BILINEAR | Min-pooling |
| Kullanım | Sürekli yoğunluk | İkili sınıflandırma |

- Eski `u4` (4-bit) veri dosyaları (`images_u4.bin`) bu scriptlerle uyumlu değildir.
