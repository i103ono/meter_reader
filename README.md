# meter_reader

## Rename images
Use `scripts/rename_images.py` to rename image files in `data/images/` to sequential
`meter_0001`, `meter_0002`, ... names. The script targets `.jpg`, `.jpeg`, and
`.png` files (case-insensitive) and normalizes the extension to lowercase.

Dry run (default):

```
python scripts/rename_images.py
```

Apply changes:

```
python scripts/rename_images.py --apply
```

## List images
`data/images/` is ignored by Git, so place image files there locally before running.
The script lists each file name, image size, and channel count.

Requires Pillow via the conda environment defined in `environment.yml`:

```
conda env create -f environment.yml
conda activate meter_reader
```

```
python src/list_images.py
```

## Preprocess image
Preprocess a single image and save grayscale and binary outputs to `outputs/`
(ignored by Git).

```
python src/preprocess.py data/images/meter_0001.jpg
```

Optional parameters:

```
python src/preprocess.py data/images/meter_0001.jpg --threshold 140 --resize 0.5 --debug
```

## OCR test
Run EasyOCR on a single image (requires `easyocr` in `environment.yml`).

```
python src/ocr_test.py data/images/meter_0001.jpg
```

Digits-only:

```
python src/preprocess.py data/images/meter_0001.jpg
python src/ocr_test.py outputs/meter_0001_binary.png --psm 7 --digits-only
```

## Crop ROI for OCR
Crop a numeric display region and run OCR on the cropped image.

```
python src/crop_roi.py data/images/meter_0005.jpg --roi 100 200 300 120 --output outputs/meter_0005_roi.png
python src/ocr_test.py outputs/meter_0005_roi.png --psm 7 --digits-only
```

Select the crop region with the mouse:

```
python src/crop_roi.py data/images/meter_0005.jpg --interactive --output outputs/meter_0005_roi.png
```

## Parse OCR reading
Extract a value and unit from OCR text.

```
python src/parse_reading.py "l23.O kWh"
```

## Read digital meter
Run OCR and extract a value from one image.

```
python src/read_digital_meter.py data/images/meter_0005.jpg --psm 7 --digits-only
```
