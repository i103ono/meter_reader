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

## Detect Meter Region
Crop an analog meter region for later processing.

```
python src/detect_meter_region.py data/images/meter_0005.jpg --roi 100 200 300 300
python src/detect_meter_region.py data/images/meter_0005.jpg --interactive --output outputs/meter_0005_meter_region.png
```

## Detect Dial
Detect a circular analog meter dial and save a debug image.

```
python src/detect_dial.py outputs/meter_0005_meter_region.png --min-radius 20
```

## Detect Needle
Detect analog meter needle candidates and estimate angle.

```
python src/detect_needle.py data/images/meter_0005.jpg --center-x 1867 --center-y 2114 --radius 1043
```

## Analog Calibration
Convert a needle angle to a meter value.

```
python src/analog_calibration.py --angle 223.2 --min-value 0 --max-value 70 --min-angle 225 --max-angle 345 --direction counterclockwise
```

Angles near the min/max endpoints on the same vertical half are clamped to the
endpoint value. The default tolerance is 5 degrees.

```
python src/analog_calibration.py --angle 223.2 --min-value 0 --max-value 70 --min-angle 225 --max-angle 345 --direction counterclockwise --endpoint-tolerance 0
```

## Read Analog Meter
Detect the dial and needle, then convert the needle angle to a value.

```
python src/read_analog_meter.py outputs/meter_0001_meter_region.png --min-value 0 --max-value 70 --unit MPa
```
