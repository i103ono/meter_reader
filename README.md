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

## Read Meter
Use one CLI for digital or analog meters.

```
python src/read_meter.py outputs/meter_0001_meter_region.png --type digital --digits-only
python src/read_meter.py outputs/meter_0005_meter_region.png --type analog --min-value 0 --max-value 70 --unit MPa
python src/read_meter.py data/images/meter_0001.jpg --config configs/sample_meter_config.json
```

CLI options override values from `--config`.

## Meter Agent
Estimate meter settings with OpenAI Vision. Digital values are read from the API
response, and analog values are read by the local needle detector using the
AI-estimated settings.
Create `.env` in the repository root and set `OPENAI_API_KEY` before running.

```
python src/meter_agent.py data/images/meter_0005.jpg --digits-only
python src/meter_agent.py data/images/meter_0001.jpg
```

Check representative digital and analog images and print CSV results:

```
python src/test_meter_agent.py --limit 2
python src/test_meter_agent.py --mock --limit 2
```

## Meter Config JSON
Meter settings can be stored in `configs/*.json`. The format is used for
AI-estimated or manually adjusted settings.

Example:

```
configs/sample_meter_config.json
```

Fields:

- `meter_type`: required, `digital` or `analog`
- `roi`: optional, `[x, y, width, height]` in original image pixels, or `null`
- `unit`: optional display unit, such as `m3` or `MPa`
- `min_value`, `max_value`: required for analog meters, optional for digital meters
- `min_angle`, `max_angle`: optional analog calibration angles in degrees
- `direction`: optional analog scale direction, `clockwise` or `counterclockwise`

Missing optional fields should be treated as defaults by the script that reads the
config. Missing required fields or invalid values should produce a clear `Error: ...`
message instead of a traceback.

Suggest a config as JSON with OpenAI Vision:

```
python src/suggest_meter_config.py data/images/meter_0005.jpg
python src/suggest_meter_config.py data/images/meter_0005.jpg --output configs/meter_0005.json
python -m json.tool --no-ensure-ascii configs/meter_0005.json
```

Omit `--output` to print the JSON without saving a file.

Apply `roi` from a config and save the cropped image under `outputs/`:

```
python src/apply_meter_config.py data/images/meter_0001.jpg configs/sample_meter_config.json
```

Suggest configs for images in filename order:

```
python src/batch_suggest_configs.py --limit 5
```

Configs are saved under `configs/`; failures are recorded in
`outputs/config_suggestion_errors.csv`.

## Batch Read
Process images in `data/images/` in filename order and save `outputs/results.csv`.

```
python src/batch_read.py --type digital --digits-only
python src/batch_read.py --type analog --min-value 0 --max-value 70 --unit MPa
python src/batch_read.py --type digital --digits-only --canvas-size 960 --limit 5
python src/batch_read.py --config-dir configs --limit 5
```

In `--config-dir` mode, digital readings use the AI-generated `value`, `raw_text`,
`unit`, and `confidence` from each config without running local OCR.
Analog readings use the AI-generated meter type, unit, and min/max values, then run
the local dial and needle detector on the original image.
