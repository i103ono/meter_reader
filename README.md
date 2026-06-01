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
