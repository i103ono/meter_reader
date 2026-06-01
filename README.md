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
