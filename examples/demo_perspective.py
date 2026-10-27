"""Create an artificial planar drawing and known correspondences; no AI/API."""
import argparse
from pathlib import Path
import sys
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from image_io import save_png_exclusive
from schemas import PerspectiveSpec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/perspective_demos"))
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve() / ("demo_" + uuid4().hex)
    root.mkdir(parents=True, exist_ok=False)
    points = [[60, 22], [340, 55], [304, 260], [25, 222]]
    spec = PerspectiveSpec(reference="rectangle", source_points=points, aspect_ratio=1,
                           evidence="Artificial known square drawing; not real meter landmarks")
    with Image.new("RGB", (320, 320), "white") as image:
        draw = ImageDraw.Draw(image)
        draw.ellipse((40, 40, 280, 280), outline="black", width=3)
        draw.line((160, 160, 90, 75), fill="red", width=4)
        draw.text((90, 300), "ARTIFICIAL TEST", fill="black")
        for x, y, color in [(40, 40, "red"), (280, 40, "green"), (280, 280, "blue"), (40, 280, "orange")]:
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color)
        save_png_exclusive(image, root / "front.png")
        matrix = cv2.getPerspectiveTransform(np.float32([[0, 0], [319, 0], [319, 319], [0, 319]]), np.float32(points))
        pixels = cv2.warpPerspective(np.array(image), matrix, (384, 288), borderValue=(210, 210, 210))
    with Image.fromarray(pixels) as image:
        save_png_exclusive(image, root / "oblique.png")
    with (root / "perspective.json").open("x", encoding="utf-8") as handle:
        handle.write(spec.model_dump_json(indent=2) + "\n")
    print(root)


if __name__ == "__main__":
    main()
