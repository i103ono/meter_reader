"""Descriptive luminance metrics; these are neither scores nor acceptance rules."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from schemas import QualityMetrics


def measure_quality(image: Image.Image) -> QualityMetrics:
    if image.width <= 0 or image.height <= 0:
        raise ValueError("Quality metrics require a nonempty image")
    with image.convert("RGB") as rgb:
        y = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2YCrCb)[:, :, 0]
    # Ignore transparent content for display metrics by compositing over white.
    if image.mode == "RGBA":
        with Image.new("RGBA", image.size, "white") as white:
            with Image.alpha_composite(white, image).convert("RGB") as visible:
                y = cv2.cvtColor(np.array(visible), cv2.COLOR_RGB2YCrCb)[:, :, 0]
    p5, p50, p95 = (float(value) for value in np.percentile(y, [5, 50, 95]))
    grid_x, grid_y = min(4, image.width), min(4, image.height)
    means = [
        float(block.mean()) for row in np.array_split(y, grid_y, axis=0)
        for block in np.array_split(row, grid_x, axis=1)
    ]
    return QualityMetrics(
        width=image.width, height=image.height,
        luminance_p5=p5, luminance_p50=p50, luminance_p95=p95,
        luminance_spread=p95 - p5,
        dark_fraction=float((y <= 10).mean()),
        saturated_fraction=float((y >= 245).mean()),
        local_mean_spread=max(means) - min(means),
        parameters={
            "channel": "OpenCV.RGB2YCrCb.Y", "percentile_method": "numpy.linear",
            "dark_at_most": 10, "saturated_at_least": 245,
            "requested_grid": [4, 4], "effective_grid": [grid_x, grid_y],
            "alpha_background": "white",
        },
    )
