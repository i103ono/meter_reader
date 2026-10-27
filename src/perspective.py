"""Validated planar correspondences, not ellipse-to-circle pose inference."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw

from image_io import save_png_exclusive
from schemas import PerspectiveSpec, ProcessingStep

VERSION = "perspective-v1"
MAX_SIDE = 2048
MAX_PIXELS = 64_000_000


class GeometryError(ValueError):
    pass


def perspective_step(spec: PerspectiveSpec, size) -> ProcessingStep:
    spec = PerspectiveSpec.model_validate(spec)
    width, height = size
    points = np.asarray(spec.source_points, dtype=np.float64)
    if width < 32 or height < 32 or not np.isfinite(points).all():
        raise GeometryError("Perspective correction requires a finite reference on an ROI of at least 32x32")
    if np.any(points < 0) or np.any(points[:, 0] > width - 1) or np.any(points[:, 1] > height - 1):
        raise GeometryError("Perspective points must be inside the original ROI pixel centers")
    edges = np.roll(points, -1, axis=0) - points
    following = np.roll(edges, -1, axis=0)
    cross = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    if np.any(cross <= 1e-6) or np.any(np.linalg.norm(edges, axis=1) < 8):
        raise GeometryError("Perspective points must be distinct and convex in the documented clockwise order")
    area = abs(float(cv2.contourArea(points.astype(np.float32))))
    if area < width * height * 0.01:
        raise GeometryError("Perspective reference covers too little of the original ROI")
    side = min(MAX_SIDE, max(width, height))
    if spec.reference == "dial_cardinals":
        out_width = out_height = side
        mid = (side - 1) / 2
        destination = np.array([[mid, 0], [side - 1, mid], [mid, side - 1], [0, mid]], dtype=np.float64)
    else:
        ratio = spec.aspect_ratio
        out_width = side if ratio >= 1 else max(2, round(side * ratio))
        out_height = side if ratio <= 1 else max(2, round(side / ratio))
        destination = np.array([[0, 0], [out_width - 1, 0], [out_width - 1, out_height - 1], [0, out_height - 1]], dtype=np.float64)
    if out_width * out_height > MAX_PIXELS:
        raise GeometryError("Perspective output exceeds the pixel limit")
    try:
        matrix = cv2.getPerspectiveTransform(points.astype(np.float32), destination.astype(np.float32))
        if abs(matrix[2, 2]) < 1e-10:
            raise GeometryError("Perspective transform has an unstable origin")
        matrix /= matrix[2, 2]
        # Condition on unit-size coordinates, independent of translation/pixel dimensions.
        src_scale = np.diag([width - 1, height - 1, 1.0])
        dst_scale = np.diag([1 / (out_width - 1), 1 / (out_height - 1), 1.0])
        condition = float(np.linalg.cond(dst_scale @ matrix @ src_scale))
        if not np.isfinite(matrix).all() or not np.isfinite(condition) or condition > 100:
            raise GeometryError("Perspective transform is singular or too ill-conditioned")
        inverse = np.linalg.inv(matrix)
        corners = np.array([[0, 0, 1], [out_width - 1, 0, 1],
                            [out_width - 1, out_height - 1, 1], [0, out_height - 1, 1]], dtype=float)
        mapped = (inverse @ corners.T).T
        denominators = mapped[:, 2]
        if np.any(abs(denominators) < 1e-8) or not (np.all(denominators > 0) or np.all(denominators < 0)):
            raise GeometryError("Perspective output crosses a projective horizon")
        mapped = mapped[:, :2] / denominators[:, None]
        if (np.any(mapped < -1e-3) or np.any(mapped[:, 0] > width - 1 + 1e-3)
                or np.any(mapped[:, 1] > height - 1 + 1e-3)):
            raise GeometryError("Perspective output would require pixels outside the original ROI")
    except (cv2.error, np.linalg.LinAlgError, FloatingPointError) as exc:
        raise GeometryError("Could not construct a stable perspective transform") from exc
    return ProcessingStep(operation="perspective", parameters={
        "version": VERSION, "spec": spec.model_dump(mode="json"),
        "input_size": [width, height], "output_size": [out_width, out_height],
        "destination_points": destination.tolist(), "matrix": matrix.tolist(),
        "normalized_condition": condition, "interpolation": "OpenCV.INTER_LINEAR",
        "border_mode": "OpenCV.BORDER_CONSTANT", "border_value": [0, 0, 0, 0],
        "alpha_processing": "premultiplied", "max_output_pixels": MAX_PIXELS,
    })


def warp(original: Image.Image, step: ProcessingStep) -> Image.Image:
    if original.mode not in ("RGB", "RGBA"):
        raise GeometryError("Perspective input must be normalized RGB/RGBA")
    params = step.parameters
    pixels = np.array(original)
    alpha = original.mode == "RGBA"
    if alpha:
        pixels = pixels.astype(np.float32) / 255
        pixels[:, :, :3] *= pixels[:, :, 3:4]
    try:
        output = cv2.warpPerspective(pixels, np.asarray(params["matrix"], dtype=np.float64),
                                     tuple(params["output_size"]), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    except cv2.error as exc:
        raise GeometryError("OpenCV could not apply the validated perspective transform") from exc
    if alpha:
        np.divide(output[:, :, :3], output[:, :, 3:4], out=output[:, :, :3], where=output[:, :, 3:4] > 1e-8)
        output = np.floor(np.clip(output, 0, 1) * 255 + 0.5).astype(np.uint8)
    result = Image.fromarray(output)
    result.info.clear()
    return result


def validate_saved_step(spec, size, step):
    """Allow library roundoff when checking a matrix, then replay the SAVED matrix."""
    expected = perspective_step(spec, size)
    if step.operation != "perspective" or set(step.parameters) != set(expected.parameters):
        raise GeometryError("Saved perspective step has unexpected fields")
    for key, value in expected.parameters.items():
        saved = step.parameters[key]
        if key in ("matrix", "normalized_condition"):
            try:
                a, b = np.asarray(saved, dtype=float), np.asarray(value, dtype=float)
                matches = a.shape == b.shape and np.isfinite(a).all() and np.allclose(a, b, rtol=1e-10, atol=1e-10)
            except (ValueError, TypeError):
                matches = False
            if not matches:
                raise GeometryError("Saved perspective matrix/conditioning does not match the reference")
        elif saved != value:
            raise GeometryError("Saved perspective parameters do not match the registered version")


def save_landmark_preview(image, spec, path):
    with image.convert("RGB") as preview:
        draw = ImageDraw.Draw(preview)
        names = ("top", "right", "bottom", "left") if spec.reference == "dial_cardinals" else ("TL", "TR", "BR", "BL")
        points = list(spec.source_points)
        draw.line([*points, points[0]], fill="lime", width=2)
        for name, (x, y) in zip(names, points):
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="red")
            draw.text((x + 4, y + 4), name, fill="red")
        save_png_exclusive(preview, path)
