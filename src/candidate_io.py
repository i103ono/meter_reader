"""One exclusive correction at a time, always starting from the unchanged ROI."""
from __future__ import annotations

import hashlib

from PIL import Image

from image_io import ImageLoadError, save_png_exclusive
from quality import measure_quality
from reader import ReaderError
from schemas import CandidateResult, GeometryRecord
from tools import CorrectionError, apply_recipe, pixel_sha256, recipe_steps, runtime_versions, validate_recipes


class CandidateExecutor:
    def __init__(self, run_dir, read_asset, check_budget, *, perspective_config=None, geometry_locator=None,
                 target_type="auto", question=None):
        self.run_dir, self.read_asset, self.check_budget = run_dir, read_asset, check_budget
        self.roi_path = run_dir / "roi.png"
        self.roi_hash = hashlib.sha256(self.roi_path.read_bytes()).hexdigest()
        self.runtime = runtime_versions()
        self.tried = []
        self.saved_assets = {}
        self.perspective_config, self.geometry_locator = perspective_config, geometry_locator
        self.target_type, self.question = target_type, question

    def verify(self):
        if hashlib.sha256(self.roi_path.read_bytes()).hexdigest() != self.roi_hash:
            raise ImageLoadError("Original ROI changed during candidate processing")
        for path, digest in self.saved_assets.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ImageLoadError("Saved candidate changed during processing")

    def baseline(self, observation):
        with Image.open(self.roi_path) as image:
            image.load()
            return CandidateResult(
                candidate_id="c0", parent_candidate_id=None, recipe_id=None, runtime=self.runtime,
                input_roi_sha256=self.roi_hash, image_path="roi.png", image_sha256=self.roi_hash,
                pixel_sha256=pixel_sha256(image), image_size=image.size, image_mode=image.mode,
                quality=measure_quality(image), status=observation["status"], reading=observation["reading"],
                inference=observation["inference"], api_calls=observation["api_calls"], error=observation["error"],
                error_code=observation["reasons"][0] if observation["status"] == "error" else None,
            )

    def apply(self, recipe_id, index):
        # Validate again at the execution boundary, even for non-VLM callers.
        validate_recipes([*self.tried, recipe_id])
        if index != len(self.tried) + 1:
            raise CorrectionError("Candidate index must follow the bounded recipe history")
        self.tried.append(recipe_id)
        self.verify()
        with Image.open(self.roi_path) as image:
            image.load()
            steps, geometry = [], None
            info = dict(candidate_id=f"c{index}", parent_candidate_id="c0", recipe_id=recipe_id,
                        steps=steps, runtime=self.runtime, input_roi_sha256=self.roi_hash)
            if recipe_id == "perspective_rectify":
                info["processing_version"] = "perspective-v1"
            path = self.run_dir / "candidates" / f"c{index}_{recipe_id}.png"
            try:
                self.check_budget()
                spec = None
                if recipe_id == "perspective_rectify":
                    if self.perspective_config is not None:
                        geometry = GeometryRecord(source="manual", status="ready", input_roi_sha256=self.roi_hash,
                                                  spec=self.perspective_config)
                    elif self.geometry_locator is not None:
                        geometry = self.geometry_locator.locate(self.roi_path, target_type=self.target_type, question=self.question)
                    else:
                        raise CorrectionError("Perspective recipe needs manual correspondences or an enabled geometry locator")
                    info["geometry"] = geometry
                    if geometry.input_roi_sha256 != self.roi_hash:
                        raise CorrectionError("Geometry refers to a different original ROI")
                    if geometry.status != "ready":
                        raise ReaderError(geometry.error or "Perspective correspondence could not be established",
                                          code=geometry.error_code or "geometry_" + geometry.status)
                    spec = geometry.spec
                    # Geometry localization consumes the same budget as the Reader.
                    self.check_budget()
                steps = recipe_steps(recipe_id, image.size, perspective=spec)
                info["steps"] = steps
                path.parent.mkdir(exist_ok=True)
                if spec is not None:
                    from perspective import save_landmark_preview
                    save_landmark_preview(image, spec, path.parent / f"c{index}_perspective_points.png")
                correction = apply_recipe(image, recipe_id, perspective=spec) if spec else apply_recipe(image, recipe_id)
                try:
                    save_png_exclusive(correction.image, path)
                    image_info = dict(
                        image_path=str(path.relative_to(self.run_dir)),
                        image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        pixel_sha256=pixel_sha256(correction.image), image_size=correction.image.size,
                        image_mode=correction.image.mode, quality=measure_quality(correction.image),
                    )
                finally:
                    correction.image.close()
            except (CorrectionError, OSError, ReaderError) as exc:
                return CandidateResult(
                    **info, image_path=None, image_sha256=None, pixel_sha256=None,
                    image_size=None, image_mode=None, quality=None, status="error", reading=None,
                    inference=None, api_calls=geometry.api_calls if geometry else 0,
                    error_code=exc.code if isinstance(exc, ReaderError) else "correction_error", error=str(exc),
                )
        self.saved_assets[path] = image_info["image_sha256"]
        observation = self.read_asset(path)
        self.verify()
        return CandidateResult(
            **info, **image_info, status=observation["status"], reading=observation["reading"],
            inference=observation["inference"],
            api_calls=observation["api_calls"] + (geometry.api_calls if geometry else 0) if observation["api_calls"] is not None else None,
            error=observation["error"],
            error_code=observation["reasons"][0] if observation["status"] == "error" else None,
        )
