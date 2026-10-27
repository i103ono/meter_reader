"""One exclusive correction at a time, always starting from the unchanged ROI."""
from __future__ import annotations

import hashlib

from PIL import Image

from image_io import ImageLoadError, save_png_exclusive
from quality import measure_quality
from reader import ReaderError
from schemas import CandidateResult
from tools import CorrectionError, apply_recipe, pixel_sha256, recipe_steps, runtime_versions, validate_recipes


class CandidateExecutor:
    def __init__(self, run_dir, read_asset, check_budget):
        self.run_dir, self.read_asset, self.check_budget = run_dir, read_asset, check_budget
        self.roi_path = run_dir / "roi.png"
        self.roi_hash = hashlib.sha256(self.roi_path.read_bytes()).hexdigest()
        self.runtime = runtime_versions()
        self.tried = []
        self.saved_assets = {}

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
            steps = recipe_steps(recipe_id, image.size)
            info = dict(candidate_id=f"c{index}", parent_candidate_id="c0", recipe_id=recipe_id,
                        steps=steps, runtime=self.runtime, input_roi_sha256=self.roi_hash)
            path = self.run_dir / "candidates" / f"c{index}_{recipe_id}.png"
            try:
                self.check_budget()
                path.parent.mkdir(exist_ok=True)
                correction = apply_recipe(image, recipe_id)
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
                    inference=None, api_calls=0,
                    error_code=exc.code if isinstance(exc, ReaderError) else "correction_error", error=str(exc),
                )
        self.saved_assets[path] = image_info["image_sha256"]
        observation = self.read_asset(path)
        self.verify()
        return CandidateResult(
            **info, **image_info, status=observation["status"], reading=observation["reading"],
            inference=observation["inference"], api_calls=observation["api_calls"], error=observation["error"],
            error_code=observation["reasons"][0] if observation["status"] == "error" else None,
        )
