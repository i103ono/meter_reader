"""SQLite history with immutable run snapshots and owned, hashed asset copies."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
from uuid import uuid4

from history_models import MeterRecord, ReviewInput
from schemas import RunResult


def now():
    return datetime.now(timezone.utc).isoformat()


def utc(value):
    return value.astimezone(timezone.utc).isoformat() if value is not None else None


def dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def asset_in_run(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name or not relative.parts:
        raise ValueError("Run assets must be relative paths inside the run")
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Referenced run asset is missing or escapes its run")
    return path


class Repository:
    def __init__(self, database):
        self.path = Path(database).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.asset_root = self.path.parent / (self.path.stem + "_assets")
        if self.asset_root.is_symlink():
            raise ValueError("History asset root must not be a symlink")
        self.connection = sqlite3.connect(self.path, timeout=5)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        version = self.connection.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, 1):
            self.close()
            raise ValueError("Unsupported history schema version")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS meters(meter_id TEXT PRIMARY KEY, data TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS meter_versions(
                meter_id TEXT NOT NULL REFERENCES meters(meter_id), series_version TEXT NOT NULL,
                data TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY(meter_id,series_version));
            CREATE TABLE IF NOT EXISTS observations(
                observation_id TEXT PRIMARY KEY, meter_id TEXT REFERENCES meters(meter_id),
                series_version TEXT, input_sha256 TEXT NOT NULL, captured_at TEXT, captured_timezone TEXT,
                timestamp_source TEXT NOT NULL, original_roi TEXT, selected_run_id TEXT, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS read_runs(
                run_id TEXT PRIMARY KEY, observation_id TEXT NOT NULL REFERENCES observations(observation_id),
                result_json TEXT NOT NULL, result_sha256 TEXT NOT NULL, processed_at TEXT NOT NULL,
                mode TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS assets(
                asset_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES read_runs(run_id),
                name TEXT NOT NULL, relative_path TEXT NOT NULL UNIQUE, sha256 TEXT NOT NULL, size INTEGER NOT NULL,
                UNIQUE(run_id,name));
            CREATE TABLE IF NOT EXISTS reviews(
                review_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES read_runs(run_id),
                request_json TEXT NOT NULL, resolution_json TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS run_selections(
                selection_id INTEGER PRIMARY KEY, observation_id TEXT NOT NULL REFERENCES observations(observation_id),
                run_id TEXT NOT NULL REFERENCES read_runs(run_id), created_at TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS runs_observation ON read_runs(observation_id);
            CREATE INDEX IF NOT EXISTS reviews_run ON reviews(run_id);
            CREATE TRIGGER IF NOT EXISTS immutable_runs_update BEFORE UPDATE ON read_runs BEGIN SELECT RAISE(ABORT,'immutable run'); END;
            CREATE TRIGGER IF NOT EXISTS immutable_runs_delete BEFORE DELETE ON read_runs BEGIN SELECT RAISE(ABORT,'immutable run'); END;
            CREATE TRIGGER IF NOT EXISTS immutable_reviews_update BEFORE UPDATE ON reviews BEGIN SELECT RAISE(ABORT,'append-only review'); END;
            CREATE TRIGGER IF NOT EXISTS immutable_reviews_delete BEFORE DELETE ON reviews BEGIN SELECT RAISE(ABORT,'append-only review'); END;
            PRAGMA user_version=1;
        """)
        with self.connection:
            for row in self.connection.execute("SELECT meter_id,data,created_at FROM meters").fetchall():
                self.connection.execute("INSERT OR IGNORE INTO meter_versions VALUES(?,?,?,?)",
                                        (row[0], json.loads(row[1])["series_version"], row[1], row[2]))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.connection.close()

    def register_meter(self, record):
        record = MeterRecord.model_validate(record)
        data = record.model_dump(mode="json")
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            old = self.connection.execute("SELECT data FROM meters WHERE meter_id=?", (record.meter_id,)).fetchone()
            version = self.connection.execute("SELECT data FROM meter_versions WHERE meter_id=? AND series_version=?",
                                              (record.meter_id, record.series_version)).fetchone()
            if version and any(json.loads(version[0])[k] != data[k] for k in ("meter_type", "registered_unit", "registered_scale")):
                raise ValueError("Series version already describes a different measurement reference")
            if old and json.loads(old[0]) != data:
                old_data = json.loads(old[0])
                changed_reference = any(old_data[k] != data[k] for k in ("meter_type", "registered_unit", "registered_scale"))
                if changed_reference and old_data["series_version"] == record.series_version:
                    raise ValueError("Changing the measurement reference requires a new series_version")
            self.connection.execute("INSERT INTO meters VALUES(?,?,?) ON CONFLICT(meter_id) DO UPDATE SET data=excluded.data",
                                    (record.meter_id, dump(data), now()))
            self.connection.execute("INSERT OR IGNORE INTO meter_versions VALUES(?,?,?,?)",
                                    (record.meter_id, record.series_version, dump(data), now()))
        return data

    def list_meters(self):
        return [json.loads(row[0]) for row in self.connection.execute("SELECT data FROM meters ORDER BY meter_id")]

    def _collect(self, result_path, result, raw):
        root = result_path.parent
        names = {"result.json"}
        source = result.source_image
        names.update(name for name in (source.stored_path, source.normalized_path, source.roi_path, result.roi_preview_path) if name)
        expected = {source.stored_path: result.input_sha256}
        for candidate in result.candidates:
            if candidate.image_path:
                names.add(candidate.image_path)
                if candidate.image_sha256:
                    expected[candidate.image_path] = candidate.image_sha256
            if candidate.geometry and (root / f"candidates/{candidate.candidate_id}_perspective_points.png").exists():
                names.add(f"candidates/{candidate.candidate_id}_perspective_points.png")
        for name in ("roi_proposal.json", "roi_locator_input.png"):
            if (root / name).exists():
                names.add(name)
        assets = {}
        for name in sorted(names):
            content = raw if name == "result.json" else asset_in_run(root, name).read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            if name in expected and expected[name] != digest:
                raise ValueError("Run asset hash does not match the result: " + name)
            assets[name] = (content, digest)
        return assets

    def import_run(self, result_path, *, meter_id=None, select=True):
        result_path = Path(result_path).expanduser().resolve()
        if result_path.name != "result.json":
            raise ValueError("Import the result.json inside its original run directory")
        raw = result_path.read_bytes()
        result = RunResult.model_validate_json(raw)
        if result.meter_id is not None and meter_id is not None and result.meter_id != meter_id:
            raise ValueError("Explicit meter ID conflicts with the run")
        meter_id = meter_id or result.meter_id
        assets = self._collect(result_path, result, raw)
        digest = hashlib.sha256(raw).hexdigest()
        directory = self.asset_root / ("run_" + uuid4().hex)
        created = False
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            old = self.connection.execute("SELECT * FROM read_runs WHERE run_id=?", (result.run_id,)).fetchone()
            if old:
                observation = self.connection.execute("SELECT meter_id FROM observations WHERE observation_id=?", (old["observation_id"],)).fetchone()
                if old["result_sha256"] != digest or (meter_id is not None and observation[0] != meter_id):
                    raise ValueError("Run ID already exists with different content or meter")
                stored = {r["name"]: r["sha256"] for r in self.connection.execute("SELECT name,sha256 FROM assets WHERE run_id=?", (result.run_id,))}
                if stored != {name: value[1] for name, value in assets.items()}:
                    raise ValueError("Previously imported assets differ")
                self.connection.commit()
                return result.run_id
            if meter_id is not None:
                meter = self.connection.execute("SELECT data FROM meters WHERE meter_id=?", (meter_id,)).fetchone()
                if meter is None:
                    data = MeterRecord(meter_id=meter_id, name=meter_id).model_dump(mode="json")
                    self.connection.execute("INSERT INTO meters VALUES(?,?,?)", (meter_id, dump(data), now()))
                    self.connection.execute("INSERT INTO meter_versions VALUES(?,?,?,?)", (meter_id, data["series_version"], dump(data), now()))
                else:
                    data = json.loads(meter[0])
                series = data["series_version"]
            else:
                series = None
            observation = self.connection.execute("SELECT * FROM observations WHERE observation_id=?", (result.observation_id,)).fetchone()
            if observation:
                if any(observation[key] != value for key, value in (
                    ("meter_id", meter_id), ("input_sha256", result.input_sha256), ("captured_at", utc(result.captured_at)))):
                    raise ValueError("Observation ID already refers to a different image, meter or capture time")
            else:
                self.connection.execute("INSERT INTO observations VALUES(?,?,?,?,?,?,?,?,?,?)", (
                    result.observation_id, meter_id, series, result.input_sha256, utc(result.captured_at),
                    result.captured_timezone, result.timestamp_source, dump(result.roi), None, now()))
            directory.mkdir(parents=True, exist_ok=False)
            created = True
            self.connection.execute("INSERT INTO read_runs VALUES(?,?,?,?,?,?,?,?)", (
                result.run_id, result.observation_id, raw.decode("utf-8"), digest, utc(result.processed_at), result.mode, result.status, now()))
            for name, (content, sha) in assets.items():
                # name was validated relative to the source run before this transaction.
                target = directory / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as handle:
                    handle.write(content)
                self.connection.execute("INSERT INTO assets VALUES(?,?,?,?,?,?)", (
                    "asset_" + uuid4().hex, result.run_id, name, str(target.relative_to(self.asset_root)), sha, len(content)))
            if select or observation is None or observation["selected_run_id"] is None:
                self._select(result.observation_id, result.run_id)
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            if created:
                shutil.rmtree(directory)
            raise
        return result.run_id

    def _select(self, observation_id, run_id):
        self.connection.execute("UPDATE observations SET selected_run_id=? WHERE observation_id=?", (run_id, observation_id))
        self.connection.execute("INSERT INTO run_selections(observation_id,run_id,created_at) VALUES(?,?,?)", (observation_id, run_id, now()))

    def select_run(self, observation_id, run_id):
        with self.connection:
            row = self.connection.execute("SELECT observation_id FROM read_runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None or row[0] != observation_id:
                raise ValueError("Selected run does not belong to this observation")
            self._select(observation_id, run_id)

    def get_asset(self, asset_id):
        row = self.connection.execute("SELECT * FROM assets WHERE asset_id=?", (asset_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown asset ID")
        path = asset_in_run(self.asset_root.resolve(), row["relative_path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Stored asset integrity check failed")
        return path

    def get_run(self, run_id):
        row = self.connection.execute("SELECT * FROM read_runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown run ID")
        return {"result": json.loads(row["result_json"]),
                "assets": [dict(r) for r in self.connection.execute("SELECT asset_id,name,sha256,size FROM assets WHERE run_id=? ORDER BY name", (run_id,))],
                "reviews": [{"review_id": r["review_id"], "request": json.loads(r["request_json"]),
                             "resolution": json.loads(r["resolution_json"]), "created_at": r["created_at"]}
                            for r in self.connection.execute("SELECT * FROM reviews WHERE run_id=? ORDER BY rowid", (run_id,))]}

    def add_review(self, request):
        request = ReviewInput.model_validate(request)
        run = self.get_run(request.run_id)["result"]
        candidate_id = request.candidate_id
        if candidate_id is None:
            candidate_id = run["selected_candidate_id"]
        known_candidates = {c["candidate_id"] for c in run["candidates"]} or {"c0"}
        if candidate_id is not None and candidate_id not in known_candidates:
            raise ValueError("Unknown review candidate ID")
        if request.action == "confirm":
            candidate = next((c for c in run["candidates"] if c["candidate_id"] == candidate_id), None)
            reading = candidate["reading"] if candidate else (run["reading"] if candidate_id == "c0" and not run["candidates"] else None)
            if reading is None or reading["value"] is None or reading["unit"] is None:
                raise ValueError("Confirm requires an existing candidate with known value and unit")
            resolution = {"review_status": "confirmed", "value": str(Decimal(str(reading["value"]))),
                          "unit": reading["unit"], "scale": reading["scale"], "candidate_id": candidate_id, "corrected": False}
        elif request.action == "correct":
            if candidate_id is not None and candidate_id != "c0" and not any(c["candidate_id"] == candidate_id for c in run["candidates"]):
                raise ValueError("Unknown review candidate ID")
            resolution = {"review_status": "confirmed", "value": str(request.value), "unit": request.unit,
                          "scale": request.scale.model_dump(mode="json") if request.scale else None,
                          "candidate_id": candidate_id, "corrected": True}
        else:
            resolution = {"review_status": "rejected", "value": None, "unit": None,
                          "scale": None, "candidate_id": candidate_id, "corrected": False}
        review_id = request.review_id or "review_" + uuid4().hex
        data = request.model_dump(mode="json")
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            old = self.connection.execute("SELECT request_json FROM reviews WHERE review_id=?", (review_id,)).fetchone()
            if old:
                if json.loads(old[0]) != data:
                    raise ValueError("Review ID already exists with different content")
                return review_id
            self.connection.execute("INSERT INTO reviews VALUES(?,?,?,?,?)", (review_id, request.run_id, dump(data), dump(resolution), now()))
        return review_id

    def get_observation(self, observation_id):
        row = self.connection.execute("SELECT * FROM observations WHERE observation_id=?", (observation_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown observation ID")
        result = dict(row)
        result["original_roi"] = json.loads(result["original_roi"])
        result["run_ids"] = [r[0] for r in self.connection.execute("SELECT run_id FROM read_runs WHERE observation_id=? ORDER BY rowid", (observation_id,))]
        selected = self.get_run(result["selected_run_id"]) if result["selected_run_id"] else None
        result["selected_roi"] = selected["result"]["roi"] if selected else None
        result["read_status"] = selected["result"]["status"] if selected else None
        version = self.connection.execute("SELECT data FROM meter_versions WHERE meter_id=? AND series_version=?",
                                          (result["meter_id"], result["series_version"])).fetchone()
        result["meter"] = json.loads(version[0]) if version else None
        reviews = selected["reviews"] if selected else []
        latest = reviews[-1]["resolution"] if reviews else None
        result.update(review_status=latest["review_status"] if latest else "unreviewed",
                      effective_value=latest["value"] if latest and latest["review_status"] == "confirmed" else None,
                      effective_unit=latest["unit"] if latest and latest["review_status"] == "confirmed" else None,
                      effective_scale=latest["scale"] if latest and latest["review_status"] == "confirmed" else None,
                      corrected=latest["corrected"] if latest else False)
        return result

    def list_observations(self, meter_id=None):
        query = "SELECT observation_id FROM observations"
        rows = self.connection.execute(query + (" WHERE meter_id=?" if meter_id else "") + " ORDER BY created_at,observation_id",
                                       (meter_id,) if meter_id else ())
        return [self.get_observation(row[0]) for row in rows.fetchall()]
