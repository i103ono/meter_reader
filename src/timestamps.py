"""Capture time is evidence, never the processing clock."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from schemas import ObservationInput, TimestampMetadata


@dataclass(frozen=True)
class CaptureTime:
    captured_at: datetime | None
    timezone: str | None
    source: str
    metadata: TimestampMetadata


def resolve_capture_time(
    request: ObservationInput,
    exif_datetime: str | None,
    exif_offset: str | None,
    exif_timezone: str | None = None,
) -> CaptureTime:
    metadata = TimestampMetadata(
        exif_datetime_original=exif_datetime, exif_offset_original=exif_offset,
    )
    configured_zone = None
    if exif_timezone is not None:
        try:
            configured_zone = ZoneInfo(exif_timezone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"Unknown EXIF timezone: {exif_timezone}") from exc
    if request.captured_at is not None:
        captured = request.captured_at
        offset = captured.strftime("%z")
        zone = getattr(captured.tzinfo, "key", None) or (
            offset[:3] + ":" + offset[3:5] + (":" + offset[5:] if len(offset) > 5 else "")
        )
        return CaptureTime(captured.astimezone(timezone.utc), zone, request.timestamp_source, metadata)
    if exif_datetime is None:
        return CaptureTime(None, None, "unknown", metadata)
    try:
        naive = datetime.strptime(exif_datetime, "%Y:%m:%d %H:%M:%S")
        if exif_offset is not None:
            # Require the EXIF offset syntax, including a colon.
            import re
            if not re.fullmatch(r"[+-]\d{2}:\d{2}", exif_offset):
                raise ValueError("invalid EXIF offset")
            captured = datetime.fromisoformat(naive.isoformat() + exif_offset)
            zone = exif_offset
        elif configured_zone is not None:
            captured = naive.replace(tzinfo=configured_zone)
            alternate = naive.replace(tzinfo=configured_zone, fold=1)
            if (
                captured.utcoffset() != alternate.utcoffset()
                or captured.astimezone(timezone.utc).astimezone(configured_zone).replace(tzinfo=None) != naive
            ):
                raise ValueError("ambiguous or nonexistent local EXIF timestamp")
            zone = exif_timezone
            metadata = metadata.model_copy(update={"assumed_timezone": exif_timezone})
        else:
            metadata = metadata.model_copy(update={"notes": ["exif_timezone_missing"]})
            return CaptureTime(None, None, "unknown", metadata)
    except ValueError:
        metadata = metadata.model_copy(update={"notes": ["invalid_or_ambiguous_exif_timestamp"]})
        return CaptureTime(None, None, "unknown", metadata)
    return CaptureTime(captured.astimezone(timezone.utc), zone, "exif", metadata)
