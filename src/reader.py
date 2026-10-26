"""Reader boundary: no labels, past readings, or evaluation metadata enter it."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from schemas import ReaderOutput, TargetType


@dataclass(frozen=True)
class ReaderInput:
    image_path: Path
    target_type: TargetType
    question: str | None


class Reader(Protocol):
    mode: Literal["mock", "inference"]
    reader_id: str

    def read(self, request: ReaderInput) -> ReaderOutput: ...


class ReaderError(RuntimeError):
    pass


class MockReader:
    """Exercise plumbing only; never inspect pixels or fabricate measurements."""
    mode: Literal["mock"] = "mock"
    reader_id = "mock-unknown-v1"

    def read(self, request: ReaderInput) -> ReaderOutput:
        issues = ["mock_not_a_measurement"]
        if not request.question:
            issues.append("target_selection_unverified")
        return ReaderOutput(issue_codes=issues)
