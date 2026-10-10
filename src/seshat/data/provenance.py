"""Provenance helpers. The canonical label is ``SourceType`` (seshat.data.schemas.enums).

``DataOrigin`` is the coarse real/synthetic split used by label policies: a record's
outcome is ``real`` only when it comes from a real-usage source *and* its label is
not synthetic.
"""

from __future__ import annotations

from enum import StrEnum

from seshat.data.schemas.enums import REAL_USAGE_SOURCES, SourceType

PIPELINE_VERSION = "1.0.0"


class DataOrigin(StrEnum):
    REAL = "real"
    SYNTHETIC = "synthetic"


def data_origin(source_type: SourceType | str, label_quality: str | None = None) -> DataOrigin:
    source = SourceType(source_type)
    if source in REAL_USAGE_SOURCES and label_quality != "synthetic":
        return DataOrigin.REAL
    return DataOrigin.SYNTHETIC
