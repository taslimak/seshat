"""Explicit provenance labels for every dataset record.

Seshat must never treat a synthetic outcome as a real one. Every stored record
therefore carries a ``Provenance`` that states whether its values were observed from
a real provider (``REAL``) or generated (``SYNTHETIC``), and where it came from.

Rules enforced here:

* Sources that can only produce generated data (the mock adapter, simulations) are
  always ``SYNTHETIC``. Labelling them ``REAL`` is a validation error.
* Provenance is never inferred as ``REAL`` by default. Ingestion requires it to be
  present on the record or supplied explicitly by the caller.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, model_validator

PIPELINE_VERSION = "0.2.0"


class DataOrigin(StrEnum):
    """Whether a record's values were observed from a real provider or generated."""

    REAL = "real"
    SYNTHETIC = "synthetic"


class DataSource(StrEnum):
    """Where a record entered Seshat."""

    CONTROLLED_EXPERIMENT = "controlled_experiment"
    PRODUCTION_LOG = "production_log"
    IMPORT = "import"
    MOCK_ADAPTER = "mock_adapter"
    SIMULATION = "simulation"


SYNTHETIC_ONLY_SOURCES = frozenset({DataSource.MOCK_ADAPTER, DataSource.SIMULATION})


class Provenance(BaseModel):
    """Origin label attached to every request-feature and observed-usage record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    data_origin: DataOrigin
    source: DataSource
    generator: str | None = None
    experiment_id: str | None = None
    pipeline_version: str = PIPELINE_VERSION

    @model_validator(mode="after")
    def synthetic_sources_stay_synthetic(self) -> Provenance:
        if self.source in SYNTHETIC_ONLY_SOURCES and self.data_origin != DataOrigin.SYNTHETIC:
            raise ValueError(
                f"source={self.source.value} only produces generated data; "
                "data_origin must be 'synthetic'"
            )
        return self
