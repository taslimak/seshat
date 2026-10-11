"""Test helper: mock-provider collections relabelled as *real-shaped* rows.

Only for unit tests of readiness/evaluation arithmetic inside ``tmp_path``.
Never use this outside tests: relabelled fixtures are not real usage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml

from seshat.data.cli import main as cli_main


def mock_collection(
    root: Path, *, experiment_id: str = "exp_a", count: int = 120, seed: int = 5, **extra: Any
) -> Path:
    """Run an offline mock collection into ``root/data``; returns the data directory."""
    root.mkdir(parents=True, exist_ok=True)
    data = root / "data"
    config = {
        "experiment_id": experiment_id,
        "seed": seed,
        "prompts": {"generate": {"count": count, "seed": seed}},
        "targets": [
            {
                "provider": "mock-openai",
                "model": "mock-gpt",
                "efforts": ["low", "high"],
                "max_output_tokens": [512],
            }
        ],
        "sampling": {"configs_per_prompt": 1},
        "repetitions": {"fraction": 0.0, "count": 1},
        "budget": {"max_requests": count},
        "execution": {"concurrency": 1, "backoff_base_seconds": 0, "backoff_max_seconds": 0},
        "storage": {"root": str(data)},
        **extra,
    }
    path = root / f"{experiment_id}.yml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert cli_main(["collect", "--config", str(path)]) == 0
    return data


def relabel_as_real(data: Path) -> None:
    """Rewrite normalized rows to controlled-experiment provenance (tests only)."""
    import pyarrow as pa
    import pyarrow.compute as pc

    for table in ("requests", "observations"):
        for part in sorted((data / "normalized" / table).rglob("part-*.parquet")):
            frame = pq.read_table(part)
            n = frame.num_rows
            frame = frame.set_column(
                frame.schema.get_field_index("source_type"),
                "source_type",
                pa.array(["controlled_experiment"] * n),
            )
            # An unregistered provider name, so validation does not flag mock provenance.
            frame = frame.set_column(
                frame.schema.get_field_index("provider"), "provider", pa.array(["relabelled"] * n)
            )
            if table == "observations":
                truncated = pc.fill_null(frame["is_truncated"], False).to_pylist()
                quality = ["censored" if t else "verified" for t in truncated]
                frame = frame.set_column(
                    frame.schema.get_field_index("label_quality"),
                    "label_quality",
                    pa.array(quality),
                )
            pq.write_table(frame, part)
    assert cli_main(["validate", "--root", str(data)]) == 0
