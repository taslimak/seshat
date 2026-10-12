"""Guards on reviewed batch configs before a paid workflow run (no API calls)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts import use_batch_config
from seshat.data.generators import generate_prompts

BATCH = Path("configs/experiments/batches")


def _write(name: str, **overrides: Any) -> Path:
    config: dict[str, Any] = {
        "experiment_id": "adaptive_x",
        "prompts": {"generate": {"count": 34, "seed": 1}},
        "targets": [{"provider": "openai", "model": "gpt-6.1-sol", "max_output_tokens": [1024]}],
        "budget": {"max_requests": 10, "max_cost_usd": "2.00"},
        **overrides,
    }
    BATCH.mkdir(parents=True, exist_ok=True)
    path = BATCH / name
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


@pytest.fixture
def tracked(monkeypatch: pytest.MonkeyPatch) -> None:
    real_run = subprocess.run

    def fake_run(args: list[str], **kwargs: Any) -> Any:
        if args[:2] == ["git", "ls-files"]:
            return subprocess.CompletedProcess(args, 0)
        return real_run(args, **kwargs)

    monkeypatch.setattr(use_batch_config.subprocess, "run", fake_run)


@pytest.fixture
def cleanup() -> Any:
    created: list[Path] = []
    yield created
    for path in created:
        path.unlink(missing_ok=True)


def test_untracked_config_is_refused(cleanup: list[Path]) -> None:
    path = _write("_test_untracked.yaml")
    cleanup.append(path)
    with pytest.raises(ValueError, match="not committed"):
        use_batch_config.prepare(
            str(path), provider="openai", model="gpt-6.1-sol", run_id="1", data_root="data"
        )


def test_config_outside_batch_dir_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "x.yaml"
    path.write_text("{}")
    with pytest.raises(ValueError, match="under configs/experiments/batches"):
        use_batch_config.prepare(
            str(path), provider="openai", model="gpt-6.1-sol", run_id="1", data_root="data"
        )


def test_budget_and_target_guards(tracked: None, cleanup: list[Path]) -> None:
    over = _write("_test_over.yaml", budget={"max_requests": 10, "max_cost_usd": "2.50"})
    cleanup.append(over)
    with pytest.raises(ValueError, match="max_cost_usd"):
        use_batch_config.prepare(
            str(over), provider="openai", model="gpt-6.1-sol", run_id="1", data_root="data"
        )
    ok = _write("_test_ok.yaml")
    cleanup.append(ok)
    with pytest.raises(ValueError, match="workflow selected"):
        use_batch_config.prepare(
            str(ok), provider="openai", model="gpt-6-astra", run_id="1", data_root="data"
        )
    config = use_batch_config.prepare(
        str(ok), provider="openai", model="gpt-6.1-sol", run_id="987", data_root="data"
    )
    assert config["experiment_id"] == "adaptive_x_r987"
    assert Path(str(config["storage"]["root"])).is_absolute()  # type: ignore[index]


def test_reasoning_prompts_are_not_one_family() -> None:
    families: dict[str, set[str]] = {}
    for prompt in generate_prompts(340, 3):
        families.setdefault(prompt.category, set()).add(prompt.prompt_family_id)
    assert len(families["reasoning"]) > 5
    assert len(families["mathematics"]) > 5
    assert len(families["classification"]) > 3


def test_whitespace_is_tolerated_and_missing_files_are_explained(tracked: None) -> None:
    with pytest.raises(ValueError, match="does not exist on this branch"):
        use_batch_config.prepare(
            "  configs/experiments/batches/_missing.yaml ",
            provider="openai",
            model="gpt-6.1-sol",
            run_id="1",
            data_root="data",
        )


def test_batches_that_cannot_finish_in_time_are_refused(tracked: None, cleanup: list[Path]) -> None:
    slow = _write(
        "_test_slow.yaml",
        budget={"max_requests": 400, "max_cost_usd": "2.00"},
        execution={"rate_limits": {"openai": {"requests_per_minute": 2}}},
    )
    cleanup.append(slow)
    with pytest.raises(ValueError, match="over the 90 min"):
        use_batch_config.prepare(
            str(slow), provider="openai", model="gpt-6.1-sol", run_id="1", data_root="data"
        )


def test_shadow_batches_need_native_preflight_and_a_verified_artifact(
    tracked: None, cleanup: list[Path], tmp_path: Path
) -> None:
    import hashlib

    artifact = tmp_path / "forecaster.json"
    artifact.write_text('{"predictor_version": "v"}')
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    shadow = {
        "artifact_ref": "models/v/forecaster.json",
        "artifact_sha256": digest,
        "predictor_version": "v",
        "calibration_version": "c",
        "dataset_version": "d",
    }
    approx = _write(
        "_test_shadow_approx.yaml", shadow=shadow, execution={"preflight": "approximate"}
    )
    cleanup.append(approx)
    with pytest.raises(ValueError, match="preflight: native"):
        use_batch_config.prepare(
            str(approx),
            provider="openai",
            model="gpt-6.1-sol",
            run_id="1",
            data_root="data",
            shadow_artifact=str(artifact),
        )
    native = _write("_test_shadow_native.yaml", shadow=shadow, execution={"preflight": "native"})
    cleanup.append(native)
    with pytest.raises(ValueError, match="no paid request"):
        use_batch_config.prepare(
            str(native), provider="openai", model="gpt-6.1-sol", run_id="1", data_root="data"
        )
    other = tmp_path / "other.json"
    other.write_text("{}")
    with pytest.raises(ValueError, match="digest"):
        use_batch_config.prepare(
            str(native),
            provider="openai",
            model="gpt-6.1-sol",
            run_id="1",
            data_root="data",
            shadow_artifact=str(other),
        )
    config = use_batch_config.prepare(
        str(native),
        provider="openai",
        model="gpt-6.1-sol",
        run_id="1",
        data_root="data",
        shadow_artifact=str(artifact),
    )
    assert config["shadow"]["artifact_path"] == str(artifact.resolve())  # type: ignore[index]


def test_reviewed_batch_is_a_native_shadow_run() -> None:
    config = yaml.safe_load(
        Path("configs/experiments/batches/adaptive_20261011b.yaml").read_text(encoding="utf-8")
    )
    assert config["execution"]["preflight"] == "native"
    assert config["shadow"]["predictor_version"] == "cln-b0d10f58ce"
    assert config["shadow"]["calibration_version"] == "cal-2fe70064b6"
    assert config["shadow"]["dataset_version"] == "ds-8def61fb7ca4"
    assert config["shadow"]["required"] is True
    assert "artifact_path" not in config["shadow"]  # never committed
    assert config["budget"]["max_cost_usd"] == "2.00"
