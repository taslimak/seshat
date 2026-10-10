import json
from pathlib import Path

import pytest
import yaml

from seshat.data.cli import main
from tests.data.conftest import REPO_ROOT


def _config(tmp_path: Path, **changes: object) -> Path:
    raw = yaml.safe_load((REPO_ROOT / "configs" / "data_collection.yaml").read_text())
    raw["prompts"]["generate"]["count"] = 34
    raw["budget"]["max_requests"] = 60
    raw["storage"]["root"] = str(tmp_path / "data")
    raw.update(changes)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def _json(capsys: pytest.CaptureFixture[str]) -> object:
    return json.loads(capsys.readouterr().out)


def test_offline_cli_workflow(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = str(tmp_path / "data")
    assert (
        main(["generate", "--count", "100", "--seed", "42", "--output", str(tmp_path / "p.jsonl")])
        == 0
    )
    assert _json(capsys)["written"] == 100  # type: ignore[index]
    cfg = str(_config(tmp_path))
    assert main(["collect", "--config", cfg, "--dry-run"]) == 0
    assert _json(capsys)["collection"]["mode"] == "dry_run"  # type: ignore[index]
    assert main(["collect", "--config", cfg]) == 0
    out = _json(capsys)
    assert (
        out["collection"]["status_counts"] == {"completed": 60}
        and out["normalized"]["observations"] == 60
    )  # type: ignore[index]
    assert main(["validate", "--root", root]) == 0
    capsys.readouterr()
    assert main(["export", "--format", "csv", "--root", root]) == 0
    assert Path(_json(capsys)["csv"]).exists()  # type: ignore[index]
    assert main(["prepare-training", "--root", root, "--version", "1"]) == 2  # real labels required
    assert "label-origin synthetic" in capsys.readouterr().err
    assert (
        main(["prepare-training", "--root", root, "--version", "1", "--label-origin", "synthetic"])
        == 0
    )
    capsys.readouterr()
    assert main(["stats", "--root", root]) == 0
    assert "mock-openai/mock-gpt" in _json(capsys)["by_provider_model"]  # type: ignore[index]
    assert main(["predict", "--root", root]) == 0 and main(["compare", "--root", root]) == 0


def test_live_plans_default_to_dry_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = str(
        _config(
            tmp_path,
            targets=[
                {"provider": "anthropic", "model": "claude-haiku-5-5", "max_output_tokens": [300]}
            ],
        )
    )
    assert main(["collect", "--config", cfg]) == 0
    out = _json(capsys)
    assert out["collection"]["mode"] == "dry_run" and "execute-live" in out["note"]  # type: ignore[index]
    assert main(["collect", "--config", cfg, "--execute-live"]) == 2
    assert "max-cost-usd" in capsys.readouterr().err


def test_prompt_bank_commands(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["prompts", "check", "--bank-dir", str(REPO_ROOT / "prompts")]) == 0
    assert _json(capsys)["ok"] is True  # type: ignore[index]
    assert main(["prompts", "report", "--bank-dir", str(REPO_ROOT / "prompts")]) == 0
    assert _json(capsys)["prompts"] >= 36  # type: ignore[index]
