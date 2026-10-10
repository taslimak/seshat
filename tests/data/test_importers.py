import json
from pathlib import Path

import pytest

from seshat.data.collectors.log_import import import_usage_log
from seshat.data.collectors.telemetry import TelemetryRecorder
from seshat.data.models import PromptRecord
from seshat.data.pricing import PriceTable
from seshat.data.privacy import RetentionPolicy
from seshat.data.schemas import ConsentStatus, LLMRequest, SourceType
from seshat.data.storage import StorageLayout
from seshat.prompts.bank import load_bank, write_prompt_file
from seshat.prompts.importer import import_prompts
from seshat.prompts.policy import load_policy
from seshat.prompts.report import bank_violations, coverage_report
from seshat.prompts.review import set_review_status
from seshat.providers.catalog import load_catalog
from tests.data.conftest import REPO_ROOT, provider_fixture

POLICY = load_policy(REPO_ROOT / "configs" / "prompt_collection.json")
DEFAULTS = {
    "source": "example_dataset",
    "source_detail": "Example corpus v1 (CC-BY-4.0), https://example.org",
    "license_id": "CC-BY-4.0",
    "commercial_use_allowed": True,
    "category": "question_answering",
    "complexity": "medium",
    "language": "en",
}


def _write(path: Path, rows: list[object]) -> Path:
    path.write_text(
        "\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    return path


# --- prompt dataset import (public_permitted_dataset) ---------------------------


def test_prompt_import_validates_dedupes_and_groups_near_duplicates(tmp_path: Path) -> None:
    bank = tmp_path / "prompts"
    bank.mkdir()
    rows = [
        {"text": "Explain how vaccines train the immune system to recognise pathogens quickly."},
        {
            "text": "Explain how vaccines train the immune system to recognise pathogens quickly."
        },  # exact dup
        {
            "text": (
                "Explain how vaccines train the immune system to recognise pathogens very quickly."
            )
        },  # near dup
        {"text": "Email me at jane@example.com about the results please."},  # PII
        {"text": "Summarize the novel in detail.", "license_id": "Proprietary-EULA"},  # license
        {"text": "Short"},  # too short
        {
            "text": "A prompt whose review status the source tries to set.",
            "review_status": "approved",
            "reviewed_by": "x",
        },
        "{broken json",
    ]
    report = import_prompts(
        _write(tmp_path / "in.jsonl", rows),
        bank_dir=bank,
        target_file=bank / "imported.jsonl",
        policy=POLICY,
        defaults=DEFAULTS,
        report_dir=tmp_path / "reports",
    )
    assert (report.imported, report.duplicates, report.near_duplicates_joined) == (3, 1, 1)
    assert report.reasons == {"invalid_json": 1, "policy_violation": 3}
    imported = load_bank(bank)[bank / "imported.jsonl"]
    assert all(p.review_status == "unreviewed" for p in imported)  # imports cannot self-approve
    near = next(p for p in imported if "very" in p.text)
    original = next(p for p in imported if p.text.endswith("quickly.") and "very" not in p.text)
    assert near.prompt_family_id == original.prompt_family_id
    rejections = Path(report.rejection_report or "").read_text()
    assert "jane@example.com" not in rejections
    again = import_prompts(
        _write(tmp_path / "in.jsonl", rows[:1]),
        bank_dir=bank,
        target_file=bank / "imported.jsonl",
        policy=POLICY,
        defaults=DEFAULTS,
        report_dir=tmp_path / "reports",
    )
    assert (again.imported, again.duplicates) == (0, 1)  # deterministic re-import


def test_prompt_import_csv_and_required_metadata(tmp_path: Path) -> None:
    bank = tmp_path / "prompts"
    bank.mkdir()
    csv_path = tmp_path / "in.csv"
    csv_path.write_text(
        "text,category,complexity\n"
        "Translate 'good morning' into French please.,translation,low\n"
        "No category given here at all,,low\n",
        encoding="utf-8",
    )
    defaults = {k: v for k, v in DEFAULTS.items() if k not in {"category", "complexity"}}
    report = import_prompts(
        csv_path,
        bank_dir=bank,
        target_file=bank / "csv.jsonl",
        policy=POLICY,
        defaults=defaults,
        report_dir=tmp_path / "r",
    )
    assert report.imported == 1 and report.reasons == {"missing_metadata": 1}


def test_bank_check_review_and_report(tmp_path: Path) -> None:
    bank = tmp_path / "prompts"
    good = PromptRecord(
        prompt_id="p1",
        prompt_family_id="f1",
        category="qa",
        text="What causes ocean tides on Earth?",
        source="hand_authored",
        license_id="project-authored",
        commercial_use_allowed=True,
    )
    twin = good.model_copy(
        update={
            "prompt_id": "p2",
            "prompt_family_id": "f2",
            "text": "What causes ocean tides on Earth ?",
        }
    )
    write_prompt_file(bank / "a.jsonl", [good, twin])
    codes = {v["code"] for v in bank_violations(bank, POLICY)}
    assert {"duplicate_content", "near_duplicates_in_different_families"} & codes
    write_prompt_file(bank / "a.jsonl", [good])
    assert bank_violations(bank, POLICY) == []
    set_review_status(bank, ["p1"], "approved", reviewer="alice", policy=POLICY)
    assert load_bank(bank)[bank / "a.jsonl"][0].reviewed_by == "alice"
    with pytest.raises(ValueError, match="Unknown prompt ids"):
        set_review_status(bank, ["nope"], "approved", reviewer="alice", policy=POLICY)
    report = coverage_report(bank, POLICY)
    assert report["approved"] == 1 and report["categories_below_min_approved"] == ["qa"]


def test_repository_prompt_bank_passes_checks() -> None:
    assert bank_violations(REPO_ROOT / "prompts", POLICY) == []


# --- usage-log import (authorized_log_import) ------------------------------------


def _log_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "request_id": "req_a",
        "timestamp_utc": "2026-10-10T12:00:00Z",  # on/after the price table effective date
        "consent_status": "granted",
        "prompt_category": "support",
        "latency_ms": 900.0,
        "request": {
            "provider": "anthropic",
            "model": "claude-opus-5-5",
            "system": "You help customers.",
            "messages": [
                {"role": "user", "content": "Where is order 123? Contact me at jane@example.com"}
            ],
            "params": {"max_output_tokens": 1024},
        },
        "response": provider_fixture("anthropic_cache_thinking.json"),
    }
    row.update(overrides)
    return row


def test_usage_log_import_normalizes_dedupes_and_protects_text(tmp_path: Path) -> None:
    root = tmp_path / "data"
    rows = [
        _log_row(),
        _log_row(),
        _log_row(
            request_id="req_b",
            response={**provider_fixture("openai_reasoning_cached.json")},
            request={
                "provider": "openai",
                "model": "gpt-6-sol",
                "messages": [{"role": "user", "content": "hi"}],
            },
        ),
        "not json",
        {"request": {}, "response": {}},
    ]
    path = _write(tmp_path / "log.jsonl", rows)
    kwargs = {
        "root": root,
        "source_type": SourceType.AUTHORIZED_LOG_IMPORT,
        "authorization_ref": "DPA-2026-01",
        "dataset_origin": "support-bot",
        "catalog": load_catalog(),
        "prices": PriceTable.load(),
        "policy": RetentionPolicy(),
    }
    report = import_usage_log(path, **kwargs)
    assert (report.imported, report.duplicates) == (2, 1)
    assert report.reasons == {"invalid_json": 1, "schema_error": 1}
    layout = StorageLayout(root)
    obs = layout.read_frame("observations").set_index("request_id")
    assert (
        obs.loc["req_a", "actual_input_tokens"] == 4096
        and obs.loc["req_a", "label_quality"] == "verified"
    )
    assert obs.loc["req_a", "calculated_cost_usd"] is not None
    assert obs.loc["req_a", "retain_until"] is not None  # log imports expire
    req = layout.read_frame("requests")
    assert req["full_prompt_storage_ref"].isna().all()  # text not kept by default
    raw = "".join(p.read_text() for p in (root / "raw" / "imports").rglob("*.jsonl"))
    assert "jane@example.com" not in raw and "Where is order" not in raw
    assert import_usage_log(path, **kwargs).duplicates == 3  # idempotent across batches


def test_usage_log_import_guards(tmp_path: Path) -> None:
    path = _write(tmp_path / "log.jsonl", [_log_row(consent_status="denied")])
    common = {
        "root": tmp_path / "d",
        "dataset_origin": "x",
        "catalog": load_catalog(),
        "prices": PriceTable.load(),
        "policy": RetentionPolicy(),
    }
    with pytest.raises(ValueError, match="authorization_ref"):
        import_usage_log(
            path, source_type=SourceType.AUTHORIZED_LOG_IMPORT, authorization_ref=" ", **common
        )
    with pytest.raises(ValueError, match="real-usage source"):
        import_usage_log(path, source_type=SourceType.TEST_FIXTURE, authorization_ref="x", **common)
    report = import_usage_log(
        path, source_type=SourceType.AUTHORIZED_LOG_IMPORT, authorization_ref="x", **common
    )
    assert report.reasons == {"normalization_error": 1}  # consent denied => not recorded


def test_imported_real_labels_need_explicit_opt_in_for_training(tmp_path: Path) -> None:
    from seshat.data.training import prepare_training_dataset
    from seshat.data.validators import validate_storage

    root = tmp_path / "data"
    fixture = provider_fixture("anthropic_cache_thinking.json")
    categories = ["support", "billing", "shipping"]
    rows = [
        _log_row(
            request_id=f"req_{i}",
            prompt_category=categories[i % 3],
            response={**fixture, "id": f"msg_{i}"},
            request={
                "provider": "anthropic",
                "model": "claude-opus-5-5",
                "messages": [{"role": "user", "content": f"Question {i} on topic {i * 7}"}],
                "params": {"max_output_tokens": 1024},
            },
        )
        for i in range(24)
    ]
    import_usage_log(
        _write(tmp_path / "log.jsonl", rows),
        root=root,
        source_type=SourceType.AUTHORIZED_LOG_IMPORT,
        authorization_ref="DPA-2026-01",
        dataset_origin="support-bot",
        catalog=load_catalog(),
        prices=PriceTable.load(),
        policy=RetentionPolicy(),
    )
    validate_storage(root)
    with pytest.raises(ValueError, match="excluded by source policy"):
        prepare_training_dataset(root, "1")
    with pytest.raises(ValueError, match="include_sources"):
        prepare_training_dataset(root, "1", include_sources=("test_fixture",))
    manifest = prepare_training_dataset(root, "1", include_sources=("authorized_log_import",))
    assert manifest["label_origin"] == "real"
    assert manifest["source_types"] == ["authorized_log_import"]
    assert manifest["rows"] == 24


# --- consented telemetry ---------------------------------------------------------


def _request() -> LLMRequest:
    return LLMRequest.model_validate(
        {
            "provider": "openai",
            "model": "gpt-6-sol",
            "messages": [{"role": "user", "content": "Secret plan: launch on Friday"}],
            "params": {"max_output_tokens": 500},
        }
    )


def test_telemetry_is_opt_in_and_keeps_text_only_with_consent(tmp_path: Path) -> None:
    disabled = TelemetryRecorder(tmp_path / "off")
    assert disabled.start(_request(), request_id="r0") is None
    assert disabled.flush() == 0
    recorder = TelemetryRecorder(tmp_path / "on", enabled=True, capture_prompt_text=True)
    unknown = recorder.start(_request(), request_id="r1")  # consent unknown
    recorder.finish(unknown, provider_fixture("openai_reasoning_cached.json"), latency_ms=10)
    granted = recorder.start(_request(), request_id="r2", consent_status=ConsentStatus.GRANTED)
    recorder.finish(
        granted,
        {**provider_fixture("openai_reasoning_cached.json"), "id": "resp_other"},
        latency_ms=10,
    )
    assert recorder.start(_request(), request_id="r3", consent_status=ConsentStatus.DENIED) is None
    assert recorder.flush() == 2
    req = StorageLayout(tmp_path / "on").read_frame("requests").set_index("request_id")
    assert req.loc["r1", "full_prompt_storage_ref"] is None
    assert req.loc["r2", "full_prompt_storage_ref"] is not None
    assert set(req["source_type"]) == {"consented_api_telemetry"}


def test_usage_before_price_effective_date_has_no_cost(tmp_path: Path) -> None:
    path = _write(tmp_path / "old.jsonl", [_log_row(timestamp_utc="2026-01-01T00:00:00Z")])
    import_usage_log(
        path,
        root=tmp_path / "d",
        source_type=SourceType.AUTHORIZED_LOG_IMPORT,
        authorization_ref="x",
        dataset_origin="x",
        catalog=load_catalog(),
        prices=PriceTable.load(),
        policy=RetentionPolicy(),
    )
    obs = StorageLayout(tmp_path / "d").read_frame("observations").iloc[0]
    assert (
        obs["calculated_cost_usd"] is None and obs["cost_method"] == "unavailable"
    )  # never back-dated
