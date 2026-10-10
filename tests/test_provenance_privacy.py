from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from seshat.data.privacy import PromptHashScheme, RetentionPolicy, hash_prompt, redact_text
from seshat.data.provenance import DataOrigin, DataSource, Provenance
from seshat.providers.registry import provenance_for_provider


@pytest.mark.parametrize("source", [DataSource.MOCK_ADAPTER, DataSource.SIMULATION])
def test_generated_sources_cannot_be_labelled_real(source: DataSource) -> None:
    with pytest.raises(ValidationError, match="must be 'synthetic'"):
        Provenance(data_origin=DataOrigin.REAL, source=source)


def test_mock_adapter_results_are_synthetic_and_real_adapters_are_real() -> None:
    assert provenance_for_provider("mock").data_origin == DataOrigin.SYNTHETIC
    assert provenance_for_provider("mock").source == DataSource.MOCK_ADAPTER
    openai = provenance_for_provider("openai", "exp_1")
    assert openai.data_origin == DataOrigin.REAL
    assert openai.source == DataSource.CONTROLLED_EXPERIMENT
    assert openai.experiment_id == "exp_1"


def test_keyed_hash_differs_from_plain_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SESHAT_PROMPT_HASH_KEY", raising=False)
    plain, plain_scheme = hash_prompt("hello")
    keyed, keyed_scheme = hash_prompt("hello", key="secret")
    assert plain_scheme == PromptHashScheme.SHA256
    assert keyed_scheme == PromptHashScheme.HMAC_SHA256
    assert plain != keyed
    monkeypatch.setenv("SESHAT_PROMPT_HASH_KEY", "secret")
    assert hash_prompt("hello") == (keyed, keyed_scheme)


def test_redaction_scrubs_secrets_and_caps_length() -> None:
    text = (
        "user jane.doe@example.com sent Authorization: Bearer abc.def-123 "
        "with key sk-ABCDEFGHIJKLMNOP and card 4111111111111111"
    )
    redacted = redact_text(text)
    assert redacted is not None
    secrets = ("jane.doe@example.com", "abc.def-123", "sk-ABCDEFGHIJKLMNOP", "4111111111111111")
    for secret in secrets:
        assert secret not in redacted
    assert len(redact_text("x" * 1000) or "") == 500
    assert redact_text(None) is None


def test_retention_depends_on_origin_and_source() -> None:
    policy = RetentionPolicy(production_log_days=30, import_days=7)
    now = datetime(2026, 1, 1, tzinfo=UTC)

    def until(origin: DataOrigin, source: DataSource) -> datetime | None:
        return policy.retain_until(Provenance(data_origin=origin, source=source), now)

    assert until(DataOrigin.REAL, DataSource.PRODUCTION_LOG) == now + timedelta(days=30)
    assert until(DataOrigin.REAL, DataSource.IMPORT) == now + timedelta(days=7)
    assert until(DataOrigin.REAL, DataSource.CONTROLLED_EXPERIMENT) is None
    assert until(DataOrigin.SYNTHETIC, DataSource.SIMULATION) is None
    with pytest.raises(ValueError):
        RetentionPolicy(production_log_days=0)
