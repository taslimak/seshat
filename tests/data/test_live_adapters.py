"""Live adapters: request translation and response handling with fake SDK clients.

Ordinary tests never touch the network. Real calls run only when
SESHAT_LIVE_TESTS=1 and the provider's API key are set (see bottom of file).
"""

import os
from types import SimpleNamespace
from typing import Any

import pytest

from seshat.data.schemas import LLMRequest, PreflightMethod, ResponseStatus
from seshat.providers import anthropic as anthropic_adapter
from seshat.providers import gemini as gemini_adapter
from seshat.providers import openai as openai_adapter
from seshat.providers.base import ErrorCategory, ProviderError
from tests.data.conftest import provider_fixture


def _req(provider: str, model: str, **params: Any) -> LLMRequest:
    return LLMRequest.model_validate(
        {
            "provider": provider,
            "model": model,
            "system": "Be concise.",
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "Weather in Oslo?"},
            ],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Weather",
                    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
                }
            ],
            "params": {"max_output_tokens": 800, **params},
            "provider_options": {"_mock_category": "internal", "metadata": {"user_id": "hash"}},
        }
    )


class _Dumpable(SimpleNamespace):
    def __init__(self, payload: dict[str, Any], **kw: Any) -> None:
        super().__init__(**kw)
        self._payload = payload

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        return self._payload


def test_anthropic_translation() -> None:
    kwargs = anthropic_adapter.build_request_kwargs(
        _req(
            "anthropic",
            "claude-opus-5-5",
            effort="high",
            thinking_mode="adaptive",
            tool_choice="auto",
            parallel_tool_calls=False,
            cache={"enabled": True, "ttl": "1h"},
        )
    )
    assert kwargs["max_tokens"] == 800 and kwargs["system"] == "Be concise."
    assert [m["role"] for m in kwargs["messages"]] == ["user", "assistant", "user"]
    assert kwargs["output_config"] == {"effort": "high"} and kwargs["thinking"] == {
        "type": "adaptive"
    }
    assert kwargs["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert kwargs["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert kwargs["tools"][0]["input_schema"]["properties"]["city"]["type"] == "string"
    assert "_mock_category" not in kwargs and kwargs["metadata"] == {"user_id": "hash"}
    assert "temperature" not in kwargs
    with pytest.raises(ProviderError):
        anthropic_adapter.build_request_kwargs(
            _req("anthropic", "x", thinking_mode="enabled")
        )  # budget required


def test_openai_translation() -> None:
    kwargs = openai_adapter.build_request_kwargs(
        _req(
            "openai",
            "gpt-6-sol",
            effort="low",
            verbosity="low",
            response_format="json",
            tool_choice="any",
            cache={"enabled": True, "key": "fam1"},
        )
    )
    assert (
        kwargs["instructions"] == "Be concise."
        and kwargs["max_output_tokens"] == 800
        and kwargs["store"] is False
    )
    assert kwargs["reasoning"] == {"effort": "low"}
    assert kwargs["text"] == {"verbosity": "low", "format": {"type": "json_object"}}
    assert kwargs["tool_choice"] == "required" and kwargs["tools"][0]["type"] == "function"
    assert kwargs["prompt_cache_key"] == "fam1"


def test_gemini_translation() -> None:
    config = gemini_adapter.build_config(
        _req(
            "gemini",
            "gemini-3.8-flash",
            effort="low",
            temperature=0.2,
            seed=7,
            tool_choice="get_weather",
        )
    )
    contents = gemini_adapter.build_contents(_req("gemini", "gemini-3.8-flash"))
    assert [c["role"] for c in contents] == ["user", "model", "user"]
    assert config["thinking_config"] == {"thinking_level": "LOW"}
    assert config["system_instruction"] == "Be concise." and config["seed"] == 7
    assert config["automatic_function_calling"] == {"disable": True}
    assert config["tool_config"]["function_calling_config"] == {
        "mode": "ANY",
        "allowed_function_names": ["get_weather"],
    }


def test_adapters_with_fake_clients() -> None:
    created: dict[str, Any] = {}

    class Messages:
        def create(self, **kw: Any) -> _Dumpable:
            created.update(kw)
            return _Dumpable(provider_fixture("anthropic_cache_thinking.json"))

        def count_tokens(self, **kw: Any) -> SimpleNamespace:
            assert "max_tokens" not in kw
            return SimpleNamespace(input_tokens=4100)

    adapter = anthropic_adapter.AnthropicAdapter(client=SimpleNamespace(messages=Messages()))
    req = _req("anthropic", "claude-opus-5-5")
    pre = adapter.count_input_tokens(req)
    assert (
        pre.method == PreflightMethod.PROVIDER_NATIVE and pre.input_tokens == 4100 and pre.complete
    )
    response = adapter.execute_request(req, timeout_seconds=5)
    assert created["timeout"] == 5 and response.status == ResponseStatus.COMPLETED
    assert adapter.normalize_usage(response).actual_input_tokens == 4096

    class Responses:
        input_tokens = SimpleNamespace(count=lambda **kw: SimpleNamespace(input_tokens=2048))

        def create(self, **kw: Any) -> _Dumpable:
            return _Dumpable(provider_fixture("openai_truncated.json"))

    oa = openai_adapter.OpenAIAdapter(client=SimpleNamespace(responses=Responses()))
    assert oa.count_input_tokens(_req("openai", "gpt-6-sol")).input_tokens == 2048
    out = oa.execute_request(_req("openai", "gpt-6-sol"), timeout_seconds=5)
    assert out.is_truncated and oa.normalize_usage(out).actual_visible_output_tokens == 0

    models = SimpleNamespace(
        count_tokens=lambda **kw: SimpleNamespace(total_tokens=None),
        generate_content=lambda **kw: _Dumpable(provider_fixture("gemini_thinking_cached.json")),
    )
    ge = gemini_adapter.GeminiAdapter(client=SimpleNamespace(models=models))
    pre = ge.count_input_tokens(_req("gemini", "gemini-3.8-flash"))
    assert pre.input_tokens is None and pre.error  # missing count => error, never 0
    assert (
        ge.normalize_usage(
            ge.execute_request(_req("gemini", "gemini-3.8-flash"), timeout_seconds=5)
        ).actual_output_tokens
        == 520
    )


def test_error_classification() -> None:
    assert (
        anthropic_adapter.classify_exception(
            TypeError("Could not resolve authentication method")
        ).category
        == ErrorCategory.AUTHENTICATION
    )
    assert (
        openai_adapter.classify_exception(ValueError("bad param")).category
        == ErrorCategory.INVALID_REQUEST
    )
    weird = gemini_adapter.classify_exception(OSError("reset"))
    assert weird.category == ErrorCategory.UNKNOWN and weird.outcome_uncertain
    rate = ProviderError(ErrorCategory.RATE_LIMIT, "429")
    assert rate.retryable and not rate.outcome_uncertain
    timeout = ProviderError(ErrorCategory.TIMEOUT, "t")
    assert timeout.outcome_uncertain and not timeout.retryable


LIVE = os.environ.get("SESHAT_LIVE_TESTS") == "1"


@pytest.mark.skipif(
    not (LIVE and os.environ.get("ANTHROPIC_API_KEY")),
    reason="set SESHAT_LIVE_TESTS=1 and ANTHROPIC_API_KEY",
)
def test_live_anthropic_count_tokens() -> None:  # free endpoint: no generation is billed
    adapter = anthropic_adapter.AnthropicAdapter()
    assert (adapter.count_input_tokens(_req("anthropic", "claude-haiku-5-5")).input_tokens or 0) > 0


@pytest.mark.skipif(
    not (LIVE and os.environ.get("OPENAI_API_KEY")),
    reason="set SESHAT_LIVE_TESTS=1 and OPENAI_API_KEY",
)
def test_live_openai_count_tokens() -> None:
    adapter = openai_adapter.OpenAIAdapter()
    assert (adapter.count_input_tokens(_req("openai", "gpt-6-luna")).input_tokens or 0) > 0


@pytest.mark.skipif(
    not (LIVE and (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))),
    reason="set SESHAT_LIVE_TESTS=1 and GEMINI_API_KEY",
)
def test_live_gemini_count_tokens() -> None:
    adapter = gemini_adapter.GeminiAdapter()
    assert (
        adapter.count_input_tokens(_req("gemini", "gemini-3.5-flash-lite")).input_tokens or 0
    ) > 0
