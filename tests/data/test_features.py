from seshat.data.features import FEATURE_DEFINITIONS, RequestFeatures, extract_request_features
from seshat.data.features.extract import explicit_length_hint, requested_output_format
from seshat.data.schemas import LLMRequest


def _req(content: object = "Hello there", **kw: object) -> LLMRequest:
    payload: dict[str, object] = {
        "provider": "mock-openai",
        "model": "mock-gpt",
        "messages": [{"role": "user", "content": content}],
    }
    payload.update(kw)
    return LLMRequest.model_validate(payload)


def test_counts_cover_system_and_all_messages() -> None:
    req = _req(
        system="Sys prompt.",
        messages=[
            {"role": "user", "content": "First question?"},
            {"role": "assistant", "content": "An answer."},
            {"role": "user", "content": "Follow-up ✓"},
        ],
    )
    f = extract_request_features(req)
    text = "Sys prompt.\n\nFirst question?\n\nAn answer.\n\nFollow-up ✓"
    assert f.prompt_char_count == len(text)
    assert f.prompt_byte_count == len(text.encode())
    assert f.system_prompt_char_count == len("Sys prompt.")
    assert (f.message_count, f.conversation_turn_count, f.context_message_count) == (3, 2, 2)
    assert f.role_sequence == ["system", "user", "assistant", "user"]
    assert f.current_user_message_char_count == len("Follow-up ✓")
    assert f.context_character_count == len("Sys prompt.") + len("First question?") + len(
        "An answer."
    )
    assert f.conversation_hash is not None


def test_features_are_deterministic_and_hash_tracks_content() -> None:
    a, b = (
        extract_request_features(_req(), hash_key="k"),
        extract_request_features(_req(), hash_key="k"),
    )
    assert a == b
    assert extract_request_features(_req("Hello there!"), hash_key="k").prompt_hash != a.prompt_hash
    assert a.conversation_hash is None


def test_indicators() -> None:
    f = extract_request_features(
        _req(
            "Fix:\n```python\ndef f(x):\n    return x\n```\n"
            'Return {"a": 1}. Solve $x^2 = 4$.\n## Notes\n- item'
        )
    )
    assert f.contains_code and f.contains_json and f.contains_mathematics and f.contains_markdown
    plain = extract_request_features(_req("Tell me a story about a cat"))
    assert not (
        plain.contains_code
        or plain.contains_json
        or plain.contains_mathematics
        or plain.contains_markdown
    )


def test_output_format_and_length_hints() -> None:
    assert requested_output_format(_req("Reply as JSON")) == "json"
    assert requested_output_format(_req("hi", params={"response_format": "json_schema"})) == "json"
    assert requested_output_format(_req("Write a Python function that sorts")) == "code"
    assert requested_output_format(_req("Tell me about cats")) == "unspecified"
    assert explicit_length_hint("Answer in under 120 words.") == "max_words:120"
    assert explicit_length_hint("Answer in three bullet points") == "count_bullet_points:3"
    assert explicit_length_hint("Write a one-sentence summary") == "count_sentences:1"
    assert explicit_length_hint("Explain in detail") == "qualitative:detailed"
    assert explicit_length_hint("What is 2+2") is None


def test_tools_config_and_multimodal_metadata() -> None:
    req = _req(
        [
            {"type": "text", "text": "Describe these"},
            {
                "type": "image",
                "mime_type": "image/png",
                "byte_size": 1000,
                "width": 10,
                "height": 20,
            },
            {"type": "image", "mime_type": "image/jpeg"},
            {"type": "document", "mime_type": "application/pdf", "page_count": 3},
        ],
        tools=[
            {"name": "search", "description": "d"},
            {"name": "web_search", "type": "web_search"},
        ],
        params={
            "max_output_tokens": 100,
            "stop_sequences": ["END"],
            "cache": {"enabled": True, "ttl": "1h"},
            "effort": "high",
        },
        provider_options={"metadata": {"k": "v"}},
    )
    f = extract_request_features(req, context_window_limit=1000)
    assert f.tool_count == 2 and f.tool_schema_bytes > 0 and f.tool_schema_hash
    assert f.enabled_tool_types == ["function", "web_search"]
    assert (f.image_count, f.image_total_bytes, f.document_count, f.document_page_count) == (
        2,
        1000,
        1,
        3,
    )
    assert f.audio_duration_seconds is None  # no audio: unknown, not 0
    assert f.modality_types == ["document", "image", "text"]
    assert (f.stop_sequence_count, f.caching_enabled, f.requested_cache_ttl) == (1, True, "1h")
    assert f.context_window_limit == 1000 and f.provider_settings == {"metadata": {"k": "v"}}
    assert f.temperature is None  # unset stays null


def test_every_computed_feature_is_defined_or_configuration() -> None:
    config_fields = {
        "max_output_tokens",
        "requested_effort",
        "resolved_effort",
        "thinking_mode",
        "thinking_budget_tokens",
        "temperature",
        "top_p",
        "top_k",
        "seed",
        "response_format",
        "verbosity",
        "tool_choice",
        "parallel_tool_calls",
        "streaming_enabled",
        "requested_cache_ttl",
        "service_tier",
        "region",
        "routing_mode",
        "provider_settings",
        "context_window_limit",
        "image_metadata",
        "attachment_metadata",
    }
    undocumented = set(RequestFeatures.model_fields) - set(FEATURE_DEFINITIONS) - config_fields
    assert not undocumented, undocumented
