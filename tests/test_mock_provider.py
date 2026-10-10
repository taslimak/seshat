from seshat.providers.mock import MockAdapter


def _call(adapter: MockAdapter) -> tuple[str | None, int | None]:
    result = adapter.execute(
        prompt="Explain entropy briefly.",
        model="mock-v1",
        effort="low",
        max_output_tokens=512,
        temperature=None,
    )
    return result.provider_request_id, result.usage.output_tokens


def test_identical_requests_get_distinct_request_ids_but_same_usage() -> None:
    adapter = MockAdapter()
    first_id, first_tokens = _call(adapter)
    second_id, second_tokens = _call(adapter)
    assert first_id != second_id
    assert first_tokens == second_tokens


def test_request_ids_are_unique_across_adapter_instances() -> None:
    assert _call(MockAdapter())[0] != _call(MockAdapter())[0]
