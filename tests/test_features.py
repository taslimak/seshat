from seshat.features import extract_prompt_features


def test_extract_prompt_features() -> None:
    features = extract_prompt_features("Hello world!\nValue: 42")

    assert features.char_count == 22
    assert features.word_count == 4
    assert features.line_count == 2
    assert features.digit_count == 2
    assert features.sentence_count == 1
