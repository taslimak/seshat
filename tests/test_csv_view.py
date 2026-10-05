from pathlib import Path

import pandas as pd

from seshat.data.csv_view import export_csv


def test_export_csv_from_jsonl(tmp_path: Path) -> None:
    source = tmp_path / "records.jsonl"
    source.write_text('{"a":1,"b":"x"}\n{"a":2,"b":"y"}\n', encoding="utf-8")
    target = tmp_path / "records.csv"
    export_csv(source, target, limit=1)
    frame = pd.read_csv(target)
    assert frame.to_dict(orient="records") == [{"a": 1, "b": "x"}]
