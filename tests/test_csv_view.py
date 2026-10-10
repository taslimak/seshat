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


def test_export_csv_from_store_directory_neutralizes_formulas(tmp_path: Path) -> None:
    table = tmp_path / "observed_usage"
    table.mkdir()
    pd.DataFrame({"id": ["a"], "note": ['=HYPERLINK("x")']}).to_parquet(table / "part-1.parquet")
    pd.DataFrame({"id": ["b"], "note": ["plain"]}).to_parquet(table / "part-2.parquet")
    target = export_csv(table, tmp_path / "out.csv", columns=["id", "note"])
    frame = pd.read_csv(target)
    assert frame["id"].tolist() == ["a", "b"]
    assert frame["note"].tolist() == ['\'=HYPERLINK("x")', "plain"]
