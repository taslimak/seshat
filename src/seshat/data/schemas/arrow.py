"""Arrow/Parquet schemas derived from the canonical Pydantic records.

Type mapping (all columns nullable):

=====================================  ==========================================
Pydantic annotation                    Arrow type
=====================================  ==========================================
str, StrEnum, Literal[str]             string
int / float / bool                     int64 / float64 / bool
AwareDatetime                          timestamp[us, tz=UTC]
date                                   date32
Decimal                                decimal128(24, 12)
UUID                                   string (canonical hyphenated form)
list[str]                              list<string>
list[BaseModel]                        list<struct<...>>
dict[str, Any]                         string holding canonical JSON (``*_json``)
=====================================  ==========================================

``dict`` fields are stored as JSON strings in a column named ``<field>_json`` so the
Parquet files stay readable by any tool; ``from_table`` restores them.
"""

from __future__ import annotations

import datetime as dt
import json
import types
import typing
import uuid
from decimal import Decimal
from enum import Enum
from typing import Any, Literal, TypeVar

import pyarrow as pa
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)
DECIMAL_TYPE = pa.decimal128(24, 12)
JSON_SUFFIX = "_json"


def _strip_optional(annotation: Any) -> Any:
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        args = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
        if all(arg in (str, int, float, bool) for arg in args):
            return str  # scalar unions are stored as strings
        return args[0]
    return annotation


def _arrow_type(annotation: Any) -> pa.DataType:
    annotation = _strip_optional(annotation)
    origin = typing.get_origin(annotation)
    if origin is list:
        (item,) = typing.get_args(annotation)
        return pa.list_(_arrow_type(item))
    if origin is dict or annotation is dict:
        return pa.string()
    if origin is Literal:
        return pa.string()
    if isinstance(annotation, type):
        if issubclass(annotation, BaseModel):
            return pa.struct(
                [
                    pa.field(name, _arrow_type(info.annotation), nullable=True)
                    for name, info in annotation.model_fields.items()
                ]
            )
        if issubclass(annotation, Enum) or annotation is str or annotation is uuid.UUID:
            return pa.string()
        if annotation is bool:
            return pa.bool_()
        if annotation is int:
            return pa.int64()
        if annotation is float:
            return pa.float64()
        if annotation is Decimal:
            return DECIMAL_TYPE
        if annotation is dt.datetime:
            return pa.timestamp("us", tz="UTC")
        if annotation is dt.date:
            return pa.date32()
    if getattr(annotation, "__name__", "") == "AwareDatetime" or "AwareDatetime" in str(annotation):
        return pa.timestamp("us", tz="UTC")
    raise TypeError(f"No Arrow mapping for annotation {annotation!r}")


def _is_json_field(annotation: Any) -> bool:
    annotation = _strip_optional(annotation)
    return typing.get_origin(annotation) is dict or annotation is dict


def column_name(model: type[BaseModel], field: str) -> str:
    info = model.model_fields[field]
    return f"{field}{JSON_SUFFIX}" if _is_json_field(info.annotation) else field


def arrow_schema(model: type[BaseModel]) -> pa.Schema:
    """Arrow schema for a canonical record model."""
    fields = []
    for name, info in model.model_fields.items():
        column = column_name(model, name)
        fields.append(pa.field(column, _arrow_type(info.annotation), nullable=True))
    return pa.schema(fields, metadata={b"seshat_model": model.__name__.encode()})


def _to_column_value(value: Any) -> Any:
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseModel):
        return {key: _to_column_value(item) for key, item in value.__dict__.items()}
    if isinstance(value, list):
        return [_to_column_value(item) for item in value]
    return value


def to_rows(records: typing.Iterable[BaseModel]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        model = type(record)
        row: dict[str, Any] = {}
        for name in model.model_fields:
            value = getattr(record, name)
            if _is_json_field(model.model_fields[name].annotation):
                row[column_name(model, name)] = (
                    None
                    if value is None
                    else json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
                )
            else:
                row[name] = _to_column_value(value)
        rows.append(row)
    return rows


def to_table(records: typing.Sequence[BaseModel], model: type[BaseModel]) -> pa.Table:
    """Records -> Arrow table with the canonical schema (validates types)."""
    if any(type(record) is not model for record in records):
        raise TypeError(f"All records must be {model.__name__}")
    return pa.Table.from_pylist(to_rows(records), schema=arrow_schema(model))


def from_table(table: pa.Table, model: type[T]) -> list[T]:
    """Arrow table -> validated records (inverse of ``to_table``)."""
    records: list[T] = []
    for row in table.to_pylist():
        payload: dict[str, Any] = {}
        for name in model.model_fields:
            column = column_name(model, name)
            if column not in row:
                continue
            value = row[column]
            if column.endswith(JSON_SUFFIX) and column != name:
                value = json.loads(value) if value is not None else None
            if value is None and model.model_fields[name].default is not None:
                continue
            payload[name] = value
        records.append(model.model_validate(payload))
    return records
