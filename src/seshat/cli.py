"""Command-line interface for the Seshat dataset pipeline (``seshat-data``).

Controlled experiments:   build-manifest → run → normalize → validate → features →
                          split → release (→ profile, export-csv)
Versioned record store:   synth / ingest / ingest-experiment → predict → compare,
                          features --from-store, export-csv, purge-expired, store-summary
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from seshat.data.build_features import build_features_to_parquet
from seshat.data.compare import compare_predictions, prediction_records, summarize_comparison
from seshat.data.csv_view import export_csv
from seshat.data.ingest import ingest_experiment, ingest_jsonl, ingest_records
from seshat.data.io import read_parquet, write_json, write_jsonl, write_parquet
from seshat.data.normalize import normalize_to_parquet
from seshat.data.privacy import RetentionPolicy
from seshat.data.profile import profile_dataset
from seshat.data.provenance import DataOrigin, DataSource, Provenance
from seshat.data.records import RecordType
from seshat.data.release import freeze_release
from seshat.data.split import split_to_parquet
from seshat.data.store import DatasetStore
from seshat.data.synthetic import generate_synthetic_records
from seshat.data.validate import validate_dataframe
from seshat.experiments.manifest import build_manifest_from_config, load_prompts
from seshat.experiments.runner import run_manifest
from seshat.predictor import BaselinePredictor
from seshat.providers.registry import ProviderCallsNotAllowedError

DEFAULT_STORE = "data/store"


def _add_retention_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--keep-prompt-text",
        action="store_true",
        help="Store prompt text (default: hash + features only).",
    )
    parser.add_argument("--production-days", type=int, default=90)
    parser.add_argument("--import-days", type=int, default=90)
    parser.add_argument("--quarantine-dir")


def _policy(args: argparse.Namespace) -> RetentionPolicy:
    return RetentionPolicy(
        production_log_days=args.production_days,
        import_days=args.import_days,
        store_prompt_text=args.keep_prompt_text,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="seshat-data", description="Seshat dataset pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-manifest", help="Expand an experiment config into runs")
    build.add_argument("--config", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--root", help="Directory prompt_sources are relative to")

    run = sub.add_parser("run", help="Execute manifest runs (mock unless explicitly allowed)")
    run.add_argument("--manifest", required=True)
    run.add_argument("--prompts", nargs="+", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--retry-failed", action="store_true")
    run.add_argument(
        "--allow-provider-calls",
        action="store_true",
        default=None,
        help="Permit real (paid) provider requests; also SESHAT_ALLOW_PROVIDER_CALLS=1",
    )

    normalize = sub.add_parser("normalize")
    normalize.add_argument("--input", required=True)
    normalize.add_argument("--output", required=True)

    validate = sub.add_parser("validate")
    validate.add_argument("--input", required=True)
    validate.add_argument("--output", required=True)

    features = sub.add_parser("features", help="Build a leakage-safe feature matrix")
    source = features.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Validated normalized Parquet")
    source.add_argument("--from-store", help="Record store root")
    features.add_argument("--output", required=True)
    features.add_argument(
        "--label-origin",
        choices=[origin.value for origin in DataOrigin],
        default=DataOrigin.REAL.value,
        help="Outcome origin to use as labels (default: real; never mixed)",
    )

    split = sub.add_parser("split")
    split.add_argument("--input", required=True)
    split.add_argument("--output", required=True)
    split.add_argument("--seed", type=int, default=1729)

    profile = sub.add_parser("profile")
    profile.add_argument("--input", required=True)
    profile.add_argument("--output", required=True)

    release = sub.add_parser("release")
    release.add_argument("--features", required=True)
    release.add_argument("--splits", required=True)
    release.add_argument("--output-dir", default="data/releases")
    release.add_argument("--version", required=True)

    csv = sub.add_parser("export-csv", help="CSV view of a Parquet/JSONL file or store table")
    csv.add_argument("--input", required=True, help="File, or a store table directory")
    csv.add_argument("--output", default="data/records.csv")
    csv.add_argument("--limit", type=int)
    csv.add_argument("--columns", nargs="+")

    synth = sub.add_parser("synth", help="Generate labelled synthetic records")
    synth.add_argument("--prompts", nargs="+", required=True)
    synth.add_argument("--n", type=int, default=48)
    synth.add_argument("--seed", type=int, default=1729)
    synth.add_argument("--output-dir", required=True)

    ingest = sub.add_parser("ingest", help="Ingest a JSONL file of versioned records")
    ingest.add_argument("--store", default=DEFAULT_STORE)
    ingest.add_argument(
        "--record-type", required=True, choices=[record.value for record in RecordType]
    )
    ingest.add_argument("--input", required=True)
    ingest.add_argument(
        "--origin",
        choices=[origin.value for origin in DataOrigin],
        help="Provenance for records that lack one (requires --source)",
    )
    ingest.add_argument("--source", choices=[item.value for item in DataSource])
    _add_retention_args(ingest)

    ingest_exp = sub.add_parser("ingest-experiment", help="Load raw experiment JSONL into store")
    ingest_exp.add_argument("--store", default=DEFAULT_STORE)
    ingest_exp.add_argument("--raw", required=True)
    ingest_exp.add_argument("--prompts", nargs="+", required=True)
    _add_retention_args(ingest_exp)

    predict = sub.add_parser("predict", help="Score stored requests with the baseline predictor")
    predict.add_argument("--store", default=DEFAULT_STORE)

    compare = sub.add_parser("compare", help="Predicted vs observed usage")
    compare.add_argument("--store", default=DEFAULT_STORE)
    compare.add_argument("--output", help="Optional CSV of matched pairs")

    purge = sub.add_parser("purge-expired", help="Delete records past retain_until")
    purge.add_argument("--store", default=DEFAULT_STORE)
    purge.add_argument("--as-of", help="ISO timestamp with timezone (default: now)")

    summary = sub.add_parser("store-summary", help="Counts per record type and origin")
    summary.add_argument("--store", default=DEFAULT_STORE)
    return parser


def _default_provenance(args: argparse.Namespace) -> Provenance | None:
    if (args.origin is None) != (args.source is None):
        raise ValueError("--origin and --source must be given together")
    if args.origin is None:
        return None
    return Provenance(data_origin=DataOrigin(args.origin), source=DataSource(args.source))


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _store_summary(store: DatasetStore) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for record_type in RecordType:
        frame = store.read(record_type)
        entry: dict[str, Any] = {"rows": int(len(frame))}
        if not frame.empty and "data_origin" in frame.columns:
            entry["by_origin"] = {
                str(key): int(value) for key, value in frame["data_origin"].value_counts().items()
            }
        result[record_type.value] = entry
    return result


def _dispatch(args: argparse.Namespace) -> None:
    if args.command == "build-manifest":
        frame = build_manifest_from_config(args.config, args.output, root=args.root)
        print(f"wrote {len(frame)} planned runs to {args.output}")
    elif args.command == "run":
        prompts = load_prompts(args.prompts)
        written = run_manifest(
            pd.read_parquet(args.manifest),
            prompts,
            args.output,
            retry_failed=args.retry_failed,
            allow_provider_calls=args.allow_provider_calls,
        )
        print(f"wrote {written} observations to {args.output}")
    elif args.command == "normalize":
        frame = normalize_to_parquet(args.input, args.output)
        print(f"normalized {len(frame)} observations to {args.output}")
    elif args.command == "validate":
        frame = validate_dataframe(read_parquet(args.input))
        write_parquet(args.output, frame)
        print(frame["quality_decision"].value_counts().to_string())
    elif args.command == "features":
        frame = build_features_to_parquet(
            args.input,
            args.output,
            label_origin=args.label_origin,
            store_root=args.from_store,
        )
        print(f"wrote {len(frame)} eligible {args.label_origin}-label rows to {args.output}")
    elif args.command == "split":
        frame = split_to_parquet(args.input, args.output, seed=args.seed)
        print(frame["split"].value_counts().to_string())
    elif args.command == "profile":
        profile = profile_dataset(read_parquet(args.input))
        write_json(args.output, profile)
        _print_json(profile)
    elif args.command == "release":
        manifest = freeze_release(args.features, args.splits, args.output_dir, version=args.version)
        _print_json(manifest)
    elif args.command == "export-csv":
        output = export_csv(args.input, args.output, limit=args.limit, columns=args.columns)
        print(f"wrote CSV view to {output}")
    elif args.command == "synth":
        requests, observations = generate_synthetic_records(
            load_prompts(args.prompts), n=args.n, seed=args.seed
        )
        out = Path(args.output_dir)
        write_jsonl(out / "request_features.jsonl", requests)
        write_jsonl(out / "observed_usage.jsonl", observations)
        print(f"wrote {len(requests)} synthetic request/observation pairs to {out}")
    elif args.command == "ingest":
        report = ingest_jsonl(
            DatasetStore(args.store),
            RecordType(args.record_type),
            args.input,
            default_provenance=_default_provenance(args),
            policy=_policy(args),
            quarantine_dir=args.quarantine_dir,
        )
        _print_json(report.to_dict())
    elif args.command == "ingest-experiment":
        reports = ingest_experiment(
            DatasetStore(args.store),
            args.raw,
            load_prompts(args.prompts),
            policy=_policy(args),
            quarantine_dir=args.quarantine_dir,
        )
        _print_json([report.to_dict() for report in reports])
    elif args.command == "predict":
        store = DatasetStore(args.store)
        predictor = BaselinePredictor()
        report = ingest_records(
            store,
            RecordType.PREDICTION,
            prediction_records(store, predictor),
            source_name=f"predictor:{predictor.version}",
        )
        _print_json(report.to_dict())
    elif args.command == "compare":
        frame = compare_predictions(DatasetStore(args.store))
        if args.output:
            compare_csv = Path(args.output)
            compare_csv.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(compare_csv, index=False)
        _print_json(summarize_comparison(frame))
    elif args.command == "purge-expired":
        as_of = datetime.fromisoformat(args.as_of) if args.as_of else datetime.now(UTC)
        _print_json(DatasetStore(args.store).purge_expired(as_of))
    elif args.command == "store-summary":
        _print_json(_store_summary(DatasetStore(args.store)))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _dispatch(args)
    except (ValueError, FileExistsError, FileNotFoundError, ProviderCallsNotAllowedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
