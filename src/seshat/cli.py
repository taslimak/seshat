"""Command-line interface for the Seshat data factory."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

import pandas as pd

from seshat.data.build_features import build_features_to_parquet
from seshat.data.csv_view import export_csv
from seshat.data.io import read_parquet, write_json, write_parquet
from seshat.data.normalize import normalize_to_parquet
from seshat.data.profile import profile_dataset
from seshat.data.release import freeze_release
from seshat.data.split import split_to_parquet
from seshat.data.validate import validate_dataframe
from seshat.experiments.manifest import build_manifest_from_config, load_prompts
from seshat.experiments.runner import run_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="seshat-data", description="Seshat dataset factory")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-manifest")
    build.add_argument("--config", required=True)
    build.add_argument("--output", required=True)

    run = sub.add_parser("run")
    run.add_argument("--manifest", required=True)
    run.add_argument("--prompts", nargs="+", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--retry-failed", action="store_true")

    normalize = sub.add_parser("normalize")
    normalize.add_argument("--input", required=True)
    normalize.add_argument("--output", required=True)

    validate = sub.add_parser("validate")
    validate.add_argument("--input", required=True)
    validate.add_argument("--output", required=True)

    features = sub.add_parser("features")
    features.add_argument("--input", required=True)
    features.add_argument("--output", required=True)

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

    csv = sub.add_parser("export-csv")
    csv.add_argument("--input", required=True)
    csv.add_argument("--output", default="data/records.csv")
    csv.add_argument("--limit", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build-manifest":
        frame = build_manifest_from_config(args.config, args.output)
        print(f"wrote {len(frame)} planned runs to {args.output}")
    elif args.command == "run":
        prompts = load_prompts(args.prompts)
        written = run_manifest(
            pd.read_parquet(args.manifest), prompts, args.output, retry_failed=args.retry_failed
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
        frame = build_features_to_parquet(args.input, args.output)
        print(f"wrote {len(frame)} eligible rows to {args.output}")
    elif args.command == "split":
        frame = split_to_parquet(args.input, args.output, seed=args.seed)
        print(frame["split"].value_counts().to_string())
    elif args.command == "profile":
        profile = profile_dataset(read_parquet(args.input))
        write_json(args.output, profile)
        print(json.dumps(profile, indent=2, sort_keys=True))
    elif args.command == "release":
        manifest = freeze_release(args.features, args.splits, args.output_dir, version=args.version)
        print(json.dumps(manifest, indent=2, sort_keys=True))
    elif args.command == "export-csv":
        output = export_csv(args.input, args.output, limit=args.limit)
        print(f"wrote CSV view to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
