"""Seshat dataset pipeline CLI.

Run as ``python -m seshat.data.cli <command>`` or ``seshat-data <command>``.

Typical offline workflow (no API keys, no network)::

    python -m seshat.data.cli generate --count 120 --seed 42
    python -m seshat.data.cli collect --config configs/data_collection.yaml
    python -m seshat.data.cli validate
    python -m seshat.data.cli export --format csv
    python -m seshat.data.cli prepare-training --version 0.1.0 --label-origin synthetic
    python -m seshat.data.cli stats

Live collection (paid) needs explicit authorization and a budget::

    python -m seshat.data.cli collect --config configs/experiments/live_pilot.yaml \\
        --execute-live --max-cost-usd 5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

DEFAULT_ROOT = "data"


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _decimal(value: str) -> Decimal:
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"not a number: {value}") from exc
    if result < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return result


def _newest_predictor(store: str) -> str | None:
    ledger = Path(store) / "quality" / "model_evaluations.jsonl"
    if not ledger.exists():
        return None
    lines = [line for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    return str(json.loads(lines[-1])["predictor_version"]) if lines else None


def _key_value(item: str) -> tuple[str, str]:
    key, sep, value = item.partition("=")
    if not sep or not key:
        raise ValueError(f"Expected KEY=VALUE, got {item!r}")
    return key.strip(), value.strip()


def _ratios(value: str | None) -> dict[str, float] | None:
    if not value:
        return None
    if value == "real_label":
        from seshat.data.training import REAL_LABEL_RATIOS

        return dict(REAL_LABEL_RATIOS)
    pairs = [_key_value(part) for part in value.split(",") if part.strip()]
    return {key: float(val) for key, val in pairs}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seshat-data",
        description="Seshat dataset collection pipeline (offline / dry-run by default).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Docs: docs/DATA_COLLECTION_GUIDE.md",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate deterministic experimental prompts (offline)")
    gen.add_argument("--count", type=int, default=120)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument(
        "--output", help="JSONL path (default data/raw/generated/prompts_seed<seed>_n<count>.jsonl)"
    )
    gen.add_argument("--templates", nargs="+", help="Restrict to these template ids")

    collect = sub.add_parser("collect", help="Plan and run a collection experiment")
    collect.add_argument("--config", default="configs/data_collection.yaml")
    mode = collect.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true", help="Plan + estimate cost only; send nothing"
    )
    mode.add_argument(
        "--execute-live", action="store_true", help="Authorize real (paid) provider calls"
    )
    collect.add_argument(
        "--max-cost-usd", type=_decimal, help="Hard spending cap (required with --execute-live)"
    )
    collect.add_argument(
        "--retry-failed", action="store_true", help="Re-attempt samples that failed permanently"
    )
    collect.add_argument(
        "--retry-uncertain",
        action="store_true",
        help="Re-send samples whose outcome is unknown (may duplicate paid executions!)",
    )
    collect.add_argument("--no-normalize", action="store_true")

    norm = sub.add_parser(
        "normalize", help="Rebuild normalized tables for an experiment from raw events"
    )
    norm.add_argument("--config", default="configs/data_collection.yaml")

    imp = sub.add_parser("import", help="Import prompts (unlabeled) or authorized usage logs")
    imp.add_argument("--file", required=True)
    imp.add_argument("--kind", choices=["prompts", "usage-log"], default="usage-log")
    imp.add_argument("--root", default=DEFAULT_ROOT)
    imp.add_argument("--bank-dir", default="prompts")
    imp.add_argument(
        "--target", help="prompts: bank file to append to (default prompts/imported_<stem>.jsonl)"
    )
    imp.add_argument("--source", help="prompts: provenance name of the dataset")
    imp.add_argument("--source-detail", help="prompts: attribution / dataset URL / generator")
    imp.add_argument("--license-id", help="License (must be in configs/prompt_collection.json)")
    imp.add_argument("--commercial-use", choices=["true", "false"])
    imp.add_argument("--category")
    imp.add_argument("--complexity", choices=["low", "medium", "high"])
    imp.add_argument("--language")
    imp.add_argument("--synthetic", action="store_true")
    imp.add_argument(
        "--source-type",
        default="authorized_log_import",
        choices=["authorized_log_import", "consented_api_telemetry", "controlled_experiment"],
    )
    imp.add_argument(
        "--authorization-ref", help="usage-log: who/what authorized this import (required)"
    )
    imp.add_argument("--dataset-origin", help="usage-log: name of the originating system")
    imp.add_argument(
        "--keep-prompt-text",
        action="store_true",
        help="usage-log: store text when rows have consent=granted",
    )

    val = sub.add_parser(
        "validate", help="Validate the normalized layer and write the quality report"
    )
    val.add_argument("--root", default=DEFAULT_ROOT)
    val.add_argument("--preflight-tolerance", type=float, default=0.1)

    exp = sub.add_parser("export", help="Export a readable CSV (+ data dictionary)")
    exp.add_argument("--format", choices=["csv"], default="csv")
    exp.add_argument("--root", default=DEFAULT_ROOT)
    exp.add_argument("--output")
    exp.add_argument("--limit", type=int)

    prep = sub.add_parser(
        "prepare-training", help="Build a versioned, leakage-safe training dataset"
    )
    prep.add_argument("--root", default=DEFAULT_ROOT)
    prep.add_argument("--version", required=True)
    prep.add_argument("--label-origin", choices=["real", "synthetic"], default="real")
    prep.add_argument("--seed", type=int, default=1729)
    prep.add_argument(
        "--temporal-holdout",
        type=float,
        default=0.0,
        help="Fraction of newest groups held out as test",
    )
    prep.add_argument(
        "--include-source",
        action="append",
        default=[],
        choices=["consented_api_telemetry", "authorized_log_import"],
        help="Also use real labels from this source (default: controlled experiments only)",
    )
    prep.add_argument(
        "--ratios",
        help="'real_label' preset or e.g. train=0.6,validation=0.1,calibration=0.15,test=0.15",
    )
    prep.add_argument(
        "--group-by-template",
        action="store_true",
        help="Also keep each prompt template in one split",
    )

    stats = sub.add_parser("stats", help="Provider/model collection statistics")
    stats.add_argument("--root", default=DEFAULT_ROOT)

    pred = sub.add_parser("predict", help="Score stored requests with the baseline predictor")
    pred.add_argument("--root", default=DEFAULT_ROOT)
    cmp_ = sub.add_parser("compare", help="Predicted vs observed output tokens (per data origin)")
    cmp_.add_argument("--root", default=DEFAULT_ROOT)
    cmp_.add_argument("--output")

    purge = sub.add_parser("purge-expired", help="Delete telemetry/log-import rows past retention")
    purge.add_argument("--root", default=DEFAULT_ROOT)
    purge.add_argument("--as-of", help="ISO timestamp with timezone (default: now)")

    prompts = sub.add_parser("prompts", help="Prompt bank tools")
    psub = prompts.add_subparsers(dest="prompts_command", required=True)
    for name, text in (
        ("check", "Fail on policy/duplicate violations"),
        ("report", "Coverage report"),
    ):
        cmd = psub.add_parser(name, help=text)
        cmd.add_argument("--bank-dir", default="prompts")
    review = psub.add_parser(
        "review", help="Approve/reject prompts (approval is required for live runs)"
    )
    review.add_argument("--bank-dir", default="prompts")
    review.add_argument("--ids", nargs="+", required=True)
    review.add_argument("--status", choices=["approved", "rejected", "unreviewed"], required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--notes")

    cons = sub.add_parser(
        "consolidate",
        help="Merge collection artifacts (ZIP or directory) into a durable canonical store",
    )
    cons.add_argument("--source", nargs="+", required=True, help="Artifact ZIP(s) or dir(s)")
    cons.add_argument("--store", required=True, help="Store root (keep it in private storage)")
    cons.add_argument("--store-kind", choices=["real", "fixture"], default="real")
    cons.add_argument(
        "--meta",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Provenance metadata, e.g. github_run_id=123 artifact_id=456 (repeatable)",
    )
    cons.add_argument("--no-archive", action="store_true", help="Do not keep the source ZIP")

    srep = sub.add_parser("store-report", help="Re-validate a store and regenerate its reports")
    srep.add_argument("--store", required=True)

    ready = sub.add_parser(
        "readiness", help="Per-split usable-label counts from the exact training split"
    )
    ready.add_argument("--root", default=DEFAULT_ROOT)
    ready.add_argument("--seed", type=int, default=1729)
    ready.add_argument("--markdown", action="store_true", help="Print Markdown instead of JSON")

    ev = sub.add_parser(
        "evaluate-baseline",
        help="Score the heuristic baseline on real observations (censoring-aware, bootstrap CIs)",
    )
    ev.add_argument("--root", default=DEFAULT_ROOT)
    ev.add_argument("--output-dir", help="Write baseline_evaluation.{json,md} here")
    ev.add_argument("--n-boot", type=int, default=2000)

    plan = sub.add_parser(
        "plan-batch",
        help="Plan the next adaptive paid batch from the store (dry run; sends nothing)",
    )
    plan.add_argument("--store", help="Consolidated real store (omit to plan from priors)")
    plan.add_argument("--provider", required=True)
    plan.add_argument("--model", required=True)
    plan.add_argument("--experiment-id", required=True)
    plan.add_argument("--budget-usd", type=_decimal, default=Decimal("2.00"))
    plan.add_argument(
        "--efforts", nargs="+", help="Effort levels (default: low medium high if supported)"
    )
    plan.add_argument("--seed", type=int)
    plan.add_argument("--pool-size", type=int, default=1020)
    plan.add_argument("--max-censor-rate", type=float, default=0.10)
    plan.add_argument(
        "--monthly-target-usd",
        type=_decimal,
        default=Decimal("10.00"),
        help="Refuse if store month-to-date estimated spend + budget exceeds this",
    )
    plan.add_argument(
        "--quality-predictor",
        help="Weight categories by this predictor's realized errors (default: newest registered "
        "forecaster, else the baseline)",
    )
    plan.add_argument(
        "--shadow-artifact",
        help="Local forecaster.json to run as an observation-only shadow forecaster before each "
        "dispatch; only its SHA-256 and versions are written to the config",
    )
    plan.add_argument(
        "--shadow-artifact-ref",
        help="Path of that artifact inside the private store (default models/<version>/...)",
    )
    plan.add_argument("--output", required=True, help="Collector config to write (YAML)")
    plan.add_argument("--report", help="Dry-run report path (Markdown; JSON alongside)")

    tf = sub.add_parser(
        "train-forecaster",
        help="Train/calibrate/evaluate forecasters on a store (protected test; no deployment)",
    )
    tf.add_argument("--root", required=True)
    tf.add_argument("--output-dir", required=True)
    tf.add_argument("--confidence-level", type=float, default=0.90)
    tf.add_argument("--register", action="store_true", help="Store artifact + backtests in --root")

    qr = sub.add_parser(
        "quality-refresh",
        help="Rebuild prediction outcomes, quality snapshots, drift and run reports",
    )
    qr.add_argument("--root", required=True)
    return parser


def _plan(config_path: str) -> tuple[Any, Any]:
    from seshat.data.generators import LengthThresholds, generate_prompts
    from seshat.experiments.config import load_collection_config
    from seshat.experiments.design import build_plan
    from seshat.providers.catalog import load_catalog

    config = load_collection_config(config_path)
    prompts = []
    if config.prompts.generate is not None:
        prompts += generate_prompts(
            config.prompts.generate.count,
            config.prompts.generate.seed,
            thresholds=LengthThresholds(
                config.strata.short_max_words, config.strata.medium_max_words
            ),
            templates=config.prompts.generate.templates,
        )
    if config.prompts.bank_files:
        from seshat.experiments.prompts_io import load_prompt_files

        prompts += load_prompt_files(config.prompts.bank_files)
    return config, build_plan(config, prompts, load_catalog())


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "generate":
        from seshat.data.generators import generate_prompts
        from seshat.data.io import write_jsonl

        prompts = generate_prompts(args.count, args.seed, templates=args.templates)
        out = Path(
            args.output
            or f"{DEFAULT_ROOT}/raw/generated/prompts_seed{args.seed}_n{args.count}.jsonl"
        )
        write_jsonl(
            out,
            [
                p.model_dump(mode="json", exclude_defaults=True) | {"prompt_id": p.prompt_id}
                for p in prompts
            ],
        )
        categories = sorted({p.category for p in prompts})
        _print(
            {
                "written": len(prompts),
                "output": str(out),
                "categories": len(categories),
                "source_type": "synthetic_unexecuted",
            }
        )
        return 0

    if args.command in {"collect", "normalize"}:
        from seshat.data.collectors.collector import Collector
        from seshat.data.collectors.normalize import normalize_experiment
        from seshat.data.pricing import PriceTable
        from seshat.data.privacy import RetentionPolicy
        from seshat.providers.catalog import load_catalog
        from seshat.providers.registry import live_providers

        config, plan = _plan(args.config)
        policy = RetentionPolicy(store_prompt_text=config.privacy.store_prompt_text)
        if args.command == "normalize":
            _print(
                normalize_experiment(
                    config.storage.root,
                    config.experiment_id,
                    catalog=load_catalog(),
                    prices=PriceTable.load(),
                    policy=policy,
                )
            )
            return 0
        live = live_providers({s.provider for s in plan.samples})
        if args.execute_live:
            mode = "live"
        elif args.dry_run or live:
            mode = "dry_run"
        else:
            mode = "offline"
        collector = Collector(
            config,
            plan,
            mode=mode,  # type: ignore[arg-type]
            execute_live=args.execute_live,
            max_cost_usd=args.max_cost_usd,
            retry_failed=args.retry_failed,
            retry_uncertain=args.retry_uncertain,
        )
        result = collector.run()
        payload: dict[str, Any] = {
            "collection": result.to_dict(),
            "design": {
                k: plan.report[k]
                for k in (
                    "planned_requests",
                    "full_cartesian_requests",
                    "repeated_cells",
                    "excluded_combinations",
                )
            },
        }
        if mode == "dry_run" and live and not args.dry_run:
            payload["note"] = (
                "Plan contains live providers: ran as a dry run. "
                "Add --execute-live --max-cost-usd N to send requests."
            )
        if mode != "dry_run" and not args.no_normalize:
            payload["normalized"] = normalize_experiment(
                config.storage.root,
                config.experiment_id,
                catalog=load_catalog(),
                prices=PriceTable.load(),
                policy=policy,
            )
        _print(payload)
        return 0

    if args.command == "import":
        if args.kind == "prompts":
            from seshat.prompts.importer import import_prompts
            from seshat.prompts.policy import load_policy

            defaults = {
                "source": args.source,
                "source_detail": args.source_detail,
                "license_id": args.license_id,
                "commercial_use_allowed": None
                if args.commercial_use is None
                else args.commercial_use == "true",
                "category": args.category,
                "complexity": args.complexity,
                "language": args.language,
                "is_synthetic": True if args.synthetic else None,
            }
            target = args.target or f"{args.bank_dir}/imported_{Path(args.file).stem}.jsonl"
            prompt_report = import_prompts(
                args.file,
                bank_dir=args.bank_dir,
                target_file=target,
                policy=load_policy(),
                defaults=defaults,
                report_dir=f"{args.root}/raw/imports",
            )
            _print(
                {
                    **prompt_report.to_dict(),
                    "source_type": "public_permitted_dataset",
                    "note": (
                        "Imported prompts are unlabeled and unreviewed until approved and executed."
                    ),
                }
            )
            return 0
        from seshat.data.collectors.log_import import import_usage_log
        from seshat.data.pricing import PriceTable
        from seshat.data.privacy import RetentionPolicy
        from seshat.data.schemas.enums import SourceType
        from seshat.providers.catalog import load_catalog

        if not args.authorization_ref:
            raise ValueError("--authorization-ref is required for usage-log imports")
        log_report = import_usage_log(
            args.file,
            root=args.root,
            source_type=SourceType(args.source_type),
            authorization_ref=args.authorization_ref,
            dataset_origin=args.dataset_origin or Path(args.file).stem,
            catalog=load_catalog(),
            prices=PriceTable.load(),
            policy=RetentionPolicy(store_prompt_text=args.keep_prompt_text),
            data_license=args.license_id,
        )
        _print(log_report.to_dict())
        return 0

    if args.command == "validate":
        from seshat.data.validators import validate_storage

        quality = validate_storage(args.root, preflight_tolerance=args.preflight_tolerance)
        _print(
            {
                "totals": quality["totals"],
                "by_decision": quality["by_decision"],
                "by_flag": quality["by_flag"],
                "report": f"{args.root}/normalized/_validation/quality_report.md",
            }
        )
        return 0

    if args.command == "export":
        from seshat.data.exports import export_csv

        path = export_csv(args.root, args.output, limit=args.limit)
        _print({"csv": str(path), "data_dictionary": str(path.with_name("data_dictionary.md"))})
        return 0

    if args.command == "prepare-training":
        from seshat.data.training import DEFAULT_GROUP_BY, prepare_training_dataset

        group_by = DEFAULT_GROUP_BY + (("prompt_template_id",) if args.group_by_template else ())
        manifest = prepare_training_dataset(
            args.root,
            args.version,
            label_origin=args.label_origin,
            seed=args.seed,
            temporal_holdout_fraction=args.temporal_holdout,
            group_by=group_by,
            include_sources=tuple(args.include_source),
            ratios=_ratios(args.ratios),
        )
        _print(manifest)
        return 0

    if args.command == "stats":
        from seshat.data.stats import collection_stats

        _print(collection_stats(args.root))
        return 0

    if args.command == "predict":
        from seshat.data.predictions import score_requests
        from seshat.predictor import BaselinePredictor

        _print({"predictions": score_requests(args.root, BaselinePredictor())})
        return 0

    if args.command == "compare":
        from seshat.data.predictions import compare_predictions, summarize

        frame = compare_predictions(args.root)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(args.output, index=False)
        _print(summarize(frame))
        return 0

    if args.command == "purge-expired":
        from seshat.data.storage import StorageLayout

        as_of = datetime.fromisoformat(args.as_of) if args.as_of else datetime.now(UTC)
        _print(StorageLayout(args.root).purge_expired(as_of))
        return 0

    if args.command == "prompts":
        from seshat.prompts.policy import load_policy

        collection_policy = load_policy()
        if args.prompts_command == "check":
            from seshat.prompts.report import bank_violations

            problems = bank_violations(args.bank_dir, collection_policy)
            _print({"violations": problems, "ok": not problems})
            return 1 if problems else 0
        if args.prompts_command == "report":
            from seshat.prompts.report import coverage_report

            _print(coverage_report(args.bank_dir, collection_policy))
            return 0
        from seshat.prompts.review import set_review_status

        changed = set_review_status(
            args.bank_dir,
            args.ids,
            args.status,
            reviewer=args.reviewer,
            policy=collection_policy,
            notes=args.notes,
        )
        _print({"updated": changed, "status": args.status})
        return 0
    if args.command == "consolidate":
        from seshat.data.consolidate import consolidate

        metadata = dict(_key_value(item) for item in args.meta)
        results = [
            consolidate(
                source,
                args.store,
                store_kind=args.store_kind,
                metadata=metadata,
                keep_archive=not args.no_archive,
                write_reports=index == len(args.source) - 1,
            )
            for index, source in enumerate(args.source)
        ]
        _print(
            [{k: v for k, v in result.items() if k not in {"source_files"}} for result in results]
        )
        return 0

    if args.command == "store-report":
        from seshat.data.consolidate import refresh_reports

        _print(refresh_reports(args.store))
        return 0

    if args.command == "readiness":
        from seshat.data.readiness import readiness_report, render_readiness_markdown

        report = readiness_report(args.root, seed=args.seed)
        if args.markdown:
            print(render_readiness_markdown(report))
        else:
            _print(report)
        return 0

    if args.command == "evaluate-baseline":
        from seshat.data.evaluation import evaluate_baseline, render_evaluation_markdown

        evaluation = evaluate_baseline(args.root, n_boot=args.n_boot)
        if args.output_dir:
            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "baseline_evaluation.json").write_text(
                json.dumps(evaluation, indent=2, default=str) + "\n", encoding="utf-8"
            )
            (out / "baseline_evaluation.md").write_text(
                render_evaluation_markdown(evaluation, "Baseline evaluation on real observations"),
                encoding="utf-8",
            )
        _print(
            {
                k: evaluation[k]
                for k in ("rows", "uncensored_rows", "censored_rows", "point", "interval")
            }
            | {"censored": evaluation["censored"], "bootstrap_95": evaluation["bootstrap_95"]}
        )
        return 0

    if args.command == "plan-batch":
        import yaml

        from seshat.data.readiness import readiness_report
        from seshat.experiments.adaptive import plan_next_batch, render_plan_markdown
        from seshat.experiments.config import CollectionConfig

        efforts = (
            [None if e in {"none_unset", "unset"} else e for e in args.efforts]
            if args.efforts
            else None
        )
        month = datetime.now(UTC).strftime("%Y-%m")
        if args.store:
            from seshat.data.consolidate import month_to_date_spend

            spent = month_to_date_spend(args.store, month)
            if spent + args.budget_usd > args.monthly_target_usd:
                raise ValueError(
                    f"Month-to-date estimated spend ${spent} + batch budget ${args.budget_usd} "
                    f"exceeds the monthly target ${args.monthly_target_usd} for {month}"
                )
        segment_quality = None
        if args.store:
            from seshat.quality.outcomes import read_outcomes
            from seshat.quality.snapshots import segment_quality as _segment_quality

            outcomes = read_outcomes(args.store)
            if not outcomes.empty:
                version = args.quality_predictor or _newest_predictor(args.store) or "baseline-0.1"
                segment_quality = _segment_quality(outcomes, version)
        batch = plan_next_batch(
            store_root=args.store,
            provider=args.provider,
            model=args.model,
            experiment_id=args.experiment_id,
            budget_usd=args.budget_usd,
            efforts=efforts,
            seed=args.seed,
            prompt_pool_size=args.pool_size,
            max_censor_rate=args.max_censor_rate,
            segment_quality=segment_quality,
        )
        if args.shadow_artifact:
            import hashlib

            artifact_bytes = Path(args.shadow_artifact).read_bytes()
            artifact = json.loads(artifact_bytes)
            batch["config"]["shadow"] = {
                "artifact_ref": args.shadow_artifact_ref
                or f"models/{artifact['predictor_version']}/forecaster.json",
                "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
                "predictor_version": artifact["predictor_version"],
                "calibration_version": artifact["calibration_version"],
                "dataset_version": artifact["dataset_version"],
                "required": True,
            }
            batch["report"]["shadow"] = batch["config"]["shadow"]
        CollectionConfig.model_validate(batch["config"])  # fail before writing
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)

        # JSON round-trip removes shared references, so the YAML has no anchors/aliases.
        plain = json.loads(json.dumps(batch["config"]))
        output.write_text(yaml.safe_dump(plain, sort_keys=False), encoding="utf-8")
        if args.report:
            readiness = readiness_report(args.store) if args.store else None
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                render_plan_markdown(batch["report"], readiness), encoding="utf-8"
            )
            report_path.with_suffix(".json").write_text(
                json.dumps(batch["report"], indent=2) + "\n", encoding="utf-8"
            )
        summary_keys = (
            "experiment_id",
            "planned_requests",
            "expected_cost_usd",
            "expected_new_usable_rows",
            "approval_required",
        )
        _print({k: batch["report"][k] for k in summary_keys} | {"config": str(output)})
        return 0

    if args.command == "train-forecaster":
        from seshat.modeling.pipeline import train_and_evaluate

        evaluation = train_and_evaluate(
            args.root, args.output_dir, confidence_level=args.confidence_level
        )
        trained: dict[str, Any] = {
            k: evaluation[k]
            for k in (
                "predictor_version",
                "uncertainty_model_version",
                "calibration_version",
                "dataset_version",
                "counts",
            )
        }
        trained["selected"] = evaluation["selection"]["chosen"]
        trained["promotion"] = evaluation["promotion"]["decision"]
        trained["failed_checks"] = [
            k for k, v in evaluation["promotion"]["checks"].items() if v is not True
        ]
        if args.register:
            from seshat.quality.refresh import register_model

            trained["registered"] = register_model(args.root, args.output_dir)
        _print(trained)
        return 0

    if args.command == "quality-refresh":
        from seshat.quality.refresh import refresh_quality

        _print(refresh_quality(args.root))
        return 0

    raise AssertionError(f"unhandled command {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from seshat.providers.registry import ProviderCallsNotAllowedError

    try:
        return _dispatch(args)
    except (ValueError, FileExistsError, FileNotFoundError, ProviderCallsNotAllowedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
