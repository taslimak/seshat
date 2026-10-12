"""Forward-validation report: how well did a model predict new requests before they ran?

Regimes are reported side by side and never pooled:

* ``true_forward_shadow`` -- ``live_shadow_preflight`` forecasts persisted before
  dispatch and joined to the observation afterwards (verified
  ``prediction_precedes_observation``);
* ``out_of_fold_backtest`` -- cross-validation forecasts on development rows;
* ``protected_test`` -- one-time forecasts on the protected test set.

Only the first answers "how well does Seshat predict completely new requests before
they are executed"; the backtests are retrospective by construction.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from seshat.quality.metrics import aggregate_metrics

REGIME_ORDER = ("true_forward_shadow", "out_of_fold_backtest", "protected_test")
SLICES = (
    "effort",
    "prompt_category",
    "actual_output_band",
    "predicted_output_band",
    "prompt_length_band",
)
HEADLINE = (
    ("n_observations", "matched observations", 0),
    ("n_censored", "censored", 0),
    ("mae_tokens", "MAE (latent, uncensored)", 1),
    ("median_absolute_error_tokens", "median AE", 1),
    ("wape_pct", "WAPE %", 1),
    ("median_absolute_percentage_error", "median APE %", 1),
    ("smape_pct", "sMAPE %", 1),
    ("rmse_tokens", "RMSE", 1),
    ("mean_signed_error_tokens", "bias (tokens)", 1),
    ("aggregate_percentage_bias", "bias %", 1),
    ("p50_absolute_error_tokens", "p50 |error|", 1),
    ("p90_absolute_error_tokens", "p90 |error|", 1),
    ("p95_absolute_error_tokens", "p95 |error|", 1),
    ("underprediction_rate_pct", "under-prediction %", 1),
    ("overprediction_rate_pct", "over-prediction %", 1),
    ("catastrophic_gt25pct_rate_pct", "catastrophic >25% %", 1),
    ("catastrophic_gt50pct_rate_pct", "catastrophic >50% %", 1),
    ("catastrophic_gt100pct_rate_pct", "catastrophic >100% %", 1),
    ("realized_mae_tokens", "realized MAE (all rows)", 1),
    ("realized_wape_pct", "realized WAPE %", 1),
    ("realized_aggregate_percentage_bias", "realized bias %", 1),
    ("nominal_coverage_pct", "nominal coverage %", 0),
    ("interval_coverage_pct", "coverage % (latent, known rows)", 1),
    ("realized_interval_coverage_pct", "coverage % (realized)", 1),
    ("mean_confidence_margin_tokens", "mean margin (tokens)", 1),
    ("median_confidence_margin_tokens", "median margin (tokens)", 1),
    ("mean_confidence_margin_pct", "mean margin %", 1),
    ("median_confidence_margin_pct", "median margin %", 1),
    ("cost_wape_pct", "cost WAPE %", 1),
    ("preflight_input_mae_tokens", "preflight input MAE (tokens)", 1),
    ("preflight_input_wape_pct", "preflight input WAPE %", 2),
)


def forward_validation_report(
    outcomes: pd.DataFrame, predictor_version: str, *, n_live_predictions: int = 0
) -> dict[str, Any]:
    rows = outcomes[
        (outcomes["predictor_version"] == predictor_version)
        & (outcomes["label_origin"] == "real")
        & outcomes["valid_out_of_sample"].astype(bool)
    ]
    report: dict[str, Any] = {
        "predictor_version": predictor_version,
        "label_origin": "real",
        "live_shadow_predictions": int(n_live_predictions),
        "regimes": {},
    }
    for regime in REGIME_ORDER:
        part = rows[rows["evaluation_regime"] == regime]
        if part.empty:
            report["regimes"][regime] = {"n_observations": 0}
            continue
        entry: dict[str, Any] = {"overall": aggregate_metrics(part), "slices": {}}
        if regime == "true_forward_shadow":
            entry["preflight_methods"] = (
                part["forecast_preflight_method"].fillna("unknown").value_counts().to_dict()
                if "forecast_preflight_method" in part
                else {}
            )
            entry["forecast_used_measured_preflight"] = bool(
                (
                    part["forecast_preflight_input_tokens"].astype(float)
                    == part["preflight_input_tokens"].astype(float)
                ).all()
            )
            entry["all_predictions_precede_observations"] = bool(
                part["prediction_precedes_observation"].all()
            )
        for column in SLICES:
            if column in part:
                entry["slices"][column] = {
                    str(k): aggregate_metrics(g) for k, g in part.groupby(part[column].astype(str))
                }
        report["regimes"][regime] = entry
    shadow = report["regimes"]["true_forward_shadow"]
    report["matched_live_shadow_observations"] = (
        shadow["overall"]["n_observations"] if "overall" in shadow else 0
    )
    return report


def _fmt(value: Any, digits: int) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    return f"{value:,.{digits}f}" if isinstance(value, (int, float)) else str(value)


def render_forward_markdown(report: dict[str, Any]) -> str:
    regimes = report["regimes"]
    lines = [
        f"# Forward validation · `{report['predictor_version']}`",
        "",
        f"- Live shadow predictions persisted before dispatch: "
        f"**{report['live_shadow_predictions']}**; matched to real observations: "
        f"**{report['matched_live_shadow_observations']}**",
        "- Only `true_forward_shadow` measures prediction of new requests before execution; "
        "the other columns are retrospective backtests and are never pooled with it.",
        "",
        "| Metric | " + " | ".join(REGIME_ORDER) + " |",
        "|---|" + "---:|" * len(REGIME_ORDER),
    ]
    for key, label, digits in HEADLINE:
        cells = []
        for regime in REGIME_ORDER:
            overall = regimes[regime].get("overall")
            cells.append(_fmt(overall.get(key), digits) if overall else "–")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    shadow = regimes["true_forward_shadow"]
    if "overall" in shadow:
        lines += [
            "",
            f"Preflight sources: {shadow['preflight_methods']}; all forecasts precede their "
            f"observations: **{shadow['all_predictions_precede_observations']}**.",
        ]
        for column, table in shadow["slices"].items():
            lines += [
                "",
                f"## true_forward_shadow by {column}",
                "",
                "| Level | n | censored | MAE | WAPE % | bias % | realized WAPE % "
                "| coverage % | median margin % |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for level, m in table.items():
                lines.append(
                    f"| {level} | {m['n_observations']} | {m['n_censored']} | "
                    f"{_fmt(m['mae_tokens'], 1)} | {_fmt(m['wape_pct'], 1)} | "
                    f"{_fmt(m['aggregate_percentage_bias'], 1)} | "
                    f"{_fmt(m.get('realized_wape_pct'), 1)} | "
                    f"{_fmt(m.get('realized_interval_coverage_pct'), 1)} | "
                    f"{_fmt(m.get('median_confidence_margin_pct'), 1)} |"
                )
    return "\n".join(lines) + "\n"
