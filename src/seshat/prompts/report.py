"""Bank-wide checks (``prompts check``) and coverage report (``prompts report``)."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from seshat.data.models import PromptRecord
from seshat.prompts.bank import all_prompts, content_hash, load_bank
from seshat.prompts.checks import check_prompt
from seshat.prompts.dedup import near_duplicate_pairs
from seshat.prompts.policy import CollectionPolicy


def bank_violations(bank_dir: str | Path, policy: CollectionPolicy) -> list[dict[str, Any]]:
    """Errors that make the bank unusable (empty list = OK)."""
    bank = load_bank(bank_dir)
    prompts = all_prompts(bank)
    problems: list[dict[str, Any]] = []
    location = {p.prompt_id: str(path) for path, items in bank.items() for p in items}
    ids = Counter(p.prompt_id for p in prompts)
    problems += [
        {"code": "duplicate_prompt_id", "prompt_id": pid} for pid, n in ids.items() if n > 1
    ]
    hashes: dict[str, list[str]] = defaultdict(list)
    for prompt in prompts:
        hashes[content_hash(prompt)].append(prompt.prompt_id)
        for issue in check_prompt(prompt, policy):
            problems.append(
                {
                    "code": issue.code,
                    "prompt_id": prompt.prompt_id,
                    "file": location[prompt.prompt_id],
                    "message": issue.message,
                }
            )
    problems += [
        {"code": "duplicate_content", "prompt_ids": sorted(v)}
        for v in hashes.values()
        if len(v) > 1
    ]
    categories: dict[str, set[str]] = defaultdict(set)
    for prompt in prompts:
        categories[prompt.prompt_family_id].add(prompt.category)
    problems += [
        {"code": "family_spans_categories", "family": f, "categories": sorted(c)}
        for f, c in categories.items()
        if len(c) > 1
    ]
    family = {p.prompt_id: p.prompt_family_id for p in prompts}
    for pair in near_duplicate_pairs(prompts, policy.near_duplicate_threshold):
        if family[pair.prompt_id] != family[pair.other_id]:
            problems.append(
                {
                    "code": "near_duplicates_in_different_families",
                    "prompt_ids": [pair.prompt_id, pair.other_id],
                    "similarity": pair.similarity,
                }
            )
    share = sum(p.is_synthetic for p in prompts) / len(prompts) if prompts else 0.0
    if share > policy.max_synthetic_share:
        problems.append(
            {
                "code": "synthetic_share_exceeded",
                "share": round(share, 3),
                "max": policy.max_synthetic_share,
            }
        )
    return problems


def _length_bucket(prompt: PromptRecord) -> str:
    words = len(prompt.text.split())
    return "<20" if words < 20 else "20-99" if words < 100 else "100-499" if words < 500 else "500+"


def coverage_report(bank_dir: str | Path, policy: CollectionPolicy) -> dict[str, Any]:
    prompts = all_prompts(load_bank(bank_dir))
    approved = [p for p in prompts if p.review_status == "approved"]
    per_category: dict[str, dict[str, Any]] = {}
    for category in sorted({p.category for p in prompts}):
        items = [p for p in prompts if p.category == category]
        per_category[category] = {
            "total": len(items),
            "approved": sum(p.review_status == "approved" for p in items),
            "by_complexity": dict(Counter(p.complexity for p in items)),
        }
    gaps = [c for c, v in per_category.items() if v["approved"] < policy.min_approved_per_category]
    return {
        "prompts": len(prompts),
        "approved": len(approved),
        "by_review_status": dict(Counter(p.review_status for p in prompts)),
        "by_source": dict(Counter(p.source for p in prompts)),
        "by_language": dict(Counter(p.language for p in prompts)),
        "by_complexity": dict(Counter(p.complexity for p in prompts)),
        "by_length_words": dict(Counter(_length_bucket(p) for p in prompts)),
        "structure": {
            "single_turn": sum(p.message_count == 1 for p in prompts),
            "multi_message": sum(p.message_count > 1 for p in prompts),
            "with_tools": sum(bool(p.tools) for p in prompts),
        },
        "synthetic_share": round(sum(p.is_synthetic for p in prompts) / len(prompts), 3)
        if prompts
        else 0.0,
        "per_category": per_category,
        "categories_below_min_approved": gaps,
    }
