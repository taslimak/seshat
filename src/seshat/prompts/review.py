"""Review workflow: approve or reject bank prompts (required before live collection)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from seshat.prompts.bank import load_bank, write_prompt_file
from seshat.prompts.checks import check_prompt
from seshat.prompts.policy import CollectionPolicy


def set_review_status(
    bank_dir: str | Path,
    prompt_ids: list[str],
    status: Literal["approved", "rejected", "unreviewed"],
    *,
    reviewer: str,
    policy: CollectionPolicy,
    notes: str | None = None,
) -> int:
    """Update review fields in place. Prompts violating the policy cannot be approved."""
    if not reviewer.strip() and status != "unreviewed":
        raise ValueError("reviewer is required")
    wanted = set(prompt_ids)
    bank = load_bank(bank_dir)
    found: set[str] = set()
    for path, prompts in bank.items():
        changed = False
        updated = []
        for prompt in prompts:
            if prompt.prompt_id in wanted:
                found.add(prompt.prompt_id)
                if status == "approved":
                    issues = check_prompt(prompt, policy)
                    if issues:
                        raise ValueError(
                            f"{prompt.prompt_id} cannot be approved: {[i.code for i in issues]}"
                        )
                prompt = prompt.model_copy(
                    update={
                        "review_status": status,
                        "reviewed_by": None if status == "unreviewed" else reviewer,
                        "reviewed_at": None if status == "unreviewed" else datetime.now(UTC),
                        "review_notes": notes,
                    }
                )
                changed = True
            updated.append(prompt)
        if changed:
            write_prompt_file(path, updated)
    missing = wanted - found
    if missing:
        raise ValueError(f"Unknown prompt ids: {sorted(missing)}")
    return len(found)
