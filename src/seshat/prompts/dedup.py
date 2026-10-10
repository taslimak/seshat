"""Exact and near-duplicate detection for prompts.

* **Exact**: identical ``content_hash`` (normalized text + tool names).
* **Near**: Jaccard similarity of shingle sets (word unigrams + word bigrams over
  punctuation-free, case-folded tokens) at or above the policy threshold. Mixing
  unigrams and bigrams keeps one inserted word or changed punctuation from hiding
  a paraphrase while word order still matters. Candidate pairs come from an
  inverted index over shingles, so only prompts sharing a shingle are compared.

Near-duplicates must share a ``prompt_family_id`` so they land in the same split;
otherwise a paraphrase in train would leak information about its twin in test.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from seshat.data.models import PromptRecord
from seshat.prompts.bank import normalize_text

_TOKEN = re.compile(r"\w+")


def shingles(text: str) -> frozenset[str]:
    words = _TOKEN.findall(normalize_text(text))
    unigrams = {f"1:{w}" for w in words}
    bigrams = {f"2:{a} {b}" for a, b in zip(words, words[1:], strict=False)}
    return frozenset(unigrams | bigrams)


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


@dataclass(frozen=True)
class NearDuplicate:
    prompt_id: str
    other_id: str
    similarity: float


class SimilarityIndex:
    """Incremental index answering "which indexed prompts is this one close to?"."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self._shingles: dict[str, frozenset[str]] = {}
        self._postings: dict[str, set[str]] = defaultdict(set)

    def add(self, prompt_id: str, text: str) -> None:
        tokens = shingles(text)
        self._shingles[prompt_id] = tokens
        for token in tokens:
            self._postings[token].add(prompt_id)

    def matches(self, text: str, *, exclude: str | None = None) -> list[tuple[str, float]]:
        """Indexed prompt IDs with similarity >= threshold, most similar first."""
        tokens = shingles(text)
        candidates = {pid for token in tokens for pid in self._postings.get(token, ())}
        candidates.discard(exclude or "")
        scored = [(pid, jaccard(tokens, self._shingles[pid])) for pid in candidates]
        hits = [(pid, score) for pid, score in scored if score >= self.threshold]
        return sorted(hits, key=lambda item: (-item[1], item[0]))


def near_duplicate_pairs(prompts: list[PromptRecord], threshold: float) -> list[NearDuplicate]:
    """Every near-duplicate pair in ``prompts`` (each pair reported once)."""
    index = SimilarityIndex(threshold)
    pairs: list[NearDuplicate] = []
    for prompt in sorted(prompts, key=lambda item: item.prompt_id):
        for other, score in index.matches(prompt.text):
            pairs.append(NearDuplicate(prompt.prompt_id, other, round(score, 4)))
        index.add(prompt.prompt_id, prompt.text)
    return pairs
