"""Deterministic prompt templates for controlled experiments.

Each template builds one prompt from a seeded ``random.Random`` and a target length
stratum (short / medium / long). Content comes from small built-in word banks and
procedurally assembled documents and code, so generation is fully offline and
reproducible. Generated prompts are *unexecuted* until a collection run sends them;
they never carry usage labels on their own.

Templates are reviewed as code. Changing one changes ``TEMPLATE_LIBRARY_VERSION``.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

TEMPLATE_LIBRARY_VERSION = "1.1.0"  # 1.1.0: finer family keys (reasoning, math, classification)

TOPICS = [
    "photosynthesis",
    "compound interest",
    "the water cycle",
    "TCP congestion control",
    "vaccine trials",
    "supply chain resilience",
    "black holes",
    "the French Revolution",
    "inflation targeting",
    "plate tectonics",
    "neural network overfitting",
    "urban heat islands",
    "public key cryptography",
    "coral reef bleaching",
    "the Krebs cycle",
    "database indexing",
    "carbon pricing",
    "antibiotic resistance",
    "the printing press",
    "solar panel efficiency",
]
ENTITIES = [
    ("Northwind Logistics", "Lisbon"),
    ("Aster Biotech", "Toronto"),
    ("Helio Foods", "Nairobi"),
    ("Quanta Retail", "Osaka"),
    ("Brightline Health", "Denver"),
    ("Kestrel Energy", "Oslo"),
    ("Morrow Insurance", "Dublin"),
    ("Palisade Robotics", "Seoul"),
    ("Juniper Credit", "Austin"),
    ("Tidewater Shipping", "Rotterdam"),
    # 1.1.0: wider pools => more independent prompt families per category.
    ("Lumen Analytics", "Melbourne"),
    ("Cobalt Mining", "Santiago"),
    ("Verdant Farms", "Des Moines"),
    ("Harbor Telecom", "Singapore"),
    ("Stratus Airlines", "Doha"),
    ("Granite Construction Group", "Calgary"),
    ("Saffron Hospitality", "Mumbai"),
    ("Polar Cold Chain", "Helsinki"),
    ("Meridian Pharmacy", "Cape Town"),
    ("Atlas Payroll", "Warsaw"),
    ("Beacon Education", "Bogota"),
    ("Cascade Water Utility", "Portland"),
    ("Ember Games", "Montreal"),
    ("Orchid Cosmetics", "Bangkok"),
    ("Summit Outdoor Gear", "Zurich"),
    ("Riverbend Hospital", "Leeds"),
    ("Nimbus Cloud Hosting", "Frankfurt"),
    ("Coral Reef Tourism", "Cairns"),
    ("Ironclad Security", "Tel Aviv"),
    ("Willow Home Furnishings", "Copenhagen"),
]
PEOPLE = [
    "Amara Okafor",
    "Liam Chen",
    "Sofia Rossi",
    "Diego Alvarez",
    "Priya Nair",
    "Jonas Berg",
    "Mei Tanaka",
    "Omar Haddad",
]
LANGUAGES = [
    ("Spanish", "es"),
    ("French", "fr"),
    ("German", "de"),
    ("Japanese", "ja"),
    ("Portuguese", "pt"),
    ("Italian", "it"),
    ("Dutch", "nl"),
    ("Polish", "pl"),
    ("Turkish", "tr"),
    ("Korean", "ko"),
    ("Indonesian", "id"),
    ("Swahili", "sw"),
]
SENTIMENTS = ["positive", "negative", "neutral", "mixed"]
_SENTENCE_PARTS = {
    "subject": [
        "The quarterly review",
        "Our field team",
        "The pilot program",
        "Customer feedback",
        "The audit committee",
        "The regional office",
        "The new pricing model",
        "Warehouse throughput",
        "The support backlog",
        "Vendor onboarding",
    ],
    "verb": [
        "showed",
        "highlighted",
        "confirmed",
        "suggested",
        "revealed",
        "documented",
        "questioned",
        "underscored",
    ],
    "object": [
        "a steady rise in repeat orders",
        "delays in invoice processing",
        "higher churn among annual plans",
        "strong demand in coastal markets",
        "gaps in safety training",
        "unexpected savings from route planning",
        "inconsistent data entry across sites",
        "a need for clearer escalation paths",
        "improved first-contact resolution",
        "rising costs for cold storage",
    ],
    "tail": [
        "during the third quarter.",
        "after the March update.",
        "despite staffing changes.",
        "in two of five regions.",
        "according to the latest survey.",
        "relative to last year.",
        "once the backlog was cleared.",
        "for enterprise accounts.",
    ],
}
_FUNCTION_NAMES = [
    "merge_intervals",
    "parse_invoice",
    "rolling_average",
    "dedupe_contacts",
    "normalize_phone",
    "rate_limiter",
    "flatten_config",
    "top_k_frequent",
    "validate_iban",
    "schedule_jobs",
    "lru_cache_put",
    "parse_csv_line",
    "topological_sort",
    "debounce_events",
    "compress_runs",
    "match_wildcard",
    "group_anagrams",
    "retry_with_backoff",
    "diff_dicts",
    "paginate_results",
    "slugify_title",
    "detect_cycle",
    "bucket_timestamps",
    "reconcile_ledgers",
    "mask_card_number",
    "levenshtein_distance",
    "split_bill",
    "parse_duration",
    "merge_sorted_streams",
    "validate_password",
]
_BUG_KINDS = [
    "an off-by-one error in a loop bound",
    "a mutable default argument",
    "integer division where float division is needed",
    "a missing None check",
    "shadowing of a built-in name",
    "an exception that is caught too broadly",
]


def sentence(rng: random.Random) -> str:
    return " ".join(
        rng.choice(_SENTENCE_PARTS[key]) for key in ("subject", "verb", "object", "tail")
    )


def document(rng: random.Random, target_words: int, title: str) -> str:
    """Procedural report text of roughly ``target_words`` words with headings."""
    paragraphs: list[str] = [f"# {title}"]
    words = 0
    section = 1
    while words < target_words:
        if words and rng.random() < 0.25:
            paragraphs.append(f"## Section {section}")
            section += 1
        para = " ".join(sentence(rng) for _ in range(rng.randint(3, 6)))
        paragraphs.append(para)
        words += len(para.split())
    return "\n\n".join(paragraphs)


def code_snippet(rng: random.Random, lines: int, name: str) -> str:
    body = [
        f"def {name}(items, threshold=0):",
        '    """Process items and return a summary."""',
        "    result = []",
    ]
    while len(body) < lines - 1:
        i = len(body)
        body.append(
            rng.choice(
                [
                    "    for index in range(len(items)):",
                    f"        value = items[index] * {rng.randint(2, 9)}",
                    "        if value > threshold:",
                    f"            result.append(value - {i})",
                    "    total = sum(result) / max(len(result), 1)",
                    f"    cache_{i} = {{k: v for k, v in enumerate(result)}}",
                ]
            )
        )
    body.append("    return result")
    return "\n".join(body)


@dataclass(frozen=True)
class Stratum:
    name: str
    min_words: int
    max_words: int

    def target(self, rng: random.Random) -> int:
        return rng.randint(self.min_words, self.max_words)


@dataclass
class Built:
    """Template output before it becomes a PromptRecord."""

    user: str
    system: str | None = None
    history: list[tuple[str, str]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    subcategory: str | None = None
    difficulty: str = "medium"
    topic_key: str = ""


Builder = Callable[[random.Random, Stratum], Built]


def _length_instruction(rng: random.Random) -> str:
    return rng.choice(
        [
            "",
            "",
            " Keep it under 80 words.",
            " Answer in three bullet points.",
            " Be concise.",
            " Explain in detail.",
            " Use at most 150 words.",
            " Respond in two paragraphs.",
        ]
    )


def _qa(rng: random.Random, s: Stratum) -> Built:
    topic = rng.choice(TOPICS)
    context = (
        ""
        if s.name == "short"
        else "\n\nBackground notes:\n" + document(rng, s.target(rng) - 20, f"Notes on {topic}")
    )
    q = rng.choice(
        [
            "Explain how {t} works",
            "What are the main misconceptions about {t}?",
            "Why does {t} matter to non-specialists?",
        ]
    )
    return Built(
        q.format(t=topic) + "." + _length_instruction(rng) + context,
        subcategory="explanatory",
        topic_key=topic,
        difficulty="low" if s.name == "short" else "medium",
    )


def _short_answer(rng: random.Random, s: Stratum) -> Built:
    topic = rng.choice(TOPICS)
    q = rng.choice(
        [
            "In one sentence, define {t}.",
            "Give a one word answer: is {t} mainly a physical or social phenomenon?",
            "Name the single most important factor in {t}.",
        ]
    )
    return Built(q.format(t=topic), subcategory="factoid", topic_key=topic, difficulty="low")


def _verbose(rng: random.Random, s: Stratum) -> Built:
    topic = rng.choice(TOPICS)
    extra = (
        ""
        if s.name == "short"
        else "\n\nUse this outline as a starting point:\n"
        + document(rng, s.target(rng) - 30, "Outline")
    )
    return Built(
        f"Write a comprehensive, step-by-step explanation of {topic} for a university course, with examples, history and open questions."
        + extra,
        subcategory="long_form",
        topic_key=topic,
        difficulty="high",
    )


def _math(rng: random.Random, s: Stratum) -> Built:
    a, b, c = rng.randint(2, 40), rng.randint(2, 40), rng.randint(2, 9)
    problems = [
        f"A tank fills at {a} liters per minute and drains at {b} liters per minute. Starting with {c * 10} liters, when is it empty or full at 500 liters?",
        f"Prove that the sum of the first n odd numbers equals n^2, then compute it for n = {a}.",
        f"Solve for x: {a}x + {b} = {c}x^2. Show each step.",
        f"How many integers between 1 and {a * 25} are divisible by {c} but not by {c + 1}?",
    ]
    kind = rng.randrange(len(problems))  # same RNG draw as rng.choice
    problem = problems[kind]
    if s.name != "short":
        problem += "\n\nAlso consider these related cases:\n" + "\n".join(
            f"{i + 1}. Replace {a} with {a + i * 3} and explain what changes."
            for i in range(max(1, s.target(rng) // 25))
        )
    return Built(
        problem, subcategory="word_problem", topic_key=f"math-{kind}-{a}", difficulty="high"
    )


def _reasoning(rng: random.Random, s: Stratum) -> Built:
    people = rng.sample(PEOPLE, 4)
    clues = [
        f"{people[0]} sits left of {people[1]}.",
        f"{people[2]} is not at either end.",
        f"{people[3]} sits right of {people[0]}.",
    ]
    if s.name != "short":
        clues += [sentence(rng) for _ in range(max(1, s.target(rng) // 18))]
    return Built(
        "Solve this seating puzzle and justify each deduction.\n"
        + "\n".join(f"- {c}" for c in clues),
        subcategory="logic_puzzle",
        # One family per cast of people (was a single family for the whole category,
        # which forced every reasoning prompt into one split).
        topic_key="seating-" + "|".join(sorted(people)),
        difficulty="high",
    )


def _debugging(rng: random.Random, s: Stratum) -> Built:
    name = rng.choice(_FUNCTION_NAMES)
    lines = {"short": 8, "medium": 30, "long": 90}[s.name]
    bug = rng.choice(_BUG_KINDS)
    return Built(
        f"This Python function has {bug}. Find and fix it, then explain the fix.\n```python\n{code_snippet(rng, lines, name)}\n```",
        subcategory="python_bugfix",
        topic_key=name,
        difficulty="medium",
    )


def _code_generation(rng: random.Random, s: Stratum) -> Built:
    name = rng.choice(_FUNCTION_NAMES)
    spec = f"Write a Python function `{name}` with type hints and docstrings."
    if s.name != "short":
        spec += " Requirements:\n" + "\n".join(
            f"- Handle case {i + 1}: {sentence(rng)}" for i in range(max(2, s.target(rng) // 20))
        )
    return Built(
        spec + " Include unit tests.",
        subcategory="python_function",
        topic_key=name,
        difficulty="medium",
    )


def _code_review(rng: random.Random, s: Stratum) -> Built:
    name = rng.choice(_FUNCTION_NAMES)
    lines = {"short": 10, "medium": 40, "long": 120}[s.name]
    return Built(
        f"Review this code for correctness, performance and readability. List issues by severity.\n```python\n{code_snippet(rng, lines, name)}\n```",
        subcategory="python_review",
        topic_key=name,
        difficulty="medium",
    )


def _summarization(rng: random.Random, s: Stratum) -> Built:
    company, city = rng.choice(ENTITIES)
    doc = document(rng, max(40, s.target(rng)), f"{company} operations report ({city})")
    ask = rng.choice(
        [
            "Summarize the report in three bullet points.",
            "Write a one-paragraph executive summary.",
            "Summarize the key risks in under 100 words.",
        ]
    )
    return Built(
        f"{ask}\n\n{doc}", subcategory="report_summary", topic_key=company, difficulty="low"
    )


def _classification(rng: random.Random, s: Stratum) -> Built:
    reviews = [sentence(rng) for _ in range({"short": 1, "medium": 6, "long": 25}[s.name])]
    labels = ", ".join(SENTIMENTS)
    return Built(
        f"Classify the sentiment of each statement as one of: {labels}. Reply with one label per line.\n"
        + "\n".join(f"{i + 1}. {r}" for i, r in enumerate(reviews)),
        subcategory="sentiment",
        topic_key=f"reviews-{len(reviews)}-{reviews[0]}",
        difficulty="low",
    )


def _extraction(rng: random.Random, s: Stratum) -> Built:
    company, city = rng.choice(ENTITIES)
    person = rng.choice(PEOPLE)
    doc = f"Memo from {person} at {company}, {city}. " + " ".join(
        sentence(rng) for _ in range(max(2, s.target(rng) // 15))
    )
    return Built(
        f"Extract every organization, person, location and date mentioned. Return a table with columns type and value.\n\n{doc}",
        subcategory="entity_extraction",
        topic_key=company,
        difficulty="medium",
    )


def _translation(rng: random.Random, s: Stratum) -> Built:
    language, _ = rng.choice(LANGUAGES)
    text = " ".join(sentence(rng) for _ in range(max(1, s.target(rng) // 15)))
    return Built(
        f"Translate the following text into {language}, preserving tone.\n\n{text}",
        subcategory=f"to_{language.lower()}",
        topic_key=language,
        difficulty="low",
    )


def _creative(rng: random.Random, s: Stratum) -> Built:
    topic = rng.choice(TOPICS)
    form = rng.choice(
        ["a short story", "a poem", "a dialogue between two scientists", "a product launch speech"]
    )
    constraints = (
        ""
        if s.name == "short"
        else "\nConstraints:\n"
        + "\n".join(f"- {sentence(rng)}" for _ in range(max(1, s.target(rng) // 20)))
    )
    return Built(
        f"Write {form} inspired by {topic}." + _length_instruction(rng) + constraints,
        subcategory=form.split()[-1],
        topic_key=topic,
        difficulty="medium",
    )


def _structured_json(rng: random.Random, s: Stratum) -> Built:
    company, city = rng.choice(ENTITIES)
    doc = " ".join(sentence(rng) for _ in range(max(2, s.target(rng) // 15)))
    return Built(
        f'Return a JSON object with keys "company", "city", "risks" (array of strings) and "priority" (low|medium|high) describing this update from {company} in {city}:\n\n{doc}',
        system="You only output valid JSON. No prose.",
        subcategory="json_extraction",
        topic_key=company,
        difficulty="medium",
    )


def _tool_schema(rng: random.Random, index: int, complexity: str) -> dict[str, Any]:
    names = [
        "get_weather",
        "search_orders",
        "create_ticket",
        "lookup_customer",
        "convert_currency",
        "schedule_meeting",
        "fetch_invoice",
    ]
    props: dict[str, Any] = {"query": {"type": "string", "description": "Free-text query"}}
    if complexity != "simple":
        props["filters"] = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["open", "closed", "pending"]},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            },
        }
    if complexity == "complex":
        props["items"] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string"},
                    "quantity": {"type": "integer"},
                    "notes": {"type": "string"},
                },
                "required": ["sku"],
            },
        }
    return {
        "name": names[index % len(names)],
        "description": f"Tool {index + 1}: {sentence(rng)}",
        "input_schema": {"type": "object", "properties": props, "required": ["query"]},
    }


def _tool_calling(rng: random.Random, s: Stratum) -> Built:
    count = {"short": 1, "medium": 3, "long": 6}[s.name]
    complexity = {"short": "simple", "medium": "nested", "long": "complex"}[s.name]
    company, city = rng.choice(ENTITIES)
    tools = [_tool_schema(rng, i, complexity) for i in range(count)]
    return Built(
        f"Using the available tools, find the open orders for {company} in {city} and draft a status update.",
        system="You are an operations assistant. Call tools when you need data.",
        tools=tools,
        subcategory=f"tools_{complexity}",
        topic_key=company,
        difficulty="medium",
    )


def _multi_turn(rng: random.Random, s: Stratum) -> Built:
    topic = rng.choice(TOPICS)
    depth = {"short": 1, "medium": 3, "long": 6}[s.name]
    history: list[tuple[str, str]] = []
    for turn in range(depth):
        history.append(
            ("user", f"Question {turn + 1} about {topic}: {sentence(rng)} What does that imply?")
        )
        history.append(("assistant", " ".join(sentence(rng) for _ in range(2 + turn % 3))))
    system = (
        None
        if rng.random() < 0.4
        else "You are a patient tutor. " + " ".join(sentence(rng) for _ in range(rng.randint(1, 4)))
    )
    return Built(
        f"Thanks. Now connect all of that back to {topic}." + _length_instruction(rng),
        system=system,
        history=history,
        subcategory=f"depth_{depth}",
        topic_key=topic,
        difficulty="medium",
    )


def _long_context(rng: random.Random, s: Stratum) -> Built:
    company, city = rng.choice(ENTITIES)
    words = {"short": 300, "medium": 1500, "long": 4000}[s.name]
    doc = document(rng, words, f"{company} annual review ({city})")
    q = rng.choice(
        [
            "Which sections mention costs, and what do they conclude?",
            "List every recommendation in the document with its section.",
            "What are the three biggest operational risks, with supporting quotes?",
        ]
    )
    return Built(
        f"{doc}\n\nQuestion: {q}",
        subcategory=f"document_{words}w",
        topic_key=company,
        difficulty="high",
    )


TEMPLATES: dict[str, tuple[str, Builder]] = {
    "qa_explain_v1": ("question_answering", _qa),
    "short_answer_v1": ("short_answer", _short_answer),
    "verbose_explanation_v1": ("verbose_explanation", _verbose),
    "math_problem_v1": ("mathematics", _math),
    "logic_puzzle_v1": ("reasoning", _reasoning),
    "debug_python_v1": ("debugging", _debugging),
    "codegen_python_v1": ("code_generation", _code_generation),
    "code_review_v1": ("code_review", _code_review),
    "summarize_report_v1": ("summarization", _summarization),
    "classify_sentiment_v1": ("classification", _classification),
    "extract_entities_v1": ("extraction", _extraction),
    "translate_v1": ("translation", _translation),
    "creative_v1": ("creative_writing", _creative),
    "json_output_v1": ("structured_json", _structured_json),
    "tool_use_v1": ("tool_calling", _tool_calling),
    "multi_turn_v1": ("multi_turn", _multi_turn),
    "long_document_v1": ("long_context_analysis", _long_context),
}
