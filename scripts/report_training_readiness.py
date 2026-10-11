"""Report real-data training readiness (thin wrapper around ``seshat.data.readiness``).

Only validated, verified, uncensored, controlled real API observations count as
usable, and they are counted per split of the exact leakage-safe assignment that
``prepare-training`` would make. This command sends no API requests and trains
nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from seshat.data.readiness import readiness_report, render_readiness_markdown


def readiness(root: str | Path) -> dict[str, Any]:
    return readiness_report(root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = readiness(args.root)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    path.with_suffix(".md").write_text(render_readiness_markdown(output), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
