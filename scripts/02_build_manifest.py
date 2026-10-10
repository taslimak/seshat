"""Thin wrapper around the Seshat dataset CLI."""

from __future__ import annotations

import sys

from seshat.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["build-manifest", *sys.argv[1:]]))
