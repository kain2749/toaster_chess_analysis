#!/usr/bin/env python3
"""
scripts/rebuild_pretty_analysis.py

Thin rebuild entrypoint.

Chess analysis lives in lib/analysis.py.
Ollama/MariaDB lives in lib/ollama_phrase_memory.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path.home() / "repos" / "toaster_chess_analysis"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import argparse
import os

from lib.analysis import AnalysisPaths, NoopNarrator, rebuild_all
from lib.ollama_phrase_memory import MysqlConfig, OllamaConfig, OllamaPhraseMemory, ToasterOllamaNarrator

def build_narrator(paths: AnalysisPaths, *, force: bool = False):
    if os.getenv("TOASTER_USE_OLLAMA", "1") == "0":
        return NoopNarrator()

    memory = None
    if os.getenv("TOASTER_USE_PHRASE_MEMORY", "1") != "0":
        memory = OllamaPhraseMemory(MysqlConfig())
        try:
            memory.init_db()
        except Exception as exc:
            print(f"Phrase memory init failed; continuing without DB memory: {exc}")
            memory = None

    debug_dir = None
    if os.getenv("TOASTER_DEBUG_OLLAMA", "0") == "1":
        debug_dir = paths.analysis_dir / "debug_ollama_io"

    return ToasterOllamaNarrator(
        memory=memory,
        ollama_cfg=OllamaConfig(),
        debug_dir=debug_dir,
        force_llm_regen=force,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Regenerate all reports even if markdown already exists.")
    args = parser.parse_args()

    paths = AnalysisPaths.from_repo(REPO)
    narrator = build_narrator(paths, force=args.force)

    try:
        rebuild_all(paths, narrator, force=args.force)
    finally:
        stop = getattr(narrator, "stop_model", None)
        if callable(stop):
            stop()


if __name__ == "__main__":
    main()
