#!/usr/bin/env python3
"""
scripts/poll_pgns_push_analysis.py

Executable workflow script. This belongs in scripts/, not lib/.

Why not lib/?
- It shells out to git.
- It moves/imports files.
- It is a user workflow entrypoint, not reusable domain logic.

Responsibilities:
1. find PGNs delivered by KDE Connect / phone export
2. copy/move them into games/raw_pgn
3. run chess analysis rebuild
4. git add/commit/push if anything changed

Tune paths/env vars as needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path.home() / "repos" / "toaster_chess_analysis"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import argparse
import hashlib
import shutil
import subprocess
from datetime import datetime

DEFAULT_INBOX = Path.home() / "Downloads" / "kde_connect"
RAW_DIR = REPO / "games" / "raw_pgn"
REBUILD_SCRIPT = REPO / "scripts" / "rebuild_pretty_analysis.py"


def run(cmd: list[str], *, cwd: Path = REPO, check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, text=True, check=check)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique_dest_for_pgn(src: Path, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)

    stem = src.stem
    suffix = src.suffix.lower() or ".pgn"
    digest = file_sha256(src)[:12]

    candidate = raw_dir / f"{stem}_{digest}{suffix}"
    if not candidate.exists():
        return candidate

    # Same file already imported.
    if file_sha256(candidate) == file_sha256(src):
        return candidate

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return raw_dir / f"{stem}_{timestamp}_{digest}{suffix}"


def import_pgns(inbox: Path, *, move: bool = True) -> list[Path]:
    if not inbox.exists():
        print(f"Inbox does not exist: {inbox}")
        return []

    imported = []
    for src in sorted(inbox.glob("*.pgn")):
        dest = unique_dest_for_pgn(src, RAW_DIR)

        if dest.exists() and file_sha256(dest) == file_sha256(src):
            print(f"Already imported: {src.name} -> {dest.relative_to(REPO)}")
            if move:
                src.unlink()
            imported.append(dest)
            continue

        if move:
            shutil.move(str(src), dest)
        else:
            shutil.copy2(src, dest)

        print(f"Imported: {src.name} -> {dest.relative_to(REPO)}")
        imported.append(dest)

    return imported


def git_has_changes() -> bool:
    result = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, text=True, capture_output=True, check=True)
    return bool(result.stdout.strip())


def commit_and_push(message: str) -> None:
    if not git_has_changes():
        print("No git changes to commit.")
        return

    run(["git", "add", "."])
    run(["git", "commit", "-m", message])
    run(["git", "push"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inbox", type=Path, default=DEFAULT_INBOX, help="Directory containing incoming PGN files.")
    parser.add_argument("--copy", action="store_true", help="Copy PGNs instead of moving them out of inbox.")
    parser.add_argument("--force", action="store_true", help="Force rebuild all existing reports.")
    parser.add_argument("--no-push", action="store_true", help="Run import/rebuild but do not git commit/push.")
    args = parser.parse_args()

    imported = import_pgns(args.inbox, move=not args.copy)

    rebuild_cmd = ["python", str(REBUILD_SCRIPT)]
    if args.force:
        rebuild_cmd.append("--force")
    run(rebuild_cmd)

    if args.no_push:
        print("Skipping git commit/push due to --no-push.")
        return

    if imported:
        msg = f"Analyze {len(imported)} imported PGN(s)"
    else:
        msg = "Rebuild chess analysis"

    commit_and_push(msg)


if __name__ == "__main__":
    main()
