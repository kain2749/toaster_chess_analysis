#!/usr/bin/env python3

import datetime as dt
import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path


REPO = Path.home() / "repos" / "toaster_chess_analysis"

INCOMING = Path.home() / "Downloads" / "kde_connect"
PROCESSED = INCOMING / "processed"
RAW_DIR = REPO / "games" / "raw_pgn"

REBUILD_SCRIPT = REPO / "scripts" / "rebuild_pretty_analysis.py"


def safe_slug(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "game"


def sha256_short(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def run(cmd: list[str], cwd: Path = REPO) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def git_has_changes() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def main() -> int:
    INCOMING.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    pgns = sorted(INCOMING.glob("*.pgn"))

    if not pgns:
        print(f"No PGNs found in {INCOMING}")
        return 0

    copied = []

    for src in pgns:
        digest = sha256_short(src)
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = safe_slug(src.stem)

        dst = RAW_DIR / f"{timestamp}_{safe_name}_{digest}.pgn"

        print(f"Copying {src} -> {dst}")
        shutil.copy2(src, dst)
        copied.append(dst)

        processed_dst = PROCESSED / src.name
        if processed_dst.exists():
            processed_dst = PROCESSED / f"{timestamp}_{src.name}"

        print(f"Moving processed source {src} -> {processed_dst}")
        shutil.move(str(src), processed_dst)

    print("Rebuilding analysis...")
    run([sys.executable, str(REBUILD_SCRIPT)])

    run(["git", "add", "."])

    if not git_has_changes():
        print("No git changes after rebuild.")
        return 0

    msg = f"Analyze {len(copied)} new phone chess game{'s' if len(copied) != 1 else ''}"
    run(["git", "commit", "-m", msg])
    run(["git", "push"])

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
