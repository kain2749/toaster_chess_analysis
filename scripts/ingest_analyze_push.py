#!/usr/bin/env python3

import datetime as dt
import glob
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import chess
import chess.engine
import chess.pgn


REPO = Path.home() / "repos" / "toaster_chess_analysis"
INCOMING = Path.home() / "Downloads" / "kde_connect"
PROCESSED = INCOMING / "processed"

RAW_DIR = REPO / "games" / "raw_pgn"
ANALYSIS_DIR = REPO / "analysis"
INDEX = ANALYSIS_DIR / "index.md"

DEPTH = 12
SWING_THRESHOLD_CP = 150


def safe_slug(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "game"


def find_stockfish() -> str:
    for candidate in ("stockfish", "/usr/games/stockfish", "/usr/bin/stockfish"):
        path = shutil.which(candidate) if candidate == "stockfish" else candidate
        if path and Path(path).exists():
            return path
    raise RuntimeError("Stockfish not found. Try: sudo apt install stockfish")


def score_to_cp(score: chess.engine.PovScore) -> int:
    pov = score.white()
    mate = pov.mate()
    if mate is not None:
        return 100000 if mate > 0 else -100000
    cp = pov.score()
    return cp if cp is not None else 0


def fmt_eval(cp: int) -> str:
    if abs(cp) >= 90000:
        return "M" if cp > 0 else "-M"
    return f"{cp / 100:+.2f}"


def game_name(game: chess.pgn.Game, src: Path) -> str:
    date = game.headers.get("Date", "unknown").replace(".", "-")
    white = safe_slug(game.headers.get("White", "white"))
    black = safe_slug(game.headers.get("Black", "black"))
    base = safe_slug(src.stem)
    return f"{date}_{white}_vs_{black}_{base}"


def analyze_game(pgn_path: Path, engine_path: str) -> Path:
    with pgn_path.open("r", encoding="utf-8", errors="replace") as f:
        game = chess.pgn.read_game(f)

    if game is None:
        raise RuntimeError(f"No PGN game found in {pgn_path}")

    name = game_name(game, pgn_path)
    out_path = ANALYSIS_DIR / f"{name}.md"

    board = game.board()
    rows = []
    critical = []

    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        for ply, move in enumerate(game.mainline_moves(), start=1):
            before = board.copy()
            mover = "White" if before.turn == chess.WHITE else "Black"

            info_before = engine.analyse(before, chess.engine.Limit(depth=DEPTH))
            eval_before = score_to_cp(info_before["score"])
            best = info_before.get("pv", [None])[0]

            board.push(move)

            info_after = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
            eval_after = score_to_cp(info_after["score"])

            # From mover's perspective: bad if White's eval drops or Black's eval rises.
            if mover == "White":
                swing = eval_after - eval_before
                badness = -swing
            else:
                swing = eval_after - eval_before
                badness = swing

            san = before.san(move)
            best_san = before.san(best) if best else "unknown"

            rows.append({
                "ply": ply,
                "move_no": before.fullmove_number,
                "mover": mover,
                "san": san,
                "eval_before": eval_before,
                "eval_after": eval_after,
                "best": best_san,
                "badness": badness,
            })

            if badness >= SWING_THRESHOLD_CP:
                critical.append(rows[-1])

    headers = game.headers

    lines = []
    lines.append(f"# Chess Analysis: {headers.get('White', 'White')} vs {headers.get('Black', 'Black')}")
    lines.append("")
    lines.append(f"- **Result:** {headers.get('Result', '?')}")
    lines.append(f"- **Date:** {headers.get('Date', 'unknown')}")
    lines.append(f"- **Source PGN:** `{pgn_path.name}`")
    lines.append(f"- **Engine:** Stockfish")
    lines.append(f"- **Depth:** {DEPTH}")
    lines.append("")

    lines.append("## Quick Read")
    lines.append("")
    if critical:
        worst = max(critical, key=lambda r: r["badness"])
        lines.append(
            f"The biggest engine complaint is **{worst['move_no']}. {worst['san']}** "
            f"by {worst['mover']}, where the eval moved from "
            f"**{fmt_eval(worst['eval_before'])}** to **{fmt_eval(worst['eval_after'])}**. "
            f"Stockfish preferred **{worst['best']}**."
        )
    else:
        lines.append("No huge eval crashes found at the configured threshold. Either the game was clean, short, or Stockfish depth is too shallow to yell properly.")
    lines.append("")

    lines.append("## Critical Moments")
    lines.append("")
    if critical:
        for r in critical[:10]:
            lines.append(f"### Move {r['move_no']}: {r['mover']} played {r['san']}")
            lines.append("")
            lines.append(f"- Eval before: **{fmt_eval(r['eval_before'])}**")
            lines.append(f"- Eval after: **{fmt_eval(r['eval_after'])}**")
            lines.append(f"- Stockfish preferred: **{r['best']}**")
            lines.append(f"- Swing severity: **{r['badness']} cp**")
            lines.append("")
    else:
        lines.append("No critical moves above threshold.")
        lines.append("")

    lines.append("## Move-by-Move Engine Table")
    lines.append("")
    lines.append("| Ply | Move | Side | Played | Eval Before | Eval After | Stockfish Preferred |")
    lines.append("|---:|---:|---|---|---:|---:|---|")
    for r in rows:
        lines.append(
            f"| {r['ply']} | {r['move_no']} | {r['mover']} | {r['san']} | "
            f"{fmt_eval(r['eval_before'])} | {fmt_eval(r['eval_after'])} | {r['best']} |"
        )

    lines.append("")
    lines.append("## PGN")
    lines.append("")
    lines.append("```pgn")
    lines.append(pgn_path.read_text(encoding="utf-8", errors="replace").strip())
    lines.append("```")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def update_index():
    reports = sorted(ANALYSIS_DIR.glob("*.md"))
    reports = [p for p in reports if p.name != "index.md"]

    lines = ["# Chess Analysis Index", ""]
    if not reports:
        lines.append("No games analyzed yet.")
    else:
        for p in reports:
            lines.append(f"- [{p.stem}]({p.name})")

    lines.append("")
    INDEX.write_text("\n".join(lines), encoding="utf-8")


def git_commit_push(files_analyzed: int):
    subprocess.run(["git", "add", "."], cwd=REPO, check=True)

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )

    if not status.stdout.strip():
        print("Nothing to commit.")
        return

    msg = f"Analyze {files_analyzed} phone chess game{'s' if files_analyzed != 1 else ''}"
    subprocess.run(["git", "commit", "-m", msg], cwd=REPO, check=True)
    subprocess.run(["git", "push"], cwd=REPO, check=True)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    pgns = sorted(INCOMING.glob("*.pgn"))
    if not pgns:
        print(f"No PGNs found in {INCOMING}")
        return 0

    engine_path = find_stockfish()
    analyzed = 0

    for src in pgns:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = RAW_DIR / f"{timestamp}_{safe_slug(src.name)}"

        print(f"Ingesting {src.name} -> {dst.relative_to(REPO)}")
        shutil.copy2(src, dst)

        report = analyze_game(dst, engine_path)
        print(f"Wrote {report.relative_to(REPO)}")

        processed_dst = PROCESSED / src.name
        if processed_dst.exists():
            processed_dst = PROCESSED / f"{timestamp}_{src.name}"
        shutil.move(str(src), processed_dst)

        analyzed += 1

    update_index()
    git_commit_push(analyzed)

    print(f"Done. Analyzed {analyzed} game(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
