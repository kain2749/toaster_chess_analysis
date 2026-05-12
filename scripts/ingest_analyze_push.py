#!/usr/bin/env python3

import datetime as dt
import re
import shutil
import subprocess
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
        found = shutil.which(candidate)
        if found:
            return found
        p = Path(candidate)
        if p.exists():
            return str(p)
    raise RuntimeError("Stockfish not found. Try: sudo apt install stockfish")


def white_cp(score: chess.engine.PovScore) -> int:
    pov = score.white()
    mate = pov.mate()
    if mate is not None:
        return 100000 if mate > 0 else -100000
    return pov.score() or 0


def fmt_eval(cp: int) -> str:
    if cp >= 90000:
        return "White mate"
    if cp <= -90000:
        return "Black mate"
    return f"{cp / 100:+.2f}"


def fmt_swing(cp: int) -> str:
    return f"{cp} cp"


def game_report_name(game: chess.pgn.Game, src: Path) -> str:
    date = game.headers.get("Date", "unknown").replace(".", "-")
    white = safe_slug(game.headers.get("White", "white"))
    black = safe_slug(game.headers.get("Black", "black"))
    base = safe_slug(src.stem)
    return f"{date}_{white}_vs_{black}_{base}"


def classify_loss(loss_cp: int) -> str:
    if loss_cp >= 700:
        return "Blunder"
    if loss_cp >= 300:
        return "Mistake"
    if loss_cp >= 150:
        return "Inaccuracy"
    return "Normal"


def analyze_game(pgn_path: Path, engine_path: str) -> Path:
    with pgn_path.open("r", encoding="utf-8", errors="replace") as f:
        game = chess.pgn.read_game(f)

    if game is None:
        raise RuntimeError(f"No PGN game found in {pgn_path}")

    name = game_report_name(game, pgn_path)
    out_path = ANALYSIS_DIR / f"{name}.md"

    board = game.board()
    rows = []
    critical = []

    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        for ply, move in enumerate(game.mainline_moves(), start=1):
            before = board.copy()
            mover_is_white = before.turn == chess.WHITE
            mover = "White" if mover_is_white else "Black"

            info_before = engine.analyse(before, chess.engine.Limit(depth=DEPTH))
            eval_before = white_cp(info_before["score"])
            best = info_before.get("pv", [None])[0]

            san = before.san(move)
            best_san = before.san(best) if best else "unknown"

            board.push(move)

            info_after = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
            eval_after = white_cp(info_after["score"])

            # Loss from the mover's perspective.
            # White wants eval to rise. Black wants eval to fall.
            loss_cp = (eval_before - eval_after) if mover_is_white else (eval_after - eval_before)

            row = {
                "ply": ply,
                "move_no": before.fullmove_number,
                "side": mover,
                "san": san,
                "eval_before": eval_before,
                "eval_after": eval_after,
                "best": best_san,
                "loss_cp": loss_cp,
                "classification": classify_loss(loss_cp),
            }

            rows.append(row)

            if loss_cp >= SWING_THRESHOLD_CP:
                critical.append(row)

    headers = game.headers
    result = headers.get("Result", "?")
    white = headers.get("White", "White")
    black = headers.get("Black", "Black")

    lines = []

    lines += [
        f"# {white} vs {black}",
        "",
        f"**Result:** {result}  ",
        f"**Date:** {headers.get('Date', 'unknown')}  ",
        f"**Source PGN:** `{pgn_path.name}`  ",
        f"**Engine:** Stockfish depth {DEPTH}",
        "",
        "---",
        "",
        "## Quick Read",
        "",
    ]

    if critical:
        worst = max(critical, key=lambda r: r["loss_cp"])
        lines += [
            f"Biggest engine complaint: **{worst['move_no']}. {worst['san']}** by **{worst['side']}**.",
            "",
            f"- Classification: **{worst['classification']}**",
            f"- Eval before: **{fmt_eval(worst['eval_before'])}**",
            f"- Eval after: **{fmt_eval(worst['eval_after'])}**",
            f"- Engine preferred: **{worst['best']}**",
            f"- Loss: **{fmt_swing(worst['loss_cp'])}**",
            "",
        ]
    else:
        lines += [
            "No major eval crashes found at the current threshold.",
            "",
        ]

    lines += [
        "---",
        "",
        "## Critical Moments",
        "",
    ]

    if critical:
        for r in critical[:10]:
            lines += [
                f"### {r['classification']}: {r['move_no']}. {r['san']} by {r['side']}",
                "",
                f"- Eval before: **{fmt_eval(r['eval_before'])}**",
                f"- Eval after: **{fmt_eval(r['eval_after'])}**",
                f"- Engine preferred: **{r['best']}**",
                f"- Loss: **{fmt_swing(r['loss_cp'])}**",
                "",
            ]
    else:
        lines += ["No critical moves above threshold.", ""]

    lines += [
        "---",
        "",
        "## Move-by-Move Engine Table",
        "",
        "| Ply | Move | Side | Played | Eval Before | Eval After | Preferred | Loss | Label |",
        "|---:|---:|---|---|---:|---:|---|---:|---|",
    ]

    for r in rows:
        lines.append(
            f"| {r['ply']} | {r['move_no']} | {r['side']} | {r['san']} | "
            f"{fmt_eval(r['eval_before'])} | {fmt_eval(r['eval_after'])} | "
            f"{r['best']} | {r['loss_cp']} | {r['classification']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## PGN",
        "",
        "```pgn",
        pgn_path.read_text(encoding="utf-8", errors="replace").strip(),
        "```",
        "",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def update_index():
    reports = sorted(p for p in ANALYSIS_DIR.glob("*.md") if p.name != "index.md")

    lines = [
        "# Chess Analysis Index",
        "",
    ]

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
