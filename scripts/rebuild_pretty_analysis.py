#!/usr/bin/env python3

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import chess
import chess.engine
import chess.pgn
import chess.svg


REPO = Path.home() / "repos" / "toaster_chess_analysis"
RAW_DIR = REPO / "games" / "raw_pgn"
ANALYSIS_DIR = REPO / "analysis"
ASSET_DIR = ANALYSIS_DIR / "assets"
INDEX = ANALYSIS_DIR / "index.md"

DEPTH = 14
MULTIPV = 3

LOSS_INACCURACY = 120
LOSS_MISTAKE = 300
LOSS_BLUNDER = 600

MATE_CP = 100000

MAX_KEY_MOMENTS_WITH_BOARDS = 6
BOARD_SIZE = 520

USE_OLLAMA = os.getenv("TOASTER_USE_OLLAMA", "1") != "0"
OLLAMA_MODEL = os.getenv("TOASTER_OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_URL = os.getenv("TOASTER_OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_TIMEOUT = int(os.getenv("TOASTER_OLLAMA_TIMEOUT", "90"))
OLLAMA_MAX_NOTES = int(os.getenv("TOASTER_OLLAMA_MAX_NOTES", "999999"))  # Kept for compatibility; default is effectively unlimited.

LLM_CACHE_DIR = ANALYSIS_DIR / "llm_cache"


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


def read_clean_pgn_text(path: Path) -> str:
    data = path.read_bytes()
    data = data.replace(b"\x00", b"")
    return data.decode("utf-8", errors="replace").strip()


def parse_game(path: Path) -> chess.pgn.Game:
    text = read_clean_pgn_text(path)
    game = chess.pgn.read_game(io.StringIO(text))

    if game is None:
        raise RuntimeError(f"No PGN game found in {path}")

    return game


def white_cp(score: chess.engine.PovScore) -> int:
    pov = score.white()
    mate = pov.mate()

    if mate is not None:
        return MATE_CP if mate > 0 else -MATE_CP

    return pov.score() or 0


def fmt_eval(cp: int) -> str:
    if cp >= MATE_CP:
        return "White has mate"
    if cp <= -MATE_CP:
        return "Black has mate"
    return f"{cp / 100:+.2f}"


def mover_loss(eval_before: int, eval_after: int, mover_is_white: bool) -> int:
    if mover_is_white:
        return eval_before - eval_after
    return eval_after - eval_before


def san_or_unknown(board: chess.Board, move: Optional[chess.Move]) -> str:
    if move is None:
        return "unknown"

    try:
        return board.san(move)
    except Exception:
        return "unknown"


def game_to_clean_pgn_text(game: chess.pgn.Game) -> str:
    exporter = chess.pgn.StringExporter(
        headers=True,
        variations=False,
        comments=False,
    )
    return game.accept(exporter).strip()


def report_stem(game: chess.pgn.Game, src: Path) -> str:
    date = game.headers.get("Date", "unknown").replace(".", "-")
    white = safe_slug(game.headers.get("White", "white"))
    black = safe_slug(game.headers.get("Black", "black"))
    base = safe_slug(src.stem)
    return f"{date}_{white}_vs_{black}_{base}"


def report_filename(game: chess.pgn.Game, src: Path) -> str:
    return f"{report_stem(game, src)}.md"


def classify_loss(loss_cp: int) -> str:
    if loss_cp >= LOSS_BLUNDER:
        return "Blunder"
    if loss_cp >= LOSS_MISTAKE:
        return "Mistake"
    if loss_cp >= LOSS_INACCURACY:
        return "Inaccuracy"
    if loss_cp <= 15:
        return "Best"
    if loss_cp <= 50:
        return "Great"
    return "Good"


def label_emoji(label: str) -> str:
    return {
        "Book": "📖",
        "Checkmate": "🏁",
        "Brilliant-ish": "💎",
        "Best": "✅",
        "Great": "🔥",
        "Good": "👍",
        "Inaccuracy": "⚠️",
        "Mistake": "❌",
        "Blunder": "💀",
    }.get(label, "•")


def classify_move(
    board_before: chess.Board,
    move: chess.Move,
    best_move: Optional[chess.Move],
    loss_cp: int,
    terminal: bool,
    ply: int,
) -> tuple[str, str]:
    if terminal:
        if board_before.gives_check(move):
            return (
                "Checkmate",
                "Game-ending forcing move. The toaster is not allowed to call a mating move a blunder.",
            )

        return (
            "Good",
            "Final move before the game ended. Treat this as resignation/adjudication territory.",
        )

    if ply <= 6 and loss_cp < LOSS_INACCURACY:
        return (
            "Book",
            "Early opening move. Not using a real book database yet, so this is only a soft label.",
        )

    label = classify_loss(loss_cp)

    if label == "Blunder":
        return (
            label,
            "Major eval loss. Stockfish thinks this seriously changed the game.",
        )

    if label == "Mistake":
        return (
            label,
            "Significant eval loss. Probably missed a tactic, defense, or forcing move.",
        )

    if label == "Inaccuracy":
        return (
            label,
            "Small-to-medium eval loss. Playable, but Stockfish wanted cleaner.",
        )

    is_check = board_before.gives_check(move)
    is_capture = board_before.is_capture(move)
    is_best = best_move == move

    if loss_cp <= 30 and (is_check or is_capture) and is_best:
        return (
            "Brilliant-ish",
            "Forcing move that Stockfish likes. Not official Chess.com magic, but tactically notable.",
        )

    if label == "Best":
        return (
            label,
            "At or very near Stockfish's preferred move.",
        )

    if label == "Great":
        return (
            label,
            "Very close to best. No practical complaint.",
        )

    return (
        "Good",
        "Reasonable move. Some engine loss, but not enough to care much.",
    )


def side_from_bool(is_white: bool) -> str:
    return "White" if is_white else "Black"


def board_asset_rel(path: Path) -> str:
    return path.relative_to(ANALYSIS_DIR).as_posix()


def write_board_svg(
    board: chess.Board,
    out_path: Path,
    lastmove: Optional[chess.Move] = None,
    best_move: Optional[chess.Move] = None,
    flipped: bool = False,
) -> None:
    arrows = []

    if best_move is not None:
        arrows.append(chess.svg.Arrow(best_move.from_square, best_move.to_square, color="#cc0000"))

    svg = chess.svg.board(
        board=board,
        size=BOARD_SIZE,
        coordinates=True,
        lastmove=lastmove,
        arrows=arrows,
        flipped=flipped,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg, encoding="utf-8")


def final_board_from_game(game: chess.pgn.Game) -> tuple[chess.Board, Optional[chess.Move]]:
    board = game.board()
    last_move = None

    for move in game.mainline_moves():
        board.push(move)
        last_move = move

    return board, last_move


def row_cache_key(row: dict) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "fen": row["board_before"].fen(),
        "side": row["side"],
        "played": row["played"],
        "best": row["best"],
        "label": row["label"],
        "eval_before": fmt_eval(row["eval_before"]),
        "eval_after": fmt_eval(row["eval_after"]),
        "loss_cp": row["loss_cp"],
        "prompt_version": "toaster_ollama_v2_one_sentence",
    }

    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def ollama_prompt_for_row(row: dict) -> str:
    return f"""You are writing ONE sentence of chess commentary for a casual player reviewing a phone game.

Use Stockfish's verdict as truth. Do not disagree with the engine.
Do not invent long forced lines.
Do not guess material facts.
Do not call something a sacrifice unless the position clearly supports that.
Do not say "official brilliant" or "Chess.com brilliant."
Do not use bullet points, markdown lists, headings, or preambles.
Do not write "Here's a note" or anything like that.
Write exactly one concise sentence, maximum 35 words.
Make the sentence position-specific: mention the played move, the engine's preferred move if relevant, or the practical idea in the position.
If unsure, say what Stockfish's eval says changed and keep it humble.

Position before the move:
FEN: {row["board_before"].fen()}

Move being reviewed:
Side: {row["side"]}
Played move: {row["played"]}
Label: {row["label"]}

Engine facts:
Eval before: {fmt_eval(row["eval_before"])}
Eval after: {fmt_eval(row["eval_after"])}
Stockfish preferred: {row["best"]}
Centipawn loss: {row["loss_cp"]}

Fallback explanation for context, not for copying:
{row["reason"]}

One sentence only.
"""


def call_ollama(prompt: str) -> str:
    body = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 80,
        },
    }

    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    return data.get("response", "").strip()


def clean_ollama_sentence(text: str) -> str:
    text = text.strip()

    # Models love ignoring instructions and returning bullets/preambles. Beat it into one sentence.
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*•]+\s*", "", line)
        line = re.sub(r"^\d+[.)]\s*", "", line)
        if line.lower().startswith(("here's", "here is", "note:", "chess note:")):
            continue
        lines.append(line)

    text = " ".join(lines).strip()
    text = re.sub(r"\s+", " ", text)

    # Keep first sentence-ish chunk.
    m = re.search(r"(.+?[.!?])(?:\s|$)", text)
    if m:
        text = m.group(1).strip()

    words = text.split()
    if len(words) > 35:
        text = " ".join(words[:35]).rstrip(",;:") + "."

    return text or "Stockfish flags this as a position worth reviewing, but the local model did not produce a useful note."


def ollama_explanation_for_row(row: dict, note_index: int) -> str:
    if not USE_OLLAMA:
        return row["reason"]

    # OLLAMA_MAX_NOTES is kept as an emergency throttle only. Default is effectively unlimited.
    if note_index >= OLLAMA_MAX_NOTES:
        return row["reason"]

    LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache_key = row_cache_key(row)
    cache_path = LLM_CACHE_DIR / f"{cache_key}.md"

    if cache_path.exists():
        cached = cache_path.read_text(encoding="utf-8").strip()
        if cached:
            return cached

    prompt = ollama_prompt_for_row(row)

    try:
        explanation = call_ollama(prompt)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
        return f"{row['reason']}\n\n_Ollama note unavailable: `{type(e).__name__}`._"

    if not explanation:
        return row["reason"]

    explanation = clean_ollama_sentence(explanation)
    cache_path.write_text(explanation + "\n", encoding="utf-8")
    return explanation


def analyze_game(engine: chess.engine.SimpleEngine, pgn_path: Path) -> Path:
    game = parse_game(pgn_path)
    board = game.board()

    stem = report_stem(game, pgn_path)
    out_path = ANALYSIS_DIR / f"{stem}.md"
    game_asset_dir = ASSET_DIR / stem

    rows = []
    key_moments = []

    for ply, move in enumerate(game.mainline_moves(), start=1):
        before = board.copy()
        mover_is_white = before.turn == chess.WHITE
        side = side_from_bool(mover_is_white)

        played_san = before.san(move)

        info_before = engine.analyse(
            before,
            chess.engine.Limit(depth=DEPTH),
            multipv=MULTIPV,
        )

        if isinstance(info_before, dict):
            info_before = [info_before]

        best_info = info_before[0]
        best_move = best_info.get("pv", [None])[0]
        best_san = san_or_unknown(before, best_move)
        eval_before = white_cp(best_info["score"])

        after = before.copy()
        after.push(move)
        terminal = after.is_game_over(claim_draw=True)

        if after.is_checkmate():
            eval_after = MATE_CP if mover_is_white else -MATE_CP
            loss_cp = 0
        elif terminal:
            eval_after = eval_before
            loss_cp = 0
        else:
            info_after = engine.analyse(after, chess.engine.Limit(depth=DEPTH))
            eval_after = white_cp(info_after["score"])
            loss_cp = max(0, mover_loss(eval_before, eval_after, mover_is_white))

        label, reason = classify_move(
            board_before=before,
            move=move,
            best_move=best_move,
            loss_cp=loss_cp,
            terminal=terminal,
            ply=ply,
        )

        row = {
            "ply": ply,
            "move_no": before.fullmove_number,
            "side": side,
            "played": played_san,
            "best": best_san,
            "best_move": best_move,
            "eval_before": eval_before,
            "eval_after": eval_after,
            "loss_cp": loss_cp,
            "label": label,
            "reason": reason,
            "board_before": before,
            "board_after": after,
            "move": move,
            "before_svg": None,
            "after_svg": None,
        }

        rows.append(row)

        if label in {
            "Checkmate",
            "Brilliant-ish",
            "Blunder",
            "Mistake",
            "Inaccuracy",
        }:
            key_moments.append(row)

        board.push(move)

    # Board assets for key moments.
    for row in key_moments[:MAX_KEY_MOMENTS_WITH_BOARDS]:
        prefix = f"ply_{row['ply']:03d}_{safe_slug(row['played'])}"

        before_svg = game_asset_dir / f"{prefix}_before.svg"
        after_svg = game_asset_dir / f"{prefix}_after.svg"

        write_board_svg(
            row["board_before"],
            before_svg,
            lastmove=None,
            best_move=row["best_move"],
            flipped=False,
        )

        write_board_svg(
            row["board_after"],
            after_svg,
            lastmove=row["move"],
            best_move=None,
            flipped=False,
        )

        row["before_svg"] = board_asset_rel(before_svg)
        row["after_svg"] = board_asset_rel(after_svg)

    # Final board asset.
    final_board, final_lastmove = final_board_from_game(game)
    final_svg = game_asset_dir / "final_position.svg"
    write_board_svg(final_board, final_svg, lastmove=final_lastmove, best_move=None, flipped=False)

    write_report(
        game=game,
        pgn_path=pgn_path,
        out_path=out_path,
        rows=rows,
        key_moments=key_moments,
        final_svg=board_asset_rel(final_svg),
    )

    return out_path


def write_report(
    game: chess.pgn.Game,
    pgn_path: Path,
    out_path: Path,
    rows: list[dict],
    key_moments: list[dict],
    final_svg: str,
) -> None:
    headers = game.headers
    white = headers.get("White", "White")
    black = headers.get("Black", "Black")
    result = headers.get("Result", "?")
    date = headers.get("Date", "unknown")

    blunders = [r for r in rows if r["label"] == "Blunder"]
    mistakes = [r for r in rows if r["label"] == "Mistake"]
    inaccuracies = [r for r in rows if r["label"] == "Inaccuracy"]
    good_stuff = [r for r in rows if r["label"] in {"Best", "Great", "Brilliant-ish", "Checkmate"}]

    worst = max(rows, key=lambda r: r["loss_cp"], default=None)
    final_row = rows[-1] if rows else None

    lines = []

    lines += [
        f"# {white} vs {black}",
        "",
        f"**Result:** {result}  ",
        f"**Date:** {date}  ",
        f"**Source:** `{pgn_path.name}`  ",
        f"**Engine:** Stockfish depth {DEPTH}",
        "",
        "---",
        "",
        "## Toaster Summary",
        "",
    ]

    if worst and worst["loss_cp"] >= LOSS_INACCURACY:
        lines += [
            f"Biggest toaster scream: **{worst['move_no']}. {worst['played']}** by **{worst['side']}**.",
            "",
            f"- **Label:** {label_emoji(worst['label'])} **{worst['label']}**",
            f"- **Eval:** {fmt_eval(worst['eval_before'])} → {fmt_eval(worst['eval_after'])}",
            f"- **Stockfish preferred:** `{worst['best']}`",
            f"- **Loss:** {worst['loss_cp']} cp",
            "",
        ]
    else:
        lines += [
            "No giant tactical crime detected at this depth.",
            "",
        ]

    if final_row:
        lines += [
            f"Final move recorded by parser: **{final_row['move_no']}. {final_row['played']}** by **{final_row['side']}**.",
            "",
        ]

    lines += [
        f"- 💀 Blunders: **{len(blunders)}**",
        f"- ❌ Mistakes: **{len(mistakes)}**",
        f"- ⚠️ Inaccuracies: **{len(inaccuracies)}**",
        f"- ✅ Good / best / tactical moves: **{len(good_stuff)}**",
        "",
        "---",
        "",
        "## Final Position",
        "",
        f"![Final position]({final_svg})",
        "",
        "---",
        "",
        "## Key Moments",
        "",
    ]

    if not key_moments:
        lines += [
            "Nothing crossed the highlight threshold. Either the game was clean, too short, or Stockfish did not find enough to yell about.",
            "",
        ]
    else:
        for note_index, r in enumerate(key_moments[:12]):
            explanation = ollama_explanation_for_row(r, note_index)

            lines += [
                f"### {label_emoji(r['label'])} {r['label']}: {r['move_no']}. {r['played']} by {r['side']}",
                "",
                explanation,
                "",
                f"- **Eval:** {fmt_eval(r['eval_before'])} → {fmt_eval(r['eval_after'])}",
                f"- **Preferred:** `{r['best']}`",
                f"- **Loss:** {r['loss_cp']} cp",
                "",
            ]

            if r.get("before_svg"):
                lines += [
                    f"**Before {r['move_no']}. {r['played']}**",
                    "",
                    f"![Before {r['move_no']}. {r['played']}]({r['before_svg']})",
                    "",
                ]

            if r.get("after_svg"):
                lines += [
                    f"**After {r['move_no']}. {r['played']}**",
                    "",
                    f"![After {r['move_no']}. {r['played']}]({r['after_svg']})",
                    "",
                ]

    lines += [
        "---",
        "",
        "## Human Notes",
        "",
        human_notes(rows),
        "",
        "---",
        "",
        "<details>",
        "<summary>Compact move list</summary>",
        "",
    ]

    for r in rows:
        lines += [
            f"- {label_emoji(r['label'])} **{r['move_no']}. {r['played']}** "
            f"({r['side']}, {r['label']}) — "
            f"{fmt_eval(r['eval_before'])} → {fmt_eval(r['eval_after'])}, "
            f"preferred `{r['best']}`, loss {r['loss_cp']} cp",
        ]

    lines += [
        "",
        "</details>",
        "",
        "---",
        "",
        "<details>",
        "<summary>Raw engine table</summary>",
        "",
        "| Ply | Move | Side | Played | Label | Eval Before | Eval After | Preferred | Loss |",
        "|---:|---:|---|---|---|---:|---:|---|---:|",
    ]

    for r in rows:
        lines.append(
            f"| {r['ply']} | {r['move_no']} | {r['side']} | {r['played']} | "
            f"{r['label']} | {fmt_eval(r['eval_before'])} | {fmt_eval(r['eval_after'])} | "
            f"{r['best']} | {r['loss_cp']} |"
        )

    lines += [
        "",
        "</details>",
        "",
        "---",
        "",
        "<details>",
        "<summary>Clean PGN</summary>",
        "",
        "```pgn",
        game_to_clean_pgn_text(game),
        "```",
        "",
        "</details>",
        "",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")


def human_notes(rows: list[dict]) -> str:
    if not rows:
        return "No moves found."

    notes = []

    blunders = [r for r in rows if r["label"] == "Blunder"]
    mistakes = [r for r in rows if r["label"] == "Mistake"]
    mates = [r for r in rows if r["label"] == "Checkmate"]

    if blunders:
        first = blunders[0]
        notes.append(
            f"The first major collapse was **{first['move_no']}. {first['played']}** by **{first['side']}**. "
            f"That is probably the main position to review."
        )

    if mistakes:
        first = mistakes[0]
        notes.append(
            f"There was also a notable mistake at **{first['move_no']}. {first['played']}** by **{first['side']}**."
        )

    if mates:
        mate = mates[-1]
        notes.append(
            f"The game-ending tactic was **{mate['move_no']}. {mate['played']}**."
        )

    if not notes:
        notes.append(
            "Nothing dramatic stood out. Review the compact move list if you want smaller engine complaints."
        )

    return "\n\n".join(notes)


def update_index() -> None:
    reports = sorted(p for p in ANALYSIS_DIR.glob("*.md") if p.name != "index.md")

    lines = [
        "# Chess Analysis Index",
        "",
    ]

    if not reports:
        lines.append("No games analyzed yet.")
    else:
        for p in reports:
            title = p.stem.replace("_", " ")
            lines.append(f"- [{title}]({p.name})")

    lines.append("")
    INDEX.write_text("\n".join(lines), encoding="utf-8")


def remove_existing_assets_for_report(game: chess.pgn.Game, pgn_path: Path) -> None:
    stem = report_stem(game, pgn_path)
    asset_dir = ASSET_DIR / stem

    if asset_dir.exists():
        shutil.rmtree(asset_dir)


def rebuild_all(force: bool = False) -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    engine_path = find_stockfish()
    pgns = sorted(RAW_DIR.glob("*.pgn"))

    if not pgns:
        INDEX.write_text("# Chess Analysis Index\n\nNo PGNs found.\n", encoding="utf-8")
        print(f"No PGNs found in {RAW_DIR}")
        return

    analyzed = 0
    skipped = 0

    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        for pgn in pgns:
            game = parse_game(pgn)
            out_path = ANALYSIS_DIR / report_filename(game, pgn)

            if out_path.exists() and not force:
                print(f"Skipping existing report: {out_path.relative_to(REPO)}")
                skipped += 1
                continue

            if force:
                remove_existing_assets_for_report(game, pgn)

            print(f"Analyzing {pgn.relative_to(REPO)}")
            out = analyze_game(engine, pgn)
            print(f"Wrote {out.relative_to(REPO)}")
            analyzed += 1

    update_index()
    print(f"Done. Analyzed {analyzed}, skipped {skipped}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate all reports even if analysis markdown already exists.",
    )
    args = parser.parse_args()

    rebuild_all(force=args.force)


if __name__ == "__main__":
    main()
