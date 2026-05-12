#!/usr/bin/env python3

import io
import re
import shutil
from pathlib import Path
from typing import Optional

import chess
import chess.engine
import chess.pgn


REPO = Path.home() / "repos" / "toaster_chess_analysis"
RAW_DIR = REPO / "games" / "raw_pgn"
ANALYSIS_DIR = REPO / "analysis"
INDEX = ANALYSIS_DIR / "index.md"

DEPTH = 14
MULTIPV = 3

LOSS_INACCURACY = 120
LOSS_MISTAKE = 300
LOSS_BLUNDER = 600

MATE_CP = 100000


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

    # Android/app export garbage. Strip NUL padding.
    data = data.replace(b"\x00", b"")

    text = data.decode("utf-8", errors="replace").strip()

    # Normalize excessive blank space at end, preserve PGN structure.
    return text


def parse_game(path: Path) -> chess.pgn.Game:
    text = read_clean_pgn_text(path)
    game = chess.pgn.read_game(io.StringIO(text))

    if game is None:
        raise RuntimeError(f"No PGN game found in {path}")

    return game


def original_pgn_result_marker(path: Path) -> str:
    text = read_clean_pgn_text(path)
    # Grab last non-empty game text line.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    return lines[-1]


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
    # White wants eval to rise.
    # Black wants eval to fall.
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


def report_filename(game: chess.pgn.Game, src: Path) -> str:
    date = game.headers.get("Date", "unknown").replace(".", "-")
    white = safe_slug(game.headers.get("White", "white"))
    black = safe_slug(game.headers.get("Black", "black"))
    base = safe_slug(src.stem)
    return f"{date}_{white}_vs_{black}_{base}.md"


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
    played_san: str,
    best_move: Optional[chess.Move],
    loss_cp: int,
    eval_before: int,
    eval_after: int,
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

    # Crude book-ish phase. Real book labels require an opening database.
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

    # Fake-but-useful “brilliant-ish” detector.
    # Near-best forcing move that does not damage eval.
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


def analyze_game(engine: chess.engine.SimpleEngine, pgn_path: Path) -> Path:
    game = parse_game(pgn_path)
    board = game.board()

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

        board.push(move)

        terminal = board.is_game_over(claim_draw=True)

        if board.is_checkmate():
            eval_after = MATE_CP if mover_is_white else -MATE_CP
            loss_cp = 0
        elif terminal:
            # Don't ask Stockfish to make sense of a finished/adjudicated game.
            eval_after = eval_before
            loss_cp = 0
        else:
            info_after = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
            eval_after = white_cp(info_after["score"])
            loss_cp = max(0, mover_loss(eval_before, eval_after, mover_is_white))

        label, reason = classify_move(
            board_before=before,
            move=move,
            played_san=played_san,
            best_move=best_move,
            loss_cp=loss_cp,
            eval_before=eval_before,
            eval_after=eval_after,
            terminal=terminal,
            ply=ply,
        )

        row = {
            "ply": ply,
            "move_no": before.fullmove_number,
            "side": side,
            "played": played_san,
            "best": best_san,
            "eval_before": eval_before,
            "eval_after": eval_after,
            "loss_cp": loss_cp,
            "label": label,
            "reason": reason,
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

    out_path = ANALYSIS_DIR / report_filename(game, pgn_path)
    write_report(game, pgn_path, out_path, rows, key_moments)
    return out_path


def write_report(
    game: chess.pgn.Game,
    pgn_path: Path,
    out_path: Path,
    rows: list[dict],
    key_moments: list[dict],
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
        "## Key Moments",
        "",
    ]

    if not key_moments:
        lines += [
            "Nothing crossed the highlight threshold. Either the game was clean, too short, or Stockfish did not find enough to yell about.",
            "",
        ]
    else:
        for r in key_moments[:12]:
            lines += [
                f"### {label_emoji(r['label'])} {r['label']}: {r['move_no']}. {r['played']} by {r['side']}",
                "",
                r["reason"],
                "",
                f"- **Eval:** {fmt_eval(r['eval_before'])} → {fmt_eval(r['eval_after'])}",
                f"- **Preferred:** `{r['best']}`",
                f"- **Loss:** {r['loss_cp']} cp",
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


def rebuild_all() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    for old in ANALYSIS_DIR.glob("*.md"):
        old.unlink()

    engine_path = find_stockfish()
    pgns = sorted(RAW_DIR.glob("*.pgn"))

    if not pgns:
        INDEX.write_text("# Chess Analysis Index\n\nNo PGNs found.\n", encoding="utf-8")
        print(f"No PGNs found in {RAW_DIR}")
        return

    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        for pgn in pgns:
            print(f"Analyzing {pgn.relative_to(REPO)}")
            out = analyze_game(engine, pgn)
            print(f"Wrote {out.relative_to(REPO)}")

    update_index()


def main() -> None:
    rebuild_all()


if __name__ == "__main__":
    main()
