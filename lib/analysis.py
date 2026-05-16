#!/usr/bin/env python3
"""
lib/analysis.py

Pure chess-analysis/report-rendering layer.

This module should NOT know:
- MariaDB
- Ollama HTTP
- shell polling
- git push mechanics

It owns:
- PGN parsing
- Stockfish evals
- move classification
- board SVGs
- markdown report rendering
- index rebuilding

Narration is injected through a small interface:
    narrator.game_story(context)
    narrator.move_note(context, row, note_index)

So the caller can provide:
- a real Ollama narrator
- a no-op narrator
- a test/fake narrator
"""

from __future__ import annotations

import hashlib
import io
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

import chess
import chess.engine
import chess.pgn
import chess.svg


DEPTH = 14
MULTIPV = 3

LOSS_INACCURACY = 120
LOSS_MISTAKE = 300
LOSS_BLUNDER = 600
MATE_CP = 100000
BOARD_SIZE = 520

MAX_INACCURACIES = 2
MAX_BEST_MOVES = 2
MAX_KEY_MOMENTS_IN_REPORT = 8
IGNORE_KEY_MOMENTS_BEFORE_PLY = 14


class Narrator(Protocol):
    def game_story(self, context: "GameContext") -> str:
        ...

    def move_note(self, context: "GameContext", row: dict, note_index: int) -> str:
        ...


class NoopNarrator:
    def game_story(self, context: "GameContext") -> str:
        return ""

    def move_note(self, context: "GameContext", row: dict, note_index: int) -> str:
        return row["reason"]


@dataclass(frozen=True)
class AnalysisPaths:
    repo: Path
    raw_dir: Path
    analysis_dir: Path
    asset_dir: Path
    index: Path
    llm_cache_dir: Path

    @classmethod
    def from_repo(cls, repo: Path) -> "AnalysisPaths":
        analysis_dir = repo / "analysis"
        return cls(
            repo=repo,
            raw_dir=repo / "games" / "raw_pgn",
            analysis_dir=analysis_dir,
            asset_dir=analysis_dir / "assets",
            index=analysis_dir / "index.md",
            llm_cache_dir=analysis_dir / "llm_cache",
        )


@dataclass
class GameContext:
    game: chess.pgn.Game
    roles: dict
    pgn_path: Path
    out_path: Path
    stem: str
    game_id: str
    rows: list[dict]
    key_moments: list[dict]
    final_svg: str


# ---------- utilities ----------

def safe_slug(text: str) -> str:
    text = str(text).lower().strip()
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
    data = path.read_bytes().replace(b"\x00", b"")
    return data.decode("utf-8", errors="replace").strip()


def parse_game(path: Path) -> chess.pgn.Game:
    text = read_clean_pgn_text(path)
    game = chess.pgn.read_game(io.StringIO(text))
    if game is None:
        raise RuntimeError(f"No PGN game found in {path}")
    return game


def game_to_clean_pgn_text(game: chess.pgn.Game) -> str:
    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
    return game.accept(exporter).strip()


def game_id_from_game(game: chess.pgn.Game) -> str:
    clean_pgn = game_to_clean_pgn_text(game)
    return hashlib.sha256(clean_pgn.encode("utf-8")).hexdigest()[:32]


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


def side_from_bool(is_white: bool) -> str:
    return "White" if is_white else "Black"


def san_or_unknown(board: chess.Board, move: Optional[chess.Move]) -> str:
    if move is None:
        return "unknown"
    try:
        return board.san(move)
    except Exception:
        return "unknown"


def report_stem(game: chess.pgn.Game, src: Path) -> str:
    date = game.headers.get("Date", "unknown").replace(".", "-")
    white = safe_slug(game.headers.get("White", "white"))
    black = safe_slug(game.headers.get("Black", "black"))
    base = safe_slug(src.stem)
    return f"{date}_{white}_vs_{black}_{base}"


def report_filename(game: chess.pgn.Game, src: Path) -> str:
    return f"{report_stem(game, src)}.md"


def board_asset_rel(paths: AnalysisPaths, path: Path) -> str:
    return path.relative_to(paths.analysis_dir).as_posix()


# ---------- identity / perspective ----------

def identify_player_roles(game: chess.pgn.Game) -> dict:
    white_name = game.headers.get("White", "White")
    black_name = game.headers.get("Black", "Black")

    white_is_you = white_name.strip().lower() == "you"
    black_is_you = black_name.strip().lower() == "you"

    if white_is_you:
        you_side = "White"
        cpu_side = "Black"
        you_name = white_name
        cpu_name = black_name
    elif black_is_you:
        you_side = "Black"
        cpu_side = "White"
        you_name = black_name
        cpu_name = white_name
    else:
        you_side = None
        cpu_side = None
        you_name = None
        cpu_name = None

    return {
        "white_name": white_name,
        "black_name": black_name,
        "white_is_you": white_is_you,
        "black_is_you": black_is_you,
        "you_side": you_side,
        "cpu_side": cpu_side,
        "you_name": you_name,
        "cpu_name": cpu_name,
        "flipped": black_is_you,
    }


def actor_label(side: Optional[str], roles: dict) -> str:
    if side is None:
        return "None"
    if roles["you_side"] == side:
        return f"You ({side})"
    if roles["cpu_side"] == side:
        return f"CPU ({side})"
    return side


def matchup_title(roles: dict) -> str:
    if roles["you_side"] == "White":
        return "You (White) vs CPU (Black)"
    if roles["you_side"] == "Black":
        return "You (Black) vs CPU (White)"
    return f"{roles['white_name']} vs {roles['black_name']}"


def game_result_facts(game: chess.pgn.Game, roles: dict) -> dict:
    result = game.headers.get("Result", "?")
    you_side = roles.get("you_side")

    if result == "1-0":
        winner_side = "White"
        loser_side = "Black"
    elif result == "0-1":
        winner_side = "Black"
        loser_side = "White"
    else:
        winner_side = None
        loser_side = None

    if result == "1/2-1/2":
        user_outcome = "USER DREW"
    elif winner_side is None:
        user_outcome = "UNKNOWN"
    elif you_side == winner_side:
        user_outcome = "USER WON"
    else:
        user_outcome = "USER LOST"

    return {
        "result": result,
        "you_side": you_side or "Unknown",
        "cpu_side": roles.get("cpu_side") or "Unknown",
        "winner_side": winner_side or "None",
        "loser_side": loser_side or "None",
        "winner_actor": actor_label(winner_side, roles) if winner_side else "None",
        "loser_actor": actor_label(loser_side, roles) if loser_side else "None",
        "user_outcome": user_outcome,
    }


# ---------- move labels ----------

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
            return "Checkmate", "Game-ending forcing move."
        return "Good", "Final move before the game ended."

    if ply <= 6 and loss_cp < LOSS_INACCURACY:
        return "Book", "Early opening move."

    label = classify_loss(loss_cp)

    if label == "Blunder":
        return label, "Major eval loss."
    if label == "Mistake":
        return label, "Significant eval loss."
    if label == "Inaccuracy":
        return label, "Smaller but real eval loss."

    is_check = board_before.gives_check(move)
    is_capture = board_before.is_capture(move)
    is_best = best_move == move

    if loss_cp <= 30 and (is_check or is_capture) and is_best:
        return "Best", "Forcing engine-approved move."

    if label == "Best":
        return label, "At or very near Stockfish's preferred move."
    if label == "Great":
        return label, "Very close to best."

    return "Good", "Reasonable move."


# ---------- board generation ----------

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


# ---------- selection ----------

def is_opening_noise(row: dict) -> bool:
    if row["ply"] >= IGNORE_KEY_MOMENTS_BEFORE_PLY:
        return False

    if row["label"] in {"Blunder", "Mistake", "Checkmate"}:
        return False

    if row["loss_cp"] >= LOSS_MISTAKE:
        return False

    return True


def actor_bucket(row: dict) -> str:
    actor = row.get("actor", "")
    if actor.startswith("You"):
        return "You"
    if actor.startswith("CPU"):
        return "CPU"
    return row.get("side", "Unknown")


def pick_worst_by_actor(rows: list[dict], labels: set[str], limit_per_actor: int = 1) -> list[dict]:
    picked = []
    for actor in ("You", "CPU"):
        actor_rows = [
            r for r in rows
            if r["label"] in labels and actor_bucket(r) == actor
        ]
        actor_rows = sorted(actor_rows, key=lambda r: (-r["loss_cp"], r["ply"]))
        picked.extend(actor_rows[:limit_per_actor])
    return picked


def append_unique(target: list[dict], candidates: list[dict], max_total: int) -> None:
    seen = {r["ply"] for r in target}
    for r in candidates:
        if len(target) >= max_total:
            break
        if r["ply"] in seen:
            continue
        target.append(r)
        seen.add(r["ply"])


def select_key_moments(rows: list[dict]) -> list[dict]:
    candidate_rows = [r for r in rows if not is_opening_noise(r)]

    blunders = [r for r in candidate_rows if r["label"] == "Blunder"]
    mistakes = [r for r in candidate_rows if r["label"] == "Mistake"]
    checkmates = [r for r in candidate_rows if r["label"] == "Checkmate"]
    serious_exists = bool(blunders or mistakes)

    selected = []

    append_unique(selected, sorted(checkmates, key=lambda r: r["ply"]), MAX_KEY_MOMENTS_IN_REPORT)
    append_unique(selected, pick_worst_by_actor(candidate_rows, {"Blunder"}, 1), MAX_KEY_MOMENTS_IN_REPORT)
    append_unique(selected, pick_worst_by_actor(candidate_rows, {"Mistake"}, 1), MAX_KEY_MOMENTS_IN_REPORT)

    append_unique(selected, sorted(blunders, key=lambda r: (-r["loss_cp"], r["ply"])), MAX_KEY_MOMENTS_IN_REPORT)
    append_unique(selected, sorted(mistakes, key=lambda r: (-r["loss_cp"], r["ply"])), MAX_KEY_MOMENTS_IN_REPORT)

    if not serious_exists:
        inaccuracies = sorted(
            [r for r in candidate_rows if r["label"] == "Inaccuracy"],
            key=lambda r: (-r["loss_cp"], r["ply"]),
        )
        append_unique(selected, inaccuracies[:MAX_INACCURACIES], MAX_KEY_MOMENTS_IN_REPORT)

    if len(selected) < MAX_KEY_MOMENTS_IN_REPORT:
        best_moves = [r for r in candidate_rows if r["label"] == "Best"]
        append_unique(selected, pick_worst_by_actor(best_moves, {"Best"}, 1), MAX_KEY_MOMENTS_IN_REPORT)
        if len(selected) < MAX_KEY_MOMENTS_IN_REPORT:
            append_unique(selected, sorted(best_moves, key=lambda r: r["ply"])[:MAX_BEST_MOVES], MAX_KEY_MOMENTS_IN_REPORT)

    return sorted(selected, key=lambda r: r["ply"])


# ---------- analysis ----------

def analyze_game(engine: chess.engine.SimpleEngine, pgn_path: Path, paths: AnalysisPaths, narrator: Narrator) -> Path:
    game = parse_game(pgn_path)
    roles = identify_player_roles(game)
    board = game.board()

    stem = report_stem(game, pgn_path)
    out_path = paths.analysis_dir / f"{stem}.md"
    game_asset_dir = paths.asset_dir / stem
    game_id = game_id_from_game(game)

    rows: list[dict] = []

    for ply, move in enumerate(game.mainline_moves(), start=1):
        before = board.copy()
        mover_is_white = before.turn == chess.WHITE
        side = side_from_bool(mover_is_white)
        actor = actor_label(side, roles)
        played_san = before.san(move)

        info_before = engine.analyse(before, chess.engine.Limit(depth=DEPTH), multipv=MULTIPV)
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

        label, reason = classify_move(before, move, best_move, loss_cp, terminal, ply)

        row = {
            "ply": ply,
            "move_no": before.fullmove_number,
            "side": side,
            "actor": actor,
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
        board.push(move)

    key_moments = select_key_moments(rows)

    for row in key_moments:
        prefix = f"ply_{row['ply']:03d}_{safe_slug(row['played'])}"
        before_svg = game_asset_dir / f"{prefix}_before.svg"
        after_svg = game_asset_dir / f"{prefix}_after.svg"

        write_board_svg(row["board_before"], before_svg, best_move=row["best_move"], flipped=roles["flipped"])
        write_board_svg(row["board_after"], after_svg, lastmove=row["move"], flipped=roles["flipped"])

        row["before_svg"] = board_asset_rel(paths, before_svg)
        row["after_svg"] = board_asset_rel(paths, after_svg)

    final_board, final_lastmove = final_board_from_game(game)
    final_svg = game_asset_dir / "final_position.svg"
    write_board_svg(final_board, final_svg, lastmove=final_lastmove, flipped=roles["flipped"])

    context = GameContext(
        game=game,
        roles=roles,
        pgn_path=pgn_path,
        out_path=out_path,
        stem=stem,
        game_id=game_id,
        rows=rows,
        key_moments=key_moments,
        final_svg=board_asset_rel(paths, final_svg),
    )

    write_report(context, paths, narrator)
    return out_path


# ---------- report rendering ----------

def human_notes(rows: list[dict], roles: dict) -> str:
    if not rows:
        return "No moves found."

    notes = []
    blunders = [r for r in rows if r["label"] == "Blunder"]
    mistakes = [r for r in rows if r["label"] == "Mistake"]
    mates = [r for r in rows if r["label"] == "Checkmate"]

    if blunders:
        first = blunders[0]
        notes.append(f"Biggest caveman lesson: {first['actor']} blew it with **{first['move_no']}. {first['played']}**.")
    if mistakes:
        first = mistakes[0]
        notes.append(f"There was another real screw-up at **{first['move_no']}. {first['played']}** by **{first['actor']}**.")
    if mates:
        mate = mates[-1]
        notes.append(f"The game-ending shot was **{mate['move_no']}. {mate['played']}** by **{mate['actor']}**.")
    if not notes:
        notes.append("Nothing dramatic happened; this one was mostly normal move trading.")

    return "\n\n".join(notes)


def write_report(context: GameContext, paths: AnalysisPaths, narrator: Narrator) -> None:
    game = context.game
    roles = context.roles
    rows = context.rows
    key_moments = context.key_moments
    headers = game.headers
    result = headers.get("Result", "?")
    date = headers.get("Date", "unknown")

    your_rows = [r for r in rows if r["actor"].startswith("You")]

    blunders = [r for r in your_rows if r["label"] == "Blunder"]
    mistakes = [r for r in your_rows if r["label"] == "Mistake"]
    inaccuracies = [r for r in your_rows if r["label"] == "Inaccuracy"]
    bests = [r for r in your_rows if r["label"] == "Best"]

    worst = max(rows, key=lambda r: r["loss_cp"], default=None)
    final_row = rows[-1] if rows else None

    lines = [
        f"# {matchup_title(roles)}",
        "",
        f"**Result:** {result}  ",
        f"**Date:** {date}  ",
        f"**Source:** `{context.pgn_path.name}`  ",
        f"**Engine:** Stockfish depth {DEPTH}  ",
        f"**Your side:** {roles['you_side'] or 'Unknown'}  ",
        f"**Computer side:** {roles['cpu_side'] or 'Unknown'}  ",
        f"**Board perspective:** {'Your perspective' if roles['flipped'] else 'Standard White-at-bottom'}",
        "",
        "---",
        "",
        "## Who Is Who",
        "",
        f"- **You:** {roles['you_name'] or 'Unknown'} ({roles['you_side'] or 'Unknown'})",
        f"- **Computer:** {roles['cpu_name'] or 'Unknown'} ({roles['cpu_side'] or 'Unknown'})",
        "",
        "---",
        "",
        "## Toaster Summary",
        "",
    ]

    game_story = narrator.game_story(context)
    if game_story:
        lines += [f"**Game Story:** {game_story}", ""]

    if worst and worst["loss_cp"] >= LOSS_INACCURACY:
        lines += [
            f"Biggest toaster scream: **{worst['move_no']}. {worst['played']}** by **{worst['actor']}**.",
            "",
            f"- **Label:** {label_emoji(worst['label'])} **{worst['label']}**",
            f"- **Eval:** {fmt_eval(worst['eval_before'])} → {fmt_eval(worst['eval_after'])}",
            f"- **Stockfish preferred:** `{worst['best']}`",
            f"- **Loss:** {worst['loss_cp']} cp",
            "",
        ]
    else:
        lines += ["No giant tactical crime detected at this depth.", ""]

    if final_row:
        lines += [f"Final move recorded: **{final_row['move_no']}. {final_row['played']}** by **{final_row['actor']}**.", ""]

    lines += [
        f"- 💀 Your blunders: **{len(blunders)}**",
        f"- ❌ Your mistakes: **{len(mistakes)}**",
        f"- ⚠️ Your inaccuracies: **{len(inaccuracies)}**",
        f"- ✅ Your best moves called out: **{len(bests)}**",
        "",
        "---",
        "",
        "## Final Position",
        "",
        f"![Final position]({context.final_svg})",
        "",
        "---",
        "",
        "## Key Moments",
        "",
    ]

    if not key_moments:
        lines += ["Nothing crossed the highlight threshold.", ""]
    else:
        for note_index, row in enumerate(key_moments):
            explanation = narrator.move_note(context, row, note_index)
            lines += [
                f"### {label_emoji(row['label'])} {row['label']}: {row['move_no']}. {row['played']} by {row['actor']}",
                "",
                explanation,
                "",
                f"- **Eval:** {fmt_eval(row['eval_before'])} → {fmt_eval(row['eval_after'])}",
                f"- **Preferred:** `{row['best']}`",
                f"- **Loss:** {row['loss_cp']} cp",
                "",
            ]

            if row.get("before_svg"):
                lines += [f"**Before {row['move_no']}. {row['played']}**", "", f"![Before {row['move_no']}. {row['played']}]({row['before_svg']})", ""]
            if row.get("after_svg"):
                lines += [f"**After {row['move_no']}. {row['played']}**", "", f"![After {row['move_no']}. {row['played']}]({row['after_svg']})", ""]

    lines += [
        "---",
        "",
        "## Human Notes",
        "",
        human_notes(rows, roles),
        "",
        "---",
        "",
        "<details>",
        "<summary>Compact move list</summary>",
        "",
    ]

    for row in rows:
        lines.append(
            f"- {label_emoji(row['label'])} **{row['move_no']}. {row['played']}** ({row['actor']}, {row['label']}) — "
            f"{fmt_eval(row['eval_before'])} → {fmt_eval(row['eval_after'])}, preferred `{row['best']}`, loss {row['loss_cp']} cp"
        )

    lines += [
        "",
        "</details>",
        "",
        "---",
        "",
        "<details>",
        "<summary>Raw engine table</summary>",
        "",
        "| Ply | Move | Actor | Played | Label | Eval Before | Eval After | Preferred | Loss |",
        "|---:|---:|---|---|---|---:|---:|---|---:|",
    ]

    for row in rows:
        lines.append(
            f"| {row['ply']} | {row['move_no']} | {row['actor']} | {row['played']} | {row['label']} | "
            f"{fmt_eval(row['eval_before'])} | {fmt_eval(row['eval_after'])} | {row['best']} | {row['loss_cp']} |"
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

    context.out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------- rebuild orchestration ----------

def update_index(paths: AnalysisPaths) -> None:
    reports = sorted(p for p in paths.analysis_dir.glob("*.md") if p.name != "index.md")

    lines = ["# Chess Analysis Index", ""]
    if not reports:
        lines.append("No games analyzed yet.")
    else:
        for p in reports:
            title = p.stem.replace("_", " ")
            lines.append(f"- [{title}]({p.name})")
    lines.append("")

    paths.index.write_text("\n".join(lines), encoding="utf-8")


def remove_existing_assets_for_report(game: chess.pgn.Game, pgn_path: Path, paths: AnalysisPaths) -> None:
    stem = report_stem(game, pgn_path)
    asset_dir = paths.asset_dir / stem
    if asset_dir.exists():
        shutil.rmtree(asset_dir)


def rebuild_all(paths: AnalysisPaths, narrator: Narrator, *, force: bool = False) -> None:
    paths.analysis_dir.mkdir(parents=True, exist_ok=True)
    paths.asset_dir.mkdir(parents=True, exist_ok=True)
    paths.llm_cache_dir.mkdir(parents=True, exist_ok=True)

    engine_path = find_stockfish()
    pgns = sorted(paths.raw_dir.glob("*.pgn"))

    if not pgns:
        paths.index.write_text("# Chess Analysis Index\n\nNo PGNs found.\n", encoding="utf-8")
        print(f"No PGNs found in {paths.raw_dir}")
        return

    analyzed = 0
    skipped = 0

    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        for pgn in pgns:
            game = parse_game(pgn)
            out_path = paths.analysis_dir / report_filename(game, pgn)

            if out_path.exists() and not force:
                print(f"Skipping existing report: {out_path.relative_to(paths.repo)}")
                skipped += 1
                continue

            if force:
                remove_existing_assets_for_report(game, pgn, paths)

            print(f"Analyzing {pgn.relative_to(paths.repo)}")
            out = analyze_game(engine, pgn, paths, narrator)
            print(f"Wrote {out.relative_to(paths.repo)}")
            analyzed += 1

    update_index(paths)
    print(f"Done. Analyzed {analyzed}, skipped {skipped}.")
