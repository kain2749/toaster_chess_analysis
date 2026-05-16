#!/usr/bin/env python3

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import sys
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
LLM_CACHE_DIR = ANALYSIS_DIR / "llm_cache"

# Let this script import repo-local modules when run from scripts/.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lib.ollama_phrase_memory import OllamaPhraseMemory

DEPTH = 14
MULTIPV = 3

LOSS_INACCURACY = 120
LOSS_MISTAKE = 300
LOSS_BLUNDER = 600
MATE_CP = 100000
BOARD_SIZE = 520

# Selection controls.
MAX_INACCURACIES = 2
MAX_BEST_MOVES = 2
MAX_KEY_MOMENTS_IN_REPORT = 8

# Ignore opening fluff before this ply unless it is a real crime.
# Ply 14 = after Black's 7th move.
IGNORE_KEY_MOMENTS_BEFORE_PLY = 14

# Ollama defaults:
# - on by default
# - disable with TOASTER_USE_OLLAMA=0
USE_OLLAMA = os.getenv("TOASTER_USE_OLLAMA", "1") != "0"
OLLAMA_MODEL = os.getenv("TOASTER_OLLAMA_MODEL", "dolphin-mistral")
OLLAMA_URL = os.getenv("TOASTER_OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_TIMEOUT = int(os.getenv("TOASTER_OLLAMA_TIMEOUT", "120"))
OLLAMA_MAX_NOTES = int(os.getenv("TOASTER_OLLAMA_MAX_NOTES", "9999"))
OLLAMA_NUM_PREDICT = int(os.getenv("TOASTER_OLLAMA_NUM_PREDICT", "320"))

USE_PHRASE_MEMORY = os.getenv("TOASTER_USE_PHRASE_MEMORY", "1") != "0"
FORCE_LLM_REGEN = False
PHRASE_MEMORY: Optional[OllamaPhraseMemory] = OllamaPhraseMemory() if USE_PHRASE_MEMORY else None


# ---------- utility ----------

def init_phrase_memory() -> None:
    """Initialize phrase memory. If DB is dead, analysis still runs."""
    global PHRASE_MEMORY

    if PHRASE_MEMORY is None:
        return

    try:
        PHRASE_MEMORY.init_db()
    except Exception as exc:
        print(f"Warning: phrase memory disabled: {exc}")
        PHRASE_MEMORY = None


def safe_move_note_avoidance_block(game_id: str, row: dict, limit: int = 8) -> str:
    if PHRASE_MEMORY is None:
        return "Already-used wording for similar move notes in this game:\n- Phrase memory disabled."

    try:
        return PHRASE_MEMORY.move_note_avoidance_block(game_id, row, limit=limit)
    except Exception as exc:
        return f"Already-used wording for similar move notes in this game:\n- Phrase memory read failed: {exc}"


def safe_game_summary_avoidance_block(limit: int = 12) -> str:
    if PHRASE_MEMORY is None:
        return "Recent game summaries already written:\n- Phrase memory disabled."

    try:
        return PHRASE_MEMORY.game_summary_avoidance_block(limit=limit)
    except Exception as exc:
        return f"Recent game summaries already written:\n- Phrase memory read failed: {exc}"


def safe_remember_move_note(game_id: str, row: dict, note_text: str) -> None:
    if PHRASE_MEMORY is None:
        return
    try:
        PHRASE_MEMORY.remember_move_note(game_id, row, note_text)
    except Exception as exc:
        print(f"Warning: failed to remember move note: {exc}")


def safe_remember_game_summary(
    game_id: str,
    summary_text: str,
    result: Optional[str] = None,
    you_side: Optional[str] = None,
    cpu_side: Optional[str] = None,
) -> None:
    if PHRASE_MEMORY is None:
        return
    try:
        PHRASE_MEMORY.remember_game_summary(
            game_id,
            summary_text,
            result=result,
            you_side=you_side,
            cpu_side=cpu_side,
        )
    except Exception as exc:
        print(f"Warning: failed to remember game summary: {exc}")


def game_id_for_game(game: chess.pgn.Game) -> str:
    clean_pgn = game_to_clean_pgn_text(game)
    if PHRASE_MEMORY is not None:
        return PHRASE_MEMORY.game_id_from_pgn_text(clean_pgn)
    return hashlib.sha256(clean_pgn.encode("utf-8")).hexdigest()[:32]


def stop_ollama_model() -> None:
    if not USE_OLLAMA:
        return

    try:
        req = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps({
                "model": OLLAMA_MODEL,
                "prompt": "",
                "stream": False,
                "keep_alive": 0,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        urllib.request.urlopen(req, timeout=15).read()
    except Exception:
        # Cleanup failure should not break the analysis pipeline.
        pass


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
    # White wants eval to rise. Black wants eval to fall.
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


def report_stem(game: chess.pgn.Game, src: Path) -> str:
    date = game.headers.get("Date", "unknown").replace(".", "-")
    white = safe_slug(game.headers.get("White", "white"))
    black = safe_slug(game.headers.get("Black", "black"))
    base = safe_slug(src.stem)
    return f"{date}_{white}_vs_{black}_{base}"


def report_filename(game: chess.pgn.Game, src: Path) -> str:
    return f"{report_stem(game, src)}.md"


def board_asset_rel(path: Path) -> str:
    return path.relative_to(ANALYSIS_DIR).as_posix()


# ---------- identity / perspective ----------

def side_from_bool(is_white: bool) -> str:
    return "White" if is_white else "Black"


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
        # If you are Black, render boards from your side.
        "flipped": black_is_you,
    }


def actor_label(side: str, roles: dict) -> str:
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

    # No more "brilliant-ish" nonsense. Just call it Best.
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


# ---------- ollama ----------

def row_cache_key(row: dict) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "fen": row["board_before"].fen(),
        "actor": row["actor"],
        "side": row["side"],
        "played": row["played"],
        "best": row["best"],
        "label": row["label"],
        "eval_before": fmt_eval(row["eval_before"]),
        "eval_after": fmt_eval(row["eval_after"]),
        "loss_cp": row["loss_cp"],
        "game_id": row.get("game_id"),
        "avoidance_hash": row.get("avoidance_hash"),
        "prompt_version": "toaster_ollama_v8_mysql_phrase_memory",
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def ollama_prompt_for_row(row: dict, roles: dict, avoidance_block: str = "") -> str:
    you_side = roles["you_side"] or "Unknown"
    cpu_side = roles["cpu_side"] or "Unknown"

    return f"""Move Note rules:
- Use only the engine facts provided.
- Do not invent tactics.
- Do not mention numeric eval scores, centipawns, loss points, or "cp".
- Do not summarize the whole game.
- Do not list moves one by one like an engine table.
- Mention only the played move and, if useful, the preferred move.
- Keep it to 1-3 short sentences.
- Sound like Toaster Chess: rude, blunt, mildly profane, and annoyed that it had to watch this.
- If there was checkmate, end by naming the mating move.
- If both sides played badly, say so.
- Do not say "won cleanly" if the winner survived major blunders or mistakes.
- Do not write bullets.
- Do not use an intro like "Here's a note".
- End with a complete sentence.
- If you do not know the real opening name, describe the opening structure plainly.
- Do not invent named openings.

Player info:
- You are: {you_side}
- Computer is: {cpu_side}

Position before the move:
FEN: {row['board_before'].fen()}

Move being reviewed:
Actor: {row['actor']}
Played move: {row['played']}
Label: {row['label']}

Engine facts:
Eval before: {fmt_eval(row['eval_before'])}
Eval after: {fmt_eval(row['eval_after'])}
Stockfish preferred: {row['best']}
Centipawn loss: {row['loss_cp']}

Plain-English target:
Explain why {row['actor']} played something useful or stupid at move {row['move_no']} with {row['played']}. If the preferred move differs, mention {row['best']} only if it helps.

Fallback explanation:
{row['reason']}

Anti-repetition memory:
{avoidance_block}

Do not reuse the same sentence shape, joke, metaphor, insult, or opening phrase from the already-used wording.
Prefer concrete chess consequences over generic roasting.

Write the note now.
"""


def clean_llm_note(text: str) -> str:
    text = " ".join(text.split()).strip()
    text = text.strip(' "\'“”‘’')

    if not text:
        return text

    # Kill common dumb intros.
    for prefix in (
        "Here's a note:",
        "Here is a note:",
        "Note:",
        "Here's the note:",
        "Here is the note:",
    ):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()

    # Keep only complete sentences if possible.
    last_end = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    if last_end != -1:
        text = text[: last_end + 1]

    # Hard character cap for phone readability.
    if len(text) > 500:
        clipped = text[:500]
        last_end = max(clipped.rfind("."), clipped.rfind("!"), clipped.rfind("?"))
        if last_end != -1:
            text = clipped[: last_end + 1]
        else:
            text = clipped.rstrip() + "..."

    return text


def call_ollama(prompt: str) -> str:
    body = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.45,
            "top_p": 0.9,
            "repeat_penalty": 1.15,
            "num_predict": OLLAMA_NUM_PREDICT,
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

    text = data.get("response", "").strip()
    return " ".join(text.split()).strip()


def ollama_explanation_for_row(row: dict, roles: dict, note_index: int, game_id: str) -> str:
    if not USE_OLLAMA or note_index >= OLLAMA_MAX_NOTES:
        return row["reason"]

    avoidance_block = safe_move_note_avoidance_block(game_id, row, limit=8)
    row["game_id"] = game_id
    row["avoidance_hash"] = hashlib.sha256(avoidance_block.encode("utf-8")).hexdigest()

    LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = row_cache_key(row)
    cache_path = LLM_CACHE_DIR / f"{cache_key}.txt"

    if not FORCE_LLM_REGEN and cache_path.exists():
        cached = cache_path.read_text(encoding="utf-8").strip()
        if cached:
            return cached

    prompt = ollama_prompt_for_row(row, roles, avoidance_block)

    try:
        explanation = call_ollama(prompt)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return row["reason"]

    explanation = clean_llm_note(explanation)

    if not explanation:
        return row["reason"]

    cache_path.write_text(explanation + "\n", encoding="utf-8")
    safe_remember_move_note(game_id, row, explanation)
    return explanation


def game_summary_cache_key(
    game: chess.pgn.Game,
    roles: dict,
    rows: list[dict],
    key_moments: list[dict],
    game_id: str,
    avoidance_block: str,
) -> str:
    compact_rows = [
        {
            "ply": r["ply"],
            "move_no": r["move_no"],
            "actor": r["actor"],
            "played": r["played"],
            "label": r["label"],
            "best": r["best"],
            "loss_cp": r["loss_cp"],
            "eval_before": fmt_eval(r["eval_before"]),
            "eval_after": fmt_eval(r["eval_after"]),
        }
        for r in key_moments
    ]

    payload = {
        "model": OLLAMA_MODEL,
        "result": game.headers.get("Result", "?"),
        "white": game.headers.get("White", "White"),
        "black": game.headers.get("Black", "Black"),
        "you_side": roles.get("you_side"),
        "cpu_side": roles.get("cpu_side"),
        "pgn": game_to_clean_pgn_text(game),
        "key_moments": compact_rows,
        "game_id": game_id,
        "avoidance_hash": hashlib.sha256(avoidance_block.encode("utf-8")).hexdigest(),
        "prompt_version": "toaster_game_story_v2_mysql_phrase_memory",
    }

    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def compact_key_moments_for_prompt(key_moments: list[dict]) -> str:
    if not key_moments:
        return "No promoted key moments."

    lines = []
    for r in key_moments:
        lines.append(
            f"- Move {r['move_no']} {r['played']} by {r['actor']}: "
            f"{r['label']}; preferred {r['best']}; "
            f"eval {fmt_eval(r['eval_before'])} to {fmt_eval(r['eval_after'])}; "
            f"loss {r['loss_cp']} cp"
        )

    return "\n".join(lines)


def ollama_game_summary_prompt(
    game: chess.pgn.Game,
    roles: dict,
    rows: list[dict],
    key_moments: list[dict],
    avoidance_block: str = "",
) -> str:
    result = game.headers.get("Result", "?")
    final_row = rows[-1] if rows else None
    final_move = "unknown"
    if final_row:
        final_move = f"{final_row['move_no']}. {final_row['played']} by {final_row['actor']}"

    worst = max(rows, key=lambda r: r["loss_cp"], default=None)
    worst_text = "No major engine complaint."
    if worst and worst["loss_cp"] >= LOSS_INACCURACY:
        worst_text = (
            f"Move {worst['move_no']} {worst['played']} by {worst['actor']}: "
            f"{worst['label']}; preferred {worst['best']}; "
            f"eval {fmt_eval(worst['eval_before'])} to {fmt_eval(worst['eval_after'])}; "
            f"loss {worst['loss_cp']} cp"
        )

    pgn = game_to_clean_pgn_text(game)

    return f"""Game Story rules:
- Write 2-4 short sentences.
- No bullets.
- No intro.
- Do not mention Stockfish by name.
- Do not mention eval scores, centipawns, loss points, or numeric engine data.
- Do not define chess terms like a textbook.
- Do not insult the user personally; roast the game, move quality, pieces, or position.
- Summarize the whole game, not one move.
- Mention only 1-3 specific moves max.
- If both sides played badly, say so.
- Do not say "won cleanly" if the winner survived major blunders or mistakes.
- Keep it readable on a phone.
- Sound like Toaster Chess: rude, blunt, mildly profane, and annoyed that it had to watch this.
- End with a complete sentence.

Game info:
Result: {result}
You are: {roles.get('you_side') or 'Unknown'}
Computer is: {roles.get('cpu_side') or 'Unknown'}
Final move: {final_move}

Biggest engine complaint:
{worst_text}

Promoted key moments:
{compact_key_moments_for_prompt(key_moments)}

Full clean PGN:
{pgn}

Anti-repetition memory:
{avoidance_block}

Do not reuse the same structure, opening sentence, joke, metaphor, or insult from the prior summaries.
Prefer concrete chess consequences over generic roasting.

Write the game story now.
"""


def clean_game_summary(text: str) -> str:
    text = " ".join(text.split()).strip()
    text = text.strip(' "\'“”‘’')

    if not text:
        return text

    for prefix in (
        "Here's a summary:",
        "Here is a summary:",
        "Game summary:",
        "Summary:",
        "Here's the game story:",
        "Here is the game story:",
    ):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()

    last_end = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    if last_end != -1:
        text = text[: last_end + 1]

    return text


def ollama_game_summary(
    game: chess.pgn.Game,
    roles: dict,
    rows: list[dict],
    key_moments: list[dict],
    game_id: str,
) -> str:
    if not USE_OLLAMA:
        return ""

    avoidance_block = safe_game_summary_avoidance_block(limit=12)

    LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = game_summary_cache_key(game, roles, rows, key_moments, game_id, avoidance_block)
    cache_path = LLM_CACHE_DIR / f"game_story_{cache_key}.txt"

    if not FORCE_LLM_REGEN and cache_path.exists():
        cached = cache_path.read_text(encoding="utf-8").strip()
        if cached:
            return cached

    prompt = ollama_game_summary_prompt(game, roles, rows, key_moments, avoidance_block)

    try:
        summary = call_ollama(prompt)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return ""

    summary = clean_game_summary(summary)

    if not summary:
        return ""

    cache_path.write_text(summary + "\n", encoding="utf-8")
    safe_remember_game_summary(
        game_id,
        summary,
        result=game.headers.get("Result", "?"),
        you_side=roles.get("you_side"),
        cpu_side=roles.get("cpu_side"),
    )
    return summary


# ---------- selection ----------

def is_opening_noise(row: dict) -> bool:
    if row["ply"] >= IGNORE_KEY_MOMENTS_BEFORE_PLY:
        return False

    # Early disasters still matter. Hanging a queen on move 5 is not "book, bro."
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

    # Always keep any game-ending shot if it exists.
    append_unique(selected, sorted(checkmates, key=lambda r: r["ply"]), MAX_KEY_MOMENTS_IN_REPORT)

    # Biggest crimes by each side first.
    append_unique(selected, pick_worst_by_actor(candidate_rows, {"Blunder"}, 1), MAX_KEY_MOMENTS_IN_REPORT)
    append_unique(selected, pick_worst_by_actor(candidate_rows, {"Mistake"}, 1), MAX_KEY_MOMENTS_IN_REPORT)

    # Then remaining serious moments by severity.
    remaining_blunders = sorted(blunders, key=lambda r: (-r["loss_cp"], r["ply"]))
    remaining_mistakes = sorted(mistakes, key=lambda r: (-r["loss_cp"], r["ply"]))
    append_unique(selected, remaining_blunders, MAX_KEY_MOMENTS_IN_REPORT)
    append_unique(selected, remaining_mistakes, MAX_KEY_MOMENTS_IN_REPORT)

    # Only show inaccuracies if the game does not already have real crimes.
    if not serious_exists:
        inaccuracies = sorted(
            [r for r in candidate_rows if r["label"] == "Inaccuracy"],
            key=lambda r: (-r["loss_cp"], r["ply"]),
        )
        append_unique(selected, inaccuracies[:MAX_INACCURACIES], MAX_KEY_MOMENTS_IN_REPORT)

    # Best moves are seasoning, not dinner. Only include a couple if there is still room.
    if len(selected) < MAX_KEY_MOMENTS_IN_REPORT:
        best_moves = [r for r in candidate_rows if r["label"] == "Best"]
        append_unique(selected, pick_worst_by_actor(best_moves, {"Best"}, 1), MAX_KEY_MOMENTS_IN_REPORT)
        if len(selected) < MAX_KEY_MOMENTS_IN_REPORT:
            best_remaining = sorted(best_moves, key=lambda r: (r["ply"]))
            append_unique(selected, best_remaining[:MAX_BEST_MOVES], MAX_KEY_MOMENTS_IN_REPORT)

    return sorted(selected, key=lambda r: r["ply"])


# ---------- analysis ----------

def analyze_game(engine: chess.engine.SimpleEngine, pgn_path: Path) -> Path:
    game = parse_game(pgn_path)
    roles = identify_player_roles(game)
    board = game.board()
    game_id = game_id_for_game(game)

    stem = report_stem(game, pgn_path)
    out_path = ANALYSIS_DIR / f"{stem}.md"
    game_asset_dir = ASSET_DIR / stem

    rows = []

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

        write_board_svg(
            row["board_before"],
            before_svg,
            lastmove=None,
            best_move=row["best_move"],
            flipped=roles["flipped"],
        )
        write_board_svg(
            row["board_after"],
            after_svg,
            lastmove=row["move"],
            best_move=None,
            flipped=roles["flipped"],
        )

        row["before_svg"] = board_asset_rel(before_svg)
        row["after_svg"] = board_asset_rel(after_svg)

    final_board, final_lastmove = final_board_from_game(game)
    final_svg = game_asset_dir / "final_position.svg"
    write_board_svg(final_board, final_svg, lastmove=final_lastmove, best_move=None, flipped=roles["flipped"])

    write_report(game, roles, pgn_path, out_path, rows, key_moments, board_asset_rel(final_svg), game_id)
    return out_path


# ---------- report ----------

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


def write_report(
    game: chess.pgn.Game,
    roles: dict,
    pgn_path: Path,
    out_path: Path,
    rows: list[dict],
    key_moments: list[dict],
    final_svg: str,
    game_id: str,
) -> None:
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

    lines = []
    lines += [
        f"# {matchup_title(roles)}",
        "",
        f"**Result:** {result}  ",
        f"**Date:** {date}  ",
        f"**Source:** `{pgn_path.name}`  ",
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

    game_story = ollama_game_summary(game, roles, rows, key_moments, game_id)
    if game_story:
        lines += [
            f"**Game Story:** {game_story}",
            "",
        ]

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
        f"![Final position]({final_svg})",
        "",
        "---",
        "",
        "## Key Moments",
        "",
    ]

    if not key_moments:
        lines += ["Nothing crossed the highlight threshold.", ""]
    else:
        for note_index, r in enumerate(key_moments):
            explanation = ollama_explanation_for_row(r, roles, note_index, game_id)
            lines += [
                f"### {label_emoji(r['label'])} {r['label']}: {r['move_no']}. {r['played']} by {r['actor']}",
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
        human_notes(rows, roles),
        "",
        "---",
        "",
        "<details>",
        "<summary>Compact move list</summary>",
        "",
    ]

    for r in rows:
        lines.append(
            f"- {label_emoji(r['label'])} **{r['move_no']}. {r['played']}** ({r['actor']}, {r['label']}) — "
            f"{fmt_eval(r['eval_before'])} → {fmt_eval(r['eval_after'])}, preferred `{r['best']}`, loss {r['loss_cp']} cp"
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

    for r in rows:
        lines.append(
            f"| {r['ply']} | {r['move_no']} | {r['actor']} | {r['played']} | {r['label']} | "
            f"{fmt_eval(r['eval_before'])} | {fmt_eval(r['eval_after'])} | {r['best']} | {r['loss_cp']} |"
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


# ---------- rebuild ----------

def update_index() -> None:
    reports = sorted(p for p in ANALYSIS_DIR.glob("*.md") if p.name != "index.md")

    lines = ["# Chess Analysis Index", ""]
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
    global FORCE_LLM_REGEN
    FORCE_LLM_REGEN = force

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    init_phrase_memory()

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
    parser.add_argument("--force", action="store_true", help="Regenerate all reports even if markdown already exists.")
    args = parser.parse_args()
    try:
        rebuild_all(force=args.force)
    finally:
        stop_ollama_model()  # let my vram be free plz


if __name__ == "__main__":
    main()
