#!/usr/bin/env python3
"""
lib/ollama_phrase_memory.py

Ollama + MariaDB boundary.

This module owns:
- MariaDB connection
- phrase-memory tables
- Ollama HTTP calls
- prompt/response logging
- move-note and game-story prompt construction

analysis.py deliberately does not import mysql.connector or urllib.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mysql.connector
from mysql.connector import MySQLConnection

from lib.analysis import (
    LOSS_BLUNDER,
    LOSS_INACCURACY,
    LOSS_MISTAKE,
    GameContext,
    fmt_eval,
    game_result_facts,
    game_to_clean_pgn_text,
    safe_slug,
)


DEFAULT_DB_NAME = "toaster_chess_ollama"


@dataclass(frozen=True)
class MysqlConfig:
    host: str = os.getenv("TOASTER_MYSQL_HOST", "127.0.0.1")
    port: int = int(os.getenv("TOASTER_MYSQL_PORT", "3306"))
    user: str = os.getenv("TOASTER_MYSQL_USER", "toaster")
    password: str = os.getenv("TOASTER_MYSQL_PASSWORD", "password")
    database: str = os.getenv("TOASTER_MYSQL_DATABASE", DEFAULT_DB_NAME)


@dataclass(frozen=True)
class OllamaConfig:
    model: str = os.getenv("TOASTER_OLLAMA_MODEL", "dolphin-mistral")
    url: str = os.getenv("TOASTER_OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
    timeout: int = int(os.getenv("TOASTER_OLLAMA_TIMEOUT", "120"))
    num_predict: int = int(os.getenv("TOASTER_OLLAMA_NUM_PREDICT", "320"))
    enabled: bool = os.getenv("TOASTER_USE_OLLAMA", "1") != "0"


class OllamaPhraseMemory:
    def __init__(self, mysql_cfg: Optional[MysqlConfig] = None) -> None:
        self.cfg = mysql_cfg or MysqlConfig()

    def _connect_server(self) -> MySQLConnection:
        return mysql.connector.connect(
            host=self.cfg.host,
            port=self.cfg.port,
            user=self.cfg.user,
            password=self.cfg.password,
            autocommit=True,
        )

    def _connect_db(self) -> MySQLConnection:
        return mysql.connector.connect(
            host=self.cfg.host,
            port=self.cfg.port,
            user=self.cfg.user,
            password=self.cfg.password,
            database=self.cfg.database,
            autocommit=True,
        )

    @staticmethod
    def _hash_text(text: str) -> str:
        normalized = " ".join((text or "").split()).strip()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def init_db(self) -> None:
        with self._connect_server() as conn:
            cur = conn.cursor()
            cur.execute(
                f"""
                CREATE DATABASE IF NOT EXISTS `{self.cfg.database}`
                CHARACTER SET utf8mb4
                COLLATE utf8mb4_unicode_ci
                """
            )

        with self._connect_db() as conn:
            cur = conn.cursor()

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS game_move_notes (
                    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
                    game_id VARCHAR(128) NOT NULL,
                    ply INT NOT NULL,
                    move_no INT NULL,
                    actor VARCHAR(64) NULL,
                    side VARCHAR(16) NULL,
                    label VARCHAR(32) NULL,
                    played VARCHAR(64) NULL,
                    best VARCHAR(64) NULL,
                    note_hash CHAR(64) NOT NULL,
                    note_text TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

                    UNIQUE KEY uq_game_ply_hash (game_id, ply, note_hash),
                    KEY idx_game_created (game_id, created_at),
                    KEY idx_game_label (game_id, label),
                    KEY idx_game_actor (game_id, actor)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS game_summaries (
                    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
                    game_id VARCHAR(128) NOT NULL,
                    result VARCHAR(16) NULL,
                    you_side VARCHAR(16) NULL,
                    cpu_side VARCHAR(16) NULL,
                    summary_hash CHAR(64) NOT NULL,
                    summary_text TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

                    UNIQUE KEY uq_summary_hash (summary_hash),
                    KEY idx_created (created_at),
                    KEY idx_game_id (game_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS ollama_calls (
                    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
                    game_id VARCHAR(128) NULL,
                    kind VARCHAR(64) NOT NULL,
                    prompt_hash CHAR(64) NOT NULL,
                    response_hash CHAR(64) NULL,
                    model VARCHAR(128) NULL,
                    prompt_text LONGTEXT NOT NULL,
                    response_text LONGTEXT NULL,
                    error_text TEXT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

                    KEY idx_game_kind_created (game_id, kind, created_at),
                    KEY idx_prompt_hash (prompt_hash),
                    KEY idx_response_hash (response_hash)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS summary_avoid_phrases (
                    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
                    phrase VARCHAR(255) NOT NULL,
                    reason VARCHAR(255) NULL,
                    source VARCHAR(64) NOT NULL DEFAULT 'manual',
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

                    UNIQUE KEY uq_phrase (phrase)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
            )

            cur.execute(
                """
                INSERT IGNORE INTO summary_avoid_phrases (phrase, reason, source) VALUES
                    ('won cleanly', 'contradicts sloppy/blunder games', 'seed'),
                    ('sealed your fate', 'overused and implies loss', 'seed'),
                    ('bitter taste', 'overused loss framing', 'seed'),
                    ('congratulations on losing', 'wrong if user won', 'seed'),
                    ('bad joke at a funeral', 'overused metaphor', 'seed'),
                    ('blindfolded baboon', 'overused metaphor', 'seed'),
                    ('drunken sailor', 'overused metaphor', 'seed')
                """
            )

    def remember_ollama_call(
        self,
        *,
        game_id: Optional[str],
        kind: str,
        model: Optional[str],
        prompt_text: str,
        response_text: Optional[str] = None,
        error_text: Optional[str] = None,
    ) -> None:
        response_text = response_text or ""
        error_text = error_text or ""

        with self._connect_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO ollama_calls
                    (game_id, kind, prompt_hash, response_hash, model, prompt_text, response_text, error_text)
                VALUES
                    (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    game_id,
                    kind,
                    self._hash_text(prompt_text),
                    self._hash_text(response_text) if response_text else None,
                    model,
                    prompt_text,
                    response_text if response_text else None,
                    error_text if error_text else None,
                ),
            )

    def remember_move_note(self, game_id: str, row: dict, note_text: str) -> None:
        note_text = " ".join(note_text.split()).strip()
        if not note_text:
            return

        with self._connect_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT IGNORE INTO game_move_notes
                    (game_id, ply, move_no, actor, side, label, played, best, note_hash, note_text)
                VALUES
                    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    game_id,
                    int(row.get("ply", 0)),
                    row.get("move_no"),
                    row.get("actor"),
                    row.get("side"),
                    row.get("label"),
                    row.get("played"),
                    row.get("best"),
                    self._hash_text(note_text),
                    note_text,
                ),
            )

    def remember_game_summary(
        self,
        game_id: str,
        summary_text: str,
        *,
        result: Optional[str] = None,
        you_side: Optional[str] = None,
        cpu_side: Optional[str] = None,
    ) -> None:
        summary_text = " ".join(summary_text.split()).strip()
        if not summary_text:
            return

        with self._connect_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT IGNORE INTO game_summaries
                    (game_id, result, you_side, cpu_side, summary_hash, summary_text)
                VALUES
                    (%s, %s, %s, %s, %s, %s)
                """,
                (
                    game_id,
                    result,
                    you_side,
                    cpu_side,
                    self._hash_text(summary_text),
                    summary_text,
                ),
            )

    def recent_move_notes_for_game(
        self,
        game_id: str,
        *,
        label: Optional[str] = None,
        actor_prefix: Optional[str] = None,
        limit: int = 8,
    ) -> list[str]:
        where = ["game_id = %s"]
        params: list[object] = [game_id]

        if label:
            where.append("label = %s")
            params.append(label)

        if actor_prefix:
            where.append("actor LIKE %s")
            params.append(f"{actor_prefix}%")

        params.append(limit)

        with self._connect_db() as conn:
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT note_text
                FROM game_move_notes
                WHERE {" AND ".join(where)}
                ORDER BY created_at DESC, id DESC
                LIMIT %s
                """,
                tuple(params),
            )
            return [row[0] for row in cur.fetchall()]

    def summary_avoid_phrases(self, *, limit: int = 20) -> list[str]:
        with self._connect_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT phrase
                FROM summary_avoid_phrases
                ORDER BY created_at DESC, id DESC
                LIMIT %s
                """,
                (limit,),
            )
            return [row[0] for row in cur.fetchall()]

    @staticmethod
    def format_avoidance_block(items: list[str], title: str) -> str:
        items = [i.strip() for i in items if i and i.strip()]
        if not items:
            return f"{title}\n- None."

        return "\n".join([title, *[f"- {item}" for item in items]])

    def move_note_avoidance_block(self, game_id: str, row: dict, *, limit: int = 8) -> str:
        actor = row.get("actor", "")
        actor_prefix = "You" if actor.startswith("You") else "CPU" if actor.startswith("CPU") else None

        notes = self.recent_move_notes_for_game(
            game_id,
            label=row.get("label"),
            actor_prefix=actor_prefix,
            limit=limit,
        )

        return self.format_avoidance_block(
            notes,
            "Already-used wording for similar move notes in this game:",
        )

    def summary_phrase_avoidance_block(self, *, limit: int = 20) -> str:
        phrases = self.summary_avoid_phrases(limit=limit)
        if not phrases:
            return "No summary phrase blacklist yet."

        return "\n".join([
            "Avoid these repeated or banned phrases:",
            *[f"- {phrase}" for phrase in phrases],
            "",
            "These phrases are wording history only. They are not facts about this game.",
        ])


class ToasterOllamaNarrator:
    """
    Narrator implementation used by analysis.py.

    analysis.py calls:
        narrator.game_story(context)
        narrator.move_note(context, row, note_index)

    This class decides how to call Ollama, what to log, and what DB memory to use.
    """

    def __init__(
        self,
        *,
        memory: Optional[OllamaPhraseMemory] = None,
        ollama_cfg: Optional[OllamaConfig] = None,
        debug_dir: Optional[Path] = None,
        force_llm_regen: bool = False,
        max_notes: Optional[int] = None,
    ) -> None:
        self.memory = memory
        self.ollama = ollama_cfg or OllamaConfig()
        self.debug_dir = debug_dir
        self.force_llm_regen = force_llm_regen
        self.max_notes = max_notes if max_notes is not None else int(os.getenv("TOASTER_OLLAMA_MAX_NOTES", "9999"))

    def stop_model(self) -> None:
        if not self.ollama.enabled:
            return

        try:
            req = urllib.request.Request(
                self.ollama.url,
                data=json.dumps({
                    "model": self.ollama.model,
                    "prompt": "",
                    "stream": False,
                    "keep_alive": 0,
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=15).read()
        except Exception:
            pass

    def _log_call(
        self,
        *,
        game_id: str,
        kind: str,
        prompt: str,
        response: str = "",
        error: str = "",
    ) -> None:
        if self.memory is not None:
            try:
                self.memory.remember_ollama_call(
                    game_id=game_id,
                    kind=kind,
                    model=self.ollama.model,
                    prompt_text=prompt,
                    response_text=response,
                    error_text=error,
                )
            except Exception as exc:
                print(f"Ollama-call DB log failed: {exc}")

        if self.debug_dir is not None:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
            safe_kind = safe_slug(kind)
            safe_game = safe_slug(game_id)[:32]
            counter = len(list(self.debug_dir.glob("*.combined.txt"))) + 1
            base = f"{counter:04d}_{safe_kind}_{safe_game}_{prompt_hash}"

            combined_path = self.debug_dir / f"{base}.combined.txt"
            combined_path.write_text(
                "\n".join([
                    f"KIND: {kind}",
                    f"GAME_ID: {game_id}",
                    f"PROMPT_HASH: {prompt_hash}",
                    "",
                    "===== PROMPT =====",
                    prompt,
                    "",
                    "===== RESPONSE =====",
                    response,
                    "",
                    "===== ERROR =====",
                    error,
                ]),
                encoding="utf-8",
            )
            print(f"DEBUG Ollama I/O: {combined_path}")

    def _call_ollama(self, *, game_id: str, kind: str, prompt: str) -> str:
        if not self.ollama.enabled:
            return ""

        body = {
            "model": self.ollama.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.45,
                "top_p": 0.9,
                "repeat_penalty": 1.15,
                "num_predict": self.ollama.num_predict,
            },
        }

        req = urllib.request.Request(
            self.ollama.url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.ollama.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            response = " ".join(data.get("response", "").split()).strip()
            self._log_call(game_id=game_id, kind=kind, prompt=prompt, response=response)
            return response
        except Exception as exc:
            self._log_call(game_id=game_id, kind=kind, prompt=prompt, error=repr(exc))
            raise

    @staticmethod
    def _clean_note(text: str, *, max_chars: int = 500) -> str:
        text = " ".join(text.split()).strip()
        text = text.strip(' "\'“”‘’')

        for prefix in (
            "Here's a note:",
            "Here is a note:",
            "Note:",
            "Here's the note:",
            "Here is the note:",
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

        if len(text) > max_chars:
            clipped = text[:max_chars]
            last_end = max(clipped.rfind("."), clipped.rfind("!"), clipped.rfind("?"))
            text = clipped[: last_end + 1] if last_end != -1 else clipped.rstrip() + "..."

        return text

    @staticmethod
    def _loss_severity(loss_cp: int) -> str:
        if loss_cp >= LOSS_BLUNDER:
            return "major"
        if loss_cp >= LOSS_MISTAKE:
            return "significant"
        if loss_cp >= LOSS_INACCURACY:
            return "noticeable"
        return "minor"

    def _move_prompt(self, context: GameContext, row: dict, avoidance_block: str) -> str:
        roles = context.roles
        return f"""Move Note rules:
- Write 1-3 short sentences.
- No bullets.
- No intro.
- Use only the engine facts provided.
- Do not invent tactics.
- Do not mention numeric eval scores, centipawns, loss points, or "cp".
- Explain this move only.
- Do not summarize the whole game.
- Do not decide who won or lost the game.
- Sound like Toaster Chess: rude, blunt, mildly profane, and annoyed that it had to watch this.
- Do not insult the user personally; roast the move, the piece, or the position.
- If the move is Best, do not call it incompetent.
- If the move is bad, explain the concrete problem.
- You may invent one dumb fake opening nickname for flavor only if it is obviously a joke.
- Fake opening nicknames are not real chess theory. Do not present them as official openings.
- End with a complete sentence.

Player info:
- You are: {roles.get("you_side") or "Unknown"}
- Computer is: {roles.get("cpu_side") or "Unknown"}

Position before the move:
FEN: {row["board_before"].fen()}

Move being reviewed:
Actor: {row["actor"]}
Played move: {row["played"]}
Label: {row["label"]}
Severity: {self._loss_severity(row["loss_cp"])}

Engine facts:
Eval before: {fmt_eval(row["eval_before"])}
Eval after: {fmt_eval(row["eval_after"])}
Preferred move: {row["best"]}

Plain-English target:
Explain why {row["actor"]} played something useful or stupid at move {row["move_no"]} with {row["played"]}.
If the preferred move differs, mention {row["best"]} only if it helps.

Fallback explanation:
{row["reason"]}

Anti-repetition memory:
{avoidance_block}

The anti-repetition memory is wording history only. It is not factual context for this move.
Do not reuse the same sentence shape, joke, metaphor, insult, or opening phrase from the already-used wording.

Write the move note now.
"""

    def move_note(self, context: GameContext, row: dict, note_index: int) -> str:
        if not self.ollama.enabled or note_index >= self.max_notes:
            return row["reason"]

        avoidance_block = ""
        if self.memory is not None:
            try:
                avoidance_block = self.memory.move_note_avoidance_block(context.game_id, row, limit=8)
            except Exception as exc:
                print(f"Move-note avoidance lookup failed: {exc}")

        prompt = self._move_prompt(context, row, avoidance_block)
        kind = f"move_note_ply_{row['ply']}_{safe_slug(row['actor'])}_{safe_slug(row['label'])}"

        try:
            note = self._call_ollama(game_id=context.game_id, kind=kind, prompt=prompt)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            print(f"Ollama move-note failed: {exc}")
            return row["reason"]

        note = self._clean_note(note)
        if not note:
            return row["reason"]

        if self.memory is not None:
            try:
                self.memory.remember_move_note(context.game_id, row, note)
            except Exception as exc:
                print(f"Move-note memory write failed: {exc}")

        return note

    @staticmethod
    def _worst_row_for_actor(rows: list[dict], actor_prefix: str) -> Optional[dict]:
        actor_rows = [
            r for r in rows
            if r["actor"].startswith(actor_prefix)
            and r["loss_cp"] >= LOSS_INACCURACY
        ]
        if not actor_rows:
            return None
        return max(actor_rows, key=lambda r: r["loss_cp"])

    @staticmethod
    def _row_brief(row: Optional[dict]) -> str:
        if row is None:
            return "None."
        return (
            f"Move {row['move_no']} {row['played']} by {row['actor']}: "
            f"{row['label']}; preferred {row['best']}; "
            f"eval {fmt_eval(row['eval_before'])} to {fmt_eval(row['eval_after'])}; "
            f"loss {row['loss_cp']} cp"
        )

    @staticmethod
    def _compact_key_moments(key_moments: list[dict]) -> str:
        if not key_moments:
            return "No promoted key moments."

        lines = []
        for row in key_moments:
            lines.append(
                f"- Move {row['move_no']} {row['played']} by {row['actor']}: "
                f"{row['label']}; preferred {row['best']}; "
                f"eval {fmt_eval(row['eval_before'])} to {fmt_eval(row['eval_after'])}; "
                f"loss {row['loss_cp']} cp"
            )
        return "\n".join(lines)

    def _summary_prompt(self, context: GameContext, avoidance_block: str) -> str:
        game = context.game
        roles = context.roles
        rows = context.rows
        key_moments = context.key_moments
        facts = game_result_facts(game, roles)

        final_row = rows[-1] if rows else None
        final_move = "unknown"
        if final_row:
            final_move = f"{final_row['move_no']}. {final_row['played']} by {final_row['actor']}"

        worst = max(rows, key=lambda r: r["loss_cp"], default=None)
        worst_text = self._row_brief(worst) if worst and worst["loss_cp"] >= LOSS_INACCURACY else "No major engine complaint."

        worst_you = self._worst_row_for_actor(rows, "You")
        worst_cpu = self._worst_row_for_actor(rows, "CPU")

        return f"""Game Story rules:
- Write 2-4 short sentences.
- No bullets.
- No intro.
- Do not mention Stockfish by name.
- Do not mention eval scores, centipawns, loss points, or numeric engine data.
- Do not define chess terms like a textbook.
- Do not invent moves.
- Do not contradict the non-negotiable outcome facts.
- If User outcome is USER WON, do not say the user lost, got defeated, was beaten, failed to survive, or got a bitter taste from losing.
- If User outcome is USER LOST, do not say the user won.
- A blunder can mean "missed a stronger win" or "missed mate"; it does not automatically mean the mover lost.
- If a player missed forced mate but still won, describe it as sloppy conversion, not defeat.
- You may say the user made mistakes even in a win.
- You may say the CPU threw the game if the worst engine complaint was by CPU.
- If both sides played badly, say so.
- Never use the phrase "won cleanly".
- If the winner made mistakes or blunders, call the win sloppy, messy, or converted anyway.
- Keep it readable on a phone.
- Sound like Toaster Chess: rude, blunt, mildly profane, and annoyed that it had to watch this.
- You may invent one dumb fake opening nickname for flavor.
- Fake opening nicknames must be obviously jokes, not plausible real chess theory.
- Do not claim fake opening nicknames are official or real.
- End with a complete sentence.

Non-negotiable outcome facts:
- PGN result: {facts["result"]}
- User side: {facts["you_side"]}
- CPU side: {facts["cpu_side"]}
- Winner: {facts["winner_actor"]}
- Loser: {facts["loser_actor"]}
- User outcome: {facts["user_outcome"]}

Game info:
Result: {game.headers.get("Result", "?")}
You are: {roles.get("you_side") or "Unknown"}
Computer is: {roles.get("cpu_side") or "Unknown"}
Final move: {final_move}

Engine story:
Biggest engine complaint overall:
{worst_text}

Worst user move:
{self._row_brief(worst_you)}

Worst CPU move:
{self._row_brief(worst_cpu)}

Important interpretation:
- The engine complaint is about move quality, not necessarily the final winner.
- A move can be labeled Blunder because it missed mate or missed a stronger continuation while still keeping a winning position.
- The non-negotiable outcome facts decide who won.

Promoted key moments:
{self._compact_key_moments(key_moments)}

Anti-repetition memory:
{avoidance_block}

The anti-repetition memory is wording history only. It is not factual context for this game.
Do not copy the structure, opening sentence, joke, metaphor, or insult from prior summaries.

Full clean PGN:
{game_to_clean_pgn_text(game)}

Write the game story now.
"""

    def game_story(self, context: GameContext) -> str:
        if not self.ollama.enabled:
            return ""

        # Important: DO NOT inject full old summaries. They can poison the current summary.
        # Future-safe extension point:
        # - keep full summaries stored for audit
        # - derive short banned/repeated phrase lists only
        # - feed only phrases, never full old paragraphs
        if self.memory is not None:
            try:
                avoidance_block = self.memory.summary_phrase_avoidance_block(limit=20)
            except Exception as exc:
                print(f"Summary phrase avoidance lookup failed: {exc}")
                avoidance_block = "No summary phrase blacklist available."
        else:
            avoidance_block = "No summary phrase blacklist available."

        prompt = self._summary_prompt(context, avoidance_block)

        try:
            summary = self._call_ollama(game_id=context.game_id, kind="game_summary", prompt=prompt)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            print(f"Ollama game-summary failed: {exc}")
            return ""

        summary = self._clean_note(summary, max_chars=1200)
        if not summary:
            return ""

        if self.memory is not None:
            try:
                self.memory.remember_game_summary(
                    context.game_id,
                    summary,
                    result=context.game.headers.get("Result", "?"),
                    you_side=context.roles.get("you_side"),
                    cpu_side=context.roles.get("cpu_side"),
                )
            except Exception as exc:
                print(f"Summary memory write failed: {exc}")

        return summary
