#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional

import mysql.connector
from mysql.connector import MySQLConnection


DEFAULT_DB_NAME = "toaster_chess_ollama"


@dataclass(frozen=True)
class MysqlConfig:
    host: str = os.getenv("TOASTER_MYSQL_HOST", "127.0.0.1")
    port: int = int(os.getenv("TOASTER_MYSQL_PORT", "3306"))
    user: str = os.getenv("TOASTER_MYSQL_USER", "toaster")
    password: str = os.getenv("TOASTER_MYSQL_PASSWORD", "")
    database: str = os.getenv("TOASTER_MYSQL_DATABASE", DEFAULT_DB_NAME)


class OllamaPhraseMemory:
    """
    Small MySQL-backed memory for reducing Ollama repetition.

    Design:
    - Move notes are scoped to one game_id.
    - Game summaries are global across all games.
    - This does NOT store engine truth. Only text Ollama already wrote.
    """

    def __init__(self, cfg: Optional[MysqlConfig] = None) -> None:
        self.cfg = cfg or MysqlConfig()

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

    def init_db(self) -> None:
        """
        Creates database and tables if missing.
        Safe to call at script startup.
        """
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

    @staticmethod
    def game_id_from_pgn_text(clean_pgn: str) -> str:
        """
        Stable game ID from clean PGN.
        Good enough for this pipeline.
        """
        return hashlib.sha256(clean_pgn.encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def _hash_text(text: str) -> str:
        normalized = " ".join(text.split()).strip()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

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
        """
        Pulls only notes from this same game.

        Optional filters help avoid giving irrelevant phrases:
        - same label: Blunder/Mistake/etc.
        - same actor prefix: "You" or "CPU"
        """
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

    def recent_game_summaries(self, *, limit: int = 12) -> list[str]:
        """
        Pulls global prior summaries across all games.
        This is for the game-story prompt only.
        """
        with self._connect_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT summary_text
                FROM game_summaries
                ORDER BY created_at DESC, id DESC
                LIMIT %s
                """,
                (limit,),
            )
            return [row[0] for row in cur.fetchall()]

    @staticmethod
    def format_avoidance_block(items: Iterable[str], title: str) -> str:
        items = [i.strip() for i in items if i and i.strip()]
        if not items:
            return f"{title}\n- None yet."

        lines = [title]
        for item in items:
            lines.append(f"- {item}")
        return "\n".join(lines)

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

    def game_summary_avoidance_block(self, *, limit: int = 12) -> str:
        summaries = self.recent_game_summaries(limit=limit)

        return self.format_avoidance_block(
            summaries,
            "Recent game summaries already written:",
        )
