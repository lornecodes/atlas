"""SQLite trigger store — persistent storage for trigger definitions."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import aiosqlite

from atlas.triggers.models import TriggerDefinition


class TriggerStore:
    """Async SQLite store for trigger persistence.

    Usage:
        store = TriggerStore("atlas_jobs.db")
        await store.init()
        await store.save(trigger)
        trigger = await store.get("trigger-abc123")
        due = await store.list_due(before=time.time())
        await store.close()
    """

    def __init__(self, db_path: str = "atlas_jobs.db") -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> None:
        """Create tables if they don't exist."""
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS triggers (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL DEFAULT '',
                trigger_type TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                agent_name TEXT DEFAULT '',
                chain_name TEXT DEFAULT '',
                input_data TEXT NOT NULL DEFAULT '{}',
                priority INTEGER DEFAULT 0,
                metadata TEXT DEFAULT '{}',
                cron_expr TEXT DEFAULT '',
                interval_seconds REAL DEFAULT 0.0,
                fire_at REAL DEFAULT 0.0,
                webhook_secret TEXT DEFAULT '',
                last_fired REAL DEFAULT 0.0,
                next_fire REAL DEFAULT 0.0,
                fire_count INTEGER DEFAULT 0,
                last_job_id TEXT DEFAULT '',
                created_at REAL DEFAULT 0.0
            )
        """)
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_triggers_enabled ON triggers(enabled)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_triggers_type ON triggers(trigger_type)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_triggers_next_fire ON triggers(next_fire)"
        )
        await self._db.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    def _ensure_init(self) -> None:
        if not self._db:
            raise RuntimeError("Store not initialized — call init() first")

    async def save(self, trigger: TriggerDefinition) -> None:
        """Upsert a trigger — insert or update all fields."""
        self._ensure_init()
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO triggers (
                id, name, trigger_type, enabled, agent_name, chain_name,
                input_data, priority, metadata, cron_expr, interval_seconds,
                fire_at, webhook_secret, last_fired, next_fire, fire_count,
                last_job_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name,
                trigger_type=excluded.trigger_type,
                enabled=excluded.enabled,
                agent_name=excluded.agent_name,
                chain_name=excluded.chain_name,
                input_data=excluded.input_data,
                priority=excluded.priority,
                metadata=excluded.metadata,
                cron_expr=excluded.cron_expr,
                interval_seconds=excluded.interval_seconds,
                fire_at=excluded.fire_at,
                webhook_secret=excluded.webhook_secret,
                last_fired=excluded.last_fired,
                next_fire=excluded.next_fire,
                fire_count=excluded.fire_count,
                last_job_id=excluded.last_job_id
            """,
            (
                trigger.id,
                trigger.name,
                trigger.trigger_type,
                1 if trigger.enabled else 0,
                trigger.agent_name,
                trigger.chain_name,
                json.dumps(trigger.input_data),
                trigger.priority,
                json.dumps(trigger.metadata),
                trigger.cron_expr,
                trigger.interval_seconds,
                trigger.fire_at,
                trigger.webhook_secret,
                trigger.last_fired,
                trigger.next_fire,
                trigger.fire_count,
                trigger.last_job_id,
                trigger.created_at,
            ),
        )
        await self._db.commit()

    async def get(self, trigger_id: str) -> TriggerDefinition | None:
        """Get a trigger by ID."""
        self._ensure_init()
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM triggers WHERE id = ?", (trigger_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return self._row_to_trigger(row)

    async def list(
        self,
        *,
        trigger_type: str | None = None,
        enabled: bool | None = None,
        limit: int = 100,
    ) -> list[TriggerDefinition]:
        """List triggers with optional filters."""
        self._ensure_init()
        assert self._db is not None

        conditions: list[str] = []
        params: list[Any] = []

        if trigger_type:
            conditions.append("trigger_type = ?")
            params.append(trigger_type)
        if enabled is not None:
            conditions.append("enabled = ?")
            params.append(1 if enabled else 0)

        where = " AND ".join(conditions) if conditions else "1=1"
        query = f"SELECT * FROM triggers WHERE {where} ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        async with self._db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_trigger(row) for row in rows]

    async def list_due(self, before: float) -> list[TriggerDefinition]:
        """List enabled, scheduled triggers whose next_fire <= before.

        Excludes webhook triggers (they fire on HTTP request, not schedule).
        """
        self._ensure_init()
        assert self._db is not None
        async with self._db.execute(
            """
            SELECT * FROM triggers
            WHERE enabled = 1
              AND trigger_type != 'webhook'
              AND next_fire > 0
              AND next_fire <= ?
            ORDER BY next_fire ASC
            """,
            (before,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_trigger(row) for row in rows]

    async def delete(self, trigger_id: str) -> bool:
        """Delete a trigger. Returns True if it existed."""
        self._ensure_init()
        assert self._db is not None
        cursor = await self._db.execute(
            "DELETE FROM triggers WHERE id = ?", (trigger_id,)
        )
        await self._db.commit()
        return cursor.rowcount > 0

    @staticmethod
    def _row_to_trigger(row: tuple) -> TriggerDefinition:
        """Convert a database row to a TriggerDefinition."""
        t = TriggerDefinition.__new__(TriggerDefinition)
        t.id = row[0]
        t.name = row[1]
        t.trigger_type = row[2]
        t.enabled = bool(row[3])
        t.agent_name = row[4] or ""
        t.chain_name = row[5] or ""
        t.input_data = json.loads(row[6]) if row[6] else {}
        t.priority = row[7]
        t.metadata = json.loads(row[8]) if row[8] else {}
        t.cron_expr = row[9] or ""
        t.interval_seconds = row[10]
        t.fire_at = row[11]
        t.webhook_secret = row[12] or ""
        t.last_fired = row[13]
        t.next_fire = row[14]
        t.fire_count = row[15]
        t.last_job_id = row[16] or ""
        t.created_at = row[17]
        return t
