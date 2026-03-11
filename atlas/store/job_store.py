"""SQLite job store — persistent storage for job data via aiosqlite."""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any

import aiosqlite

from atlas.pool.job import JobData


class JobStore:
    """Async SQLite store for job persistence.

    Usage:
        store = JobStore("jobs.db")
        await store.init()
        await store.save(job)
        job = await store.get("job-abc123")
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
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                agent_name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                input_data TEXT NOT NULL DEFAULT '{}',
                output_data TEXT,
                error TEXT DEFAULT '',
                priority INTEGER DEFAULT 0,
                created_at REAL DEFAULT 0.0,
                started_at REAL DEFAULT 0.0,
                completed_at REAL DEFAULT 0.0,
                warmup_ms REAL DEFAULT 0.0,
                execution_ms REAL DEFAULT 0.0,
                retry_count INTEGER DEFAULT 0,
                original_job_id TEXT DEFAULT '',
                metadata TEXT DEFAULT '{}'
            )
        """)
        # Migrate existing DBs that lack newer columns
        for col_def in (
            "retry_count INTEGER DEFAULT 0",
            "original_job_id TEXT DEFAULT ''",
            "metadata TEXT DEFAULT '{}'",
        ):
            try:
                await self._db.execute(f"ALTER TABLE jobs ADD COLUMN {col_def}")
            except sqlite3.OperationalError:
                pass  # Column already exists
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_agent ON jobs(agent_name)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at)"
        )
        await self._db.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    async def save(self, job: JobData) -> None:
        """Upsert a job — insert or update all fields."""
        if not self._db:
            raise RuntimeError("Store not initialized — call init() first")
        await self._db.execute(
            """
            INSERT INTO jobs (id, agent_name, status, input_data, output_data,
                            error, priority, created_at, started_at, completed_at,
                            warmup_ms, execution_ms, retry_count, original_job_id,
                            metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status=excluded.status,
                output_data=excluded.output_data,
                error=excluded.error,
                started_at=excluded.started_at,
                completed_at=excluded.completed_at,
                warmup_ms=excluded.warmup_ms,
                execution_ms=excluded.execution_ms,
                retry_count=excluded.retry_count,
                original_job_id=excluded.original_job_id,
                metadata=excluded.metadata
            """,
            (
                job.id,
                job.agent_name,
                job.status,
                json.dumps(job.input_data),
                json.dumps(job.output_data) if job.output_data is not None else None,
                job.error,
                job.priority,
                job.created_at,
                job.started_at,
                job.completed_at,
                job.warmup_ms,
                job.execution_ms,
                job.retry_count,
                job.original_job_id,
                json.dumps(job.metadata),
            ),
        )
        await self._db.commit()

    async def get(self, job_id: str) -> JobData | None:
        """Get a job by ID."""
        if not self._db:
            raise RuntimeError("Store not initialized — call init() first")
        async with self._db.execute(
            "SELECT * FROM jobs WHERE id = ?", (job_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return self._row_to_job(row)

    async def list(
        self,
        *,
        status: str | None = None,
        agent_name: str | None = None,
        since: float | None = None,
        until: float | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[JobData]:
        """List jobs with optional filters."""
        if not self._db:
            raise RuntimeError("Store not initialized — call init() first")

        conditions: list[str] = []
        params: list[Any] = []

        if status:
            conditions.append("status = ?")
            params.append(status)
        if agent_name:
            conditions.append("agent_name = ?")
            params.append(agent_name)
        if since is not None:
            conditions.append("created_at >= ?")
            params.append(since)
        if until is not None:
            conditions.append("created_at <= ?")
            params.append(until)

        where = " AND ".join(conditions) if conditions else "1=1"
        query = f"SELECT * FROM jobs WHERE {where} ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        async with self._db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_job(row) for row in rows]

    async def count(self, *, status: str | None = None) -> int:
        """Count jobs, optionally filtered by status."""
        if not self._db:
            raise RuntimeError("Store not initialized — call init() first")

        if status:
            query = "SELECT COUNT(*) FROM jobs WHERE status = ?"
            params = (status,)
        else:
            query = "SELECT COUNT(*) FROM jobs"
            params = ()

        async with self._db.execute(query, params) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

    @staticmethod
    def _row_to_job(row: tuple) -> JobData:
        """Convert a database row to a JobData instance."""
        job = JobData.__new__(JobData)
        job.id = row[0]
        job.agent_name = row[1]
        job.status = row[2]
        job.input_data = json.loads(row[3]) if row[3] else {}
        job.output_data = json.loads(row[4]) if row[4] else None
        job.error = row[5] or ""
        job.priority = row[6]
        job.created_at = row[7]
        job.started_at = row[8]
        job.completed_at = row[9]
        job.warmup_ms = row[10]
        job.execution_ms = row[11]
        job.retry_count = row[12] if len(row) > 12 else 0
        job.original_job_id = row[13] if len(row) > 13 else ""
        job.metadata = json.loads(row[14]) if len(row) > 14 and row[14] else {}
        return job
