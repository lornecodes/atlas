"""Lightweight cron expression parser — no external dependencies.

Supports standard 5-field cron expressions:
    minute hour day-of-month month day-of-week

Field syntax:
    *       — every value
    */N     — every N values
    N       — exact value
    N-M     — range (inclusive)
    N,M,O   — list
    N-M/S   — range with step
"""

from __future__ import annotations

import calendar
from datetime import datetime, timedelta, timezone

# Field boundaries: (name, min, max)
_FIELDS = [
    ("minute", 0, 59),
    ("hour", 0, 23),
    ("day", 1, 31),
    ("month", 1, 12),
    ("weekday", 0, 6),  # 0 = Monday in Python, but cron uses 0 = Sunday
]

# Map cron weekday (0=Sun) to Python weekday (0=Mon)
_CRON_TO_PY_WEEKDAY = {0: 6, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}


def _parse_field(token: str, min_val: int, max_val: int) -> set[int]:
    """Parse a single cron field into a set of matching integers."""
    values: set[int] = set()
    for part in token.split(","):
        if "/" in part:
            range_part, step_str = part.split("/", 1)
            step = int(step_str)
            if range_part == "*":
                start, end = min_val, max_val
            elif "-" in range_part:
                start, end = (int(x) for x in range_part.split("-", 1))
            else:
                start, end = int(range_part), max_val
            values.update(range(start, end + 1, step))
        elif "-" in part:
            start, end = (int(x) for x in part.split("-", 1))
            values.update(range(start, end + 1))
        elif part == "*":
            values.update(range(min_val, max_val + 1))
        else:
            values.add(int(part))
    # Clamp to valid range
    return {v for v in values if min_val <= v <= max_val}


class CronExpr:
    """Parsed cron expression with matching and next-fire computation."""

    __slots__ = ("expr", "minutes", "hours", "days", "months", "weekdays")

    def __init__(
        self,
        expr: str,
        minutes: set[int],
        hours: set[int],
        days: set[int],
        months: set[int],
        weekdays: set[int],
    ) -> None:
        self.expr = expr
        self.minutes = minutes
        self.hours = hours
        self.days = days
        self.months = months
        self.weekdays = weekdays

    @classmethod
    def parse(cls, expr: str) -> CronExpr:
        """Parse a 5-field cron expression string."""
        parts = expr.strip().split()
        if len(parts) != 5:
            raise ValueError(
                f"Cron expression must have 5 fields, got {len(parts)}: {expr!r}"
            )

        minutes = _parse_field(parts[0], 0, 59)
        hours = _parse_field(parts[1], 0, 23)
        days = _parse_field(parts[2], 1, 31)
        months = _parse_field(parts[3], 1, 12)

        # Parse weekday with cron convention (0=Sunday) -> Python (0=Monday)
        raw_weekdays = _parse_field(parts[4], 0, 7)
        weekdays = {_CRON_TO_PY_WEEKDAY.get(w, w) for w in raw_weekdays}

        for name, field_set in [
            ("minute", minutes), ("hour", hours), ("day", days),
            ("month", months), ("weekday", weekdays),
        ]:
            if not field_set:
                raise ValueError(f"Cron field {name!r} produced no valid values: {expr!r}")

        return cls(expr, minutes, hours, days, months, weekdays)

    def matches(self, dt: datetime) -> bool:
        """Check if a datetime matches this cron expression."""
        return (
            dt.minute in self.minutes
            and dt.hour in self.hours
            and dt.day in self.days
            and dt.month in self.months
            and dt.weekday() in self.weekdays
        )

    def next_fire(self, after: float) -> float:
        """Return the next fire time (unix timestamp) strictly after `after`.

        Scans forward minute-by-minute. Caps search at 366 days to avoid
        infinite loops on impossible expressions (e.g., Feb 31).
        """
        dt = datetime.fromtimestamp(after, tz=timezone.utc).replace(second=0, microsecond=0)
        dt += timedelta(minutes=1)  # strictly after
        limit = dt + timedelta(days=366)

        while dt < limit:
            if dt.month not in self.months:
                # Skip to first day of next month
                if dt.month == 12:
                    dt = dt.replace(year=dt.year + 1, month=1, day=1, hour=0, minute=0)
                else:
                    dt = dt.replace(month=dt.month + 1, day=1, hour=0, minute=0)
                continue

            max_day = calendar.monthrange(dt.year, dt.month)[1]
            if dt.day not in self.days or dt.day > max_day:
                dt += timedelta(days=1)
                dt = dt.replace(hour=0, minute=0)
                continue

            if dt.weekday() not in self.weekdays:
                dt += timedelta(days=1)
                dt = dt.replace(hour=0, minute=0)
                continue

            if dt.hour not in self.hours:
                dt += timedelta(hours=1)
                dt = dt.replace(minute=0)
                continue

            if dt.minute not in self.minutes:
                dt += timedelta(minutes=1)
                continue

            return dt.timestamp()

        raise ValueError(
            f"No matching time found within 366 days for cron expression: {self.expr!r}"
        )
