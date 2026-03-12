"""Tests for the cron expression parser."""

from datetime import datetime, timezone

import pytest

from atlas.triggers.cron import CronExpr, _parse_field


# --- _parse_field ---

class TestParseField:
    def test_wildcard(self):
        assert _parse_field("*", 0, 59) == set(range(0, 60))

    def test_exact(self):
        assert _parse_field("5", 0, 59) == {5}

    def test_range(self):
        assert _parse_field("1-5", 0, 59) == {1, 2, 3, 4, 5}

    def test_step(self):
        assert _parse_field("*/15", 0, 59) == {0, 15, 30, 45}

    def test_range_with_step(self):
        assert _parse_field("0-30/10", 0, 59) == {0, 10, 20, 30}

    def test_list(self):
        assert _parse_field("1,15,30", 0, 59) == {1, 15, 30}

    def test_combined_list_and_range(self):
        assert _parse_field("1-3,10,20-22", 0, 59) == {1, 2, 3, 10, 20, 21, 22}

    def test_clamps_to_range(self):
        # Values outside range are dropped
        assert _parse_field("60", 0, 59) == set()

    def test_step_on_value(self):
        # "5/10" means start at 5, step by 10 to max
        assert _parse_field("5/10", 0, 59) == {5, 15, 25, 35, 45, 55}


# --- CronExpr.parse ---

class TestCronExprParse:
    def test_every_minute(self):
        expr = CronExpr.parse("* * * * *")
        assert expr.minutes == set(range(60))
        assert expr.hours == set(range(24))

    def test_specific_time(self):
        expr = CronExpr.parse("30 9 * * *")
        assert expr.minutes == {30}
        assert expr.hours == {9}

    def test_every_5_minutes(self):
        expr = CronExpr.parse("*/5 * * * *")
        assert expr.minutes == {0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55}

    def test_weekday_range(self):
        # cron 1-5 = Mon-Fri (cron convention), maps to Python 0-4
        expr = CronExpr.parse("0 9 * * 1-5")
        assert expr.weekdays == {0, 1, 2, 3, 4}

    def test_sunday_zero(self):
        # cron 0 = Sunday = Python weekday 6
        expr = CronExpr.parse("0 0 * * 0")
        assert expr.weekdays == {6}

    def test_sunday_seven(self):
        # cron 7 also = Sunday
        expr = CronExpr.parse("0 0 * * 7")
        assert expr.weekdays == {6}

    def test_wrong_field_count(self):
        with pytest.raises(ValueError, match="5 fields"):
            CronExpr.parse("* * *")

    def test_empty_field_raises(self):
        with pytest.raises(ValueError):
            CronExpr.parse("60 * * * *")  # minute 60 is out of range → empty set


# --- CronExpr.matches ---

class TestCronExprMatches:
    def test_every_minute_matches(self):
        expr = CronExpr.parse("* * * * *")
        dt = datetime(2026, 3, 12, 14, 30, tzinfo=timezone.utc)
        assert expr.matches(dt) is True

    def test_specific_time_matches(self):
        expr = CronExpr.parse("30 9 * * *")
        dt = datetime(2026, 3, 12, 9, 30, tzinfo=timezone.utc)
        assert expr.matches(dt) is True

    def test_specific_time_no_match(self):
        expr = CronExpr.parse("30 9 * * *")
        dt = datetime(2026, 3, 12, 10, 30, tzinfo=timezone.utc)
        assert expr.matches(dt) is False

    def test_weekday_match(self):
        # 2026-03-12 is Thursday → Python weekday 3
        expr = CronExpr.parse("0 0 * * 4")  # cron 4 = Thursday → Python 3
        dt = datetime(2026, 3, 12, 0, 0, tzinfo=timezone.utc)
        assert expr.matches(dt) is True

    def test_weekday_no_match(self):
        expr = CronExpr.parse("0 0 * * 1")  # Monday
        dt = datetime(2026, 3, 12, 0, 0, tzinfo=timezone.utc)  # Thursday
        assert expr.matches(dt) is False


# --- CronExpr.next_fire ---

class TestCronExprNextFire:
    def test_every_minute(self):
        expr = CronExpr.parse("* * * * *")
        # 2026-03-12 14:30:00 UTC
        after = datetime(2026, 3, 12, 14, 30, 0, tzinfo=timezone.utc).timestamp()
        nf = expr.next_fire(after)
        dt = datetime.fromtimestamp(nf, tz=timezone.utc)
        assert dt.minute == 31
        assert dt.hour == 14

    def test_specific_time_next_day(self):
        expr = CronExpr.parse("0 9 * * *")
        # After 9:01 today → should be 9:00 tomorrow
        after = datetime(2026, 3, 12, 9, 1, 0, tzinfo=timezone.utc).timestamp()
        nf = expr.next_fire(after)
        dt = datetime.fromtimestamp(nf, tz=timezone.utc)
        assert dt.day == 13
        assert dt.hour == 9
        assert dt.minute == 0

    def test_every_5_minutes(self):
        expr = CronExpr.parse("*/5 * * * *")
        after = datetime(2026, 3, 12, 14, 32, 0, tzinfo=timezone.utc).timestamp()
        nf = expr.next_fire(after)
        dt = datetime.fromtimestamp(nf, tz=timezone.utc)
        assert dt.minute == 35

    def test_strictly_after(self):
        # If we're exactly at a match, next_fire should return the NEXT match
        expr = CronExpr.parse("30 9 * * *")
        after = datetime(2026, 3, 12, 9, 30, 0, tzinfo=timezone.utc).timestamp()
        nf = expr.next_fire(after)
        dt = datetime.fromtimestamp(nf, tz=timezone.utc)
        # Should be next day at 9:30, not today
        assert dt.day == 13
