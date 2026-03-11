"""Shared constants for Atlas."""

# Terminal job statuses — a job in one of these states will not change again.
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
