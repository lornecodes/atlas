"""Exec-echo agent — reads JSON envelope from stdin, echoes input back on stdout.

This is a minimal example of an exec-provider agent. Any language can
implement this protocol: read JSON from stdin, write JSON to stdout.

Stdin envelope format:
    {"input": {...}, "context": {"job_id": ..., ...}, "memory": "..."}
"""

import json
import sys


def main():
    envelope = json.loads(sys.stdin.read())
    # Just echo the input back
    print(json.dumps(envelope["input"]))


if __name__ == "__main__":
    main()
