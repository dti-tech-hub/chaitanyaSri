"""Headless smoke test of the Streamlit dashboard.

Runs the dashboard script through Streamlit's AppTest harness and fails if the
page raises an exception or loses its expected tabs. It does not click "Start",
so it is fast and does not process video.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    from streamlit.testing.v1 import AppTest

    app = PROJECT_ROOT / "dashboard" / "app.py"
    at = AppTest.from_file(str(app), default_timeout=120)
    at.run()
    errors = list(at.exception)
    tabs = len(at.tabs)
    print(f"dashboard script: {app}")
    print(f"tabs rendered: {tabs}")
    print(f"exceptions: {len(errors)}")
    for error in errors:
        print(f"ERROR: {error.value}")
    if errors:
        return 1
    if tabs < 5:
        print("ERROR: expected at least 5 dashboard tabs")
        return 1
    print("dashboard smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
