#!/usr/bin/env python3
"""Root entrypoint for the full LiDAR-robot pipeline in Main_Code."""

import sys
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parent
    main_code_dir = repo_root / "Main_Code"
    sys.path.insert(0, str(main_code_dir))

    from main_pipeline import main as run_pipeline

    run_pipeline()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Keep traceback behavior unchanged for ROS debugging.
        raise
