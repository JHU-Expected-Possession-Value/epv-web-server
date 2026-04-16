"""Path utilities for local SkillCorner tracking and events files (offline / dev only).

**Website runtime:** FastAPI replay and tactics routes must **not** import this module.
They read **`frame`**, **`detection`**, **`events`**, **`matches`**, etc. from PostgreSQL
via `api.db` + `api.services.replay_service` (filtered SQL, no full-table loads).

**When this is used:** training scripts, one-off ingestion, or local tools that still expect
`*_tracking.jsonl` / `*_dynamic_events.csv` under **`EPV_DATA_DIR`**.

If `EPV_DATA_DIR` is unset, helpers here return empty lists or raise only when called — not at import.
"""

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Load environment variables if not already loaded
load_dotenv()

# Read EPV_DATA_DIR from environment (optional).
# If unset, local file helpers will raise when called (not at import time).
EPV_DATA_DIR = os.getenv("EPV_DATA_DIR")

# Define directory paths
tracking_dir = Path(EPV_DATA_DIR) / "tracking" if EPV_DATA_DIR else None
events_dir = Path(EPV_DATA_DIR) / "dynamic_events" if EPV_DATA_DIR else None


def validate_directories() -> None:
    """Validate that tracking_dir and events_dir exist. Raise RuntimeError if missing."""
    if tracking_dir is None or events_dir is None:
        raise RuntimeError("EPV_DATA_DIR is not set; local file paths are unavailable.")
    if not tracking_dir.exists():
        raise RuntimeError(
            f"Tracking directory does not exist: {tracking_dir}\n"
            f"Please ensure EPV_DATA_DIR points to a directory containing a 'tracking' subdirectory."
        )
    if not tracking_dir.is_dir():
        raise RuntimeError(
            f"Tracking path exists but is not a directory: {tracking_dir}"
        )
    if not events_dir.exists():
        raise RuntimeError(
            f"Events directory does not exist: {events_dir}\n"
            f"Please ensure EPV_DATA_DIR points to a directory containing a 'dynamic_events' subdirectory."
        )
    if not events_dir.is_dir():
        raise RuntimeError(
            f"Events path exists but is not a directory: {events_dir}"
        )


# Do not validate on import (deployed backend runs without EPV_DATA_DIR).


def get_tracking_path(match_id: str) -> Path:
    """Get Path for tracking file: '{match_id}_tracking.jsonl' in tracking_dir."""
    validate_directories()
    return tracking_dir / f"{match_id}_tracking.jsonl"


def get_events_path(match_id: str) -> Path:
    """Get Path for events CSV file matching match_id.
    
    Checks for both '{match_id}_events.csv' and '{match_id}_dynamic_events.csv'.
    Raises RuntimeError if neither exists.
    """
    validate_directories()
    # Try both naming patterns
    events_path = events_dir / f"{match_id}_events.csv"
    dynamic_events_path = events_dir / f"{match_id}_dynamic_events.csv"
    
    if events_path.exists():
        return events_path
    elif dynamic_events_path.exists():
        return dynamic_events_path
    else:
        raise RuntimeError(
            f"No events file found for match_id '{match_id}'. "
            f"Checked:\n"
            f"  - {events_path}\n"
            f"  - {dynamic_events_path}"
        )


def list_available_match_ids() -> List[str]:
    """List available match IDs derived from files in tracking_dir matching '*_tracking.jsonl'.
    
    Returns a list of match_id strings (extracted from filenames).
    """
    match_ids = []
    if tracking_dir is None or not tracking_dir.exists():
        return match_ids
    
    for file_path in tracking_dir.glob("*_tracking.jsonl"):
        # Extract match_id from filename: "{match_id}_tracking.jsonl"
        filename = file_path.name
        if filename.endswith("_tracking.jsonl"):
            match_id = filename[:-len("_tracking.jsonl")]
            if match_id:  # Only add non-empty match IDs
                match_ids.append(match_id)
    
    return sorted(match_ids)
