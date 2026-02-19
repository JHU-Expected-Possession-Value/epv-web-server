"""Path utilities for SkillCorner tracking and events files."""

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Load environment variables if not already loaded
load_dotenv()

# Read EPV_DATA_DIR from environment (required)
EPV_DATA_DIR = os.getenv("EPV_DATA_DIR")
if not EPV_DATA_DIR:
    raise RuntimeError(
        "EPV_DATA_DIR environment variable must be set. "
        "Please set it in your .env file or environment."
    )

# Define directory paths
tracking_dir = Path(EPV_DATA_DIR) / "tracking"
events_dir = Path(EPV_DATA_DIR) / "dynamic_events"


def validate_directories() -> None:
    """Validate that tracking_dir and events_dir exist. Raise RuntimeError if missing."""
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


# Validate directories on import
validate_directories()


def get_tracking_path(match_id: str) -> Path:
    """Get Path for tracking file: '{match_id}_tracking.jsonl' in tracking_dir."""
    return tracking_dir / f"{match_id}_tracking.jsonl"


def get_events_path(match_id: str) -> Path:
    """Get Path for events CSV file matching match_id.
    
    Checks for both '{match_id}_events.csv' and '{match_id}_dynamic_events.csv'.
    Raises RuntimeError if neither exists.
    """
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
    if not tracking_dir.exists():
        return match_ids
    
    for file_path in tracking_dir.glob("*_tracking.jsonl"):
        # Extract match_id from filename: "{match_id}_tracking.jsonl"
        filename = file_path.name
        if filename.endswith("_tracking.jsonl"):
            match_id = filename[:-len("_tracking.jsonl")]
            if match_id:  # Only add non-empty match IDs
                match_ids.append(match_id)
    
    return sorted(match_ids)
