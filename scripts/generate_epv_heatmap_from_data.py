"""
Generate EPV heatmap from actual tracking data.

This script loads a specific match frame and generates an EPV heatmap
showing how the expected possession value varies across the pitch given
the actual player positions and game state.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from generate_epv_heatmap import generate_epv_heatmap_with_tracking, RESULTS_DIR
import pandas as pd

# Data paths
DATA_DIR = Path("/Users/jonathanlocala/PycharmProjects/EPV_SARG/data/skillcorner_download")

# Choose a match
MATCH_ID = "1112357"

# Paths
match_json_path = DATA_DIR / f"match_{MATCH_ID}.json"
tracking_path = DATA_DIR / f"{MATCH_ID}_tracking_extrapolated.jsonl"
events_csv_path = DATA_DIR / f"{MATCH_ID}_dynamic_events.csv"

print(f"\n{'='*60}")
print(f"GENERATING EPV HEATMAP FROM MATCH {MATCH_ID}")
print(f"{'='*60}\n")

# Load events to find an interesting frame
print("Loading events to find an interesting frame...")
events = pd.read_csv(events_csv_path)

# Filter for shots or key passes in attacking third
interesting_events = events[
    (events['event_type'].isin(['shot', 'pass'])) &
    (events['x_start'] > 70) &  # Attacking third
    (events['frame_start'].notna())
].copy()

if len(interesting_events) == 0:
    print("No interesting events found, using first event with frame data")
    interesting_events = events[events['frame_start'].notna()].copy()

# Pick a frame from middle of the data
mid_idx = len(interesting_events) // 2
selected_event = interesting_events.iloc[mid_idx]

frame_number = int(selected_event['frame_start'])
player_id = int(selected_event['player_id']) if pd.notna(selected_event['player_id']) else 1
team_id = int(selected_event['team_id']) if pd.notna(selected_event['team_id']) else 1

print(f"\nSelected event details:")
print(f"  Frame: {frame_number}")
print(f"  Player ID: {player_id}")
print(f"  Team ID: {team_id}")
print(f"  Event type: {selected_event['event_type']}")
print(f"  Position: ({selected_event['x_start']:.1f}, {selected_event['y_start']:.1f})")

# Generate heatmap
output_path = RESULTS_DIR / f"epv_heatmap_match_{MATCH_ID}_frame_{frame_number}.png"

print(f"\n{'='*60}")
print("GENERATING HEATMAP...")
print(f"{'='*60}\n")

generate_epv_heatmap_with_tracking(
    match_json_path=match_json_path,
    tracking_path=tracking_path,
    events_csv_path=events_csv_path,
    frame_number=frame_number,
    player_id=player_id,
    team_id=team_id,
    output_path=output_path,
    grid_resolution=5.0  # Coarser grid for speed (~400 points instead of ~1900)
)

print(f"\n{'='*60}")
print("DONE!")
print(f"{'='*60}")
print(f"\nHeatmap saved to: {output_path}")
