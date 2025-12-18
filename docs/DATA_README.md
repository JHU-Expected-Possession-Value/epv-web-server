# Data Requirements

This document explains the data requirements for the EPV-SARG system.

## Overview

The EPV calculation system requires three types of data files for each match:
1. Match metadata (JSON)
2. Tracking data (JSONL)
3. Event data (CSV)

## Why Data is Not Included

Large tracking and event data files are **not included** in this repository because:
- File sizes typically range from 50MB to 500MB per match
- Git repositories are optimized for code, not large binary/data files
- Data licensing restrictions may prevent redistribution
- Training data contains 100+ matches (multiple gigabytes total)

## Expected Data Structure

Place your data files in a directory structure like this:

```
/path/to/your/data/
└── skillcorner_download/
    ├── match_1112357.json
    ├── 1112357_tracking_extrapolated.jsonl
    ├── 1112357_dynamic_events.csv
    ├── match_1112359.json
    ├── 1112359_tracking_extrapolated.jsonl
    ├── 1112359_dynamic_events.csv
    └── ...
```

## File Formats

### 1. Match Metadata (`match_<id>.json`)
Contains match-level information:
- Team rosters and player IDs
- Match date and location
- Period information
- Player-to-team mappings

Example structure:
```json
{
  "match_id": "1112357",
  "home_team": {...},
  "away_team": {...},
  "pid_to_teamid": {
    "4651": 1498,
    "4652": 1498,
    ...
  }
}
```

### 2. Tracking Data (`<id>_tracking_extrapolated.jsonl`)
Frame-by-frame player and ball positions. Each line is a JSON object:

```json
{
  "frame": 0,
  "timestamp": 0.04,
  "period": 1,
  "ball_data": {"x": 52.5, "y": 0.0, "z": 0.0, "is_detected": true},
  "player_data": [
    {"player_id": 4651, "x": 45.2, "y": -10.3, "is_detected": true},
    ...
  ]
}
```

**Coordinates:**
- X: 0 to 105 meters (length)
- Y: -34 to 34 meters (width, centered)

### 3. Event Data (`<id>_dynamic_events.csv`)
Row-per-event structure with columns including:
- `event_id`, `event_type`, `event_subtype`
- `frame_start`, `frame_end`
- `player_id`, `team_id`
- `x_start`, `y_start`, `x_end`, `y_end`
- Plus 100+ additional feature columns for analysis

## Using Your Own Data

To use this code with your own data:

1. **Update paths** in scripts:
   ```python
   DATA_DIR = Path("/path/to/your/data/skillcorner_download")
   ```

2. **Verify file naming** matches the pattern:
   - `match_<id>.json`
   - `<id>_tracking_extrapolated.jsonl`
   - `<id>_dynamic_events.csv`

3. **Check coordinate system** - code assumes:
   - Pitch: 105m × 68m
   - Center-based coordinates for calculations
   - Absolute coordinates (0-105) in some visualizations

## Data Sources

This project uses:
- **SkillCorner tracking data** - High-frequency (25 Hz) player tracking
- **SkillCorner event data** - Rich event annotations with 200+ features
- **MLS 2023 season** - Training and validation data

## Obtaining Data

For academic/research use:
- Contact SkillCorner for tracking data access
- Alternative open datasets: StatsBomb, Wyscout (different formats)
- Metrica Sports provides sample tracking data

## Data Privacy & Ethics

- Player tracking data may be proprietary
- Respect data usage agreements and licenses
- Do not redistribute licensed data without permission
- Anonymize player information if required

## Helper Script

Use `generate_epv_heatmap_from_data.py` to generate heatmaps with your data:

```python
# Edit these paths in the script
DATA_DIR = Path("/Users/yourname/your_data_location")
MATCH_ID = "1112357"  # Your match ID
```

Then run:
```bash
python3 generate_epv_heatmap_from_data.py
```

## Troubleshooting

**Issue**: "File not found" errors
- **Solution**: Verify data paths and file naming match expected format

**Issue**: "KeyError" when loading data
- **Solution**: Check that JSON/CSV columns match expected structure

**Issue**: Empty tracking data
- **Solution**: Ensure tracking files use `.jsonl` format (one JSON per line)

## Questions?

For questions about data formats or access, please refer to the main README or contact the repository maintainer.
