# EPV-SARG: Expected Possession Value with Spatial Action Rating Generator

In soccer, understanding the value of possession at any moment is crucial for tactical decision-making. This project develops an Expected Possession Value (EPV) system that quantifies the goal-scoring probability from any position on the field by evaluating what players can do: shoot, pass, or dribble. By combining machine learning models trained on real tracking data with recursive action evaluation, our EPV provides a comprehensive framework for assessing attacking potential and informing strategic decisions. This work extends traditional EPV by incorporating individual player skill into the model when choosing the best decision.

## Overview

This project implements an EPV calculation system that evaluates the expected goals from any position on the field by recursively computing action values for:
- **Shooting**: Using an xG (expected goals) model
- **Passing**: Using a pass completion model
- **Dribbling**: Using a dribble success model

The system incorporates pitch control modeling, player skill individualization, defender positioning, and spatial context to provide realistic possession value estimates.


## Project Structure

```
EPV_SARG/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
│
├── src/                           # Core source code
│   ├── epv_calculator.py          # Main EPV calculation engine
│   ├── pitch_control.py           # Pitch control computation
│   ├── xg_model.py                # Expected goals model
│   ├── passing_model.py           # Pass completion model
│   └── dribbling_model.py         # Dribble success model
│
├── scripts/                       # Executable scripts
│   └── download_skillcorner_data.py  # Data download script
│
├── training/                      # Model training scripts
│   ├── train_xg_model_improved.py
│   ├── train_passing_model_improved.py
│   └── train_dribbling_model.py
│
├── models/                        # Pre-trained models
│   ├── xg_model_improved.pkl
│   ├── passing_model_improved.pkl
│   └── dribbling_model_proper_split.pkl
│
├── data/                          # Player skill data (CSVs)
│   ├── player_id_to_finishing_skill.csv
│   ├── player_id_to_passing_skill.csv
│   └── player_id_to_skill.csv
│
└── results/                       # Example visualizations
    ├── scenario1_edge_box_clear_arrows.png
    ├── scenario2_wide_position_clear_arrows.png
    ├── individuality1_shooting_penalty_FINAL.png
    ├── individuality2_edge_of_box_FINAL.png
    ├── shot_with_tracking.png
    └── shot_with_tracking_alt.png
```

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Using the EPV Calculator

The EPV calculator evaluates the expected goals from any position by considering three possible actions:
- **Shoot**: Immediate shot probability (xG)
- **Pass**: Best pass to a teammate, then their EPV
- **Dribble**: Dribble to a better position, then EPV from there

**Example usage:**

```python
import sys
from pathlib import Path
sys.path.insert(0, 'src')

from epv_calculator import EPVCalculator

# Initialize the calculator with trained models
calc = EPVCalculator(
    xg_model_path=Path('models/xg_model_improved.pkl'),
    passing_model_path=Path('models/passing_model_improved.pkl'),
    dribbling_model_path=Path('models/dribbling_model_proper_split.pkl'),
    xg_skills_path=Path('data/player_id_to_finishing_skill.csv'),
    passing_skills_path=Path('data/player_id_to_passing_skill.csv'),
    dribbling_skills_path=Path('data/player_id_to_skill.csv')
)

# Set match context (required for full EPV calculation)
# This loads tracking data and team rosters for a specific match
calc.set_match_context(
    match_json_path=Path('skillcorner_download/match_123.json'),
    tracking_path=Path('skillcorner_download/123_tracking_extrapolated.jsonl'),
    events_df=events_dataframe  # Loaded from CSV
)

# Get EPV for a possession (returns max of shoot/pass/dribble)
epv = calc.get_epv(
    x=40.0,              # Position x-coordinate (center field at x=52.5)
    y=0.0,               # Position y-coordinate (center width at y=0)
    frame=1000,          # Frame number from tracking data
    player_id=12345,     # Player with possession
    team_id=1,           # Attacking team ID
    tracking_dict=tracking_dict  # Frame-by-frame tracking data
)
print(f"EPV (best action value): {epv:.4f}")

# Or evaluate specific actions individually:
shoot_value = calc.evaluate_shoot(x, y, frame, frame_data, player_id, team_id)
pass_value = calc.evaluate_best_pass(x, y, frame, frame_data, player_id, team_id, tracking_dict, depth=2)
```

**Note:** Full EPV calculation requires tracking data. See the training scripts in `training/` for examples of how to load and use match data.

### View Results

Pre-generated EPV visualizations are available in the `results/` directory, demonstrating:
- **Scenario analysis**: Decision-making at edge of box and wide positions with clear action arrows
- **Player individualization**: How different player skills affect EPV calculations (penalty area, edge of box)
- **Real match tracking**: EPV calculations with actual player positions from tracking data

## Data

The models were trained on **SkillCorner tracking and event data from MLS 2023 season**. Large tracking data files are **not included** in this repository due to size (50-500MB per match) and licensing restrictions.
### Data Files

Each match consists of three files:

#### 1. Match Metadata (`match_<id>.json`)
Contains:
- Team rosters and player IDs
- Match date and location
- Period information
- Player-to-team mappings

#### 2. Tracking Data (`<id>_tracking_extrapolated.jsonl`)
Frame-by-frame player and ball positions (JSONL format, one JSON object per line):
```json
{
  "frame": 0,
  "timestamp": 0.04,
  "period": 1,
  "ball_data": {"x": 52.5, "y": 0.0, "z": 0.0},
  "player_data": [
    {"player_id": 4651, "x": 45.2, "y": -10.3},
    ...
  ]
}
```

**Coordinates:**
- X: 0 to 105 meters (field length)
- Y: -34 to 34 meters (field width, centered)

#### 3. Event Data (`<id>_dynamic_events.csv`)
Annotated match events with 200+ features per event:
- Event type, subtype
- Frame start/end
- Player ID, team ID
- Start/end positions

### Downloading Data

To download SkillCorner data:

1. **Edit credentials** in `scripts/download_skillcorner_data.py` (lines 29-30):
   ```python
   SKILLCORNER_USERNAME = "your_email@example.com"
   SKILLCORNER_PASSWORD = "your_password"
   ```

2. **Download matches**:
   ```bash
   cd scripts
   python3 download_skillcorner_data.py --limit 10  # Downloads 10 matches, can increase the number to download all 500+, but will take very long
   ```

Data will be saved to `skillcorner_download/` directory with the following structure:
```
skillcorner_download/
├── match_<id>.json                    # Match metadata
├── <id>_tracking_extrapolated.jsonl   # Tracking data (25 Hz)
└── <id>_dynamic_events.csv            # Event data
```

## Model Training

To retrain models with downloaded data:

```bash
cd training
python3 train_xg_model_improved.py
python3 train_passing_model_improved.py
python3 train_dribbling_model.py
```

Models are saved to the `models/` directory.

## Methodology

### EPV Calculation

EPV at position (x, y) is computed as:

```
EPV(x, y) = max(Q_shoot, Q_pass, Q_dribble)
```

Where:
- `Q_shoot = xG(x, y)` - expected goals from shooting
- `Q_pass = max_destination(P_success × EPV(destination))` - best pass value
- `Q_dribble = max_destination(P_success × EPV(destination))` - best dribble value

The system recursively evaluates downstream actions to account for multi-step attack sequences.

### Machine Learning Models

- **xG Model**: Predicts goal probability from shots (features: distance, angle, defender pressure, player skill)
- **Passing Model**: Predicts pass completion probability (features: distance, defenders in lane, pitch control, player skill)
- **Dribbling Model**: Predicts dribble success (features: distance, defender proximity, player skill)

All models are trained using scikit-learn on MLS 2023 SkillCorner data.

### Coordinate System

- Pitch dimensions: 105m × 68m
- Origin at center (0, 0)
- X-axis: -52.5 to +52.5 (left to right)
- Y-axis: -34 to +34 (bottom to top)

## References

- Fernández, J., & Bornn, L. (2018). Wide Open Spaces: A statistical technique for measuring space creation in professional soccer. *MIT Sloan Sports Analytics Conference*.
- Spearman, W. (2018). Beyond Expected Goals. *MIT Sloan Sports Analytics Conference*.

