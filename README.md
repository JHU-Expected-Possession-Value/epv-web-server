# EPV-SARG: Expected Possession Value with Spatial Action Rating Generator

A soccer analytics system that calculates Expected Possession Value (EPV) by modeling shooting, passing, and dribbling actions using machine learning models trained on real match data.

## Overview

This project implements an on-demand EPV calculation system that evaluates the expected goals from any position on the field by recursively computing action values for:
- **Shooting**: Using an xG (expected goals) model
- **Passing**: Using a pass completion model
- **Dribbling**: Using a dribble success model

The system incorporates:
- Pitch control modeling
- Player skill individualization
- Defender positioning and pressure
- Spatial context and game state

## Project Structure

```
EPV_SARG/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
│
├── src/                           # Core source code
│   ├── epv_calculator.py          # Main EPV calculation engine
│   ├── pitch_control.py           # Pitch control computation
│   ├── xg_model.py                # Expected goals model
│   ├── passing_model.py           # Pass completion model
│   └── dribbling_model.py         # Dribble success model
│
├── scripts/                       # Executable scripts
│   ├── generate_results.py        # Visualization generation
│   ├── generate_epv_heatmap.py    # EPV heatmap generation
│   └── generate_epv_heatmap_from_data.py  # Heatmap with external data
│
├── training/                      # Model training scripts
│   ├── train_xg_model_improved.py
│   ├── train_passing_model_improved.py
│   └── train_dribbling_model.py
│
├── models/                        # Trained models
│   ├── xg_model_improved.pkl
│   ├── passing_model_improved.pkl
│   ├── dribbling_model_proper_split.pkl
│   └── train_val_test_split.pkl
│
├── data/                          # Player skill data
│   ├── player_id_to_finishing_skill.csv
│   ├── player_id_to_passing_skill.csv
│   ├── player_id_to_skill.csv
│   └── mls_2023_player_possession_FIXED_NAMES.csv
│
├── results/                       # Generated visualizations
│   ├── epv_heatmap_simple.png
│   ├── scenario_compare.png
│   └── ... (9+ visualization files)
│
└── docs/                          # Additional documentation
    ├── QUICKSTART.md
    ├── DATA_README.md
    └── SUBMISSION_CHECKLIST.md
```

## Key Features

### 1. On-Demand EPV Calculation
- Computes EPV for any location on the pitch given a game state
- Uses recursive action evaluation with configurable depth
- Incorporates actual tracking data and player positions

### 2. Machine Learning Models
- **xG Model**: Predicts goal probability from shots based on distance, angle, defender pressure, and player skill
- **Passing Model**: Predicts pass completion probability based on distance, defenders in lane, pitch control, and player skill
- **Dribbling Model**: Predicts dribble success based on distance, defender proximity, and player skill

### 3. Player Individuality
- Player-specific finishing, passing, and dribbling skill ratings
- Derived from historical performance data (MLS 2023 season)

### 4. Pitch Control
- Spatial dominance calculation for both teams
- Used in passing and dribbling evaluation

## Installation

### Requirements
- Python 3.8+
- NumPy
- pandas
- scikit-learn
- matplotlib
- joblib

### Setup
```bash
# Install dependencies
pip install numpy pandas scikit-learn matplotlib joblib

# Clone the repository
git clone <repository-url>
cd EPV_SARG
```

## Usage

### Generate EPV Heatmap (Static)
Create a heatmap showing EPV across the pitch based on shooting probability:

```bash
cd scripts
python3 generate_epv_heatmap.py
```

Output: `results/epv_heatmap_simple.png`

### Generate EPV for Specific Game Frame
To generate EPV with actual tracking data:

```python
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'src')

from generate_epv_heatmap import generate_epv_heatmap_with_tracking

BASE = Path('.')
generate_epv_heatmap_with_tracking(
    match_json_path=Path("path/to/match_123.json"),
    tracking_path=Path("path/to/123_tracking.jsonl"),
    events_csv_path=Path("path/to/123_events.csv"),
    frame_number=1000,
    player_id=456,
    team_id=1,
    output_path=BASE / "results" / "epv_heatmap_frame_1000.png"
)
```

### Compute EPV for Specific Position
```python
import sys
from pathlib import Path

sys.path.insert(0, 'src')
from epv_calculator import EPVCalculator

# Initialize calculator
BASE = Path('.')
epv_calc = EPVCalculator(
    xg_model_path=BASE / "models" / "xg_model_improved.pkl",
    passing_model_path=BASE / "models" / "passing_model_improved.pkl",
    dribbling_model_path=BASE / "models" / "dribbling_model_proper_split.pkl",
    xg_skills_path=BASE / "data" / "player_id_to_finishing_skill.csv",
    passing_skills_path=BASE / "data" / "player_id_to_passing_skill.csv",
    dribbling_skills_path=BASE / "data" / "player_id_to_skill.csv"
)

# Set match context
epv_calc.set_match_context(match_json_path, tracking_path, events_df)

# Calculate EPV
epv = epv_calc.get_epv(
    x=30.0,  # meters from center
    y=0.0,   # meters from center
    frame=1000,
    player_id=456,
    team_id=1,
    tracking_dict=tracking_dict
)
```

## Data Requirements

### Training Data
The models were trained on SkillCorner tracking and event data from MLS 2023 season. Training data is **not included** in this repository due to size constraints.

### File Structure Expected
```
data/
├── skillcorner_download/
│   ├── match_<id>.json           # Match metadata
│   ├── <id>_tracking_extrapolated.jsonl  # Tracking data
│   └── <id>_dynamic_events.csv   # Event data
```

### Running with Your Own Data
1. Place tracking data in the expected directory structure
2. Ensure match metadata, tracking, and event files use matching IDs
3. Update paths in scripts to point to your data location

## Model Training

To retrain models with new data:

```bash
cd training

# Train xG model
python3 train_xg_model_improved.py

# Train passing model
python3 train_passing_model_improved.py

# Train dribbling model
python3 train_dribbling_model.py
```

## Results

The `results/` directory contains example visualizations:
- `epv_heatmap_simple.png`: Static EPV heatmap
- `scenario_compare.png`: Side-by-side attacking scenarios
- `attack_timeseries.png`: EPV evolution during attack sequence
- Additional scenario-specific visualizations

## Methodology

### EPV Calculation
EPV at position (x, y) is computed as:

```
EPV(x, y) = max(Q_shoot, Q_pass, Q_dribble)
```

Where:
- `Q_shoot = xG(x, y)` - expected goals from shooting
- `Q_pass = max_destination(P_success * EPV(destination))` - best pass value
- `Q_dribble = max_destination(P_success * EPV(destination))` - best dribble value

### Recursive Evaluation
The system recursively evaluates downstream actions with configurable depth to account for multi-step attack sequences.

### Coordinate System
- Pitch dimensions: 105m × 68m
- Origin at center (0, 0)
- X-axis: -52.5 to +52.5 (left to right)
- Y-axis: -34 to +34 (bottom to top)

## Limitations

- **Data Size**: Large tracking data files not included in repository
- **Computation Time**: Full EPV heatmap generation can take several minutes
- **Model Scope**: Models trained specifically on MLS 2023 data
- **Recursion Depth**: Limited to avoid excessive computation time

## Future Improvements

- [ ] Faster EPV computation through vectorization
- [ ] Pre-computed EPV surfaces for common game states
- [ ] Integration with real-time match tracking
- [ ] Extended model training on multi-league data
- [ ] Web-based visualization interface

## References

- Fernández, J., & Bornn, L. (2018). Wide Open Spaces: A statistical technique for measuring space creation in professional soccer. *MIT Sloan Sports Analytics Conference*.
- Spearman, W. (2018). Beyond Expected Goals. *MIT Sloan Sports Analytics Conference*.

## License

This project is for academic purposes.

## Authors

Jonathan Locala

## Acknowledgments

- SkillCorner for tracking data access
- MLS for event data
- Course instructors and teaching staff
