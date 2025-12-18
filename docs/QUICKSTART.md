# Quick Start Guide

Get the EPV-SARG system running in 5 minutes.

## Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd EPV_SARG

# 2. Install dependencies
pip install -r requirements.txt
```

## Generate Your First Visualization

### Option 1: EPV Heatmap (No Data Required)
Generate a static EPV heatmap showing goal probability across the field:

```bash
cd scripts
python3 generate_epv_heatmap.py
```

**Output**: `../results/epv_heatmap_simple.png`

This shows expected goals (EPV) values for different positions, with red/orange indicating high-value areas (near goal) and blue indicating low-value areas.

### Option 2: View Existing Results
Explore pre-generated visualizations in the `results/` directory:

```bash
ls results/
```

Notable visualizations:
- `epv_heatmap_simple.png` - EPV across the pitch
- `scenario_compare.png` - Side-by-side attack scenarios
- `attack_timeseries.png` - EPV evolution during an attack
- `shot_with_tracking.png` - Real shot with player positions

## Understanding EPV

**Expected Possession Value (EPV)** answers: *"What's the probability of scoring from this position?"*

The system evaluates three possible actions from any position:
1. **Shoot**: Direct xG (expected goals)
2. **Pass**: Success probability × EPV at destination
3. **Dribble**: Success probability × EPV after dribble

EPV = max(shoot, pass, dribble)

## Project Components

### Core Modules (`src/`)
- `epv_calculator.py` - Main EPV computation engine
- `pitch_control.py` - Spatial control modeling
- `xg_model.py` - Shot quality prediction
- `passing_model.py` - Pass completion prediction
- `dribbling_model.py` - Dribble success prediction

### Trained Models (`models/`)
- `xg_model_improved.pkl` - Trained xG model
- `passing_model_improved.pkl` - Trained passing model
- `dribbling_model_proper_split.pkl` - Trained dribble model

### Player Skills (`data/`)
- `player_id_to_finishing_skill.csv` - Shooting ability
- `player_id_to_passing_skill.csv` - Passing ability
- `player_id_to_skill.csv` - Dribbling ability

## Using Your Own Data

**Note**: Tracking data files are NOT included due to size (see `DATA_README.md`)

If you have tracking data:

1. Place files in expected structure (see `DATA_README.md`)
2. Update path in `scripts/generate_epv_heatmap_from_data.py`:
   ```python
   DATA_DIR = Path("/your/data/path/skillcorner_download")
   MATCH_ID = "your_match_id"
   ```
3. Run:
   ```bash
   cd scripts
   python3 generate_epv_heatmap_from_data.py
   ```

## Common Use Cases

### Calculate EPV for Specific Position
```python
from epv_calculator import EPVCalculator
from pathlib import Path

# Initialize
calc = EPVCalculator(
    xg_model_path=Path("xg_model_improved.pkl"),
    passing_model_path=Path("passing_model_improved.pkl"),
    dribbling_model_path=Path("dribbling_model_proper_split.pkl"),
    xg_skills_path=Path("player_id_to_finishing_skill.csv"),
    passing_skills_path=Path("player_id_to_passing_skill.csv"),
    dribbling_skills_path=Path("player_id_to_skill.csv")
)

# Get EPV (requires match context - see full README)
epv = calc.get_epv(x=30, y=0, frame=1000, player_id=123, team_id=1, tracking_dict={})
print(f"EPV: {epv:.3f}")
```

### Retrain Models
```bash
cd training

# Requires training data (not included)
python3 train_xg_model_improved.py
python3 train_passing_model_improved.py
python3 train_dribbling_model.py
```

## Troubleshooting

**Problem**: Import errors
**Solution**: Install dependencies: `pip install -r requirements.txt`

**Problem**: "File not found" errors
**Solution**: Check you're in the EPV_SARG directory: `pwd`

**Problem**: Want to use tracking data
**Solution**: See `DATA_README.md` for data setup instructions

## Next Steps

- Read the full `README.md` for detailed documentation
- Check `DATA_README.md` if you want to use tracking data
- Explore the `results/` directory for example outputs
- Review the code in `epv_calculator.py` to understand the algorithm

## Getting Help

- Check documentation: `README.md`, `DATA_README.md`
- Review code comments in source files
- Contact repository maintainer with questions

---

**Ready to dive deeper?** Check out the main `README.md` for comprehensive documentation!
