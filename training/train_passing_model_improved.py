"""
Improved Passing Model with Pitch Control and Defender Proximity

Adds critical features:
1. Pitch control at pass origin
2. Pitch control at pass destination
3. Defenders within 3m of pass origin
4. Defenders within 3m of pass destination
5. Defenders in passing lane
6. Player passing skill (individuality)
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss,
    brier_score_loss
)
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import glob
import sys

# Add repo src to path for pitch_control import
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from pitch_control import (
        PitchControlRunner,
        PitchControlCache,
        load_match_meta_robust,
        load_tracking_jsonl
    )
    PITCH_CONTROL_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: Could not import pitch_control module")
    print("   Pitch control features will be set to 0.5 (neutral)")
    PITCH_CONTROL_AVAILABLE = False
    PitchControlRunner = None
    PitchControlCache = None

from passing_model import (
    DefenderPriorityPassModel,
    PASS_DEFENDER_LANE_WEIGHT,
    PASS_DEFENDER_DEST_WEIGHT,
    PASS_MIN_DIST_CAP,
    PASS_MIN_DIST_FLOOR,
)


def load_tracking_data(tracking_path: Path) -> Dict:
    """Load tracking data from JSONL file."""
    tracking_dict = {}
    with open(tracking_path, 'r') as f:
        for line in f:
            if line.strip():
                frame_data = json.loads(line)
                frame_num = frame_data['frame']
                tracking_dict[frame_num] = frame_data
    return tracking_dict


def build_team_roster(events_df: pd.DataFrame) -> dict:
    """Build team roster mapping team_id -> set(player_ids) for this match."""
    team_roster = {}
    player_teams = events_df[['player_id', 'team_id']].dropna().drop_duplicates()

    for _, row in player_teams.iterrows():
        team_id = int(row['team_id'])
        player_id = int(row['player_id'])

        if team_id not in team_roster:
            team_roster[team_id] = set()

        team_roster[team_id].add(player_id)

    return team_roster


def calculate_defenders_near_location(
    x: float,
    y: float,
    frame_data: dict,
    passer_team_id: int,
    team_roster: dict,
    distance: float = 3.0
) -> int:
    """Count defenders within specified distance of a location."""
    if not frame_data or 'player_data' not in frame_data:
        return 0

    # Get opponent team players
    opponent_team_ids = [tid for tid in team_roster.keys() if tid != passer_team_id]
    if not opponent_team_ids:
        return 0

    opponent_players = set()
    for opp_team_id in opponent_team_ids:
        opponent_players.update(team_roster[opp_team_id])

    defender_count = 0

    for player in frame_data.get('player_data', []):
        if not player.get('is_detected', False):
            continue

        player_id = player['player_id']

        # Only count opponent players
        if player_id not in opponent_players:
            continue

        px = player['x']
        py = player['y']

        # Calculate distance
        dist = np.sqrt((x - px)**2 + (y - py)**2)

        if dist <= distance and dist > 0.1:
            defender_count += 1

    return defender_count


def calculate_defenders_in_passing_lane(
    x_origin: float,
    y_origin: float,
    x_dest: float,
    y_dest: float,
    frame_data: dict,
    passer_team_id: int,
    team_roster: dict,
    lane_width: float = 2.0
) -> int:
    """
    Count defenders in the passing lane (corridor between origin and destination).
    Uses perpendicular distance from pass line.
    """
    if not frame_data or 'player_data' not in frame_data:
        return 0

    # Get opponent team players
    opponent_team_ids = [tid for tid in team_roster.keys() if tid != passer_team_id]
    if not opponent_team_ids:
        return 0

    opponent_players = set()
    for opp_team_id in opponent_team_ids:
        opponent_players.update(team_roster[opp_team_id])

    # Pass vector
    pass_dx = x_dest - x_origin
    pass_dy = y_dest - y_origin
    pass_length = np.sqrt(pass_dx**2 + pass_dy**2)

    if pass_length < 0.1:  # Too short to define a lane
        return 0

    defender_count = 0

    for player in frame_data.get('player_data', []):
        if not player.get('is_detected', False):
            continue

        player_id = player['player_id']

        if player_id not in opponent_players:
            continue

        px = player['x']
        py = player['y']

        # Calculate perpendicular distance from pass line
        # Vector from origin to player
        to_player_x = px - x_origin
        to_player_y = py - y_origin

        # Project onto pass direction
        projection = (to_player_x * pass_dx + to_player_y * pass_dy) / pass_length

        # Check if player is between origin and destination (along pass direction)
        if projection < 0 or projection > pass_length:
            continue

        # Calculate perpendicular distance
        perp_dist = abs(to_player_x * pass_dy - to_player_y * pass_dx) / pass_length

        if perp_dist <= lane_width:
            defender_count += 1

    return defender_count


def calculate_min_defender_dist_to_passing_lane(
    x_origin: float,
    y_origin: float,
    x_dest: float,
    y_dest: float,
    frame_data: dict,
    passer_team_id: int,
    team_roster: dict,
    default_distance: float = 50.0
) -> float:
    """Compute minimum perpendicular distance of any defender to the pass lane segment."""
    if not frame_data or 'player_data' not in frame_data:
        return default_distance

    opponent_team_ids = [tid for tid in team_roster.keys() if tid != passer_team_id]
    if not opponent_team_ids:
        return default_distance

    opponent_players = set()
    for opp_team_id in opponent_team_ids:
        opponent_players.update(team_roster[opp_team_id])

    pass_dx = x_dest - x_origin
    pass_dy = y_dest - y_origin
    pass_length = np.sqrt(pass_dx**2 + pass_dy**2)
    if pass_length < 0.1:
        return default_distance

    min_dist = None

    for player in frame_data.get('player_data', []):
        if not player.get('is_detected', False):
            continue

        player_id = player['player_id']
        if player_id not in opponent_players:
            continue

        px = player['x']
        py = player['y']

        to_player_x = px - x_origin
        to_player_y = py - y_origin

        projection = (to_player_x * pass_dx + to_player_y * pass_dy) / pass_length
        if projection < 0 or projection > pass_length:
            continue

        perp_dist = abs(to_player_x * pass_dy - to_player_y * pass_dx) / pass_length
        if min_dist is None or perp_dist < min_dist:
            min_dist = perp_dist

    return float(min_dist) if min_dist is not None else default_distance


def calculate_pitch_control(
    x: float,
    y: float,
    frame_idx: int,
    pc_runner: Optional['PitchControlRunner'] = None
) -> float:
    """
    Calculate pitch control at a location.
    Returns value between 0 (defending team control) and 1 (attacking team control).
    """
    if not PITCH_CONTROL_AVAILABLE or pc_runner is None:
        return 0.5  # Neutral if pitch control not available

    try:
        # Use PitchControlRunner to get actual pitch control value
        # pc_runner.pc_at_point() returns PC for attacking team (team with ball)
        pc_value = pc_runner.pc_at_point(frame_idx, x, y)
        return float(pc_value)
    except Exception as e:
        # If calculation fails, return neutral value
        return 0.5


def calculate_pitch_control_path_min(
    x_origin: float,
    y_origin: float,
    x_dest: float,
    y_dest: float,
    frame_idx: int,
    pc_runner: Optional['PitchControlRunner'] = None,
    n_samples: int = 5
) -> float:
    """Compute minimum pitch control along the pass line segment."""
    if not PITCH_CONTROL_AVAILABLE or pc_runner is None:
        return 0.5

    if frame_idx is None or (isinstance(frame_idx, float) and np.isnan(frame_idx)):
        return 0.5

    pass_dx = x_dest - x_origin
    pass_dy = y_dest - y_origin
    pass_length = np.sqrt(pass_dx**2 + pass_dy**2)
    if pass_length < 0.5:
        pc_origin = calculate_pitch_control(x_origin, y_origin, frame_idx, pc_runner)
        pc_dest = calculate_pitch_control(x_dest, y_dest, frame_idx, pc_runner)
        return float(min(pc_origin, pc_dest))

    fractions = np.linspace(0.2, 0.8, n_samples)
    values = []
    for frac in fractions:
        x = x_origin + frac * pass_dx
        y = y_origin + frac * pass_dy
        values.append(calculate_pitch_control(x, y, frame_idx, pc_runner))

    return float(np.min(values)) if values else 0.5


def load_player_passing_skills() -> Dict[int, float]:
    """Load player passing skill lookup."""
    skill_file = Path("data/player_id_to_passing_skill.csv")

    if not skill_file.exists():
        print(f"⚠️  Player passing skill file not found: {skill_file}")
        return {}

    skill_df = pd.read_csv(skill_file)

    # Check column names
    if 'player_id' in skill_df.columns and 'player_passing_skill' in skill_df.columns:
        player_skills = dict(zip(skill_df['player_id'], skill_df['player_passing_skill']))
    elif 'player_id' in skill_df.columns and 'player_RE' in skill_df.columns:
        player_skills = dict(zip(skill_df['player_id'], skill_df['player_RE']))
    else:
        print(f"⚠️  Could not find expected columns in {skill_file}")
        return {}

    print(f"✅ Loaded passing skills for {len(player_skills)} players")
    return player_skills


def extract_pass_features(data_dir: Path, max_matches: int = None) -> pd.DataFrame:
    """
    Extract pass features from dynamic events and tracking data.
    """
    # Load player passing skills
    player_skills = load_player_passing_skills()

    # Find all dynamic event files
    event_files = sorted(
        f for f in data_dir.glob("*_dynamic_events.csv")
        if not f.name.startswith("._")
    )

    if max_matches:
        event_files = event_files[:max_matches]

    print(f"Processing {len(event_files)} matches...")

    all_pass_features = []

    for event_file in event_files:
        match_id = event_file.stem.replace("_dynamic_events", "")

        # Load events
        try:
            events_df = pd.read_csv(event_file, low_memory=False)
        except Exception as e:
            print(f"  ⚠️  Error loading {match_id}: {e}")
            continue

        # Build team roster for this match
        team_roster = build_team_roster(events_df)

        if len(team_roster) < 2:
            print(f"  ⚠️  {match_id}: Could not identify 2 teams (found {len(team_roster)})")
            continue

        # Filter for passes
        # Identify passes (from notebook logic)
        is_pp = events_df.get("event_type", pd.Series("", index=events_df.index)).eq("player_possession")
        is_pass = events_df.get("end_type", pd.Series("", index=events_df.index)).str.contains("pass", na=False)
        has_pass_cols = events_df.columns.isin(["pass_distance","pass_angle"]).any()
        fallback_pass = (is_pp & events_df.get("pass_distance", pd.Series(np.nan, index=events_df.index)).notna()) if has_pass_cols else pd.Series(False, index=events_df.index)

        passes = events_df[is_pass | fallback_pass].copy()

        # Need label
        passes = passes[passes[['x_start', 'y_start', 'x_end', 'y_end']].notna().all(axis=1)].copy()

        if len(passes) == 0:
            continue

        # Determine pass success (from notebook logic)
        success_terms = {"pass successful","successful","complete","completed","accurate","accurate_pass"}
        fail_terms = {"pass unsuccessful","unsuccessful","inaccurate","intercepted","blocked","offside","failed"}

        y = pd.Series(np.nan, index=passes.index, dtype="float")

        if "pass_outcome" in events_df.columns:
            po = events_df.loc[passes.index, "pass_outcome"].astype(str).str.lower().str.strip()
            y = np.select([po.isin(success_terms), po.isin(fail_terms)], [1, 0], default=np.nan).astype(float)

        if "received" in events_df.columns:
            rec = events_df.loc[passes.index, "received"].astype(str).str.lower().map({"true":1,"1":1,"false":0,"0":0})
            y = np.where(np.isnan(y), rec, y)

        passes["pass_completed"] = pd.to_numeric(y, errors="coerce")
        passes = passes[passes["pass_completed"].notna()].copy()

        if len(passes) == 0:
            continue

        # Load tracking data
        tracking_file = data_dir / f"{match_id}_tracking_extrapolated.jsonl"
        if not tracking_file.exists():
            print(f"  ⚠️  No tracking for {match_id}")
            continue

        try:
            tracking_dict = load_tracking_data(tracking_file)
        except Exception as e:
            print(f"  ⚠️  Error loading tracking for {match_id}: {e}")
            continue

        # Initialize PitchControlRunner for this match
        pc_runner = None
        if PITCH_CONTROL_AVAILABLE:
            try:
                # Load match metadata for pitch control
                match_json = data_dir / f"match_{match_id}.json"
                if match_json.exists():
                    meta = load_match_meta_robust(match_json)
                    frames = load_tracking_jsonl(tracking_file)
                    pc_runner = PitchControlRunner(meta, frames)
                    print(f"  ✓ Pitch control initialized for {match_id}")
                else:
                    print(f"  ⚠️  No match.json for {match_id}, using neutral PC")
            except Exception as e:
                print(f"  ⚠️  Could not initialize pitch control for {match_id}: {e}")

        print(f"  Processing {match_id}: {len(passes)} passes")

        # Extract features for each pass
        for idx, pass_event in passes.iterrows():
            x_origin = pass_event['x_start']
            y_origin = pass_event['y_start']
            x_dest = pass_event['x_end']
            y_dest = pass_event['y_end']

            frame_start = pass_event.get('frame_start')
            team_id = pass_event.get('team_id')
            player_id = pass_event.get('player_id')

            # Get tracking data for this frame
            frame_data = tracking_dict.get(frame_start, {})

            # Calculate defender proximity features
            defenders_near_origin = calculate_defenders_near_location(
                x_origin, y_origin, frame_data, team_id, team_roster, distance=3.0
            )

            defenders_near_dest = calculate_defenders_near_location(
                x_dest, y_dest, frame_data, team_id, team_roster, distance=3.0
            )

            defenders_in_lane = calculate_defenders_in_passing_lane(
                x_origin, y_origin, x_dest, y_dest,
                frame_data, team_id, team_roster, lane_width=2.0
            )

            min_defender_dist_to_lane = calculate_min_defender_dist_to_passing_lane(
                x_origin, y_origin, x_dest, y_dest,
                frame_data, team_id, team_roster
            )

            # Calculate pitch control using actual runner
            pc_origin = calculate_pitch_control(
                x_origin, y_origin, frame_start, pc_runner
            )

            pc_dest = calculate_pitch_control(
                x_dest, y_dest, frame_start, pc_runner
            )

            pc_path_min = calculate_pitch_control_path_min(
                x_origin, y_origin, x_dest, y_dest, frame_start, pc_runner
            )

            # Get player passing skill
            player_passing_skill = player_skills.get(player_id, 0.0)

            # Pass geometry
            pass_distance = np.sqrt((x_dest - x_origin)**2 + (y_dest - y_origin)**2)
            pass_angle = np.degrees(np.arctan2(y_dest - y_origin, x_dest - x_origin))
            forward_progress = x_dest - x_origin

            # Original features from notebook
            speed_avg = pass_event.get('speed_avg', 0.0)
            inside_defensive_shape = pass_event.get('inside_defensive_shape_start', False)
            last_defensive_line_x = pass_event.get('last_defensive_line_x_start', 0.0)
            last_defensive_line_height = pass_event.get('last_defensive_line_height_start', 0.0)

            all_pass_features.append({
                'match_id': match_id,
                'event_id': pass_event.get('event_id'),
                'x_origin': x_origin,
                'y_origin': y_origin,
                'x_dest': x_dest,
                'y_dest': y_dest,
                'pass_distance': pass_distance,
                'pass_angle': pass_angle,
                'forward_progress': forward_progress,
                'defenders_near_origin': defenders_near_origin,
                'defenders_near_dest': defenders_near_dest,
                'defenders_in_lane': defenders_in_lane,
                'min_defender_dist_to_lane': min_defender_dist_to_lane,
                'pitch_control_origin': pc_origin,
                'pitch_control_dest': pc_dest,
                'pitch_control_path_min': pc_path_min,
                'player_passing_skill': player_passing_skill,
                'speed_avg': speed_avg if pd.notna(speed_avg) else 0.0,
                'inside_defensive_shape': int(inside_defensive_shape) if pd.notna(inside_defensive_shape) else 0,
                'last_defensive_line_x': last_defensive_line_x if pd.notna(last_defensive_line_x) else 0.0,
                'last_defensive_line_height': last_defensive_line_height if pd.notna(last_defensive_line_height) else 0.0,
                'pass_completed': int(pass_event['pass_completed'])
            })

    return pd.DataFrame(all_pass_features)


def calculate_calibration_error(y_true, y_pred_proba, n_bins=10):
    """Calculate Expected Calibration Error (ECE)."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0

    for i in range(n_bins):
        bin_mask = bin_indices == i
        if np.sum(bin_mask) > 0:
            bin_acc = np.mean(y_true[bin_mask])
            bin_conf = np.mean(y_pred_proba[bin_mask])
            bin_count = np.sum(bin_mask)

            ece += (bin_count / len(y_true)) * abs(bin_acc - bin_conf)

    return ece


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("IMPROVED PASSING MODEL WITH TRACKING FEATURES")
    print("=" * 60)

    # Set data directory
    data_dir = Path("more_data") if Path("more_data").exists() else Path("skillcorner_download")

    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return

    # Extract features
    print("\n[1/5] Extracting pass features with tracking data...")
    passes_df = extract_pass_features(data_dir, max_matches=150)

    print(f"\n✅ Extracted features for {len(passes_df)} passes")
    print(f"   Completed: {passes_df['pass_completed'].sum()} ({passes_df['pass_completed'].mean():.1%})")

    # Show new feature distributions
    print(f"\n   Defender proximity features:")
    print(f"     defenders_near_origin: mean={passes_df['defenders_near_origin'].mean():.2f}, max={passes_df['defenders_near_origin'].max()}")
    print(f"     defenders_near_dest: mean={passes_df['defenders_near_dest'].mean():.2f}, max={passes_df['defenders_near_dest'].max()}")
    print(f"     defenders_in_lane: mean={passes_df['defenders_in_lane'].mean():.2f}, max={passes_df['defenders_in_lane'].max()}")
    print(f"     min_defender_dist_to_lane: mean={passes_df['min_defender_dist_to_lane'].mean():.2f}, max={passes_df['min_defender_dist_to_lane'].max():.2f}")
    print(f"     player_passing_skill: mean={passes_df['player_passing_skill'].mean():.2f}, max={passes_df['player_passing_skill'].max():.2f}")

    # Prepare features
    feature_cols = [
        'pass_distance', 'pass_angle', 'forward_progress',
        'defenders_near_origin', 'defenders_near_dest', 'defenders_in_lane',
        'min_defender_dist_to_lane',
        'pitch_control_origin', 'pitch_control_dest', 'pitch_control_path_min',
        'player_passing_skill',
        'speed_avg', 'inside_defensive_shape'
    ]

    X = passes_df[feature_cols].fillna(0).values
    y = passes_df['pass_completed'].values
    match_ids = passes_df['match_id'].values

    # Split data BY MATCH
    print("\n[2/5] Splitting data (grouped by match)...")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(X, y, groups=match_ids))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Check for leakage
    train_matches = set(match_ids[train_idx])
    test_matches = set(match_ids[test_idx])
    overlap = train_matches & test_matches

    print(f"   Train: {len(X_train)} passes from {len(train_matches)} matches ({y_train.mean():.1%} completed)")
    print(f"   Test:  {len(X_test)} passes from {len(test_matches)} matches ({y_test.mean():.1%} completed)")
    print(f"   Match overlap: {len(overlap)} ✅")

    # Train model
    print("\n[3/5] Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=15,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    priority_model = DefenderPriorityPassModel(
        model,
        feature_cols,
        lane_weight=PASS_DEFENDER_LANE_WEIGHT,
        dest_weight=PASS_DEFENDER_DEST_WEIGHT,
        min_dist_cap=PASS_MIN_DIST_CAP,
        min_dist_floor=PASS_MIN_DIST_FLOOR
    )

    # Feature importances
    print("\n   Feature importances:")
    importances = sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )
    for feat, imp in importances:
        print(f"     {feat:30s}: {imp:.4f}")

    # Evaluate
    print("\n[4/5] Evaluating model...")

    y_train_proba = priority_model.predict_proba(X_train)[:, 1]
    y_train_pred = (y_train_proba >= 0.5).astype(int)

    print("\n   TRAINING SET:")
    print(f"     Accuracy:     {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"     ROC AUC:      {roc_auc_score(y_train, y_train_proba):.4f}")

    train_ece = calculate_calibration_error(y_train, y_train_proba)
    print(f"     ECE:          {train_ece:.4f}")

    y_test_proba = priority_model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= 0.5).astype(int)

    print("\n   TEST SET:")
    print(f"     Accuracy:     {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"     ROC AUC:      {roc_auc_score(y_test, y_test_proba):.4f}")
    print(f"     Log Loss:     {log_loss(y_test, y_test_proba):.4f}")

    test_ece = calculate_calibration_error(y_test, y_test_proba)
    print(f"     ECE:          {test_ece:.4f}")

    if test_ece < 0.01:
        print(f"     ✅ ECE < 0.01 (target met!)")
    else:
        print(f"     ⚠️  ECE > 0.01 (target: < 0.01)")

    # Overfitting check
    train_test_gap = roc_auc_score(y_train, y_train_proba) - roc_auc_score(y_test, y_test_proba)
    print(f"\n   Overfitting check:")
    print(f"     Train-Test AUC Gap: {train_test_gap:.3f} {'✅' if train_test_gap < 0.10 else '⚠️'}")

    # Save model
    print("\n[5/5] Saving model...")
    model_file = Path("models/passing_model_improved.pkl")
    model_file.parent.mkdir(exist_ok=True)

    with open(model_file, 'wb') as f:
        pickle.dump({
            'model': priority_model,
            'feature_cols': feature_cols,
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'test_auc': roc_auc_score(y_test, y_test_proba),
            'test_ece': test_ece,
            'defender_priority': {
                'lane_weight': PASS_DEFENDER_LANE_WEIGHT,
                'dest_weight': PASS_DEFENDER_DEST_WEIGHT,
                'min_dist_cap': PASS_MIN_DIST_CAP,
                'min_dist_floor': PASS_MIN_DIST_FLOOR
            }
        }, f)

    print(f"   ✅ Model saved to {model_file}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nKey improvements:")
    print(f"  ✅ Added defenders_near_origin feature")
    print(f"  ✅ Added defenders_near_dest feature")
    print(f"  ✅ Added defenders_in_lane feature")
    print(f"  ✅ Added min_defender_dist_to_lane feature")
    print(f"  ✅ Added pitch_control features (placeholder)")
    print(f"  ✅ Added pitch_control_path_min feature")
    print(f"  ✅ Added defender-priority pass penalty")
    print(f"  ✅ Added player_passing_skill (individuality)")
    print(f"  ✅ Match-grouped split (no leakage)")
    print(f"\nTest performance:")
    print(f"  Accuracy: {accuracy_score(y_test, y_test_pred):.3f}")
    print(f"  AUC:      {roc_auc_score(y_test, y_test_proba):.3f}")
    print(f"  ECE:      {test_ece:.4f} {'✅' if test_ece < 0.01 else '⚠️'}")


if __name__ == "__main__":
    main()
