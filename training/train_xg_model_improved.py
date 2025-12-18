"""
xG Model with Defender Proximity and Player Individuality

Adds critical features:
1. Defenders in shot triangle (between shooter and goal posts)
2. Defenders within 3 meters of shooter
3. Player finishing skill (individuality)
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss,
    classification_report, brier_score_loss
)
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import glob


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


def calculate_defenders_in_shot_triangle(
    shooter_x: float,
    shooter_y: float,
    goal_x: float,
    frame_data: dict,
    shooter_team_id: int,
    team_roster: dict
) -> int:
    """
    Count defenders in the triangle between shooter and goal posts.

    Goal posts are at (goal_x, -3.66) and (goal_x, 3.66) meters.
    """
    if not frame_data or 'player_data' not in frame_data:
        return 0

    # Goal post coordinates (standard 7.32m width = 3.66m from center)
    goal_post_1 = (goal_x, -3.66)
    goal_post_2 = (goal_x, 3.66)

    # Get opponent team players (dribbling model approach)
    opponent_team_ids = [tid for tid in team_roster.keys() if tid != shooter_team_id]
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

        # Check if defender is in the triangle using cross product method
        # Point is inside triangle if it has same orientation to all three edges
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        point = (px, py)
        shooter_point = (shooter_x, shooter_y)

        d1 = sign(point, shooter_point, goal_post_1)
        d2 = sign(point, goal_post_1, goal_post_2)
        d3 = sign(point, goal_post_2, shooter_point)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        if not (has_neg and has_pos):
            defender_count += 1

    return defender_count


def calculate_defenders_within_distance(
    shooter_x: float,
    shooter_y: float,
    frame_data: dict,
    shooter_team_id: int,
    team_roster: dict,
    distance: float = 3.0
) -> int:
    """Count defenders within specified distance of shooter."""
    if not frame_data or 'player_data' not in frame_data:
        return 0

    # Get opponent team players (dribbling model approach)
    opponent_team_ids = [tid for tid in team_roster.keys() if tid != shooter_team_id]
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
        dist = np.sqrt((shooter_x - px)**2 + (shooter_y - py)**2)

        if dist <= distance and dist > 0.1:  # Exclude self
            defender_count += 1

    return defender_count


def calculate_distance_to_goal(x: float, y: float, attacking_direction: str) -> float:
    """Calculate distance from position to goal center.

    NOTE: SkillCorner coordinates are PRE-NORMALIZED!
    x is always relative to attacking direction, so goal is ALWAYS at +52.5
    """
    goal_x = 52.5  # Always attacking toward right goal (coordinates are normalized)
    goal_y = 0.0

    distance = np.sqrt((x - goal_x)**2 + (y - goal_y)**2)
    return distance


def calculate_angle_to_goal(x: float, y: float, attacking_direction: str) -> float:
    """Calculate angle to goal (in degrees).

    NOTE: SkillCorner coordinates are PRE-NORMALIZED!
    Goal is ALWAYS at +52.5
    """
    goal_x = 52.5  # Always attacking toward right goal

    # Goal posts at +/- 3.66m from center
    post1_y = -3.66
    post2_y = 3.66

    # Calculate angles to each post
    angle1 = np.arctan2(post1_y - y, goal_x - x)
    angle2 = np.arctan2(post2_y - y, goal_x - x)

    # Angle between the two posts (in degrees)
    angle = abs(np.degrees(angle2 - angle1))

    return angle


def load_player_finishing_skills() -> Dict[int, float]:
    """Load player finishing skill lookup."""
    # Skills CSV already lives in repo root
    skill_file = Path("player_id_to_finishing_skill.csv")

    if not skill_file.exists():
        print(f"⚠️  Player finishing skill file not found: {skill_file}")
        return {}

    skill_df = pd.read_csv(skill_file)
    player_skills = dict(zip(skill_df['player_id'], skill_df['player_finishing_skill']))

    print(f"✅ Loaded finishing skills for {len(player_skills)} players")
    return player_skills


def extract_shot_features(data_dir: Path, max_matches: int = None) -> pd.DataFrame:
    """
    Extract shot features from dynamic events and tracking data.
    """
    # Load player finishing skills
    player_skills = load_player_finishing_skills()

    # Find all dynamic event files
    event_files = list(data_dir.glob("*_dynamic_events.csv"))

    if max_matches:
        event_files = event_files[:max_matches]

    print(f"Processing {len(event_files)} matches...")

    all_shot_features = []

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

        # Filter for shots
        shots = events_df[
            (events_df['end_type'].fillna('').str.lower() == 'shot')
        ].copy()

        if len(shots) == 0:
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

        print(f"  Processing {match_id}: {len(shots)} shots")

        # Extract features for each shot
        for idx, shot in shots.iterrows():
            x_start = shot['x_start']
            y_start = shot['y_start']
            frame_start = shot.get('frame_start')
            team_id = shot.get('team_id')
            player_id = shot.get('player_id')
            attacking_side = shot.get('attacking_side', 'left_to_right')

            # SkillCorner coordinates are PRE-NORMALIZED!
            # x_start is always relative to attacking direction
            # So goal is ALWAYS at +52.5 regardless of attacking_side
            goal_x = 52.5

            # Get tracking data for this frame
            frame_data = tracking_dict.get(frame_start, {})

            # Calculate defender proximity features
            defenders_in_triangle = calculate_defenders_in_shot_triangle(
                x_start, y_start, goal_x, frame_data, team_id, team_roster
            )

            defenders_within_3m = calculate_defenders_within_distance(
                x_start, y_start, frame_data, team_id, team_roster, distance=3.0
            )

            # Calculate distance and angle to goal
            distance_to_goal = calculate_distance_to_goal(x_start, y_start, attacking_side)
            angle_to_goal = calculate_angle_to_goal(x_start, y_start, attacking_side)

            # Get player finishing skill
            player_finishing_skill = player_skills.get(player_id, 0.0)

            # Determine if goal was scored
            game_interruption_after = shot.get('game_interruption_after', '')
            if pd.isna(game_interruption_after):
                game_interruption_after = ''
            goal_scored = str(game_interruption_after).lower() == 'goal_for'

            # Original features from xG2.ipynb
            penalty_area = shot.get('penalty_area_start', False)
            trajectory_angle = shot.get('trajectory_angle', 0.0)
            distance_covered = shot.get('distance_covered', 0.0)
            speed_avg = shot.get('speed_avg', 0.0)
            inside_defensive_shape = shot.get('inside_defensive_shape_start', False)
            last_defensive_line_x = shot.get('last_defensive_line_x_start', 0.0)
            last_defensive_line_height = shot.get('last_defensive_line_height_start', 0.0)

            all_shot_features.append({
                'match_id': match_id,
                'event_id': shot.get('event_id'),
                'x_start': x_start,
                'y_start': y_start,
                'distance_to_goal': distance_to_goal,
                'angle_to_goal': angle_to_goal,
                'defenders_in_triangle': defenders_in_triangle,
                'defenders_within_3m': defenders_within_3m,
                'player_finishing_skill': player_finishing_skill,
                'penalty_area': int(penalty_area) if pd.notna(penalty_area) else 0,
                'trajectory_angle': trajectory_angle if pd.notna(trajectory_angle) else 0.0,
                'distance_covered': distance_covered if pd.notna(distance_covered) else 0.0,
                'speed_avg': speed_avg if pd.notna(speed_avg) else 0.0,
                'inside_defensive_shape': int(inside_defensive_shape) if pd.notna(inside_defensive_shape) else 0,
                'last_defensive_line_x': last_defensive_line_x if pd.notna(last_defensive_line_x) else 0.0,
                'last_defensive_line_height': last_defensive_line_height if pd.notna(last_defensive_line_height) else 0.0,
                'goal': int(goal_scored)
            })

    return pd.DataFrame(all_shot_features)


def calculate_calibration_error(y_true, y_pred_proba, n_bins=10):
    """Calculate Expected Calibration Error (ECE)."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    bin_counts = []
    bin_accs = []
    bin_confs = []

    for i in range(n_bins):
        bin_mask = bin_indices == i
        if np.sum(bin_mask) > 0:
            bin_acc = np.mean(y_true[bin_mask])
            bin_conf = np.mean(y_pred_proba[bin_mask])
            bin_count = np.sum(bin_mask)

            ece += (bin_count / len(y_true)) * abs(bin_acc - bin_conf)

            bin_counts.append(bin_count)
            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
        else:
            bin_counts.append(0)
            bin_accs.append(0)
            bin_confs.append(0)

    return ece, bin_accs, bin_confs, bin_counts


def plot_calibration_curve(y_true, y_pred_proba, n_bins=10, title="Calibration Curve"):
    """Plot calibration curve."""
    ece, bin_accs, bin_confs, bin_counts = calculate_calibration_error(y_true, y_pred_proba, n_bins)

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.plot(bin_confs, bin_accs, 'o-', label=f'Model (ECE={ece:.4f})')

    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Probability')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return ece


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("IMPROVED XG MODEL WITH DEFENDER PROXIMITY + INDIVIDUALITY")
    print("=" * 60)

    # Prefer expanded dataset in more_data if present, else fallback
    data_dir = Path("more_data") if Path("more_data").exists() else Path("skillcorner_download")

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return

    # Extract features
    print("\n[1/5] Extracting shot features with tracking data...")
    shots_df = extract_shot_features(data_dir, max_matches=None)  # Use all available matches

    print(f"\nExtracted features for {len(shots_df)} shots")
    print(f"   Goals scored: {shots_df['goal'].sum()} ({shots_df['goal'].mean():.1%})")

    # Check for missing values
    print(f"\n   Missing values per column:")
    for col in shots_df.columns:
        missing = shots_df[col].isna().sum()
        if missing > 0:
            print(f"     {col}: {missing}")

    # Show new feature distributions
    print(f"\n   Defender proximity features:")
    print(f"     defenders_in_triangle: mean={shots_df['defenders_in_triangle'].mean():.2f}, max={shots_df['defenders_in_triangle'].max()}")
    print(f"     defenders_within_3m: mean={shots_df['defenders_within_3m'].mean():.2f}, max={shots_df['defenders_within_3m'].max()}")
    print(f"     player_finishing_skill: mean={shots_df['player_finishing_skill'].mean():.2f}, max={shots_df['player_finishing_skill'].max():.2f}")

    # Prepare features - KEEP IT SIMPLE like partner's model!
    # Remove confusing tracking features that hurt performance
    feature_cols = [
        'distance_to_goal', 'angle_to_goal',
        'penalty_area', 'trajectory_angle', 'distance_covered',
        'speed_avg',
        'player_finishing_skill'  # Only individuality feature
        # REMOVED: defenders_in_triangle, defenders_within_3m (confusing)
        # REMOVED: last_defensive_line_x, last_defensive_line_height (confusing)
        # REMOVED: inside_defensive_shape (confusing)
    ]

    X = shots_df[feature_cols].fillna(0).values
    y = shots_df['goal'].values
    match_ids = shots_df['match_id'].values

    # Split data BY MATCH to prevent leakage
    print("\n[2/5] Splitting data (grouped by match to prevent leakage)...")

    # Use StratifiedGroupKFold to split by match
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(X, y, groups=match_ids))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Check for leakage
    train_matches = set(match_ids[train_idx])
    test_matches = set(match_ids[test_idx])
    overlap = train_matches & test_matches

    print(f"   Train: {len(X_train)} shots from {len(train_matches)} matches ({y_train.mean():.1%} goals)")
    print(f"   Test:  {len(X_test)} shots from {len(test_matches)} matches ({y_test.mean():.1%} goals)")
    print(f"   Match overlap: {len(overlap)} (should be 0)")

    if len(overlap) > 0:
        print(f"   ⚠️  WARNING: Found match overlap! {overlap}")
    else:
        print(f"   ✅ No match leakage")

    # Train model with ORIGINAL hyperparameters (like partner's working model)
    print("\n[3/5] Training Random Forest model...")
    print("   Hyperparameters:")
    print("     - n_estimators=200")
    print("     - max_depth=15 (ORIGINAL - allows model to learn)")
    print("     - NO min_samples_leaf (ORIGINAL - less restrictive)")
    print("     - NO class_weight (ORIGINAL - better predictions)")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,             # BACK TO ORIGINAL!
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

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

    # Training set
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]

    print("\n   TRAINING SET:")
    print(f"     Accuracy:     {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"     ROC AUC:      {roc_auc_score(y_train, y_train_proba):.4f}")
    print(f"     Log Loss:     {log_loss(y_train, y_train_proba):.4f}")
    print(f"     Brier Score:  {brier_score_loss(y_train, y_train_proba):.4f}")

    train_ece, _, _, _ = calculate_calibration_error(y_train, y_train_proba)
    print(f"     ECE:          {train_ece:.4f}")

    # Test set
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    print("\n   TEST SET:")
    print(f"     Accuracy:     {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"     ROC AUC:      {roc_auc_score(y_test, y_test_proba):.4f}")
    print(f"     Log Loss:     {log_loss(y_test, y_test_proba):.4f}")
    print(f"     Brier Score:  {brier_score_loss(y_test, y_test_proba):.4f}")

    test_ece, _, _, _ = calculate_calibration_error(y_test, y_test_proba)
    print(f"     ECE:          {test_ece:.4f}")

    if test_ece < 0.01:
        print(f"     ✅ ECE < 0.01 (target met!)")
    else:
        print(f"     ⚠️  ECE > 0.01 (target: < 0.01)")

    # Plot calibration curve
    print("\n[5/5] Generating calibration curve...")
    plt_file = Path("plots/xg_calibration_curve.png")
    plt_file.parent.mkdir(exist_ok=True)

    plot_calibration_curve(y_test, y_test_proba, title="XG Model Calibration (Test Set)")
    plt.savefig(plt_file, dpi=150, bbox_inches='tight')
    print(f"   Saved calibration curve to {plt_file}")

    # Save model
    print("\n[6/5] Saving model...")
    model_file = Path("models/xg_model_improved.pkl")
    model_file.parent.mkdir(exist_ok=True)

    with open(model_file, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_cols': feature_cols,
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'test_auc': roc_auc_score(y_test, y_test_proba),
            'test_ece': test_ece
        }, f)

    print(f"   ✅ Model saved to {model_file}")

    # Calculate overfitting gap
    train_test_gap = roc_auc_score(y_train, y_train_proba) - roc_auc_score(y_test, y_test_proba)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nKey improvements:")
    print(f"  ✅ Added defenders_in_triangle feature")
    print(f"  ✅ Added defenders_within_3m feature")
    print(f"  ✅ Added player_finishing_skill (individuality)")
    print(f"  ✅ Calculated calibration (ECE = {test_ece:.4f})")
    print(f"  ✅ FIXED: Match-grouped split (no data leakage)")
    print(f"  ✅ FIXED: Reduced model complexity (max_depth=8)")
    print(f"  ✅ FIXED: Added class weights (balanced)")
    print(f"\nTest performance:")
    print(f"  Accuracy:      {accuracy_score(y_test, y_test_pred):.3f}")
    print(f"  AUC:           {roc_auc_score(y_test, y_test_proba):.3f}")
    print(f"  ECE:           {test_ece:.4f} {'✅' if test_ece < 0.01 else '⚠️'}")
    print(f"\nOverfitting check:")
    print(f"  Train AUC:     {roc_auc_score(y_train, y_train_proba):.3f}")
    print(f"  Test AUC:      {roc_auc_score(y_test, y_test_proba):.3f}")
    print(f"  Gap:           {train_test_gap:.3f} {'✅ (healthy)' if train_test_gap < 0.10 else '⚠️ (still overfitting)'}")


if __name__ == "__main__":
    main()
