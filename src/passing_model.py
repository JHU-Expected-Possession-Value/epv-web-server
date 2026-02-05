"""
Passing Model with Defensive Context and Player Individuality

Improvements over original:
1. Match-based train/test split (already had this)
2. Pitch control at pass destination
3. Defender pressure on passer and receiver
4. Defenders in passing lane
5. Player passing individuality with minimum 90s threshold
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib
import pickle
from typing import Dict, Tuple, List, Set, Optional
import json
import sys
from collections import defaultdict

try:
    from src.pitch_control import PitchControlCache
    PITCH_CONTROL_AVAILABLE = True
except ImportError:
    try:
        from pitch_control import PitchControlCache
        PITCH_CONTROL_AVAILABLE = True
    except ImportError:
        PITCH_CONTROL_AVAILABLE = False
        print("⚠️  Pitch control module not available")

PASS_DEFENDER_LANE_WEIGHT = 0.4
PASS_DEFENDER_DEST_WEIGHT = 0.25
PASS_MIN_DIST_CAP = 6.0
PASS_MIN_DIST_FLOOR = 0.2


def apply_defender_priority_penalty(
    prob,
    defenders_in_lane,
    defenders_near_dest,
    min_defender_dist_to_lane,
    lane_weight=PASS_DEFENDER_LANE_WEIGHT,
    dest_weight=PASS_DEFENDER_DEST_WEIGHT,
    min_dist_cap=PASS_MIN_DIST_CAP,
    min_dist_floor=PASS_MIN_DIST_FLOOR
):
    """Apply a defender-priority penalty to pass success probability."""
    is_scalar = np.isscalar(prob)
    prob_arr = np.asarray(prob, dtype=float)
    lane_arr = np.nan_to_num(np.asarray(defenders_in_lane, dtype=float), nan=0.0)
    dest_arr = np.nan_to_num(np.asarray(defenders_near_dest, dtype=float), nan=0.0)
    dist_arr = np.nan_to_num(np.asarray(min_defender_dist_to_lane, dtype=float), nan=min_dist_cap)

    lane_factor = np.exp(-lane_weight * np.clip(lane_arr, 0.0, None))
    dest_factor = np.exp(-dest_weight * np.clip(dest_arr, 0.0, None))
    dist_factor = np.clip(dist_arr / float(min_dist_cap), min_dist_floor, 1.0)

    penalty = lane_factor * dest_factor * dist_factor
    adjusted = np.clip(prob_arr * penalty, 0.0, 1.0)

    return float(adjusted) if is_scalar else adjusted


class DefenderPriorityPassModel:
    """Wrap a pass model to emphasize defender proximity features."""

    def __init__(
        self,
        model,
        feature_cols,
        lane_weight=PASS_DEFENDER_LANE_WEIGHT,
        dest_weight=PASS_DEFENDER_DEST_WEIGHT,
        min_dist_cap=PASS_MIN_DIST_CAP,
        min_dist_floor=PASS_MIN_DIST_FLOOR
    ):
        self.model = model
        self.feature_cols = list(feature_cols) if feature_cols is not None else None
        self.lane_weight = float(lane_weight)
        self.dest_weight = float(dest_weight)
        self.min_dist_cap = float(min_dist_cap)
        self.min_dist_floor = float(min_dist_floor)

        self._lane_idx = None
        self._dest_idx = None
        self._min_dist_idx = None
        if self.feature_cols:
            if "defenders_in_lane" in self.feature_cols:
                self._lane_idx = self.feature_cols.index("defenders_in_lane")
            if "defenders_near_dest" in self.feature_cols:
                self._dest_idx = self.feature_cols.index("defenders_near_dest")
            if "min_defender_dist_to_lane" in self.feature_cols:
                self._min_dist_idx = self.feature_cols.index("min_defender_dist_to_lane")

    def predict_proba(self, X):
        proba = np.asarray(self.model.predict_proba(X))
        if self._lane_idx is None or self._dest_idx is None or self._min_dist_idx is None:
            return proba

        X_arr = np.asarray(X)
        adjusted = apply_defender_priority_penalty(
            proba[:, 1],
            X_arr[:, self._lane_idx],
            X_arr[:, self._dest_idx],
            X_arr[:, self._min_dist_idx],
            lane_weight=self.lane_weight,
            dest_weight=self.dest_weight,
            min_dist_cap=self.min_dist_cap,
            min_dist_floor=self.min_dist_floor
        )
        proba[:, 1] = adjusted
        proba[:, 0] = 1.0 - adjusted
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)

    def __getattr__(self, name):
        try:
            model = object.__getattribute__(self, "model")
        except AttributeError:
            raise AttributeError(name)
        return getattr(model, name)

class PassingModel:
    """
    Passing success model with defensive context and player individuality
    """

    def __init__(self, pitch_length: float = 107.0, pitch_width: float = 68.0):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

        # Model components
        self.model = None
        self.scaler = None
        self.is_trained = False

        # Player individuality
        self.player_id_to_passing_skill = self._load_player_skills()

        # Note: Pitch control temporarily disabled for initial training
        # TODO: Re-integrate pitch control with proper API

    def _load_player_skills(self) -> Dict[int, float]:
        """Load player passing skill from pre-built lookup table"""
        skill_file = Path("data/player_id_to_passing_skill.pkl")

        if not skill_file.exists():
            print(f"⚠️  Player passing skill file not found: {skill_file}")
            print("   Player individuality will not be used")
            return {}

        try:
            with open(skill_file, 'rb') as f:
                player_skills = pickle.load(f)
            print(f"✅ Loaded passing skills for {len(player_skills)} players")
            return player_skills
        except Exception as e:
            print(f"⚠️  Error loading player skills: {e}")
            return {}

    def load_match_data(self, match_id: str, data_dir: Path) -> Tuple:
        """Load events and tracking data for a match"""
        # Load events CSV
        events_path = data_dir / f"{match_id}_dynamic_events.csv"
        events_df = pd.read_csv(events_path)

        # Load tracking data
        tracking_path = data_dir / f"{match_id}_tracking_extrapolated.jsonl"
        if not tracking_path.exists():
            raise FileNotFoundError(f"No tracking data found for match {match_id}")

        tracking_dict = self._load_tracking_data(tracking_path)

        # Extract team roster from events
        team_roster = self._extract_team_roster_from_events(events_df)

        return events_df, tracking_dict, team_roster

    def _load_tracking_data(self, tracking_path: Path) -> Dict:
        """Load tracking data from JSONL file into a dictionary"""
        tracking_dict = {}
        with open(tracking_path, 'r') as f:
            for line in f:
                if line.strip():
                    frame_data = json.loads(line)
                    frame_num = frame_data['frame']
                    tracking_dict[frame_num] = frame_data
        return tracking_dict

    def _extract_team_roster_from_events(self, events_df: pd.DataFrame) -> Dict[int, Set[int]]:
        """Extract team rosters from events dataframe"""
        team_roster = defaultdict(set)

        # Get unique player-team pairs
        player_teams = events_df[['player_id', 'team_id']].dropna().drop_duplicates()

        for _, row in player_teams.iterrows():
            team_id = int(row['team_id'])
            player_id = int(row['player_id'])
            team_roster[team_id].add(player_id)

        return dict(team_roster)

    def extract_pass_events(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Extract pass events from match events"""
        # Filter for passes - check end_type column (SkillCorner data structure)
        is_pass = events_df['end_type'].str.lower() == 'pass'
        passes = events_df[is_pass].copy()
        return passes

    def count_defenders_in_radius(self, x: float, y: float, tracking_frame: Dict,
                                   team_roster: Dict, team_id: int, radius: float = 3.0) -> int:
        """Count opposing defenders within radius of a position"""
        defender_count = 0

        for player_data in tracking_frame.get('player_data', []):
            player_id = player_data.get('player_id')

            # Check if player is on opposing team
            is_opponent = False
            for tid, player_set in team_roster.items():
                if player_id in player_set:
                    if tid != team_id:
                        is_opponent = True
                    break

            if not is_opponent:
                continue

            # Get player position
            px = player_data.get('x', 0)
            py = player_data.get('y', 0)

            # Check if within radius
            dist = np.sqrt((px - x)**2 + (py - y)**2)
            if dist <= radius:
                defender_count += 1

        return defender_count

    def count_defenders_in_passing_lane(self, x_start: float, y_start: float,
                                        x_end: float, y_end: float,
                                        tracking_frame: Dict, team_roster: Dict,
                                        team_id: int, lane_width: float = 2.0) -> int:
        """
        Count defenders in the passing lane (rectangle between passer and receiver)
        """
        defender_count = 0

        # Calculate perpendicular direction for lane width
        pass_vec = np.array([x_end - x_start, y_end - y_start])
        pass_length = np.linalg.norm(pass_vec) + 1e-10
        pass_dir = pass_vec / pass_length

        # Perpendicular direction
        perp_dir = np.array([-pass_dir[1], pass_dir[0]])

        for player_data in tracking_frame.get('player_data', []):
            player_id = player_data.get('player_id')

            # Check if player is on opposing team
            is_opponent = False
            for tid, player_set in team_roster.items():
                if player_id in player_set:
                    if tid != team_id:
                        is_opponent = True
                    break

            if not is_opponent:
                continue

            px = player_data.get('x', 0)
            py = player_data.get('y', 0)

            # Check if defender is in the passing lane
            to_defender = np.array([px - x_start, py - y_start])
            along_pass = np.dot(to_defender, pass_dir)

            # Check if between passer and receiver
            if 0 <= along_pass <= pass_length:
                # Check perpendicular distance to pass line
                perp_dist = abs(np.dot(to_defender, perp_dir))

                if perp_dist <= lane_width:
                    defender_count += 1

        return defender_count

    def get_tracking_frame(self, tracking_dict: Dict, frame_num: int) -> Dict:
        """Get tracking frame by frame number"""
        if frame_num not in tracking_dict:
            return {'player_data': []}
        return tracking_dict[frame_num]

    def extract_features(self, pass_events: pd.DataFrame, tracking_dict: Dict,
                        team_roster: Dict, match_id: Optional[str] = None,
                        pc_cache: Optional['PitchControlCache'] = None) -> pd.DataFrame:
        """
        Extract features for each pass including:
        - Basic geometry (distance, angle, direction)
        - Pitch control at destination
        - Defender pressure on passer and receiver
        - Defenders in passing lane
        - Player passing skill
        - Pass outcome (completed = 1, failed = 0)

        Args:
            pass_events: DataFrame of pass events
            tracking_dict: Dictionary of tracking data
            team_roster: Dictionary mapping team_id to set of player_ids
            match_id: Match ID (required for pitch control)
            pc_cache: Optional PitchControlCache instance
        """
        features = []

        for _, pass_event in pass_events.iterrows():
            # Get pass start and end locations
            x_start = pass_event.get('x_start', 0)
            y_start = pass_event.get('y_start', 0)
            x_end = pass_event.get('x_end', x_start)
            y_end = pass_event.get('y_end', y_start)

            # Basic geometry
            dx = x_end - x_start
            dy = y_end - y_start
            distance = np.sqrt(dx**2 + dy**2)

            # Angle of pass
            angle = np.arctan2(dy, dx) * 180 / np.pi

            # Forward progress
            forward_progress = dx

            # Get tracking frame
            frame_num = pass_event.get('frame_start', pass_event.get('frame', 0))
            frame = self.get_tracking_frame(tracking_dict, frame_num)

            # Get player and team info
            passer_id = int(pass_event.get('player_id', 0))
            team_id = int(pass_event.get('team_id', 0))

            # Compute pitch control at destination
            if pc_cache is not None and match_id is not None and PITCH_CONTROL_AVAILABLE:
                try:
                    pc_destination = pc_cache.get_pc(match_id, frame_num, x_end, y_end)
                except Exception as e:
                    pc_destination = 0.5  # Fallback on error
            else:
                pc_destination = 0.5  # No pitch control available

            # Defender pressure on passer
            pressure_on_passer = self.count_defenders_in_radius(
                x_start, y_start, frame, team_roster, team_id, radius=3.0
            )

            # Defender pressure on receiver (at destination)
            pressure_on_receiver = self.count_defenders_in_radius(
                x_end, y_end, frame, team_roster, team_id, radius=3.0
            )

            # Defenders in passing lane
            defenders_in_lane = self.count_defenders_in_passing_lane(
                x_start, y_start, x_end, y_end, frame, team_roster, team_id, lane_width=1.5
            )

            # Distance from sideline (constrained passes are harder)
            dist_from_sideline = min(
                abs(y_start - self.pitch_width / 2),
                abs(y_start + self.pitch_width / 2)
            )

            # Get player passing skill
            passing_skill = self.player_id_to_passing_skill.get(passer_id, 0.0)

            # Determine if pass was completed
            pass_completed = 0
            if 'pass_outcome' in pass_event:
                outcome = str(pass_event['pass_outcome']).lower()
                pass_completed = 1 if 'success' in outcome or 'complete' in outcome else 0
            elif 'outcome' in pass_event:
                outcome = str(pass_event['outcome']).lower()
                pass_completed = 1 if 'success' in outcome or 'complete' in outcome else 0

            features.append({
                'event_id': pass_event.get('event_id', pass_event.name),
                'player_id': passer_id,
                'x_start': x_start,
                'y_start': y_start,
                'x_end': x_end,
                'y_end': y_end,
                'distance': distance,
                'angle': angle,
                'forward_progress': forward_progress,
                'dist_from_sideline': dist_from_sideline,
                'pitch_control_destination': pc_destination,
                'pressure_on_passer': pressure_on_passer,
                'pressure_on_receiver': pressure_on_receiver,
                'defenders_in_lane': defenders_in_lane,
                'player_passing_skill': passing_skill,
                'pass_completed': pass_completed
            })

        return pd.DataFrame(features)

    def train(self, features_df: pd.DataFrame):
        """Train passing model on pass features"""
        feature_cols = [
            'x_start', 'y_start', 'x_end', 'y_end',
            'distance', 'angle', 'forward_progress',
            'dist_from_sideline',
            'pitch_control_destination',
            'pressure_on_passer', 'pressure_on_receiver',
            'defenders_in_lane',
            'player_passing_skill'
        ]

        X = features_df[feature_cols].values
        y = features_df['pass_completed'].values

        # Handle any NaNs
        X = np.nan_to_num(X, nan=0.0)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X_scaled, y)
        self.is_trained = True

        print(f"Model trained on {len(X)} passes")
        print(f"Passes completed: {y.sum()} / {len(y)} ({y.mean():.2%})")
        print(f"\nFeature importances:")
        importances = sorted(zip(feature_cols, self.model.feature_importances_),
                           key=lambda x: x[1], reverse=True)
        for feat, imp in importances:
            print(f"  {feat}: {imp:.4f}")

    def predict_pass_success_probability(self, features_df: pd.DataFrame) -> np.ndarray:
        """Predict probability of pass completion"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        feature_cols = [
            'x_start', 'y_start', 'x_end', 'y_end',
            'distance', 'angle', 'forward_progress',
            'dist_from_sideline',
            'pitch_control_destination',
            'pressure_on_passer', 'pressure_on_receiver',
            'defenders_in_lane',
            'player_passing_skill'
        ]

        X = features_df[feature_cols].values
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = self.scaler.transform(X)

        # Return probability of completion (class 1)
        pass_prob = self.model.predict_proba(X_scaled)[:, 1]

        return pass_prob

    def save_model(self, filepath: Path):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'pitch_length': self.pitch_length,
            'pitch_width': self.pitch_width
        }

        joblib.dump(model_data, filepath)
        print(f"Model saved to: {filepath}")

    def load_model(self, filepath: Path):
        """Load trained model"""
        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.pitch_length = model_data['pitch_length']
        self.pitch_width = model_data['pitch_width']
        self.is_trained = True

        print(f"Model loaded from: {filepath}")
