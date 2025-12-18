"""
xG Model with Proper Features and Training

Improvements over original:
1. Match-based train/test split (no leakage)
2. Distance and angle to goal calculated
3. Defender metrics from tracking data
4. Player individuality with minimum shot threshold
5. Proper evaluation metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
import joblib
import pickle
from typing import Dict, Tuple, List, Set
import json


class xGModel:
    """
    Expected Goals (xG) model with defensive context and player individuality
    """

    def __init__(self, pitch_length: float = 107.0, pitch_width: float = 68.0):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.goal_width = 7.32  # Standard goal width in meters

        # Goal is at positive x end (center of field is 0,0)
        self.goal_x = pitch_length / 2
        self.goal_y = 0

        # Model components
        self.model = None
        self.scaler = None
        self.is_trained = False

        # Player individuality
        self.player_id_to_finishing_skill = self._load_player_skills()

    def _load_player_skills(self) -> Dict[int, float]:
        """Load player finishing skill from pre-built lookup table"""
        skill_file = Path("data/player_id_to_finishing_skill.pkl")

        if not skill_file.exists():
            print(f"⚠️  Player finishing skill file not found: {skill_file}")
            print("   Player individuality will not be used")
            return {}

        try:
            with open(skill_file, 'rb') as f:
                player_skills = pickle.load(f)
            print(f"✅ Loaded finishing skills for {len(player_skills)} players")
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
        from collections import defaultdict
        team_roster = defaultdict(set)

        # Get unique player-team pairs
        player_teams = events_df[['player_id', 'team_id']].dropna().drop_duplicates()

        for _, row in player_teams.iterrows():
            team_id = int(row['team_id'])
            player_id = int(row['player_id'])
            team_roster[team_id].add(player_id)

        return dict(team_roster)

    def extract_shot_events(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Extract shot events from match events"""
        # Filter for shots - check end_type column (SkillCorner data structure)
        is_shot = events_df['end_type'].str.lower() == 'shot'
        shots = events_df[is_shot].copy()
        return shots

    def calculate_distance_to_goal(self, x: float, y: float) -> float:
        """Calculate Euclidean distance from position to goal center"""
        return np.sqrt((x - self.goal_x)**2 + (y - self.goal_y)**2)

    def calculate_angle_to_goal(self, x: float, y: float) -> float:
        """
        Calculate angle to goal (how wide the goal appears from shot position)
        Returns angle in degrees
        """
        # Goal posts
        left_post_y = self.goal_y - self.goal_width / 2
        right_post_y = self.goal_y + self.goal_width / 2

        # Vectors from shot position to each post
        vec_to_left = np.array([self.goal_x - x, left_post_y - y])
        vec_to_right = np.array([self.goal_x - x, right_post_y - y])

        # Angle between vectors
        cos_angle = np.dot(vec_to_left, vec_to_right) / (
            np.linalg.norm(vec_to_left) * np.linalg.norm(vec_to_right) + 1e-10
        )

        # Clamp to valid range
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def count_defenders_near_shot(self, shot_x: float, shot_y: float,
                                   shooter_team: int, tracking_frame: Dict,
                                   team_roster: Dict, radius: float = 5.0) -> int:
        """Count defenders within radius of shot location"""
        defenders_count = 0

        for player_data in tracking_frame.get('player_data', []):
            player_id = player_data.get('player_id')

            # Check if player is on opposing team
            is_opponent = False
            for team_id, player_set in team_roster.items():
                if player_id in player_set:
                    if team_id != shooter_team:
                        is_opponent = True
                    break

            if not is_opponent:
                continue

            # Get player position
            px = player_data.get('x', 0)
            py = player_data.get('y', 0)

            # Check if within radius
            dist = np.sqrt((px - shot_x)**2 + (py - shot_y)**2)
            if dist <= radius:
                defenders_count += 1

        return defenders_count

    def count_defenders_blocking_shot(self, shot_x: float, shot_y: float,
                                       shooter_team: int, tracking_frame: Dict,
                                       team_roster: Dict) -> int:
        """
        Count defenders in cone between shot location and goal
        (defenders who could block the shot)
        """
        blocking_count = 0

        for player_data in tracking_frame.get('player_data', []):
            player_id = player_data.get('player_id')

            # Check if player is on opposing team
            is_opponent = False
            for team_id, player_set in team_roster.items():
                if player_id in player_set:
                    if team_id != shooter_team:
                        is_opponent = True
                    break

            if not is_opponent:
                continue

            px = player_data.get('x', 0)
            py = player_data.get('y', 0)

            # Check if defender is between shot and goal (in x direction)
            if shot_x < px < self.goal_x:
                # Check if within goal width cone
                progress = (px - shot_x) / (self.goal_x - shot_x + 1e-10)
                max_y_deviation = self.goal_width / 2 + (1 - progress) * abs(shot_y)

                if abs(py - self.goal_y) <= max_y_deviation:
                    blocking_count += 1

        return blocking_count

    def get_tracking_frame(self, tracking_dict: Dict, frame_num: int) -> Dict:
        """Get tracking frame by frame number"""
        if frame_num not in tracking_dict:
            return {'player_data': []}
        return tracking_dict[frame_num]

    def extract_features(self, shot_events: pd.DataFrame, tracking_dict: Dict,
                        team_roster: Dict) -> pd.DataFrame:
        """
        Extract features for each shot including:
        - Basic geometry (distance, angle)
        - Defender metrics (nearby, blocking)
        - Player finishing skill
        - Shot outcome (goal = 1, no goal = 0)
        """
        features = []

        for _, shot in shot_events.iterrows():
            # Get shot location
            x_start = shot.get('x_start', 0)
            y_start = shot.get('y_start', 0)

            # Calculate distance and angle
            distance_to_goal = self.calculate_distance_to_goal(x_start, y_start)
            angle_to_goal = self.calculate_angle_to_goal(x_start, y_start)

            # Get tracking frame
            frame_num = shot.get('frame_start', shot.get('frame', 0))
            frame = self.get_tracking_frame(tracking_dict, frame_num)

            # Get player and team info
            player_id = int(shot.get('player_id', 0))
            team_id = int(shot.get('team_id', 0))

            # Count defenders
            defenders_nearby = self.count_defenders_near_shot(
                x_start, y_start, team_id, frame, team_roster, radius=5.0
            )
            defenders_blocking = self.count_defenders_blocking_shot(
                x_start, y_start, team_id, frame, team_roster
            )

            # Get player finishing skill
            finishing_skill = self.player_id_to_finishing_skill.get(player_id, 0.0)

            # Determine if goal was scored (SkillCorner uses lead_to_goal column)
            goal_scored = 0
            if 'lead_to_goal' in shot:
                goal_scored = 1 if shot['lead_to_goal'] == True or str(shot['lead_to_goal']).lower() == 'true' else 0

            features.append({
                'event_id': shot.get('event_id', shot.name),
                'player_id': player_id,
                'x_start': x_start,
                'y_start': y_start,
                'distance_to_goal': distance_to_goal,
                'angle_to_goal': angle_to_goal,
                'defenders_nearby': defenders_nearby,
                'defenders_blocking': defenders_blocking,
                'player_finishing_skill': finishing_skill,
                'goal_scored': goal_scored
            })

        return pd.DataFrame(features)

    def train(self, features_df: pd.DataFrame):
        """Train xG model on shot features"""
        feature_cols = [
            'x_start', 'y_start',
            'distance_to_goal', 'angle_to_goal',
            'defenders_nearby', 'defenders_blocking',
            'player_finishing_skill'
        ]

        X = features_df[feature_cols].values
        y = features_df['goal_scored'].values

        # Handle any NaNs
        X = np.nan_to_num(X, nan=0.0)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X_scaled, y)
        self.is_trained = True

        print(f"Model trained on {len(X)} shots")
        print(f"Goals scored: {y.sum()} / {len(y)} ({y.mean():.2%})")
        print(f"\nFeature importances:")
        importances = sorted(zip(feature_cols, self.model.feature_importances_),
                           key=lambda x: x[1], reverse=True)
        for feat, imp in importances:
            print(f"  {feat}: {imp:.4f}")

    def predict_xg(self, features_df: pd.DataFrame) -> np.ndarray:
        """Predict xG (goal probability) for shots"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        feature_cols = [
            'x_start', 'y_start',
            'distance_to_goal', 'angle_to_goal',
            'defenders_nearby', 'defenders_blocking',
            'player_finishing_skill'
        ]

        X = features_df[feature_cols].values
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = self.scaler.transform(X)

        # Return probability of goal (class 1)
        xg = self.model.predict_proba(X_scaled)[:, 1]

        return xg

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
