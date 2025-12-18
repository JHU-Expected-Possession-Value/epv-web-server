"""
Dribbling Model for Soccer EPV

This module implements a dribbling/carry model that predicts the probability
of maintaining possession during a carry event.

Features extracted:
- Start/end position
- Distance covered
- Duration
- Number of defenders nearby (from tracking)
- Distance to goal
- Pitch control values (CRITICAL FEATURE)
- Pressure from defenders
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


class DribblingModel:
    """
    Model to predict probability of maintaining possession during a dribble/carry.

    This is a key component of an EPV (Expected Possession Value) framework.
    """

    def __init__(self, pitch_length: float = 107.0, pitch_width: float = 68.0):
        """
        Initialize the dribbling model.

        Args:
            pitch_length: Length of the pitch in meters (default: 107.0)
            pitch_width: Width of the pitch in meters (default: 68.0)
        """
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        self.scaler = StandardScaler()
        self.is_trained = False

        # Team rosters - will be populated when loading match data
        self.team_rosters = {}  # match_id -> {team_id -> set(player_ids)}

        # Load player individuality scores (player_id -> dribbling skill)
        self.player_id_to_skill = self._load_player_skills()

    def _load_player_skills(self) -> Dict[int, float]:
        """Load player individuality scores from pre-built lookup table."""
        skill_file = Path("data/player_id_to_skill.pkl")

        if not skill_file.exists():
            print(f"⚠️  Player skill file not found: {skill_file}")
            print("   Player individuality will not be used (all players treated as average)")
            return {}

        try:
            with open(skill_file, 'rb') as f:
                player_id_to_skill = pickle.load(f)
            print(f"✅ Loaded player individuality scores for {len(player_id_to_skill)} players")
            return player_id_to_skill
        except Exception as e:
            print(f"⚠️  Error loading player skills: {e}")
            return {}

    def load_match_data(self,
                       match_id: str,
                       data_dir: Path) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        Load match event data, tracking data, and team rosters for a given match.

        Args:
            match_id: Match ID (e.g., '1302757')
            data_dir: Directory containing the data files

        Returns:
            Tuple of (events_df, tracking_dict, team_roster)
            - tracking_dict maps frame -> tracking_data
            - team_roster maps team_id -> set(player_ids)
        """
        # Load events CSV
        events_path = data_dir / f"{match_id}_dynamic_events.csv"
        events_df = pd.read_csv(events_path)

        # Load tracking data (extrapolated version if available, otherwise regular)
        tracking_path = data_dir / f"{match_id}_tracking_extrapolated.jsonl"
        if not tracking_path.exists():
            # Try to load from match.json
            match_json_path = data_dir / f"{match_id}_match.json"
            if match_json_path.exists():
                tracking_dict, team_roster = self._load_tracking_from_json(match_json_path)
            else:
                raise FileNotFoundError(f"No tracking data found for match {match_id}")
        else:
            tracking_dict = self._load_tracking_data(tracking_path)
            # Load team roster from events data
            team_roster = self._extract_team_roster_from_events(events_df)

        # Store team roster
        self.team_rosters[match_id] = team_roster

        return events_df, tracking_dict, team_roster

    def _load_tracking_data(self, tracking_path: Path) -> Dict:
        """
        Load tracking data from JSONL file into a dictionary.

        Args:
            tracking_path: Path to tracking JSONL file

        Returns:
            Dictionary mapping frame number to tracking data
        """
        tracking_dict = {}

        with open(tracking_path, 'r') as f:
            for line in f:
                if line.strip():
                    frame_data = json.loads(line)
                    frame_num = frame_data['frame']
                    tracking_dict[frame_num] = frame_data

        return tracking_dict

    def _load_tracking_from_json(self, match_json_path: Path) -> Tuple[Dict, Dict]:
        """
        Load tracking data from match JSON file and extract team rosters.

        Args:
            match_json_path: Path to match JSON file

        Returns:
            Tuple of (tracking_dict, team_roster)
        """
        with open(match_json_path, 'r') as f:
            # First line is metadata
            match_metadata = json.loads(f.readline())

            # Extract team rosters from player metadata
            team_roster = self._extract_team_roster_from_metadata(match_metadata)

            # Load tracking frames
            tracking_dict = {}
            for line in f:
                if line.strip():
                    frame_data = json.loads(line)
                    frame_num = frame_data['frame']
                    tracking_dict[frame_num] = frame_data

        return tracking_dict, team_roster

    def _extract_team_roster_from_metadata(self, match_metadata: Dict) -> Dict[int, Set[int]]:
        """
        Extract team rosters from match metadata.

        Args:
            match_metadata: Match metadata dictionary

        Returns:
            Dictionary mapping team_id -> set(player_ids)
        """
        team_roster = {}

        for player in match_metadata.get('players', []):
            team_id = player['team_id']
            player_id = player['id']

            if team_id not in team_roster:
                team_roster[team_id] = set()

            team_roster[team_id].add(player_id)

        return team_roster

    def _extract_team_roster_from_events(self, events_df: pd.DataFrame) -> Dict[int, Set[int]]:
        """
        Extract team rosters from events dataframe.

        Args:
            events_df: Events dataframe

        Returns:
            Dictionary mapping team_id -> set(player_ids)
        """
        team_roster = {}

        # Get unique player-team pairs
        player_teams = events_df[['player_id', 'team_id']].dropna().drop_duplicates()

        for _, row in player_teams.iterrows():
            team_id = int(row['team_id'])
            player_id = int(row['player_id'])

            if team_id not in team_roster:
                team_roster[team_id] = set()

            team_roster[team_id].add(player_id)

        return team_roster

    def extract_carry_events(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract carry/dribble events from the events dataframe.

        Args:
            events_df: Events dataframe

        Returns:
            DataFrame containing only carry events
        """
        # Filter for player possession events where carry=True
        carry_events = events_df[
            (events_df['event_type'] == 'player_possession') &
            (events_df['carry'] == True)
        ].copy()

        return carry_events

    def calculate_distance_to_goal(self,
                                   x: float,
                                   y: float,
                                   attacking_direction: str = 'right') -> float:
        """
        Calculate distance from a position to the center of the goal.

        Args:
            x: X coordinate
            y: Y coordinate
            attacking_direction: Direction of attack ('left' or 'right')

        Returns:
            Distance to goal in meters
        """
        if attacking_direction == 'right':
            goal_x = self.pitch_length / 2
        else:
            goal_x = -self.pitch_length / 2

        goal_y = 0.0

        distance = np.sqrt((x - goal_x)**2 + (y - goal_y)**2)
        return distance

    def count_nearby_defenders(self,
                               x: float,
                               y: float,
                               frame: int,
                               tracking_dict: Dict,
                               team_id: int,
                               team_roster: Dict[int, Set[int]],
                               radius: float = 5.0) -> Tuple[int, float]:
        """
        Count the number of opposing players within a given radius and calculate pressure.

        Args:
            x: X coordinate of player with ball
            y: Y coordinate of player with ball
            frame: Frame number
            tracking_dict: Tracking data dictionary
            team_id: Team ID of player with ball
            team_roster: Dictionary mapping team_id -> set(player_ids)
            radius: Radius in meters to search for defenders

        Returns:
            Tuple of (number of defenders within radius, closest defender distance)
        """
        if frame not in tracking_dict:
            return 0, radius

        frame_data = tracking_dict[frame]
        player_data = frame_data.get('player_data', [])

        if not player_data:
            return 0, radius

        # Get opponent team players
        opponent_team_ids = [tid for tid in team_roster.keys() if tid != team_id]
        if not opponent_team_ids:
            # Fallback: just count all nearby players
            opponent_players = set()
        else:
            opponent_players = set()
            for opp_team_id in opponent_team_ids:
                opponent_players.update(team_roster[opp_team_id])

        # Count defenders within radius
        nearby_count = 0
        closest_distance = radius

        for player in player_data:
            if not player.get('is_detected', False):
                continue

            player_id = player['player_id']

            # Check if this is an opponent
            if opponent_players and player_id not in opponent_players:
                continue

            px = player['x']
            py = player['y']

            distance = np.sqrt((x - px)**2 + (y - py)**2)

            if distance < radius and distance > 0.1:  # Exclude self
                nearby_count += 1
                if distance < closest_distance:
                    closest_distance = distance

        return nearby_count, closest_distance

    def extract_features(self,
                        carry_events: pd.DataFrame,
                        tracking_dict: Dict,
                        team_roster: Dict[int, Set[int]],
                        pitch_control: Optional[Dict] = None) -> pd.DataFrame:
        """
        Extract features for each carry event.

        Args:
            carry_events: DataFrame of carry events
            tracking_dict: Tracking data dictionary
            team_roster: Dictionary mapping team_id -> set(player_ids)
            pitch_control: Optional pitch control values
                          (dict mapping (frame, x, y) -> PC value or frame -> grid)

        Returns:
            DataFrame with extracted features
        """
        features = []

        for idx, event in carry_events.iterrows():
            # Basic features from event data
            x_start = event['x_start']
            y_start = event['y_start']
            x_end = event['x_end']
            y_end = event['y_end']

            distance_covered = event.get('distance_covered', 0.0)
            duration = event.get('duration', 0.0)

            # Calculate distance to goal at start and end
            # Determine attacking direction from attacking_side column
            attacking_side = event.get('attacking_side', 'left_to_right')
            if attacking_side == 'left_to_right':
                direction = 'right'
            else:
                direction = 'left'

            dist_to_goal_start = self.calculate_distance_to_goal(x_start, y_start, direction)
            dist_to_goal_end = self.calculate_distance_to_goal(x_end, y_end, direction)
            dist_to_goal_change = dist_to_goal_start - dist_to_goal_end  # Positive = moving toward goal

            # Count defenders nearby at start and end
            frame_start = event.get('frame_start')
            frame_end = event.get('frame_end')
            team_id = event.get('team_id')

            defenders_nearby_start, pressure_start = self.count_nearby_defenders(
                x_start, y_start, frame_start, tracking_dict, team_id, team_roster, radius=5.0
            )
            defenders_nearby_end, pressure_end = self.count_nearby_defenders(
                x_end, y_end, frame_end, tracking_dict, team_id, team_roster, radius=5.0
            )

            # Calculate speed
            speed = distance_covered / duration if duration > 0 else 0.0

            # Pitch control (CRITICAL FEATURE)
            # You should pass in pitch control values from your implementation
            if pitch_control:
                # Assuming pitch_control is a dict with frame keys
                # You'll need to adapt this to your actual pitch control implementation
                pc_start = pitch_control.get(frame_start, 0.5)
                pc_end = pitch_control.get(frame_end, 0.5)
                pc_change = pc_end - pc_start
            else:
                # Placeholder values if pitch control not provided
                pc_start = 0.5
                pc_end = 0.5
                pc_change = 0.0

            # Label: whether possession was maintained
            # If end_type is 'pass' or 'shot', possession maintained
            # If it's 'possession_loss', 'interception', etc., possession lost
            end_type = event.get('end_type', '')
            possession_maintained = 1 if end_type in ['pass', 'shot'] else 0

            # Additional features
            angle_change = np.abs(np.arctan2(y_end - y_start, x_end - x_start))

            # Distance from sideline (lower = more constrained)
            dist_from_sideline = min(abs(y_start - self.pitch_width/2),
                                    abs(y_start + self.pitch_width/2))

            features.append({
                'event_id': event.get('event_id'),
                'x_start': x_start,
                'y_start': y_start,
                'x_end': x_end,
                'y_end': y_end,
                'distance_covered': distance_covered,
                'duration': duration,
                'speed': speed,
                'dist_to_goal_start': dist_to_goal_start,
                'dist_to_goal_end': dist_to_goal_end,
                'dist_to_goal_change': dist_to_goal_change,
                'defenders_nearby_start': defenders_nearby_start,
                'defenders_nearby_end': defenders_nearby_end,
                'pressure_start': pressure_start,
                'pressure_end': pressure_end,
                'pitch_control_start': pc_start,
                'pitch_control_end': pc_end,
                'pitch_control_change': pc_change,
                'angle_change': angle_change,
                'dist_from_sideline': dist_from_sideline,
                'player_dribbling_skill': self.player_id_to_skill.get(event.get('player_id'), 0.0),
                'possession_maintained': possession_maintained
            })

        return pd.DataFrame(features)

    def train(self, features_df: pd.DataFrame):
        """
        Train the dribbling model.

        Args:
            features_df: DataFrame with extracted features and labels
        """
        # Separate features and labels
        feature_cols = [
            'x_start', 'y_start', 'x_end', 'y_end',
            'distance_covered', 'duration', 'speed',
            'dist_to_goal_start', 'dist_to_goal_end', 'dist_to_goal_change',
            'defenders_nearby_start', 'defenders_nearby_end',
            'pressure_start', 'pressure_end',
            'pitch_control_start', 'pitch_control_end', 'pitch_control_change',
            'angle_change', 'dist_from_sideline',
            'player_dribbling_skill'
        ]

        X = features_df[feature_cols].values
        y = features_df['possession_maintained'].values

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True

        print(f"Model trained on {len(X)} carry events")
        print(f"Possession maintained: {y.sum()} / {len(y)} ({y.mean():.2%})")
        print(f"\nFeature importances:")
        importances = sorted(zip(feature_cols, self.model.feature_importances_),
                           key=lambda x: x[1], reverse=True)
        for feat, imp in importances:
            print(f"  {feat}: {imp:.4f}")

    def predict_possession_probability(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of maintaining possession for carry events.

        Args:
            features_df: DataFrame with extracted features

        Returns:
            Array of probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        feature_cols = [
            'x_start', 'y_start', 'x_end', 'y_end',
            'distance_covered', 'duration', 'speed',
            'dist_to_goal_start', 'dist_to_goal_end', 'dist_to_goal_change',
            'defenders_nearby_start', 'defenders_nearby_end',
            'pressure_start', 'pressure_end',
            'pitch_control_start', 'pitch_control_end', 'pitch_control_change',
            'angle_change', 'dist_from_sideline',
            'player_dribbling_skill'
        ]

        X = features_df[feature_cols].values
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = self.scaler.transform(X)

        # Return probability of class 1 (possession maintained)
        probs = self.model.predict_proba(X_scaled)[:, 1]

        return probs

    def save_model(self, filepath: Path):
        """
        Save the trained model to disk.

        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'pitch_length': self.pitch_length,
            'pitch_width': self.pitch_width,
            'team_rosters': self.team_rosters
        }

        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: Path):
        """
        Load a trained model from disk.

        Args:
            filepath: Path to the saved model
        """
        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.pitch_length = model_data['pitch_length']
        self.pitch_width = model_data['pitch_width']
        self.team_rosters = model_data.get('team_rosters', {})
        self.is_trained = True

        print(f"Model loaded from {filepath}")


def load_multiple_matches(match_ids: List[str],
                          data_dir: Path) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Load data from multiple matches and combine them.

    Args:
        match_ids: List of match IDs
        data_dir: Directory containing data files

    Returns:
        Tuple of (combined_events_df, combined_tracking_dict, combined_team_rosters)
    """
    all_events = []
    all_tracking = {}
    all_team_rosters = {}

    model = DribblingModel()

    for match_id in match_ids:
        events_df, tracking_dict, team_roster = model.load_match_data(match_id, data_dir)

        # Add match_id to events
        events_df['match_id'] = match_id
        all_events.append(events_df)

        # Store tracking with match prefix to avoid frame collisions
        for frame, data in tracking_dict.items():
            all_tracking[f"{match_id}_{frame}"] = data

        # Store team rosters
        all_team_rosters[match_id] = team_roster

    combined_events = pd.concat(all_events, ignore_index=True)

    return combined_events, all_tracking, all_team_rosters
