"""
EPV Calculator - Approach B (On-Demand with Caching)

This module implements Expected Possession Value (EPV) calculation using:
- xG model for shooting evaluation
- Passing model for pass evaluation
- Dribbling model for dribble evaluation
- Pitch control for spatial context
- Player
individuality for skill-adjusted predictions

The system computes EPV on-demand for any game state, using exact tracking data
and recursively evaluating action values.
"""

import numpy as np
import pandas as pd
import pickle
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from functools import lru_cache
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from pitch_control import PitchControlRunner, load_match_meta_robust, load_tracking_jsonl
from dribbling_model import DribblingModel
from xg_model import DampedXGModel


class EPVCalculator:
    """
    Calculate Expected Possession Value using trained models.

    Uses on-demand computation with caching for efficiency.
    """

    def __init__(
        self,
        xg_model_path: Path,
        passing_model_path: Path,
        dribbling_model_path: Path,
        xg_skills_path: Path,
        passing_skills_path: Path,
        dribbling_skills_path: Path,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0
    ):
        """
        Initialize EPV calculator with trained models.

        Args:
            xg_model_path: Path to trained xG model (.pkl)
            passing_model_path: Path to trained passing model (.pkl)
            dribbling_model_path: Path to trained dribbling model (.pkl)
            xg_skills_path: Path to player finishing skills (.csv)
            passing_skills_path: Path to player passing skills (.csv)
            dribbling_skills_path: Path to player dribbling skills (.csv)
            pitch_length: Pitch length in meters
            pitch_width: Pitch width in meters
        """
        print("Initializing EPV Calculator...")

        # Load models (they're stored as dicts with 'model' and 'feature_cols' keys)
        print("  Loading xG model...")
        with open(xg_model_path, 'rb') as f:
            xg_obj = pickle.load(f)
            if isinstance(xg_obj, dict):
                self.xg_model = xg_obj['model']
                self.xg_feature_cols = xg_obj['feature_cols']
            else:
                self.xg_model = xg_obj
                self.xg_feature_cols = None
        if isinstance(self.xg_model, DampedXGModel):
            print(
                f"  ✓ xG dampening enabled "
                f"(distance>{self.xg_model.distance_threshold:.1f}m, "
                f"angle<{self.xg_model.angle_threshold:.1f}°)"
            )
        else:
            print("  ⚠️  xG dampening not enabled (raw model)")

        print("  Loading passing model...")
        with open(passing_model_path, 'rb') as f:
            passing_obj = pickle.load(f)
            if isinstance(passing_obj, dict):
                self.passing_model = passing_obj['model']
                self.passing_feature_cols = passing_obj['feature_cols']
            else:
                self.passing_model = passing_obj
                self.passing_feature_cols = None

        print("  Loading dribbling model...")
        try:
            # DribblingModel is a class, not a direct pickle object
            self.dribbling_model = DribblingModel()
            self.dribbling_model.load_model(dribbling_model_path)
            print("  ✓ Dribbling model loaded successfully")
        except Exception as e:
            print(f"  ⚠️  Could not load dribbling model: {e}")
            print("     Using simplified dribble evaluation")
            self.dribbling_model = None

        # Load player skills
        print("  Loading player skills...")
        self.finishing_skills = self._load_skills(xg_skills_path, 'player_finishing_skill')
        self.passing_skills = self._load_skills(passing_skills_path, 'player_passing_skill')
        self.dribbling_skills = self._load_skills(dribbling_skills_path, 'player_dribbling_skill')

        # Pitch dimensions
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.goal_x = pitch_length / 2
        self.goal_y = 0

        # Pitch control runner (set per match)
        self.pc_runner: Optional[PitchControlRunner] = None
        self.current_frame_data = None
        self.current_team_roster = None
        self.match_meta = None

        # Cache for EPV values (reset per frame)
        self.cache = {}

        print("✅ EPV Calculator initialized")

    def _load_skills(self, path: Path, skill_column: str) -> Dict[int, float]:
        """Load player skills from CSV."""
        if not path.exists():
            print(f"  ⚠️  Skill file not found: {path}, using defaults")
            return {}

        df = pd.read_csv(path)

        # Try different column name variations
        for col in [skill_column, 'player_RE', 'skill']:
            if col in df.columns and 'player_id' in df.columns:
                return dict(zip(df['player_id'], df[col]))

        print(f"  ⚠️  Could not find skill column in {path}")
        return {}

    def set_match_context(
        self,
        match_json_path: Path,
        tracking_path: Path,
        events_df: pd.DataFrame
    ):
        """
        Set the match context for EPV calculation.

        Args:
            match_json_path: Path to match metadata JSON
            tracking_path: Path to tracking JSONL
            events_df: DataFrame with event data for team roster
        """
        print(f"Setting match context...")

        # Load pitch control runner
        meta = load_match_meta_robust(match_json_path)
        frames = load_tracking_jsonl(tracking_path)
        self.match_meta = meta
        self.pc_runner = PitchControlRunner(meta, frames)

        # Build team roster
        self.current_team_roster = self._build_team_roster(events_df)

        print(f"  ✓ Pitch control ready ({len(frames)} frames)")
        print(f"  ✓ Team roster: {len(self.current_team_roster)} teams")

    def _build_team_roster(self, events_df: pd.DataFrame) -> Dict[int, set]:
        """Build team roster from events."""
        team_roster = {}
        player_teams = events_df[['player_id', 'team_id']].dropna().drop_duplicates()

        for _, row in player_teams.iterrows():
            team_id = int(row['team_id'])
            player_id = int(row['player_id'])

            if team_id not in team_roster:
                team_roster[team_id] = set()

            team_roster[team_id].add(player_id)

        return team_roster

    def _opposite_side(self, side: str) -> str:
        if "left_to_right" in side:
            return "right_to_left"
        if "right_to_left" in side:
            return "left_to_right"
        return side

    def _side_to_goal_x(self, side: str) -> float:
        if "right_to_left" in side:
            return -self.pitch_length / 2
        if "left_to_right" in side:
            return self.pitch_length / 2
        return self.goal_x

    def _resolve_goal_x(self, team_id: int, frame_data: Dict) -> float:
        if not self.match_meta:
            return self.goal_x

        period = frame_data.get("period")
        try:
            period = int(period)
        except Exception:
            period = 1

        home_sides = self.match_meta.get("home_sides") or []
        if home_sides:
            home_side = home_sides[period - 1] if period - 1 < len(home_sides) else home_sides[0]
        else:
            home_side = "left_to_right"

        home_id = self.match_meta.get("home_id")
        away_id = self.match_meta.get("away_id")

        side = home_side
        if home_id is not None and team_id is not None:
            try:
                if int(team_id) == int(home_id):
                    side = home_side
                elif away_id is not None and int(team_id) == int(away_id):
                    side = self._opposite_side(home_side)
            except Exception:
                side = home_side

        return self._side_to_goal_x(side)

    def reset_cache(self):
        """Reset cache (call when frame changes)."""
        self.cache = {}

    def get_epv(
        self,
        x: float,
        y: float,
        frame: int,
        player_id: int,
        team_id: int,
        tracking_dict: Dict,
        depth: int = 0,
        max_depth: int = 3
    ) -> float:
        """
        Calculate EPV at location (x, y) for given game state.

        Args:
            x: X coordinate (meters, center-based: -52.5 to 52.5)
            y: Y coordinate (meters, center-based: -34 to 34)
            frame: Frame number for tracking data
            player_id: Player with possession
            team_id: Team with possession
            tracking_dict: Dictionary mapping frame -> tracking data
            depth: Current recursion depth
            max_depth: Maximum recursion depth

        Returns:
            EPV (expected goals) from this state
        """
        # Base case: stop recursion
        if depth >= max_depth:
            return 0.0

        # Check cache
        cache_key = (round(x, 1), round(y, 1), depth)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Get frame data
        frame_data = tracking_dict.get(frame, {})

        # Evaluate each action
        q_shoot, _ = self.evaluate_shoot(x, y, frame, frame_data, player_id, team_id)
        q_pass = self.evaluate_best_pass(x, y, frame, frame_data, player_id, team_id, tracking_dict, depth)
        q_dribble = self.evaluate_best_dribble(x, y, frame, frame_data, player_id, team_id, tracking_dict, depth)

        # EPV is best action value
        epv = max(q_shoot, q_pass, q_dribble)

        # Cache result
        self.cache[cache_key] = epv

        return epv

    def _point_to_segment_distance(
        self,
        px: float, py: float,
        x1: float, y1: float,
        x2: float, y2: float
    ) -> float:
        """Compute distance from point (px, py) to line segment (x1,y1)-(x2,y2)."""
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return np.sqrt((px - x1)**2 + (py - y1)**2)
        t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)

    def _sigmoid(self, x: float, center: float = 0.0, steepness: float = 1.0) -> float:
        """Smooth sigmoid function: 1 / (1 + exp(-steepness * (x - center)))."""
        return 1.0 / (1.0 + np.exp(-steepness * (x - center)))

    def _get_skills_for_player(self, frame_data: Dict, player_id: int) -> Tuple[float, float, float]:
        """Return (finishing, passing, dribbling) in [0,1] from frame_data skill_multipliers. Default 0.5 if missing."""
        if not frame_data or "player_data" not in frame_data:
            return (0.5, 0.5, 0.5)
        for p in frame_data["player_data"]:
            if p.get("player_id") == player_id:
                sm = p.get("skill_multipliers") or {}
                return (
                    float(sm.get("finishing", 0.5)),
                    float(sm.get("passing", 0.5)),
                    float(sm.get("dribbling", 0.5)),
                )
        return (0.5, 0.5, 0.5)

    def _pass_lane_and_receiver_defender_distances(
        self,
        x_origin: float,
        y_origin: float,
        x_dest: float,
        y_dest: float,
        frame_data: Dict,
        team_id: int,
        default_lane: float = 50.0,
        default_recv: float = 50.0
    ) -> Tuple[float, float]:
        """
        Min defender distance to pass segment and min defender distance to receiver.
        All coordinates must be in the same system (center-based: origin, dest, and
        frame_data['player_data'] x,y are all center coords).
        Returns (lane_min_def_dist, recv_min_def_dist).
        """
        if not frame_data or "player_data" not in frame_data or not self.current_team_roster:
            return (default_lane, default_recv)
        opponent_team_ids = [tid for tid in self.current_team_roster.keys() if tid != team_id]
        if not opponent_team_ids:
            return (default_lane, default_recv)
        opponent_players = set()
        for opp_team_id in opponent_team_ids:
            opponent_players.update(self.current_team_roster[opp_team_id])
        lane_min = None
        recv_min = None
        for player in frame_data.get("player_data", []):
            if not player.get("is_detected", False) or player["player_id"] not in opponent_players:
                continue
            px, py = player["x"], player["y"]
            d_lane = self._point_to_segment_distance(px, py, x_origin, y_origin, x_dest, y_dest)
            d_recv = np.sqrt((px - x_dest) ** 2 + (py - y_dest) ** 2)
            if lane_min is None or d_lane < lane_min:
                lane_min = d_lane
            if recv_min is None or d_recv < recv_min:
                recv_min = d_recv
        return (
            float(lane_min) if lane_min is not None else default_lane,
            float(recv_min) if recv_min is not None else default_recv,
        )

    def _pass_risk_from_distances(
        self,
        lane_min_def_dist: float,
        recv_min_def_dist: float,
        sigma_lane: float = 1.25,
        sigma_recv: float = 2.0
    ) -> float:
        """
        Convert lane and receiver defender distances to a [0,1] risk.
        risk = clamp(0.7 * lane_risk + 0.3 * receiver_pressure), where each term
        is exp(-(d**2)/(2*sigma**2)) so small distance -> high risk.
        """
        lane_risk = np.exp(-(lane_min_def_dist ** 2) / (2.0 * sigma_lane ** 2))
        recv_pressure = np.exp(-(recv_min_def_dist ** 2) / (2.0 * sigma_recv ** 2))
        return float(np.clip(0.7 * lane_risk + 0.3 * recv_pressure, 0.0, 1.0))

    def evaluate_shoot(
        self,
        x: float,
        y: float,
        frame: int,
        frame_data: Dict,
        player_id: int,
        team_id: int
    ) -> tuple[float, Dict]:
        """
        Evaluate shooting action using xG model with defender-aware pressure multiplier.

        Returns:
            Tuple of (q_shoot, explain_dict) where explain_dict contains:
            - base_xg: raw model output
            - shot_min_def_dist: minimum distance to any defender
            - shot_blocked: True if defender blocks shot line
            - shot_pressure_multiplier: applied multiplier
        """
        goal_x = self._resolve_goal_x(team_id, frame_data)
        goal_y = self.goal_y

        # Calculate shooting features
        distance_to_goal = np.sqrt((x - goal_x)**2 + (y - goal_y)**2)
        ang1 = np.degrees(np.arctan2(3.66 - y, goal_x - x))
        ang2 = np.degrees(np.arctan2(-3.66 - y, goal_x - x))
        angle_to_goal = abs(ang2 - ang1)
        if angle_to_goal > 180:
            angle_to_goal = 360 - angle_to_goal

        # Count defenders in shot triangle
        defenders_in_triangle = self._count_defenders_in_triangle(
            x, y, frame_data, team_id, goal_x=goal_x
        )

        # Count defenders within 3m
        defenders_within_3m = self._count_defenders_within_distance(
            x, y, frame_data, team_id, distance=3.0
        )

        # Get player finishing skill
        player_finishing_skill = self.finishing_skills.get(player_id, 0.0)

        # Pitch control at shot location
        pitch_control = 0.5
        if self.pc_runner and frame > 0:
            try:
                pitch_control = float(self.pc_runner.pc_at_point(frame, x, y))
            except Exception:
                pitch_control = 0.5

        # Other features (simplified - you can add more)
        speed_avg = 5.0  # Default
        inside_defensive_shape = int(x * goal_x < 0)  # Own half relative to goal
        last_defensive_line_x = -10.0 if goal_x > 0 else 10.0  # Default
        last_defensive_line_height = 20.0  # Default
        penalty_area = int(abs(x - goal_x) < 16.5 and abs(y) < 20.15)
        distance_covered = 0.0  # Static shot
        trajectory_angle = 0.0  # Straight

        # Create feature dict
        feature_dict = {
            'distance_to_goal': distance_to_goal,
            'angle_to_goal': angle_to_goal,
            'defenders_in_triangle': defenders_in_triangle,
            'defenders_within_3m': defenders_within_3m,
            'player_finishing_skill': player_finishing_skill,
            'pitch_control': pitch_control,
            'speed_avg': speed_avg,
            'penalty_area': penalty_area,
            'inside_defensive_shape': inside_defensive_shape,
            'last_defensive_line_x': last_defensive_line_x,
            'last_defensive_line_height': last_defensive_line_height,
            'distance_covered': distance_covered,
            'trajectory_angle': trajectory_angle,
        }

        # Build features in correct order (use saved feature columns if available)
        if self.xg_feature_cols:
            features = np.array([[feature_dict[col] for col in self.xg_feature_cols]])
        else:
            features = pd.DataFrame([feature_dict]).values

        # Predict base xG
        base_xg = 0.0
        try:
            base_xg = float(self.xg_model.predict_proba(features)[0, 1])
        except Exception as e:
            print(f"  ⚠️  xG prediction error: {e}")
            return (0.0, {
                "base_xg": 0.0,
                "shot_min_def_dist": float('inf'),
                "shot_blocked": False,
                "shot_pressure_multiplier": 1.0
            })

        # Defender-aware pressure computation
        shot_min_def_dist = float('inf')
        shot_blocked = False
        block_radius = 1.5

        if frame_data and 'player_data' in frame_data and self.current_team_roster:
            opponent_team_ids = [tid for tid in self.current_team_roster.keys() if tid != team_id]
            opponent_players = set()
            for opp_team_id in opponent_team_ids:
                opponent_players.update(self.current_team_roster[opp_team_id])

            for player in frame_data.get('player_data', []):
                if not player.get('is_detected', False):
                    continue
                if player['player_id'] not in opponent_players:
                    continue

                px = player['x']
                py = player['y']

                # Compute distance to shooter
                dist_to_shooter = np.sqrt((px - x)**2 + (py - y)**2)
                if dist_to_shooter < shot_min_def_dist:
                    shot_min_def_dist = dist_to_shooter

                # Check if defender blocks shot line (point-to-segment distance)
                # Only count if defender is between shooter and goal
                if goal_x > x:
                    between = x <= px <= goal_x
                else:
                    between = goal_x <= px <= x

                if between:
                    dist_to_line = self._point_to_segment_distance(px, py, x, y, goal_x, goal_y)
                    if dist_to_line <= block_radius:
                        shot_blocked = True

        if shot_min_def_dist == float('inf'):
            shot_min_def_dist = 100.0  # No defenders found

        # Compute shot_pressure_multiplier using sigmoid
        # Multiplier decreases as defenders get closer
        # Steeper decrease if shot is blocked
        base_multiplier = self._sigmoid(shot_min_def_dist - 3.0, center=0.0, steepness=0.8)
        if shot_blocked:
            blocked_penalty = 0.3
        else:
            blocked_penalty = 0.0
        shot_pressure_multiplier = max(0.1, base_multiplier - blocked_penalty)

        q_shoot = base_xg * shot_pressure_multiplier
        finishing_skill = self._get_skills_for_player(frame_data, player_id)[0]
        mult_finish = 0.85 + 0.30 * finishing_skill
        q_shoot *= mult_finish
        q_shoot = float(np.clip(q_shoot, 0.0, 1.0))

        explain = {
            "base_xg": float(base_xg),
            "shot_min_def_dist": float(shot_min_def_dist),
            "shot_blocked": bool(shot_blocked),
            "shot_pressure_multiplier": float(shot_pressure_multiplier),
            "mult_finish": float(mult_finish),
        }

        return (float(q_shoot), explain)

    def evaluate_best_pass(
        self,
        x: float,
        y: float,
        frame: int,
        frame_data: Dict,
        player_id: int,
        team_id: int,
        tracking_dict: Dict,
        depth: int,
        return_dest: bool = False,
        return_explain: bool = False
    ):
        """
        Evaluate best pass with defender-aware risk (lane + receiver pressure).
        All coordinates (x, y, dest_*, frame_data positions) are in the same
        center-based system used by _point_to_segment_distance.
        adjusted = base * (1 - risk). Safety override: if any candidate has
        risk <= 0.25 and adjusted >= 90% of best adjusted, choose the safest
        (lowest risk) among those; else choose best adjusted.
        """
        best_q_adj = 0.0
        best_dest = None
        best_base = 0.0
        best_risk = 0.0
        candidates: List[Dict] = []

        destinations = self._get_teammate_positions(frame_data, team_id, player_id)
        if not destinations:
            if return_explain:
                return (0.0, {
                    "best_pass_target": None,
                    "best_pass_risk": 0.0,
                    "best_pass_base": 0.0,
                    "best_pass_adjusted": 0.0,
                    "top_candidates": [],
                })
            return (0.0, best_dest) if return_dest else 0.0

        for receiver_id, dest_x, dest_y in destinations:
            if not self._is_valid_location(dest_x, dest_y):
                continue
            q_base = self.evaluate_single_pass(
                x, y, dest_x, dest_y,
                frame, frame_data, player_id, team_id,
                tracking_dict, depth
            )
            lane_min_def_dist, recv_min_def_dist = self._pass_lane_and_receiver_defender_distances(
                x, y, dest_x, dest_y, frame_data, team_id
            )
            risk = self._pass_risk_from_distances(lane_min_def_dist, recv_min_def_dist)
            q_adj = q_base * (1.0 - risk)
            candidates.append({
                "receiver_id": receiver_id,
                "receiver_xy": (float(dest_x), float(dest_y)),
                "base": float(q_base),
                "lane_min_def_dist": float(lane_min_def_dist),
                "recv_min_def_dist": float(recv_min_def_dist),
                "risk": float(risk),
                "adjusted": float(q_adj),
            })

        if not candidates:
            if return_explain:
                return (0.0, {
                    "best_pass_target": None,
                    "best_pass_risk": 0.0,
                    "best_pass_base": 0.0,
                    "best_pass_adjusted": 0.0,
                    "top_candidates": [],
                })
            return (0.0, best_dest) if return_dest else 0.0

        best_adj = max(c["adjusted"] for c in candidates)
        safe = [c for c in candidates if c["risk"] <= 0.25 and c["adjusted"] >= 0.9 * best_adj]
        if safe:
            chosen = min(safe, key=lambda c: c["risk"])
        else:
            chosen = max(candidates, key=lambda c: c["adjusted"])
        best_q_adj = chosen["adjusted"]
        best_dest = chosen["receiver_xy"]
        best_base = chosen["base"]
        best_risk = chosen["risk"]

        passing_skill = self._get_skills_for_player(frame_data, player_id)[1]
        mult_pass = 0.85 + 0.30 * passing_skill
        best_q_adj = float(np.clip(best_q_adj * mult_pass, 0.0, 1.0))

        if return_explain:
            candidates.sort(key=lambda c: c["adjusted"], reverse=True)
            top_candidates = candidates[:10]
            explain = {
                "best_pass_target": {"x": best_dest[0], "y": best_dest[1]} if best_dest else None,
                "best_pass_risk": float(best_risk),
                "best_pass_base": float(best_base),
                "best_pass_adjusted": float(best_q_adj),
                "mult_pass": float(mult_pass),
                "top_candidates": [
                    {
                        "receiver_id": c["receiver_id"],
                        "receiver_xy": c["receiver_xy"],
                        "base": c["base"],
                        "lane_min_def_dist": c["lane_min_def_dist"],
                        "recv_min_def_dist": c["recv_min_def_dist"],
                        "risk": c["risk"],
                        "adjusted": c["adjusted"],
                    }
                    for c in top_candidates
                ],
            }
            return (best_q_adj, explain)
        if return_dest:
            return (best_q_adj, (best_dest[0], best_dest[1]))
        return best_q_adj

    def evaluate_single_pass(
        self,
        x_origin: float,
        y_origin: float,
        x_dest: float,
        y_dest: float,
        frame: int,
        frame_data: Dict,
        player_id: int,
        team_id: int,
        tracking_dict: Dict,
        depth: int
    ) -> float:
        """
        Evaluate a single pass to specific destination.

        Returns:
            Q-value for this pass
        """
        # Calculate pass features (all in center coordinates).
        #
        # IMPORTANT: "forward progress" must be measured relative to the *attacking goal*.
        # The rest of the system supports both directions via `_resolve_goal_x`, but this
        # feature was previously hard-coded as (x_dest - x_origin), which implicitly
        # assumes attacking to +X. That makes away-team recommendations look incorrect
        # (away attacks toward -X in period 1 in our web usage).
        pass_distance = np.sqrt((x_dest - x_origin)**2 + (y_dest - y_origin)**2)
        pass_angle = np.degrees(np.arctan2(y_dest - y_origin, x_dest - x_origin))
        goal_x = self._resolve_goal_x(team_id, frame_data)
        forward_progress = (x_dest - x_origin) if goal_x >= 0 else (x_origin - x_dest)

        # Pitch control at origin and destination
        pc_origin = 0.5
        pc_dest = 0.5
        pc_path_min = 0.5
        if self.pc_runner and frame > 0:
            try:
                pc_origin = self.pc_runner.pc_at_point(frame, x_origin, y_origin)
                pc_dest = self.pc_runner.pc_at_point(frame, x_dest, y_dest)
                pc_path_min = self._pitch_control_path_min(
                    x_origin, y_origin, x_dest, y_dest, frame
                )
            except:
                pass

        # Defender proximity
        defenders_near_origin = self._count_defenders_within_distance(
            x_origin, y_origin, frame_data, team_id, distance=3.0
        )
        defenders_near_dest = self._count_defenders_within_distance(
            x_dest, y_dest, frame_data, team_id, distance=3.0
        )
        defenders_in_lane = self._count_defenders_in_passing_lane(
            x_origin, y_origin, x_dest, y_dest, frame_data, team_id
        )
        min_defender_dist_to_lane = self._min_defender_dist_to_passing_lane(
            x_origin, y_origin, x_dest, y_dest, frame_data, team_id
        )

        # Player passing skill
        player_passing_skill = self.passing_skills.get(player_id, 0.0)

        # Calculate ACTUAL defensive line from tracking data
        # Get defending team (the team that DOESN'T have possession)
        all_teams = list(self.current_team_roster.keys())
        defending_team_id = [t for t in all_teams if t != team_id][0] if len(all_teams) > 1 else team_id
        last_defensive_line_x, last_defensive_line_height = self._calculate_defensive_line(
            frame_data, defending_team_id
        )

        # Other features
        speed_avg = 8.0  # Default pass speed
        # Treat "inside_defensive_shape" as being behind the opponent line relative
        # to the attacking direction (not absolute X).
        inside_defensive_shape = int(x_origin < last_defensive_line_x) if goal_x >= 0 else int(x_origin > last_defensive_line_x)

        # Create feature dict
        feature_dict = {
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
            'speed_avg': speed_avg,
            'inside_defensive_shape': inside_defensive_shape,
            'last_defensive_line_x': last_defensive_line_x,
            'last_defensive_line_height': last_defensive_line_height,
        }

        # Build features in correct order (use saved feature columns if available)
        if self.passing_feature_cols:
            features = np.array([[feature_dict[col] for col in self.passing_feature_cols]])
        else:
            features = pd.DataFrame([feature_dict]).values

        # Predict pass success probability
        try:
            p_success = self.passing_model.predict_proba(features)[0, 1]
        except Exception as e:
            print(f"  ⚠️  Passing prediction error: {e}")
            p_success = 0.5

        # Value if pass succeeds (RECURSIVE)
        v_success = self.get_epv(
            x_dest, y_dest, frame + 10, player_id, team_id,
            tracking_dict, depth + 1
        )

        # Value if pass fails (turnover)
        v_fail = -0.05  # Opponent gets ball, moderate threat

        # Q-value for this pass
        q = p_success * v_success + (1 - p_success) * v_fail

        return q

    def evaluate_best_dribble(
        self,
        x: float,
        y: float,
        frame: int,
        frame_data: Dict,
        player_id: int,
        team_id: int,
        tracking_dict: Dict,
        depth: int,
        return_explain: bool = False
    ):
        """
        Evaluate best dribble with defender-aware pressure multiplier.
        Uses theta (facing direction) for open-space check. Returns float or
        (q_dribble, explain_dict) when return_explain=True.
        """
        theta = 0.0
        if frame_data and "player_data" in frame_data:
            for p in frame_data["player_data"]:
                if p.get("player_id") == player_id:
                    theta = float(p.get("theta", 0.0))
                    break

        dribble_min_def_dist = self._dribble_min_defender_distance(
            x, y, frame_data, team_id, default=50.0
        )
        dribble_nearby_defenders = self._count_defenders_within_distance(
            x, y, frame_data, team_id, distance=5.0
        )
        dribble_open_space_m = self._dribble_open_space_m(
            x, y, theta, frame_data, team_id,
            sample_distances=[2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
            safe_dist=1.5
        )

        m = 1.0
        m *= self._sigmoid(dribble_min_def_dist - 3.0, center=0.0, steepness=1.0)
        m *= 1.0 / (1.0 + 0.25 * dribble_nearby_defenders)
        m *= float(np.clip(0.6 + 0.05 * dribble_open_space_m, 0.6, 1.2))
        dribble_pressure_multiplier = float(np.clip(m, 0.1, 1.5))

        best_q = 0.0
        destinations = self._sample_dribble_destinations(x, y, team_id=team_id, frame_data=frame_data)
        for dest_x, dest_y in destinations:
            if not self._is_valid_location(dest_x, dest_y):
                continue
            distance = np.sqrt((dest_x - x)**2 + (dest_y - y)**2)
            if distance < 1.0:
                continue
            if self.dribbling_model and self.dribbling_model.is_trained:
                p_success = self._evaluate_dribble_with_model(
                    x, y, dest_x, dest_y, frame, frame_data, player_id, team_id
                )
            else:
                p_success = 0.7 if distance < 5 else 0.5
            v_success = self.get_epv(
                dest_x, dest_y, frame + 20, player_id, team_id,
                tracking_dict, depth + 1
            )
            v_fail = -0.08
            q = p_success * v_success + (1 - p_success) * v_fail
            best_q = max(best_q, q)

        base_dribble_value = best_q
        q_dribble = base_dribble_value * dribble_pressure_multiplier
        dribbling_skill = self._get_skills_for_player(frame_data, player_id)[2]
        mult_dribble = 0.85 + 0.30 * dribbling_skill
        q_dribble *= mult_dribble
        q_dribble = float(np.clip(q_dribble, 0.0, 1.0))

        if return_explain:
            explain = {
                "base_dribble": float(base_dribble_value),
                "dribble_min_def_dist": float(dribble_min_def_dist),
                "dribble_nearby_defenders": int(dribble_nearby_defenders),
                "dribble_open_space_m": float(dribble_open_space_m),
                "dribble_pressure_multiplier": float(dribble_pressure_multiplier),
                "mult_dribble": float(mult_dribble),
            }
            return (q_dribble, explain)
        return q_dribble

    def _get_teammate_positions(
        self,
        frame_data: Dict,
        team_id: int,
        player_id: int
    ) -> List[Tuple[int, float, float]]:
        """Get actual positions of teammates from tracking data. Returns (receiver_id, x, y)."""
        teammates: List[Tuple[int, float, float]] = []

        if "player_data" not in frame_data:
            return teammates

        team_roster = self.current_team_roster.get(team_id, set())
        for player in frame_data["player_data"]:
            pid = player["player_id"]
            if pid not in team_roster or pid == player_id:
                continue
            teammates.append((pid, player["x"], player["y"]))
        return teammates

    def _calculate_defensive_line(
        self,
        frame_data: Dict,
        defending_team_id: int
    ) -> Tuple[float, float]:
        """
        Calculate defensive line position from actual tracking data.

        Returns:
            (line_x, line_height): X position and vertical spread of defensive line
        """
        if 'player_data' not in frame_data or not frame_data['player_data']:
            return (-10.0, 20.0)  # Default values

        # Get all opposing team rosters
        all_teams = set(self.current_team_roster.keys())
        defending_roster = self.current_team_roster.get(defending_team_id, set())

        defender_positions = []
        for player in frame_data['player_data']:
            if player['player_id'] in defending_roster:
                defender_positions.append((player['x'], player['y']))

        if not defender_positions:
            return (-10.0, 20.0)

        # Defensive line X = average of 4 most forward defenders
        defender_x = sorted([x for x, y in defender_positions], reverse=True)[:4]
        line_x = np.mean(defender_x) if defender_x else -10.0

        # Defensive line height = std of Y positions
        defender_y = [y for x, y in defender_positions]
        line_height = np.std(defender_y) if len(defender_y) > 1 else 20.0

        return (float(line_x), float(line_height))

    def _sample_pass_destinations(
        self,
        x: float,
        y: float,
        frame_data: Dict,
        team_id: int,
        player_id: int,
        n: int = 12
    ) -> List[Tuple[float, float]]:
        """
        Sample pass destinations using ACTUAL teammate positions + strategic positions.

        This combines:
        1. Actual teammate locations (primary)
        2. Strategic positions forward (if no teammates there)
        """
        destinations = []

        # PRIMARY: Get actual teammate positions
        teammates = self._get_teammate_positions(frame_data, team_id, player_id)

        # Add all forward teammates (x > current_x)
        forward_teammates = [(tx, ty) for tx, ty in teammates if tx > x]
        destinations.extend(forward_teammates)

        # Add lateral/backward teammates if close
        nearby_teammates = [
            (tx, ty) for tx, ty in teammates
            if abs(tx - x) < 15 and abs(ty - y) < 15 and (tx, ty) not in destinations
        ]
        destinations.extend(nearby_teammates)

        # SECONDARY: Add strategic grid positions (in case teammates are clustered)
        strategic_positions = []
        for dist in [10, 20]:
            strategic_positions.append((x + dist, y))
            strategic_positions.append((x + dist, y + 7))
            strategic_positions.append((x + dist, y - 7))

        # Add strategic positions that aren't too close to existing destinations
        for sx, sy in strategic_positions:
            if not any(np.sqrt((sx-dx)**2 + (sy-dy)**2) < 5 for dx, dy in destinations):
                destinations.append((sx, sy))

        # Return top n destinations (prioritize forward positions)
        destinations = sorted(destinations, key=lambda p: -p[0])  # Sort by x (forward)
        return destinations[:n]

    def _sample_dribble_destinations(
        self,
        x: float,
        y: float,
        team_id: Optional[int] = None,
        frame_data: Optional[Dict] = None,
        n: int = 4,
    ) -> List[Tuple[float, float]]:
        """Sample reasonable dribble destinations.

        NOTE: These are *candidate* end points for the EPV recursion when evaluating
        dribbles. They must align with the attacking direction; otherwise away-team
        "dribble forward" options get sampled in the wrong direction and look wonky.
        """
        destinations = []

        # Forward dribbles (relative to the attacking goal for this team + period).
        gx = self.goal_x
        if team_id is not None:
            gx = self._resolve_goal_x(int(team_id), frame_data or {})
        direction = 1.0 if gx >= 0 else -1.0
        for dist in [3, 5]:
            destinations.append((x + direction * dist, y))
            destinations.append((x + direction * dist, y + 2))
            destinations.append((x + direction * dist, y - 2))

        return destinations[:n]

    def _is_valid_location(self, x: float, y: float) -> bool:
        """Check if location is within pitch bounds."""
        return (
            -self.pitch_length/2 <= x <= self.pitch_length/2 and
            -self.pitch_width/2 <= y <= self.pitch_width/2
        )

    def _count_defenders_in_triangle(
        self,
        x: float,
        y: float,
        frame_data: Dict,
        team_id: int,
        goal_x: Optional[float] = None
    ) -> int:
        """Count defenders in shot triangle (between shooter and goal)."""
        if not frame_data or 'player_data' not in frame_data:
            return 0

        if not self.current_team_roster:
            return 0

        # Get opponent players
        opponent_team_ids = [tid for tid in self.current_team_roster.keys() if tid != team_id]
        if not opponent_team_ids:
            return 0

        opponent_players = set()
        for opp_team_id in opponent_team_ids:
            opponent_players.update(self.current_team_roster[opp_team_id])

        gx = self.goal_x if goal_x is None else goal_x

        # Goal posts
        goal_left_y = -3.66
        goal_right_y = 3.66

        count = 0
        for player in frame_data.get('player_data', []):
            if not player.get('is_detected', False):
                continue

            player_id = player['player_id']
            if player_id not in opponent_players:
                continue

            px = player['x']
            py = player['y']

            # Check if in triangle (simplified)
            # Between shooter and goal (x-wise)
            if not (min(x, gx) <= px <= max(x, gx)):
                continue

            # Between goal posts (y-wise, with margin)
            if not (goal_left_y - 2 <= py <= goal_right_y + 2):
                continue

            count += 1

        return count

    def _count_defenders_within_distance(
        self,
        x: float,
        y: float,
        frame_data: Dict,
        team_id: int,
        distance: float = 3.0
    ) -> int:
        """Count defenders within specified distance."""
        if not frame_data or 'player_data' not in frame_data:
            return 0

        if not self.current_team_roster:
            return 0

        # Get opponent players
        opponent_team_ids = [tid for tid in self.current_team_roster.keys() if tid != team_id]
        if not opponent_team_ids:
            return 0

        opponent_players = set()
        for opp_team_id in opponent_team_ids:
            opponent_players.update(self.current_team_roster[opp_team_id])

        count = 0
        for player in frame_data.get('player_data', []):
            if not player.get('is_detected', False):
                continue

            player_id = player['player_id']
            if player_id not in opponent_players:
                continue

            px = player['x']
            py = player['y']

            dist = np.sqrt((x - px)**2 + (y - py)**2)
            if dist <= distance:
                count += 1

        return count

    def _dribble_min_defender_distance(
        self,
        x: float,
        y: float,
        frame_data: Dict,
        team_id: int,
        default: float = 50.0
    ) -> float:
        """Min Euclidean distance from (x, y) to any defender."""
        if not frame_data or "player_data" not in frame_data or not self.current_team_roster:
            return default
        opponent_team_ids = [tid for tid in self.current_team_roster.keys() if tid != team_id]
        if not opponent_team_ids:
            return default
        opponent_players = set()
        for opp_team_id in opponent_team_ids:
            opponent_players.update(self.current_team_roster[opp_team_id])
        min_dist = None
        for player in frame_data.get("player_data", []):
            if not player.get("is_detected", False) or player["player_id"] not in opponent_players:
                continue
            px, py = player["x"], player["y"]
            d = np.sqrt((x - px) ** 2 + (y - py) ** 2)
            if min_dist is None or d < min_dist:
                min_dist = d
        return float(min_dist) if min_dist is not None else default

    def _dribble_open_space_m(
        self,
        x: float,
        y: float,
        theta: float,
        frame_data: Dict,
        team_id: int,
        sample_distances: Optional[List[float]] = None,
        safe_dist: float = 1.5
    ) -> float:
        """
        Farthest sampled distance along ray (x,y) + t*(cos(theta), sin(theta))
        that is in bounds and >= safe_dist from every defender.
        theta in radians. Returns 0 if no point is open.
        """
        if sample_distances is None:
            sample_distances = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
        dx = np.cos(theta)
        dy = np.sin(theta)
        opponent_players = set()
        if frame_data and "player_data" in frame_data and self.current_team_roster:
            for tid in self.current_team_roster:
                if tid != team_id:
                    opponent_players.update(self.current_team_roster[tid])
        defender_positions = []
        if frame_data and "player_data" in frame_data:
            for player in frame_data["player_data"]:
                if player.get("is_detected", False) and player["player_id"] in opponent_players:
                    defender_positions.append((player["x"], player["y"]))
        open_farthest = 0.0
        for d in sample_distances:
            px = x + d * dx
            py = y + d * dy
            if not self._is_valid_location(px, py):
                break
            if not defender_positions:
                open_farthest = d
                continue
            min_def = min(
                np.sqrt((px - ox) ** 2 + (py - oy) ** 2)
                for ox, oy in defender_positions
            )
            if min_def >= safe_dist:
                open_farthest = d
            else:
                break
        return open_farthest

    def _count_defenders_in_passing_lane(
        self,
        x_origin: float,
        y_origin: float,
        x_dest: float,
        y_dest: float,
        frame_data: Dict,
        team_id: int,
        lane_width: float = 2.0
    ) -> int:
        """Count defenders in passing lane."""
        if not frame_data or 'player_data' not in frame_data:
            return 0

        if not self.current_team_roster:
            return 0

        # Get opponent players
        opponent_team_ids = [tid for tid in self.current_team_roster.keys() if tid != team_id]
        if not opponent_team_ids:
            return 0

        opponent_players = set()
        for opp_team_id in opponent_team_ids:
            opponent_players.update(self.current_team_roster[opp_team_id])

        # Pass vector
        pass_dx = x_dest - x_origin
        pass_dy = y_dest - y_origin
        pass_length = np.sqrt(pass_dx**2 + pass_dy**2)

        if pass_length < 0.1:
            return 0

        count = 0
        for player in frame_data.get('player_data', []):
            if not player.get('is_detected', False):
                continue

            player_id = player['player_id']
            if player_id not in opponent_players:
                continue

            px = player['x']
            py = player['y']

            # Vector from origin to player
            to_player_x = px - x_origin
            to_player_y = py - y_origin

            # Project onto pass direction
            projection = (to_player_x * pass_dx + to_player_y * pass_dy) / pass_length

            # Check if between origin and destination
            if projection < 0 or projection > pass_length:
                continue

            # Perpendicular distance
            perp_dist = abs(to_player_x * pass_dy - to_player_y * pass_dx) / pass_length

            if perp_dist <= lane_width:
                count += 1

        return count

    def _min_defender_dist_to_passing_lane(
        self,
        x_origin: float,
        y_origin: float,
        x_dest: float,
        y_dest: float,
        frame_data: Dict,
        team_id: int,
        default_distance: float = 50.0
    ) -> float:
        """Compute minimum perpendicular distance of any defender to the pass lane."""
        if not frame_data or 'player_data' not in frame_data:
            return default_distance

        if not self.current_team_roster:
            return default_distance

        opponent_team_ids = [tid for tid in self.current_team_roster.keys() if tid != team_id]
        if not opponent_team_ids:
            return default_distance

        opponent_players = set()
        for opp_team_id in opponent_team_ids:
            opponent_players.update(self.current_team_roster[opp_team_id])

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

    def _pitch_control_path_min(
        self,
        x_origin: float,
        y_origin: float,
        x_dest: float,
        y_dest: float,
        frame: int,
        n_samples: int = 5
    ) -> float:
        """Compute minimum pitch control along the pass line segment."""
        if not self.pc_runner or frame <= 0:
            return 0.5

        pass_dx = x_dest - x_origin
        pass_dy = y_dest - y_origin
        pass_length = np.sqrt(pass_dx**2 + pass_dy**2)
        if pass_length < 0.5:
            try:
                pc_origin = self.pc_runner.pc_at_point(frame, x_origin, y_origin)
                pc_dest = self.pc_runner.pc_at_point(frame, x_dest, y_dest)
                return float(min(pc_origin, pc_dest))
            except Exception:
                return 0.5

        fractions = np.linspace(0.2, 0.8, n_samples)
        values = []
        for frac in fractions:
            x = x_origin + frac * pass_dx
            y = y_origin + frac * pass_dy
            try:
                values.append(self.pc_runner.pc_at_point(frame, x, y))
            except Exception:
                continue

        return float(min(values)) if values else 0.5

    def _evaluate_dribble_with_model(
        self,
        x_start: float,
        y_start: float,
        x_end: float,
        y_end: float,
        frame: int,
        frame_data: Dict,
        player_id: int,
        team_id: int
    ) -> float:
        """
        Evaluate dribble success probability using the trained DribblingModel.

        Returns:
            Probability of dribble success (0-1)
        """
        goal_x = self._resolve_goal_x(team_id, frame_data)

        # Calculate basic features
        distance_covered = np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2)
        duration = distance_covered / 5.0  # Assume ~5 m/s dribble speed
        speed = distance_covered / duration if duration > 0 else 0.0

        dist_to_goal_start = np.sqrt((x_start - goal_x)**2 + (y_start - self.goal_y)**2)
        dist_to_goal_end = np.sqrt((x_end - goal_x)**2 + (y_end - self.goal_y)**2)
        dist_to_goal_change = dist_to_goal_end - dist_to_goal_start

        # Angle change
        angle_start = np.degrees(np.arctan2(y_start, x_start - goal_x))
        angle_end = np.degrees(np.arctan2(y_end, x_end - goal_x))
        angle_change = abs(angle_end - angle_start)
        if angle_change > 180:
            angle_change = 360 - angle_change

        # Distance from sideline
        dist_from_sideline = min(abs(y_start - self.pitch_width/2), abs(y_start + self.pitch_width/2))

        # Defender counts
        defenders_nearby_start = self._count_defenders_within_distance(
            x_start, y_start, frame_data, team_id, distance=3.0
        )
        defenders_nearby_end = self._count_defenders_within_distance(
            x_end, y_end, frame_data, team_id, distance=3.0
        )

        # Pressure (simplified - use defender proximity as proxy)
        pressure_start = min(defenders_nearby_start / 5.0, 1.0)
        pressure_end = min(defenders_nearby_end / 5.0, 1.0)

        # Pitch control
        pc_start = 0.5
        pc_end = 0.5
        if self.pc_runner and frame > 0:
            try:
                pc_start = self.pc_runner.pc_at_point(frame, x_start, y_start)
                pc_end = self.pc_runner.pc_at_point(frame, x_end, y_end)
            except:
                pass

        pitch_control_change = pc_end - pc_start

        # Build feature DataFrame
        features = pd.DataFrame([{
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
            'pitch_control_change': pitch_control_change,
            'angle_change': angle_change,
            'dist_from_sideline': dist_from_sideline,
        }])

        try:
            # Use the DribblingModel's predict method
            probs = self.dribbling_model.predict_possession_probability(features)
            return float(probs[0])
        except Exception as e:
            # Fallback to heuristic if model prediction fails
            return 0.7 if distance_covered < 5 else 0.5

    def get_best_action(
        self,
        x: float,
        y: float,
        frame: int,
        player_id: int,
        team_id: int,
        tracking_dict: Dict
    ) -> Dict:
        """
        Get the best action recommendation for current state.

        Returns:
            Dictionary with:
            - action: 'shoot', 'pass', or 'dribble'
            - epv: expected goals from this state
            - q_shoot: value of shooting
            - q_pass: value of best pass
            - q_dribble: value of best dribble
            - details: additional information
        """
        # Reset cache for fresh computation
        self.reset_cache()

        frame_data = tracking_dict.get(frame, {})

        # Evaluate each action
        q_shoot, shoot_explain = self.evaluate_shoot(x, y, frame, frame_data, player_id, team_id)
        q_pass, pass_explain = self.evaluate_best_pass(
            x, y, frame, frame_data, player_id, team_id, tracking_dict, depth=0, return_explain=True
        )
        q_dribble, dribble_explain = self.evaluate_best_dribble(
            x, y, frame, frame_data, player_id, team_id, tracking_dict, depth=0, return_explain=True
        )

        # --- Individuality: multipliers already applied at source (evaluate_shoot/pass/dribble). Build explain only. ---
        finishing_skill, passing_skill, dribbling_skill = self._get_skills_for_player(frame_data, player_id)
        mult_finish = 0.85 + 0.30 * finishing_skill
        mult_pass = 0.85 + 0.30 * passing_skill
        mult_dribble = 0.85 + 0.30 * dribbling_skill
        profile_explain = {
            "finishing": float(finishing_skill),
            "passing": float(passing_skill),
            "dribbling": float(dribbling_skill),
            "mult_finish": float(mult_finish),
            "mult_pass": float(mult_pass),
            "mult_dribble": float(mult_dribble),
            "applied_multipliers": {"shoot": mult_finish, "pass": mult_pass, "dribble": mult_dribble},
        }
        q_shoot = float(np.clip(q_shoot, 0.0, 1.0))
        q_pass = float(np.clip(q_pass, 0.0, 1.0))
        q_dribble = float(np.clip(q_dribble, 0.0, 1.0))

        # --- Contextual policy (center coords throughout) ---
        goal_x = self._resolve_goal_x(team_id, frame_data)
        goal_y = self.goal_y

        # Shoot rule: strict penalty so shooting almost never wins when blocked or high pressure
        shot_blocked = shoot_explain.get("shot_blocked", False)
        shot_pressure = shoot_explain.get("shot_pressure_multiplier", 1.0)
        base_xg = shoot_explain.get("base_xg", 0.0)
        if shot_blocked:
            q_shoot_eff = q_shoot * 0.05
            shoot_reason = "shot blocked"
        elif shot_pressure < 0.45:
            q_shoot_eff = q_shoot * 0.2
            shoot_reason = "high pressure"
        else:
            q_shoot_eff = q_shoot
            if base_xg > 0.2 and shot_pressure >= 0.6:
                q_shoot_eff = q_shoot + 0.02
                shoot_reason = "high-xG region, low pressure (prefer shoot)"
            else:
                shoot_reason = "shoot"

        # Pass rule: short safe pass sanity
        pass_candidates = pass_explain.get("top_candidates") or []
        best_overall_adjusted = pass_explain.get("best_pass_adjusted", 0.0)
        best_overall_target = pass_explain.get("best_pass_target")
        passer_xy = (x, y)
        short_safe = []
        for c in pass_candidates:
            rx, ry = c["receiver_xy"][0], c["receiver_xy"][1]
            dist = np.sqrt((rx - x) ** 2 + (ry - y) ** 2)
            if dist <= 12.0 and c.get("risk", 1.0) <= 0.25:
                short_safe.append((c["adjusted"], (rx, ry), c))
        chosen_pass_receiver_id = None
        chosen_pass_risk = None
        if short_safe:
            best_short_safe_adj = max(s[0] for s in short_safe)
            best_short_safe = max(short_safe, key=lambda s: s[0])
            if best_overall_adjusted <= best_short_safe_adj + 0.03:
                q_pass_eff = best_short_safe_adj
                chosen_pass_target = best_short_safe[1]
                chosen_pass_receiver_id = best_short_safe[2].get("receiver_id")
                chosen_pass_risk = best_short_safe[2].get("risk")
                pass_reason = "short safe pass (within 12m, risk<=0.25)"
            else:
                q_pass_eff = best_overall_adjusted
                chosen_pass_target = (best_overall_target["x"], best_overall_target["y"]) if best_overall_target else None
                pass_reason = "best adjusted pass"
                if pass_candidates:
                    chosen_pass_receiver_id = pass_candidates[0].get("receiver_id")
                    chosen_pass_risk = pass_explain.get("best_pass_risk")
        else:
            q_pass_eff = best_overall_adjusted
            chosen_pass_target = (best_overall_target["x"], best_overall_target["y"]) if best_overall_target else None
            pass_reason = "best adjusted pass"
            if pass_candidates:
                chosen_pass_receiver_id = pass_candidates[0].get("receiver_id")
                chosen_pass_risk = pass_explain.get("best_pass_risk")

        # Dribble rule: allow to beat pass when open space and low pressure
        dribble_open = dribble_explain.get("dribble_open_space_m", 0.0)
        dribble_mult = dribble_explain.get("dribble_pressure_multiplier", 0.0)
        if dribble_open >= 8.0 and dribble_mult >= 0.9 and q_dribble >= (q_pass_eff - 0.01):
            q_dribble_eff = max(q_dribble, q_pass_eff + 0.005)
            dribble_reason = "open space and low pressure (dribble preferred when close to pass)"
        else:
            q_dribble_eff = q_dribble
            dribble_reason = "dribble"

        # Choose best action
        q_values_eff = {"shoot": q_shoot_eff, "pass": q_pass_eff, "dribble": q_dribble_eff}
        best_action = max(q_values_eff, key=q_values_eff.get)
        epv = q_values_eff[best_action]

        # Reasons and raw q for reporting
        reasons = {"shoot": shoot_reason, "pass": pass_reason, "dribble": dribble_reason}
        best_action_reason = reasons[best_action]

        # Single authoritative best_action_target (center coords), same goal as evaluate_shoot for shoot
        half_len = self.pitch_length / 2
        half_w = self.pitch_width / 2
        if best_action == "shoot":
            best_action_target = (goal_x, goal_y)
        elif best_action == "pass":
            best_action_target = chosen_pass_target if chosen_pass_target is not None else (x, y)
        elif best_action == "dribble":
            # Dribble direction uses `theta` (radians). For the tactical board and replay
            # endpoints we don't have player orientation, so the API server injects a
            # consistent default: home faces +X (theta=0), away faces -X (theta=pi).
            #
            # As a safety net, if theta is missing here we align it with the attacking goal.
            theta = 0.0
            if frame_data and "player_data" in frame_data:
                for p in frame_data["player_data"]:
                    if p.get("player_id") == player_id:
                        theta = float(p.get("theta", 0.0))
                        break
            if theta == 0.0 and goal_x < 0:
                theta = float(np.pi)
            open_m = dribble_open if dribble_open is not None else 0.0
            d = min(10.0, open_m) if open_m > 0 else 3.0
            tx = x + d * np.cos(theta)
            ty = y + d * np.sin(theta)
            tx = float(np.clip(tx, -half_len, half_len))
            ty = float(np.clip(ty, -half_w, half_w))
            best_action_target = (tx, ty)
        else:
            best_action_target = (goal_x, goal_y)

        out = {
            "action": best_action,
            "epv": epv,
            "q_shoot": q_shoot,
            "q_pass": q_pass,
            "q_dribble": q_dribble,
            "best_action_reason": best_action_reason,
            "best_action_target": {"x": best_action_target[0], "y": best_action_target[1]},
            "location": (x, y),
            "details": {"player_id": player_id, "frame": frame},
            "explain": {
                "shoot": shoot_explain,
                "pass": pass_explain,
                "dribble": dribble_explain,
                "profile": profile_explain,
            },
        }
        if best_action == "pass":
            out["chosen_pass_receiver_id"] = chosen_pass_receiver_id
            out["chosen_pass_risk"] = chosen_pass_risk
        return out
