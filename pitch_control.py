"""
Pitch Control Module

Extracted from pitch control notebook and adapted for integration with EPV models.
Based on Fernandez-inspired pitch control calculations.

This module provides:
- PitchControlRunner: Core pitch control computation
- PitchControlCache: Efficient caching for repeated queries
- Simple interface functions for integration with action models
"""

import json
import csv
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict


# =======================================================
# Loaders & robust player-id → team mapping
# =======================================================
def load_tracking_jsonl(path, max_frames=None):
    """Load tracking data from JSONL file."""
    frames = []
    with open(path, "r") as f:
        for k, line in enumerate(f):
            if max_frames is not None and k >= max_frames:
                break
            if line.strip():
                frames.append(json.loads(line))
    return frames


def build_pid_team_from_match_json(path):
    """
    Build a robust mapping for BOTH possible tracking ID schemes:
      - tracking 'player_id' == match['players'][*]['id']            (roster ID)
      - tracking 'player_id' == match['players'][*]['trackable_object'] (trackable object ID)
    """
    with open(path, "r") as f:
        m = json.load(f)

    pid_to_teamid = {}
    pid_to_name = {}

    players_blob = m.get("players") or m.get("match_players") or []
    for p in players_blob:
        team_id = p.get("team_id") or (p.get("team") or {}).get("id")
        first = p.get("first_name") or (p.get("player") or {}).get("first_name") or ""
        last = p.get("last_name") or (p.get("player") or {}).get("last_name") or ""
        name = f"{first} {last}".strip()

        # 1) Roster ID
        roster_id = p.get("id")
        if roster_id is not None and team_id is not None:
            try:
                pid_to_teamid[int(roster_id)] = int(team_id)
                pid_to_name[int(roster_id)] = name
            except Exception:
                pass

        # 2) Trackable-object ID
        tobj = p.get("trackable_object") or p.get("trackable_object_id")
        if isinstance(tobj, dict):
            tobj = tobj.get("id") or tobj.get("trackable_object_id")
        if tobj is not None and team_id is not None:
            try:
                pid_to_teamid[int(tobj)] = int(team_id)
                if int(tobj) not in pid_to_name:
                    pid_to_name[int(tobj)] = name
            except Exception:
                pass

    home = m.get("home_team") or {}
    away = m.get("away_team") or {}

    meta = {
        "pid_to_teamid": pid_to_teamid,
        "pid_to_name": pid_to_name,
        "home_id": int(home.get("id")) if home.get("id") is not None else None,
        "away_id": int(away.get("id")) if away.get("id") is not None else None,
        "home_name": home.get("name") or "HOME",
        "away_name": away.get("name") or "AWAY",
        "home_sides": m.get("home_team_side") or ["left_to_right", "right_to_left"],
        "pitch": (float(m.get("pitch_length", 105)), float(m.get("pitch_width", 67))),
        "kickoffs": {
            int(mp["period"]): int(mp["start_frame"])
            for mp in m.get("match_periods", [])
            if mp.get("period") is not None and mp.get("start_frame") is not None
        },
    }
    return meta


def load_match_meta_robust(match_path):
    """Load match metadata with robust player-team mapping."""
    meta = build_pid_team_from_match_json(match_path)
    return meta


# =======================================================
# Time utils
# =======================================================
def _parse_hhmmss_ff(ts):
    """Parse HH:MM:SS.ff timestamp to seconds."""
    if ts is None:
        return None
    h, m, s = ts.split(":")
    return 3600 * int(h) + 60 * int(m) + float(s)


def infer_dt(fr_now, fr_prev, default=0.10):
    """Infer time delta between frames."""
    t0 = _parse_hhmmss_ff(fr_prev.get("timestamp"))
    t1 = _parse_hhmmss_ff(fr_now.get("timestamp"))
    return (t1 - t0) if (t0 is not None and t1 is not None and t1 > t0) else default


# =======================================================
# Period labeling
# =======================================================
def build_period_labels_from_home_sides(meta):
    """Define A/B per period: A = team on LEFT at kickoff."""
    hs = meta["home_sides"]
    home, away = meta["home_id"], meta["away_id"]
    per = {}
    if home is None or away is None:
        per[1] = {"HOME": "A", "AWAY": "B"}
        per[2] = {"HOME": "B", "AWAY": "A"}
        return per
    # Period 1
    if hs[0] == "left_to_right":
        per[1] = {home: "A", away: "B"}
    else:
        per[1] = {away: "A", home: "B"}
    # Period 2
    if len(hs) > 1:
        if hs[1] == "left_to_right":
            per[2] = {home: "A", away: "B"}
        else:
            per[2] = {away: "A", home: "B"}
    return per


def _coerce_period(val):
    """Coerce period value to int."""
    try:
        return int(val)
    except Exception:
        return None


def get_period_safe(frames, i, meta, search_radius=500):
    """Best-effort period for frame i."""
    p = _coerce_period(frames[i].get("period"))
    if p in (1, 2):
        return p
    lo = max(0, i - search_radius)
    for j in range(i - 1, lo - 1, -1):
        pj = _coerce_period(frames[j].get("period"))
        if pj in (1, 2):
            return pj
    hi = min(len(frames) - 1, i + search_radius)
    for j in range(i + 1, hi + 1):
        pj = _coerce_period(frames[j].get("period"))
        if pj in (1, 2):
            return pj
    k2 = meta.get("kickoffs", {}).get(2, None)
    if isinstance(k2, int):
        return 2 if i >= k2 else 1
    return 1


# =======================================================
# Frame adapter & possession heuristic
# =======================================================
def adapt_frame_for_pc(frame, period, meta, period_labels):
    """Adapt frame to pitch control format with A/B team labels."""
    pid_to_teamid = meta["pid_to_teamid"]
    label_by_team = period_labels[period]
    players = []
    for p in frame.get("player_data", []):
        pid = p.get("player_id")
        x = p.get("x")
        y = p.get("y")
        if pid is None or x is None or y is None:
            continue
        team_id = pid_to_teamid.get(int(pid))
        if team_id is None:
            continue
        lab = label_by_team.get(team_id)
        if lab is None:
            continue
        players.append({"id": int(pid), "team": lab, "x": float(x), "y": float(y)})
    bd = frame.get("ball_data") or {}
    bx, by = bd.get("x"), bd.get("y")
    ball = {
        "x": float(bx) if bx is not None else np.nan,
        "y": float(by) if by is not None else np.nan,
    }
    return {"players": players, "ball": ball}


def infer_team_with_ball(frame, period, meta, period_labels, radius=3.5):
    """Nearest-to-ball heuristic for possession."""
    label_by_team = period_labels[period]
    bd = frame.get("ball_data") or {}
    bx, by = bd.get("x"), bd.get("y")
    if bx is None or by is None:
        return "A"
    bx, by = float(bx), float(by)
    best = (1e18, "A")
    for p in frame.get("player_data", []):
        pid = p.get("player_id")
        x = p.get("x")
        y = p.get("y")
        if pid is None or x is None or y is None:
            continue
        team_id = meta["pid_to_teamid"].get(int(pid))
        if team_id is None:
            continue
        lab = label_by_team.get(team_id)
        if lab is None:
            continue
        d2 = (float(x) - bx) ** 2 + (float(y) - by) ** 2
        if d2 < best[0]:
            best = (d2, lab)
    return best[1] if best[0] <= radius**2 else "A"


# =======================================================
# Core pitch-control math
# =======================================================
def pitch_control_single_frame(
    query_points,
    frame_now,
    frame_prev,
    dt,
    team_with_ball,
    vmax=7.0,
    tau_react=0.70,
    beta_proj=0.50,
    lambda_control=4.0,
    speed_cap=8.5,
):
    """
    Compute pitch control at query points.

    Args:
        query_points: (N,2) XY meters
        frame_now/frame_prev: {"players":[{id,team,x,y}], "ball":{x,y}}
        dt: seconds
        team_with_ball: 'A' or 'B'
        vmax: Maximum player speed (m/s)
        tau_react: Reaction time (s)
        beta_proj: Velocity projection factor
        lambda_control: Control decay rate
        speed_cap: Maximum effective speed (m/s)

    Returns:
        pc, inf_att, inf_def each shape (N,)
    """
    now_players = {p["id"]: p for p in frame_now["players"]}
    prev_players = {p["id"]: p for p in frame_prev["players"]}

    vels = {}
    for pid, p in now_players.items():
        if pid in prev_players:
            dx = p["x"] - prev_players[pid]["x"]
            dy = p["y"] - prev_players[pid]["y"]
            vx = dx / max(dt, 1e-6)
            vy = dy / max(dt, 1e-6)
        else:
            vx = vy = 0.0
        vels[pid] = (vx, vy)

    Q = np.asarray(query_points, float)  # (N,2)
    tiny = 1e-9

    attackers, defenders = [], []
    for pid, p in now_players.items():
        vx, vy = vels.get(pid, (0.0, 0.0))
        entry = (p["x"], p["y"], vx, vy)
        if p["team"] == team_with_ball:
            attackers.append(entry)
        else:
            defenders.append(entry)

    A = np.array(attackers, float) if attackers else np.zeros((0, 4))
    D = np.array(defenders, float) if defenders else np.zeros((0, 4))

    def times_to_intercept(P):
        if P.shape[0] == 0:
            return np.full((Q.shape[0],), np.inf, float)
        px = P[:, 0][None, :]
        py = P[:, 1][None, :]
        vx = P[:, 2][None, :]
        vy = P[:, 3][None, :]
        dx = Q[:, 0:1] - px
        dy = Q[:, 1:2] - py
        dist = np.sqrt(np.maximum(dx * dx + dy * dy, tiny))
        ux = dx / dist
        uy = dy / dist
        vproj = np.maximum(0.0, vx * ux + vy * uy)
        veff = np.minimum(speed_cap, vmax + beta_proj * vproj)
        t = tau_react + dist / np.maximum(veff, 0.1)
        return np.min(t, axis=1)

    tA = times_to_intercept(A)
    tD = times_to_intercept(D)
    infA = np.exp(-lambda_control * tA)
    infD = np.exp(-lambda_control * tD)
    denom = infA + infD + tiny
    pc = infA / denom
    return pc, infA, infD


# =======================================================
# Main Runner Class
# =======================================================
class PitchControlRunner:
    """
    Main class for computing pitch control.

    Usage:
        meta = load_match_meta_robust(match_json_path)
        frames = load_tracking_jsonl(tracking_jsonl_path)
        runner = PitchControlRunner(meta, frames)

        # Compute PC at a point
        pc_value = runner.pc_at_point(frame_idx, x, y)

        # Compute PC on a grid
        result = runner.pc_at_grid(frame_idx, grid=(21, 13))
    """

    def __init__(self, meta, frames):
        """
        Initialize runner.

        Args:
            meta: Match metadata from load_match_meta_robust()
            frames: List of tracking frames from load_tracking_jsonl()
        """
        self.meta = meta
        self.frames = frames
        self.period_labels = build_period_labels_from_home_sides(meta)

    def pc_at_grid(
        self,
        i,
        grid=(21, 13),
        vmax=7.0,
        tau_react=0.70,
        beta_proj=0.50,
        lambda_control=4.0,
        speed_cap=8.5,
    ):
        """
        Compute pitch control on a grid for frame i.

        Args:
            i: Frame index (must be >= 1)
            grid: (nx, ny) grid resolution
            vmax: Maximum player speed
            tau_react: Reaction time
            beta_proj: Velocity projection factor
            lambda_control: Control decay rate
            speed_cap: Maximum effective speed

        Returns:
            Dictionary with keys: pc, att, def, X, Y, dt_used, team_with_ball, period_labels
        """
        assert 1 <= i < len(self.frames), "i must be >=1 so a previous frame exists"
        fr_now = self.frames[i]
        fr_prev = self.frames[i - 1]
        period = get_period_safe(self.frames, i, self.meta)

        f_now = adapt_frame_for_pc(fr_now, period, self.meta, self.period_labels)
        f_prev = adapt_frame_for_pc(fr_prev, period, self.meta, self.period_labels)

        team_with_ball = infer_team_with_ball(
            fr_now, period, self.meta, self.period_labels
        )

        # Ensure non-empty attackers
        nA = sum(1 for p in f_now["players"] if p["team"] == "A")
        nB = sum(1 for p in f_now["players"] if p["team"] == "B")
        if team_with_ball == "A" and nA == 0 and nB > 0:
            team_with_ball = "B"
        elif team_with_ball == "B" and nB == 0 and nA > 0:
            team_with_ball = "A"

        dt = infer_dt(fr_now, fr_prev, default=0.10)

        L, W = self.meta["pitch"]
        nx, ny = grid
        xs = np.linspace(-L / 2, L / 2, nx)
        ys = np.linspace(-W / 2, W / 2, ny)
        X, Y = np.meshgrid(xs, ys)
        Q = np.c_[X.ravel(), Y.ravel()]

        pc, infA, infD = pitch_control_single_frame(
            Q,
            f_now,
            f_prev,
            dt,
            team_with_ball,
            vmax=vmax,
            tau_react=tau_react,
            beta_proj=beta_proj,
            lambda_control=lambda_control,
            speed_cap=speed_cap,
        )

        return {
            "pc": pc.reshape(Y.shape),
            "att": infA.reshape(Y.shape),
            "def": infD.reshape(Y.shape),
            "X": X,
            "Y": Y,
            "dt_used": dt,
            "team_with_ball": team_with_ball,
            "period_labels": self.period_labels,
        }

    def pc_at_point(self, i, x, y, team_with_ball=None):
        """
        Compute pitch control at a single point for frame i.

        Args:
            i: Frame index (must be >= 1)
            x: X coordinate (meters)
            y: Y coordinate (meters)
            team_with_ball: Override team with ball ('A' or 'B'), or None to infer

        Returns:
            Float: pitch control value at (x, y) for the attacking team
        """
        assert 1 <= i < len(self.frames), "i must be >=1"
        fr_now = self.frames[i]
        fr_prev = self.frames[i - 1]
        period = get_period_safe(self.frames, i, self.meta)

        f_now = adapt_frame_for_pc(fr_now, period, self.meta, self.period_labels)
        f_prev = adapt_frame_for_pc(fr_prev, period, self.meta, self.period_labels)

        if team_with_ball is None:
            team_with_ball = infer_team_with_ball(
                fr_now, period, self.meta, self.period_labels
            )

        # Ensure non-empty attackers
        nA = sum(1 for p in f_now["players"] if p["team"] == "A")
        nB = sum(1 for p in f_now["players"] if p["team"] == "B")
        if team_with_ball == "A" and nA == 0 and nB > 0:
            team_with_ball = "B"
        elif team_with_ball == "B" and nB == 0 and nA > 0:
            team_with_ball = "A"

        dt = infer_dt(fr_now, fr_prev, default=0.10)

        Q = np.array([[float(x), float(y)]])
        pc, _, _ = pitch_control_single_frame(Q, f_now, f_prev, dt, team_with_ball)

        return float(pc[0])

    def pc_at_ball(self, i):
        """
        Compute pitch control at the ball location for frame i.

        Args:
            i: Frame index (must be >= 1)

        Returns:
            Float or None: pitch control value at ball location, or None if ball not detected
        """
        bd = self.frames[i].get("ball_data") or {}
        bx, by = bd.get("x"), bd.get("y")
        if bx is None or by is None:
            return None
        return self.pc_at_point(i, float(bx), float(by))


# =======================================================
# Caching Wrapper for Efficient Repeated Queries
# =======================================================
class PitchControlCache:
    """
    Cache for pitch control runners by match.

    This allows efficient repeated queries without reloading data.

    Usage:
        cache = PitchControlCache(data_dir)
        pc_value = cache.get_pc(match_id, frame, x, y)
    """

    def __init__(self, data_dir: Path):
        """
        Initialize cache.

        Args:
            data_dir: Directory containing match data
        """
        self.data_dir = Path(data_dir)
        self.runners = {}  # match_id -> PitchControlRunner

    def _load_runner(self, match_id: str) -> PitchControlRunner:
        """Load and cache runner for a match."""
        if match_id in self.runners:
            return self.runners[match_id]

        # Load match data
        # Try extrapolated tracking first
        tracking_path = self.data_dir / f"{match_id}_tracking_extrapolated.jsonl"
        if not tracking_path.exists():
            # Fall back to match.json
            tracking_path = self.data_dir / f"match_{match_id}.json"

        match_json_path = self.data_dir / f"match_{match_id}.json"

        print(f"Loading pitch control data for match {match_id}...")
        meta = load_match_meta_robust(match_json_path)

        # Load frames
        if tracking_path.suffix == ".jsonl":
            frames = load_tracking_jsonl(tracking_path)
        else:
            # Load from match.json (skip first line which is metadata)
            frames = []
            with open(tracking_path, "r") as f:
                _ = f.readline()  # Skip metadata
                for line in f:
                    if line.strip():
                        frames.append(json.loads(line))

        runner = PitchControlRunner(meta, frames)
        self.runners[match_id] = runner

        print(f"  Loaded {len(frames)} frames for match {match_id}")
        return runner

    def get_pc(self, match_id: str, frame: int, x: float, y: float) -> float:
        """
        Get pitch control value at a point.

        Args:
            match_id: Match ID
            frame: Frame number
            x: X coordinate (meters)
            y: Y coordinate (meters)

        Returns:
            Pitch control value (0-1) for the attacking team
        """
        runner = self._load_runner(match_id)
        return runner.pc_at_point(frame, x, y)

    def get_runner(self, match_id: str) -> PitchControlRunner:
        """
        Get the runner for a match (useful for batch queries).

        Args:
            match_id: Match ID

        Returns:
            PitchControlRunner instance
        """
        return self._load_runner(match_id)


# =======================================================
# Convenience Functions
# =======================================================
def get_pitch_control_for_events(
    events_df,
    data_dir: Path,
    cache: Optional[PitchControlCache] = None
) -> Dict[str, float]:
    """
    Get pitch control values for events.

    Args:
        events_df: DataFrame with columns: event_id, match_id, frame_start, x_start, y_start
        data_dir: Directory containing match data
        cache: Optional PitchControlCache instance

    Returns:
        Dictionary mapping event_id -> pitch control value
    """
    if cache is None:
        cache = PitchControlCache(data_dir)

    pc_values = {}

    for _, event in events_df.iterrows():
        event_id = event['event_id']
        match_id = str(event['match_id'])
        frame = int(event['frame_start'])
        x = float(event['x_start'])
        y = float(event['y_start'])

        try:
            pc = cache.get_pc(match_id, frame, x, y)
            pc_values[event_id] = pc
        except Exception as e:
            print(f"Warning: Could not compute PC for event {event_id}: {e}")
            pc_values[event_id] = 0.5  # Neutral value

    return pc_values
