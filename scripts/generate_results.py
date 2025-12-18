"""
Generate illustrative EPV diagrams using the current trained models.

Outputs (saved under results/):
  - scenario_compare.png: two similar attacking situations with different best actions.
  - attack_timeseries.png: a short attacking sequence with EPV per action.

Notes:
  - Uses simplified EPV estimates:
      * Shot EV = xG model probability at the location.
      * Pass EV = P(pass complete) * xG at destination (using xG model).
  - The new tracking renderer uses real pitch control, defender pressure, and player skills.
  - Coordinates assume SkillCorner normalized pitch (goal at +52.5).
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pitch_control import PitchControlRunner, load_match_meta_robust, load_tracking_jsonl, get_period_safe

# Load models
BASE = Path(__file__).parent
results_dir = BASE / "results"
results_dir.mkdir(exist_ok=True)


def load_model(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj["feature_cols"]


def load_dribble(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj["scaler"], obj["feature_cols"]


def shot_xg(model, feature_cols, x, y, finishing_skill=0.0, damp_long=False, goal_x=52.5):
    goal_y = 0.0
    dist = np.sqrt((goal_x - x) ** 2 + (goal_y - y) ** 2)
    post1_y, post2_y = -3.66, 3.66
    angle = abs(
        np.degrees(
            np.arctan2(post2_y - y, goal_x - x) - np.arctan2(post1_y - y, goal_x - x)
        )
    )
    penalty = 1 if ((goal_x > 0 and x > 36) or (goal_x < 0 and x < -36)) and abs(y) < 18 else 0
    # Build feature vector in the trained order
    feats = {
        "distance_to_goal": dist,
        "angle_to_goal": angle,
        "penalty_area": penalty,
        "trajectory_angle": 0.0,
        "distance_covered": 0.0,
        "speed_avg": 0.0,
        "player_finishing_skill": finishing_skill,
    }
    x_vec = np.array([[feats[c] for c in feature_cols]], dtype=float)
    base = float(model.predict_proba(x_vec)[0, 1])
    # Optional damping for low-quality long/very-wide shots
    if damp_long:
        if dist > 30:
            decay = np.exp(-(dist - 30) / 8.0)
            base *= decay
        if abs(y) > 28:
            wide_decay = np.exp(-(abs(y) - 28) / 2.5)
            base *= wide_decay
        if angle < 10:
            angle_decay = np.exp(-(10 - angle) / 4.0)
            base *= angle_decay
    return base


def pass_success(model, feature_cols, x0, y0, x1, y1):
    dist = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    ang = np.degrees(np.arctan2(y1 - y0, x1 - x0))
    fwd = x1 - x0
    feats = {
        "pass_distance": dist,
        "pass_angle": ang,
        "forward_progress": fwd,
        "defenders_near_origin": 0.0,
        "defenders_near_dest": 0.0,
        "defenders_in_lane": 0.0,
        "pitch_control_origin": 0.5,
        "pitch_control_dest": 0.5,
        "player_passing_skill": 0.0,
        "speed_avg": 0.0,
        "inside_defensive_shape": 0.0,
        "last_defensive_line_x": 0.0,
        "last_defensive_line_height": 0.0,
    }
    x_vec = np.array([[feats[c] for c in feature_cols]], dtype=float)
    return float(model.predict_proba(x_vec)[0, 1])


def dribble_success(model, scaler, feature_cols, x0, y0, x1, y1):
    dist = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    duration = max(dist / 5.0, 0.5)  # assume ~5 m/s, min half-second
    speed = dist / duration if duration > 0 else 0.0
    goal_x = 52.5
    goal_y = 0.0
    dist_start = np.sqrt((goal_x - x0) ** 2 + (goal_y - y0) ** 2)
    dist_end = np.sqrt((goal_x - x1) ** 2 + (goal_y - y1) ** 2)
    angle_change = abs(np.arctan2(y1 - y0, x1 - x0))
    dist_sideline = min(abs(y0 - 34), abs(y0 + 34))

    feats = {
        "x_start": x0,
        "y_start": y0,
        "x_end": x1,
        "y_end": y1,
        "distance_covered": dist,
        "duration": duration,
        "speed": speed,
        "dist_to_goal_start": dist_start,
        "dist_to_goal_end": dist_end,
        "dist_to_goal_change": dist_start - dist_end,
        "defenders_nearby_start": 0.0,
        "defenders_nearby_end": 0.0,
        "pressure_start": 5.0,
        "pressure_end": 5.0,
        "pitch_control_start": 0.5,
        "pitch_control_end": 0.5,
        "pitch_control_change": 0.0,
        "angle_change": angle_change,
        "dist_from_sideline": dist_sideline,
        "player_dribbling_skill": 0.0,
    }

    x_vec = np.array([[feats[c] for c in feature_cols]], dtype=float)
    x_scaled = scaler.transform(x_vec)
    return float(model.predict_proba(x_scaled)[0, 1])


def draw_pitch(ax):
    # Dimensions: length 105m (-52.5 to 52.5), width 68m (-34 to 34)
    ax.set_xlim(-55, 55)
    ax.set_ylim(-36, 36)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("#2e7d32")  # green pitch
    # Boundaries
    rect = plt.Rectangle(
        (-52.5, -34), 105, 68, fill=False, color="white", lw=1.5
    )
    ax.add_patch(rect)
    # Halfway line and center circle
    ax.plot([0, 0], [-34, 34], color="white", lw=1)
    center = plt.Circle((0, 0), 9.15, fill=False, color="white", lw=1)
    ax.add_patch(center)
    ax.scatter([0], [0], color="white", s=10)
    # Boxes
    ax.add_patch(
        plt.Rectangle((52.5 - 16.5, -20.16), 16.5, 40.32, fill=False, lw=1, color="white")
    )
    ax.add_patch(
        plt.Rectangle((-52.5, -20.16), 16.5, 40.32, fill=False, lw=1, color="white")
    )
    # 6-yard boxes
    ax.add_patch(
        plt.Rectangle((52.5 - 5.5, -9.16), 5.5, 18.32, fill=False, lw=1, color="white")
    )
    ax.add_patch(
        plt.Rectangle((-52.5, -9.16), 5.5, 18.32, fill=False, lw=1, color="white")
    )
    # Goals
    ax.add_patch(plt.Rectangle((52.5, -3.66), 2, 7.32, fill=False, lw=1, color="white"))
    ax.add_patch(plt.Rectangle((-54.5, -3.66), 2, 7.32, fill=False, lw=1, color="white"))


def load_player_name_map():
    name_map = {}
    try:
        df = pd.read_csv(BASE / "player_id_to_passing_skill.csv")
        if "player_id" in df.columns and "skillcorner_name" in df.columns:
            name_map = dict(zip(df["player_id"].astype(int), df["skillcorner_name"]))
    except Exception:
        pass
    return name_map


def load_finishing_skill_map():
    fmap = {}
    try:
        df = pd.read_csv(BASE / "player_id_to_finishing_skill.csv")
        # support either finishing_skill or player_finishing_skill column names
        if "player_id" in df.columns:
            if "finishing_skill" in df.columns:
                fmap = dict(zip(df["player_id"].astype(int), df["finishing_skill"]))
            elif "player_finishing_skill" in df.columns:
                fmap = dict(zip(df["player_id"].astype(int), df["player_finishing_skill"]))
    except Exception:
        pass
    return fmap


def load_passing_skill_map():
    pmap = {}
    try:
        df = pd.read_csv(BASE / "player_id_to_passing_skill.csv")
        if {"player_id", "player_passing_skill"} <= set(df.columns):
            pmap = dict(zip(df["player_id"].astype(int), df["player_passing_skill"]))
        elif {"player_id", "player_RE"} <= set(df.columns):
            pmap = dict(zip(df["player_id"].astype(int), df["player_RE"]))
    except Exception:
        pass
    return pmap


def load_dribbling_skill_map():
    dmap = {}
    try:
        df = pd.read_csv(BASE / "player_id_to_skill.csv")
        if {"player_id", "player_dribbling_skill"} <= set(df.columns):
            dmap = dict(zip(df["player_id"].astype(int), df["player_dribbling_skill"]))
    except Exception:
        pass
    return dmap


def _clip_pitch(x, y):
    return min(52.0, max(-52.0, x)), min(34.0, max(-34.0, y))


def _parse_time(val):
    if pd.isna(val):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        try:
            parts = str(val).split(":")
            parts = [p for p in parts if p != ""]
            if len(parts) == 2:
                m, s = parts
                return float(m) * 60 + float(s)
            if len(parts) == 3:
                h, m, s = parts
                return float(h) * 3600 + float(m) * 60 + float(s)
        except Exception:
            pass
    return None


# === Tracking-aware helpers (use real PC/pressure/skills) ====================
def _defenders_near(x, y, frame_data, team_id, roster, radius=3.0):
    """Return (count, closest_distance) of opponents near a point."""
    if team_id is None:
        return 0, None
    opp_ids = {pid for pid, tid in roster.items() if tid != team_id}
    count = 0
    closest = 99.0
    for p in frame_data or []:
        pid = p.get("player_id")
        if pid is None or pid not in opp_ids:
            continue
        if not p.get("is_detected", True):
            continue
        px, py = p.get("x"), p.get("y")
        if px is None or py is None:
            continue
        d = float(np.hypot(x - px, y - py))
        if d <= radius and d > 0.05:
            count += 1
            closest = min(closest, d)
    return count, closest if closest < 99 else None


def _defenders_in_lane(x0, y0, x1, y1, frame_data, team_id, roster, lane_width=2.0):
    if team_id is None:
        return 0
    opp_ids = {pid for pid, tid in roster.items() if tid != team_id}
    dx, dy = x1 - x0, y1 - y0
    seg_len = float(np.hypot(dx, dy))
    if seg_len < 0.1:
        return 0
    count = 0
    for p in frame_data or []:
        pid = p.get("player_id")
        if pid is None or pid not in opp_ids:
            continue
        if not p.get("is_detected", True):
            continue
        px, py = p.get("x"), p.get("y")
        if px is None or py is None:
            continue
        to_px, to_py = px - x0, py - y0
        proj = (to_px * dx + to_py * dy) / seg_len
        if proj < 0 or proj > seg_len:
            continue
        perp = abs(to_px * dy - to_py * dx) / seg_len
        if perp <= lane_width:
            count += 1
    return count


def _pitch_control(pc_runner, frame_idx, x, y):
    try:
        idx = max(1, min(frame_idx, len(pc_runner.frames) - 1))
        return float(pc_runner.pc_at_point(idx, x, y))
    except Exception:
        # If pitch control fails, return a conservative low value instead of neutral.
        return 0.0


def pass_success_real(
    model,
    feature_cols,
    x0,
    y0,
    x1,
    y1,
    frame_data,
    team_id,
    roster,
    pc_runner,
    frame_idx,
    passing_skill_map,
    player_id,
):
    dist = np.hypot(x1 - x0, y1 - y0)
    ang = np.degrees(np.arctan2(y1 - y0, x1 - x0))
    fwd = x1 - x0
    def_o, _ = _defenders_near(x0, y0, frame_data, team_id, roster, radius=3.0)
    def_d, _ = _defenders_near(x1, y1, frame_data, team_id, roster, radius=3.0)
    def_lane = _defenders_in_lane(x0, y0, x1, y1, frame_data, team_id, roster, lane_width=2.0)
    pc_o = _pitch_control(pc_runner, frame_idx, x0, y0)
    pc_d = _pitch_control(pc_runner, frame_idx, x1, y1)
    opp_positions = [
        (float(p["x"]), float(p["y"]))
        for p in (frame_data or [])
        if p.get("player_id") in {pid for pid, tid in roster.items() if tid != team_id}
        and p.get("x") is not None
        and p.get("y") is not None
    ]
    last_def_line_x = max((px for px, _ in opp_positions), default=0.0)
    line_height = (max((py for _, py in opp_positions), default=0.0) - min((py for _, py in opp_positions), default=0.0)) if opp_positions else 0.0
    inside_def_shape = 1.0 if opp_positions and any((x0 <= px <= x1 or x1 <= px <= x0) and abs(py - ((y0 + y1) / 2)) < 8 for px, py in opp_positions) else 0.0
    duration = max(dist / 18.0, 0.25)  # fast ball travel
    speed_avg = dist / duration if duration > 0 else 0.0
    feats = {
        "pass_distance": dist,
        "pass_angle": ang,
        "forward_progress": fwd,
        "defenders_near_origin": def_o,
        "defenders_near_dest": def_d,
        "defenders_in_lane": def_lane,
        "pitch_control_origin": pc_o,
        "pitch_control_dest": pc_d,
        "player_passing_skill": passing_skill_map.get(player_id, 0.0),
        "speed_avg": speed_avg,
        "inside_defensive_shape": inside_def_shape,
        "last_defensive_line_x": last_def_line_x,
        "last_defensive_line_height": line_height,
    }
    x_vec = np.array([[feats[c] for c in feature_cols]], dtype=float)
    return float(model.predict_proba(x_vec)[0, 1])


def dribble_success_real(
    model,
    scaler,
    feature_cols,
    x0,
    y0,
    x1,
    y1,
    frame_data,
    team_id,
    roster,
    pc_runner,
    frame_idx,
    dribble_skill_map,
    player_id,
    goal_x=52.5,
):
    dist = np.hypot(x1 - x0, y1 - y0)
    duration = max(dist / 5.5, 0.4)
    speed = dist / duration if duration > 0 else 0.0
    goal_y = 0.0
    d0 = np.hypot(goal_x - x0, goal_y - y0)
    d1 = np.hypot(goal_x - x1, goal_y - y1)
    def_start, press_start = _defenders_near(x0, y0, frame_data, team_id, roster, radius=5.0)
    def_end, press_end = _defenders_near(x1, y1, frame_data, team_id, roster, radius=5.0)
    pc_s = _pitch_control(pc_runner, frame_idx, x0, y0)
    pc_e = _pitch_control(pc_runner, frame_idx, x1, y1)
    angle_change = abs(np.arctan2(y1 - y0, x1 - x0))
    dist_sideline = min(abs(y0 - 34), abs(y0 + 34))
    feats = {
        "x_start": x0,
        "y_start": y0,
        "x_end": x1,
        "y_end": y1,
        "distance_covered": dist,
        "duration": duration,
        "speed": speed,
        "dist_to_goal_start": d0,
        "dist_to_goal_end": d1,
        "dist_to_goal_change": d0 - d1,
        "defenders_nearby_start": def_start,
        "defenders_nearby_end": def_end,
        "pressure_start": press_start if press_start is not None else 10.0,
        "pressure_end": press_end if press_end is not None else 10.0,
        "pitch_control_start": pc_s,
        "pitch_control_end": pc_e,
        "pitch_control_change": pc_e - pc_s,
        "angle_change": angle_change,
        "dist_from_sideline": dist_sideline,
        "player_dribbling_skill": dribble_skill_map.get(player_id, 0.0),
    }
    x_vec = np.array([[feats[c] for c in feature_cols]], dtype=float)
    x_scaled = scaler.transform(x_vec)
    return float(model.predict_proba(x_scaled)[0, 1])


def _ts_to_sec(ts):
    if ts is None:
        return None
    try:
        parts = [p for p in str(ts).split(":") if p != ""]
        if len(parts) == 3:
            h, m, s = parts
            return float(h) * 3600 + float(m) * 60 + float(s)
        if len(parts) == 2:
            m, s = parts
            return float(m) * 60 + float(s)
        return float(ts)
    except Exception:
        return None


def best_pass_candidates(x0, y0, pass_model, pass_cols, xg_model, xg_cols, max_options=2):
    """Search a small grid of forward-ish destinations and return top EV passes."""
    offsets = []
    for dx in [6, 10, 14]:
        for dy in [-12, -6, -3, 0, 3, 6, 12]:
            if dx <= 0:
                continue
            offsets.append((dx, dy))

    scored = []
    for dx, dy in offsets:
        xd, yd = _clip_pitch(x0 + dx, y0 + dy)
        p_succ = pass_success(pass_model, pass_cols, x0, y0, xd, yd)
        ev = p_succ * shot_xg(xg_model, xg_cols, xd, yd, damp_long=False)
        angle = np.degrees(np.arctan2(yd - y0, xd - x0))
        direction = "right" if angle > 5 else "left" if angle < -5 else "center"
        scored.append(
            {
                "label": f"Pass {direction}",
                "ev": ev,
                "p_succ": p_succ,
                "dest": (xd, yd),
                "angle": angle,
            }
        )
    scored.sort(key=lambda r: r["ev"], reverse=True)
    dedup = []
    seen = set()
    for r in scored:
        key = (round(r["dest"][0], 1), round(r["dest"][1], 1))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)
        if len(dedup) >= max_options:
            break
    return dedup


def best_dribble_candidate(x0, y0, drib_model, drib_scaler, drib_cols, xg_model, xg_cols):
    """Pick the dribble destination with highest EV from a small forward set."""
    offsets = [(6, -4), (8, 0), (10, 4), (12, 0)]
    best = None
    for dx, dy in offsets:
        xd, yd = _clip_pitch(x0 + dx, y0 + dy)
        p_keep = dribble_success(drib_model, drib_scaler, drib_cols, x0, y0, xd, yd)
        ev = p_keep * shot_xg(xg_model, xg_cols, xd, yd, damp_long=False)
        if best is None or ev > best["ev"]:
            best = {"dest": (xd, yd), "ev": ev, "p_keep": p_keep}
    return best


def find_goal_chain(name_map):
    """Return up to three consecutive team actions ending in a goal shot.

    Rules:
      - Use real team actions with end_type in {pass, dribble, carry, cross, shot}.
      - Prefer non-corner goals; if none found, allow corner goals and link the corner.
      - Skip metadata rows (e.g., passing_option) with no end_type.
    """
    cols = [
        "event_id",
        "team_id",
        "team_shortname",
        "player_id",
        "end_type",
        "start_type",
        "event_subtype",
        "time_start",
        "x_start",
        "y_start",
        "x_end",
        "y_end",
        "lead_to_goal",
    ]
    allowed = {"pass", "shot", "dribble", "carry", "cross"}
    corner_keys = ("corner", "corner_kick")

    for path in sorted((BASE / "more_data").glob("*_dynamic_events.csv")):
        try:
            df = pd.read_csv(path, usecols=cols, low_memory=False)
        except Exception:
            continue
        match_id_current = path.name.split("_")[0]
        # Skip previously used match if we want variety
        if match_id_current in {"1039803", "1039805"}:
            continue
        df["time_val"] = df["time_start"].apply(_parse_time)
        df = df[df["time_val"].notna()]
        if "end_type" not in df:
            continue
        all_goals = df[(df["end_type"].str.lower() == "shot") & (df["lead_to_goal"] == True)]
        if all_goals.empty:
            continue

        # Prefer non-corner goals; fall back to any goal
        def is_corner_row(row):
            st = str(row.get("start_type") or "").lower()
            sub = str(row.get("event_subtype") or "").lower()
            return any(k in st for k in corner_keys) or any(k in sub for k in corner_keys)

        goals = all_goals[~all_goals.apply(is_corner_row, axis=1)]
        if goals.empty:
            goals = all_goals

        for _, goal in goals.iterrows():
            team = goal["team_id"]
            team_df = df[df["team_id"] == team].copy()
            # actions with meaningful end_type and coordinates (shots can omit end coords)
            actions_df = team_df[
                team_df["end_type"].str.lower().isin(allowed)
                & team_df[["x_start", "y_start", "time_val"]].notna().all(axis=1)
            ].copy()
            mask_nonshot = actions_df["end_type"].str.lower() != "shot"
            actions_df = actions_df[
                (mask_nonshot & actions_df[["x_end", "y_end"]].notna().all(axis=1))
                | (~mask_nonshot)
            ]
            actions_df = actions_df.sort_values("time_val").reset_index(drop=True)

            goal_idx = actions_df.index[actions_df["event_id"] == goal["event_id"]]
            if len(goal_idx) == 0:
                continue
            gpos = int(goal_idx[0])

            is_corner = False
            st = str(goal.get("start_type") or "").lower()
            sub = str(goal.get("event_subtype") or "").lower()
            if any(k in st for k in corner_keys) or any(k in sub for k in corner_keys):
                is_corner = True

            def end_point(ev):
                et = str(ev["end_type"]).lower()
                if et in {"pass", "dribble", "carry", "cross"} and pd.notna(ev.get("x_end")) and pd.notna(ev.get("y_end")):
                    return float(ev["x_end"]), float(ev["y_end"])
                return float(ev["x_start"]), float(ev["y_start"])

            def has_coords(ev, shot_ok=False):
                req = ["x_start", "y_start", "time_val"]
                if not shot_ok:
                    req += ["x_end", "y_end"]
                return pd.Series(ev)[req].notna().all()

            # build a connected chain backward with distance/time constraints
            chain_events = [goal]
            last_point = (float(goal["x_start"]), float(goal["y_start"]))
            last_time = goal["time_val"]

            for ev in reversed(list(actions_df.iloc[:gpos].itertuples(index=False))):
                ev = ev._asdict()
                et = str(ev["end_type"]).lower()
                if et not in allowed:
                    continue
                if not has_coords(ev, shot_ok=et == "shot"):
                    continue
                cand_end = end_point(ev)
                dist = np.hypot(last_point[0] - cand_end[0], last_point[1] - cand_end[1])
                dt = last_time - ev["time_val"] if last_time is not None and pd.notna(ev["time_val"]) else None
                # require reasonable proximity
                if dt is None or dt > 12 or dist > 12:
                    continue
                chain_events.append(ev)
                last_point = (float(ev["x_start"]), float(ev["y_start"]))
                last_time = ev["time_val"]
                if len(chain_events) >= 3:
                    break

            chain_events.reverse()

            actions = []
            for ev in chain_events:
                if hasattr(ev, "_asdict"):
                    ev = ev._asdict()
                elif isinstance(ev, pd.Series):
                    ev = ev.to_dict()
                et = str(ev.get("end_type") or "").lower()
                p = int(ev["player_id"]) if pd.notna(ev.get("player_id")) else -1
                name = name_map.get(p, f"Player {p}")
                base = {
                    "player": name,
                    "player_id": p,
                    "team_id": ev["team_id"],
                    "start": (float(ev["x_start"]), float(ev["y_start"])),
                    "time": ev["time_val"],
                    "event_id": ev["event_id"],
                }
                if et == "shot":
                    actions.append({**base, "type": "shot"})
                elif et == "pass":
                    if pd.notna(ev.get("x_end")) and pd.notna(ev.get("y_end")):
                        actions.append({**base, "type": "pass", "end": (float(ev["x_end"]), float(ev["y_end"]))})
                elif et in {"dribble", "carry", "cross"}:
                    if pd.notna(ev.get("x_end")) and pd.notna(ev.get("y_end")):
                        actions.append({**base, "type": "dribble", "end": (float(ev["x_end"]), float(ev["y_end"]))})
                else:
                    # ignore unknown/unsupported types
                    continue

            has_pass = any(a["type"] == "pass" for a in actions[:-1])
            if len(actions) >= 2 and actions[-1]["type"] == "shot" and has_pass:
                match_id = path.name.split("_")[0]
                return match_id, actions[-3:]  # keep at most last 3 actions ending in shot
    return None, None


def find_outside_box_pass(name_map, x_min=20.0, x_max=40.0, y_abs_max=25.0):
    """Find a real pass event starting outside the box for decision illustration."""
    cols = [
        "event_id",
        "team_id",
        "team_shortname",
        "player_id",
        "player_name",
        "end_type",
        "time_start",
        "x_start",
        "y_start",
    ]
    for path in sorted((BASE / "more_data").glob("*_dynamic_events.csv")):
        try:
            df = pd.read_csv(path, usecols=cols, low_memory=False)
        except Exception:
            continue
        df = df[df["end_type"].str.lower() == "pass"]
        df = df[df[["x_start", "y_start", "time_start"]].notna().all(axis=1)]
        df = df[
            (df["x_start"].between(x_min, x_max))
            & (df["y_start"].abs() <= y_abs_max)
        ]
        if df.empty:
            continue
        row = df.iloc[0]
        pid = int(row["player_id"]) if pd.notna(row["player_id"]) else -1
        return {
            "match_id": path.name.split("_")[0],
            "team_id": row["team_id"],
            "team_shortname": row.get("team_shortname"),
            "player": name_map.get(pid, row.get("player_name", f"Player {pid}")),
            "player_id": pid,
            "time_start": _parse_time(row["time_start"]),
            "x": float(row["x_start"]),
            "y": float(row["y_start"]),
        }
    return None


def get_event(match_id, event_id, player_substr="Artur"):
    """Fetch a specific event by id and player substring."""
    path = BASE / "more_data" / f"{match_id}_dynamic_events.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return None
    if "event_id" not in df.columns:
        return None
    row = df[df["event_id"] == event_id]
    if row.empty:
        return None
    row = row.iloc[0]
    name = row.get("player_name") or ""
    if player_substr.lower() not in name.lower():
        return None
    ts = _parse_time(row.get("time_start"))
    if pd.isna(ts):
        return None
    return {
        "match_id": match_id,
        "team_id": row.get("team_id"),
        "team_shortname": row.get("team_shortname"),
        "player": name,
        "player_id": int(row["player_id"]) if pd.notna(row.get("player_id")) else -1,
        "time_start": ts,
        "x": float(row["x_start"]),
        "y": float(row["y_start"]),
    }


def teammate_positions_at_time(match_id, team_id, origin_time, exclude_player_id, name_map, max_players=8):
    """Approximate teammate locations at a given time using nearest event per player."""
    path = BASE / "more_data" / f"{match_id}_dynamic_events.csv"
    if not path.exists():
        path = BASE / "skillcorner_download" / f"{match_id}_dynamic_events.csv"
    try:
        df = pd.read_csv(path, usecols=["player_id", "team_id", "time_start", "x_start", "y_start"], low_memory=False)
    except Exception:
        return []
    df = df[df["team_id"] == team_id]
    df["time_start"] = df["time_start"].apply(_parse_time)
    df = df[df[["player_id", "time_start", "x_start", "y_start"]].notna().all(axis=1)]
    df["player_id"] = df["player_id"].astype(int)
    df["time_start"] = df["time_start"].astype(float)
    candidates = []
    for pid, grp in df.groupby("player_id"):
        if pid == exclude_player_id:
            continue
        idx = (grp["time_start"] - origin_time).abs().idxmin()
        row = grp.loc[idx]
        x, y = float(row["x_start"]), float(row["y_start"])
        name = name_map.get(pid, f"Player {pid}")
        candidates.append({"player_id": pid, "name": name, "pos": _clip_pitch(x, y), "dt": abs(row["time_start"] - origin_time)})
    candidates.sort(key=lambda r: r["dt"])
    return candidates[:max_players]


def best_pass_from_teammates(x0, y0, teammates, pass_model, pass_cols, xg_model, xg_cols, max_options=2):
    scored = []
    for tm in teammates:
        xd, yd = tm["pos"]
        p_succ = pass_success(pass_model, pass_cols, x0, y0, xd, yd)
        ev = p_succ * shot_xg(xg_model, xg_cols, xd, yd, damp_long=False)
        scored.append(
            {
                "label": f"Pass to {tm['name']}",
                "ev": ev,
                "p_succ": p_succ,
                "dest": (xd, yd),
            }
        )
    scored.sort(key=lambda r: r["ev"], reverse=True)
    return scored[:max_options]


def find_attacking_pass_samples(name_map, needed=2, x_thresh=20.0):
    """Find pass events starting in attacking third (positive half) from real data files."""
    samples = []
    data_dir = BASE / "more_data"
    for path in sorted(data_dir.glob("*_dynamic_events.csv")):
        if len(samples) >= needed:
            break
        mid = path.name.split("_")[0]
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        req = {"end_type", "x_start", "y_start", "x_end", "y_end"}
        if not req.issubset(df.columns):
            continue
        etype = df["end_type"].fillna("").str.lower()
        mask = (
            etype.eq("pass")
            & df[["x_start", "y_start", "x_end", "y_end"]].notna().all(axis=1)
            & (df["x_start"] > x_thresh)
        )
        passes = df[mask]
        if passes.empty:
            continue
        row = passes.iloc[0]
        pid = int(row.get("player_id", -1)) if pd.notna(row.get("player_id")) else -1
        rid = row.get("receiver_player_id")
        samples.append(
            {
                "match_id": mid,
                "x0": float(row["x_start"]),
                "y0": float(row["y_start"]),
                "x1": float(row["x_end"]),
                "y1": float(row["y_end"]),
                "time_start": _parse_time(row["time_start"]) if "time_start" in row else None,
                "player_id": pid,
                "team_id": int(row["team_id"]) if "team_id" in row and pd.notna(row["team_id"]) else None,
                "player": name_map.get(pid, f"Player {pid}"),
                "receiver": name_map.get(int(rid), "Teammate") if pd.notna(rid) else "Teammate",
            }
        )
    return samples


def sample_pass(match_id: str, nth: int, name_map):
    """Return a pass event dict (start/end/player/name)."""
    path = BASE / "more_data" / f"{match_id}_dynamic_events.csv"
    if not path.exists():
        path = BASE / "skillcorner_download" / f"{match_id}_dynamic_events.csv"
    df = pd.read_csv(path, low_memory=False)
    passes = df[df.get("end_type", "").str.lower() == "pass"]
    passes = passes[passes[["x_start", "y_start", "x_end", "y_end"]].notna().all(axis=1)]
    if passes.empty:
        return None
    row = passes.iloc[min(nth, len(passes) - 1)]
    pid = int(row.get("player_id", -1)) if pd.notna(row.get("player_id")) else -1
    name = name_map.get(pid, f"Player {pid}")
    rid = row.get("receiver_player_id")
    rname = name_map.get(int(rid), "Teammate") if pd.notna(rid) else "Teammate"
    return {
        "match_id": match_id,
        "x0": float(row["x_start"]),
        "y0": float(row["y_start"]),
        "x1": float(row["x_end"]),
        "y1": float(row["y_end"]),
        "player": name,
        "player_id": pid,
        "team_id": int(row["team_id"]) if "team_id" in row and pd.notna(row["team_id"]) else None,
        "time_start": _parse_time(row["time_start"]) if "time_start" in row else None,
        "receiver": rname,
    }


def sample_shot(match_id: str, name_map):
    path = BASE / "more_data" / f"{match_id}_dynamic_events.csv"
    if not path.exists():
        path = BASE / "skillcorner_download" / f"{match_id}_dynamic_events.csv"
    df = pd.read_csv(path, low_memory=False)
    shots = df[df.get("end_type", "").str.lower() == "shot"]
    shots = shots[shots[["x_start", "y_start"]].notna().all(axis=1)]
    if shots.empty:
        return None
    row = shots.iloc[0]
    pid = int(row.get("player_id", -1)) if pd.notna(row.get("player_id")) else -1
    name = name_map.get(pid, f"Player {pid}")
    return {
        "x": float(row["x_start"]),
        "y": float(row["y_start"]),
        "player": name,
    }


def scenario_compare(pass_model, pass_cols, xg_model, xg_cols, drib_model, drib_scaler, drib_cols, name_map):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor("#2e7d32")
    # Sample two real passes from different matches for varied names/positions
    samples = [
        ("1051321", 0),
        ("1149323", 1),
    ]
    scenarios = []
    for mid, nth in samples:
        sp = sample_pass(mid, nth, name_map)
        if sp:
            scenarios.append(
                {
                    "sample": sp,
                    "name": f"Match {mid}",
                    "pos": (sp["x0"], sp["y0"]),
                    "players": {
                        "origin": sp["player"],
                        "left": sp["receiver"],
                        "right": "Teammate",
                    },
                }
            )
    if not scenarios:
        print("❌ No pass samples found; skipping scenario_compare")
        return

    for ax, sc in zip(axes, scenarios):
        draw_pitch(ax)
        x0, y0 = sc["pos"]
        sp = sc["sample"]
        # Choose best EV pass options using real teammate locations; fallback to grid
        teammates = []
        if sp.get("team_id") is not None and sp.get("time_start") is not None and sp.get("player_id") is not None:
            teammates = teammate_positions_at_time(
                sp["match_id"],
                sp["team_id"],
                sp["time_start"],
                sp["player_id"],
                name_map,
                max_players=8,
            )
        pass_opts = best_pass_from_teammates(x0, y0, teammates, pass_model, pass_cols, xg_model, xg_cols, max_options=2)
        if not pass_opts:
            pass_opts = best_pass_candidates(x0, y0, pass_model, pass_cols, xg_model, xg_cols, max_options=2)
        shot_ev = shot_xg(xg_model, xg_cols, x0, y0, damp_long=True)
        options = []
        for i, r in enumerate(pass_opts, 1):
            label = f"{r['label']} (EV={r['ev']:.3f})"
            xd, yd = r["dest"]
            options.append((label, r["ev"], (xd, yd), r["p_succ"]))
        # Dribble: pick best EV from a small forward set
        best_dr = best_dribble_candidate(x0, y0, drib_model, drib_scaler, drib_cols, xg_model, xg_cols)
        dribble_dest = best_dr["dest"]
        dribble_ev = best_dr["ev"]
        p_drib = best_dr["p_keep"]
        # Recommend best
        best = max(
            options + [("Shoot", shot_ev, None, 1.0), ("Dribble", dribble_ev, dribble_dest, p_drib)],
            key=lambda t: t[1]
        )

        # Plot origin
        ax.scatter([x0], [y0], color="orange", s=80, zorder=5)

        # Plot passes
        for label, ev, (xd, yd), p_succ in options:
            ax.arrow(x0, y0, xd - x0, yd - y0, width=0.2, color="tab:blue", alpha=0.7, length_includes_head=True)
            ax.scatter([xd], [yd], color="tab:blue", s=50, zorder=5)
            # Offset labels slightly left/right to reduce overlap
            x_offset = -1.2 if "left" in label.lower() else 1.2
            ax.text(
                xd + x_offset,
                yd + 1.5,
                f"{label}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="white",
                bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1),
            )

        # Dribble arrow
        ax.arrow(x0, y0, dribble_dest[0] - x0, dribble_dest[1] - y0, width=0.2, color="tab:green", alpha=0.7, length_includes_head=True)
        ax.scatter([dribble_dest[0]], [dribble_dest[1]], color="tab:green", s=50, zorder=5)
        ax.text(
            dribble_dest[0],
            dribble_dest[1] + 2.0,
            f"Dribble ({sc['players']['origin']})\nEV={dribble_ev:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="white",
            bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1),
        )

        # Shot marker
        ax.text(
            x0 + 1.5,
            y0 - 3,
            f"Shot EV={shot_ev:.3f}",
            color="white",
            fontsize=8,
            bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1),
        )

        ax.set_title(
            f"{sc['name']} (best: {best[0]} EV={best[1]:.3f})",
            fontsize=11,
            color="white",
            pad=12,
        )
        # Origin label positioned below the pitch to avoid in-field overlap
        ax.text(
            0.5,
            -0.08,
            f"{sc['name']} ({sc['players']['origin']})",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            weight="bold",
            color="white",
        )

    # Legend / EV key on figure
    fig.text(
        0.02,
        0.985,
        "Legend: Orange = ball carrier | Blue arrows = pass EV (P* xG@dest) | Green arrows = dribble EV (Pkeep* xG@dest) | Shot EV shown at origin",
        color="white",
        fontsize=9,
        ha="left",
        va="top",
        bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
    )

    fig.suptitle(
        "Two similar positions, different recommended actions (pass/dribble/shot EV)",
        fontsize=14,
        color="white",
        y=0.93,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    out_path = results_dir / "scenario_compare.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


def load_match_roster(match_id, folder):
    path = Path(folder) / f"match_{match_id}.json"
    roster = {}
    if not path.exists():
        return roster
    try:
        import json
        data = json.loads(path.read_text())
        for p in data.get("players", []):
            pid = p.get("id")
            tid = p.get("team_id")
            if pid is not None and tid is not None:
                roster[int(pid)] = int(tid)
    except Exception:
        pass
    return roster


def shot_with_tracking(pass_model, pass_cols, xg_model, xg_cols, drib_model, drib_scaler, drib_cols, name_map, fin_skill_map):
    """
    Plot a real shot with player positions from tracking (if available).
    Uses match 1289900 from skillcorner_download (has non-empty player_data).
    """
    match_id = "1289900"
    ev_path = BASE / "skillcorner_download" / f"{match_id}_dynamic_events.csv"
    if not ev_path.exists():
        ev_path = BASE / "more_data" / f"{match_id}_dynamic_events.csv"
    tr_path = BASE / "skillcorner_download" / f"tracking_{match_id}.jsonl"
    roster_folder = BASE / "skillcorner_download"
    if not (roster_folder / f"match_{match_id}.json").exists():
        roster_folder = BASE / "more_data"
    roster = load_match_roster(match_id, roster_folder)

    if not ev_path.exists() or not tr_path.exists():
        print("❌ Missing event or tracking file; skipping shot_with_tracking")
        return

    import json

    df = pd.read_csv(ev_path, low_memory=False)
    shots = df[df.get("end_type", "").str.lower() == "shot"]
    shots = shots[shots[["x_start", "y_start"]].notna().all(axis=1)]
    if shots.empty:
        print("❌ No shot events found; skipping shot_with_tracking")
        return
    shot = shots.iloc[0]
    x0, y0 = float(shot["x_start"]), float(shot["y_start"])
    pid = int(shot["player_id"]) if pd.notna(shot.get("player_id")) else -1
    shooter_name = name_map.get(pid, shot.get("player_name", f"Player {pid}"))
    shooter_team = roster.get(pid)
    finishing_skill = fin_skill_map.get(pid, 0.0)
    xg = shot_xg(xg_model, xg_cols, x0, y0, finishing_skill=finishing_skill, damp_long=False)
    best_dr = best_dribble_candidate(x0, y0, drib_model, drib_scaler, drib_cols, xg_model, xg_cols)

    def ts_to_sec(ts):
        if ts is None:
            return None
        s = str(ts)
        parts = [p for p in s.split(":") if p != ""]
        if len(parts) == 3:
            h, m, sec = parts
            return float(h) * 3600 + float(m) * 60 + float(sec)
        if len(parts) == 2:
            m, sec = parts
            return float(m) * 60 + float(sec)
        try:
            return float(s)
        except Exception:
            return None

    frames = json.loads(tr_path.read_text())
    target_time = _parse_time(shot.get("time_start"))
    best_frame = None
    best_diff = float("inf")
    for frm in frames:
        pdlist = frm.get("player_data") or []
        ts = ts_to_sec(frm.get("timestamp"))
        if not pdlist or ts is None or target_time is None:
            continue
        diff = abs(ts - target_time)
        if diff < best_diff:
            best_diff = diff
            best_frame = pdlist

    # Fallback: first frame with players if no timestamp match
    if best_frame is None:
        for frm in frames:
            pdlist = frm.get("player_data") or []
            if pdlist:
                best_frame = pdlist
                break

    if best_frame is None:
        print("❌ No player positions in tracking; skipping shot_with_tracking")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#2e7d32")
    draw_pitch(ax)

    # Helper: best pass to a real teammate from tracking
    best_pass = None
    if shooter_team is not None:
        pass_candidates = []
        for p in best_frame:
            pid_p = p.get("player_id")
            if pid_p is None:
                continue
            if roster.get(int(pid_p)) != shooter_team:
                continue
            if int(pid_p) == pid:
                continue
            px, py = p.get("x"), p.get("y")
            if px is None or py is None:
                continue
            px, py = float(px), float(py)
            p_succ = pass_success(pass_model, pass_cols, x0, y0, px, py)
            ev_downstream = shot_xg(xg_model, xg_cols, px, py, finishing_skill=0.0, damp_long=False)
            ev_val = p_succ * ev_downstream
            pass_candidates.append({"dest": (px, py), "ev": ev_val})
        if pass_candidates:
            pass_candidates.sort(key=lambda r: r["ev"], reverse=True)
            best_pass = pass_candidates[0]

    # Plot players
    for p in best_frame:
        px, py = p.get("x"), p.get("y")
        pid_p = p.get("player_id")
        if px is None or py is None or pid_p is None:
            continue
        px, py = float(px), float(py)
        team_id = roster.get(int(pid_p))
        if team_id is None or shooter_team is None:
            color = "gray"
        elif team_id == shooter_team:
            color = "blue"
        else:
            color = "red"
        size = 70 if int(pid_p) == pid else 40
        ax.scatter([px], [py], color=color, s=size, alpha=0.8)

    # Shot marker and arrow
    ax.scatter([x0], [y0], color="orange", s=90, zorder=5)
    ax.arrow(x0, y0, 52.5 - x0, -y0, width=0.15, color="orange", alpha=0.7, length_includes_head=True)

    # Pass arrow (no inline text)
    if best_pass:
        px, py = best_pass["dest"]
        ax.arrow(x0, y0, px - x0, py - y0, width=0.15, color="tab:blue", alpha=0.7, length_includes_head=True)
    # Dribble arrow (no inline text)
    if best_dr:
        dx, dy = best_dr["dest"]
        ax.arrow(x0, y0, dx - x0, dy - y0, width=0.15, color="tab:green", alpha=0.7, length_includes_head=True)

    # Value box top-left
    lines = [
        f"Shooter: {shooter_name}",
        f"Shot xG={xg:.3f} (fin={finishing_skill:.2f})",
    ]
    if best_pass:
        lines.append(f"Best pass EV={best_pass['ev']:.3f}")
    if best_dr:
        lines.append(f"Best dribble EV={best_dr['ev']:.3f}")
    fig.text(
        0.02,
        0.90,
        "\n".join(lines),
        color="white",
        fontsize=9,
        ha="left",
        va="top",
        bbox=dict(facecolor="black", alpha=0.55, edgecolor="none"),
    )

    fig.text(
        0.02,
        0.985,
        "Legend: Orange = shooter | Blue = teammates | Red = opponents | Gray = unknown",
        color="white",
        fontsize=9,
        ha="left",
        va="top",
        bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
    )
    ax.set_title("Real shot with tracking positions (match 1289900)", fontsize=13, color="white", pad=18)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    out_path = results_dir / "shot_with_tracking.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


def shot_with_tracking_alt(pass_model, pass_cols, xg_model, xg_cols, drib_model, drib_scaler, drib_cols, name_map, fin_skill_map):
    """
    Alternate shot/tracking diagram from a different match (1276976) to showcase a second example.
    """
    match_id = "1276976"
    ev_path = BASE / "skillcorner_download" / f"{match_id}_dynamic_events.csv"
    if not ev_path.exists():
        ev_path = BASE / "more_data" / f"{match_id}_dynamic_events.csv"
    tr_path = BASE / "skillcorner_download" / f"tracking_{match_id}.jsonl"
    roster_folder = BASE / "skillcorner_download"
    if not (roster_folder / f"match_{match_id}.json").exists():
        roster_folder = BASE / "more_data"
    roster = load_match_roster(match_id, roster_folder)

    if not ev_path.exists() or not tr_path.exists():
        print("❌ Missing event or tracking file; skipping shot_with_tracking_alt")
        return

    import json

    df = pd.read_csv(ev_path, low_memory=False)
    shots = df[df.get("end_type", "").str.lower() == "shot"]
    shots = shots[shots[["x_start", "y_start"]].notna().all(axis=1)]
    if shots.empty:
        print("❌ No shot events found; skipping shot_with_tracking_alt")
        return
    # prefer shots closer to the attacking goal (higher x)
    shot = shots.sort_values("x_start", ascending=False).iloc[0]
    x0, y0 = float(shot["x_start"]), float(shot["y_start"])
    pid = int(shot["player_id"]) if pd.notna(shot.get("player_id")) else -1
    shooter_name = name_map.get(pid, shot.get("player_name", f"Player {pid}"))
    shooter_team = roster.get(pid)
    finishing_skill = fin_skill_map.get(pid, 0.0)
    xg = shot_xg(xg_model, xg_cols, x0, y0, finishing_skill=finishing_skill, damp_long=False)

    def ts_to_sec(ts):
        if ts is None:
            return None
        s = str(ts)
        parts = [p for p in s.split(":") if p != ""]
        if len(parts) == 3:
            h, m, sec = parts
            return float(h) * 3600 + float(m) * 60 + float(sec)
        if len(parts) == 2:
            m, sec = parts
            return float(m) * 60 + float(sec)
        try:
            return float(s)
        except Exception:
            return None

    frames = json.loads(tr_path.read_text())
    target_time = _parse_time(shot.get("time_start"))
    best_frame = None
    best_diff = float("inf")
    for frm in frames:
        pdlist = frm.get("player_data") or []
        ts = ts_to_sec(frm.get("timestamp"))
        if not pdlist or ts is None or target_time is None:
            continue
        diff = abs(ts - target_time)
        if diff < best_diff:
            best_diff = diff
            best_frame = pdlist
    if best_frame is None:
        for frm in frames:
            pdlist = frm.get("player_data") or []
            if pdlist:
                best_frame = pdlist
                break
    if best_frame is None:
        print("❌ No player positions in tracking; skipping shot_with_tracking_alt")
        return

    best_pass = None
    if shooter_team is not None:
        pass_candidates = []
        for p in best_frame:
            pid_p = p.get("player_id")
            if pid_p is None:
                continue
            if roster.get(int(pid_p)) != shooter_team or int(pid_p) == pid:
                continue
            px, py = p.get("x"), p.get("y")
            if px is None or py is None:
                continue
            px, py = float(px), float(py)
            p_succ = pass_success(pass_model, pass_cols, x0, y0, px, py)
            ev_downstream = shot_xg(xg_model, xg_cols, px, py, finishing_skill=0.0, damp_long=False)
            ev_val = p_succ * ev_downstream
            pass_candidates.append({"dest": (px, py), "ev": ev_val})
        if pass_candidates:
            pass_candidates.sort(key=lambda r: r["ev"], reverse=True)
            best_pass = pass_candidates[0]

    best_dr = best_dribble_candidate(x0, y0, drib_model, drib_scaler, drib_cols, xg_model, xg_cols)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#2e7d32")
    draw_pitch(ax)

    for p in best_frame:
        px, py = p.get("x"), p.get("y")
        pid_p = p.get("player_id")
        if px is None or py is None or pid_p is None:
            continue
        px, py = float(px), float(py)
        team_id = roster.get(int(pid_p))
        if team_id is None or shooter_team is None:
            color = "gray"
        elif team_id == shooter_team:
            color = "blue"
        else:
            color = "red"
        size = 70 if int(pid_p) == pid else 40
        ax.scatter([px], [py], color=color, s=size, alpha=0.8)

    ax.scatter([x0], [y0], color="orange", s=90, zorder=5)
    ax.arrow(x0, y0, 52.5 - x0, -y0, width=0.15, color="orange", alpha=0.7, length_includes_head=True)

    if best_pass:
        px, py = best_pass["dest"]
        ax.arrow(x0, y0, px - x0, py - y0, width=0.15, color="tab:blue", alpha=0.7, length_includes_head=True)
    if best_dr:
        dx, dy = best_dr["dest"]
        ax.arrow(x0, y0, dx - x0, dy - y0, width=0.15, color="tab:green", alpha=0.7, length_includes_head=True)

    lines = [
        f"Shooter: {shooter_name}",
        f"Shot xG={xg:.3f} (fin={finishing_skill:.2f})",
    ]
    if best_pass:
        lines.append(f"Best pass EV={best_pass['ev']:.3f}")
    if best_dr:
        lines.append(f"Best dribble EV={best_dr['ev']:.3f}")
    fig.text(
        0.02,
        0.90,
        "\n".join(lines),
        color="white",
        fontsize=9,
        ha="left",
        va="top",
        bbox=dict(facecolor="black", alpha=0.55, edgecolor="none"),
    )

    fig.text(
        0.02,
        0.985,
        "Legend: Orange = shooter | Blue = teammates | Red = opponents | Gray = unknown",
        color="white",
        fontsize=9,
        ha="left",
        va="top",
        bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
    )
    ax.set_title("Real shot with tracking positions (match 1276976)", fontsize=13, color="white", pad=18)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    out_path = results_dir / "shot_with_tracking_alt.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")




def scenario_compare_attacking_third(pass_model, pass_cols, xg_model, xg_cols, drib_model, drib_scaler, drib_cols, name_map):
    samples = find_attacking_pass_samples(name_map, needed=2, x_thresh=25.0)
    if len(samples) < 2:
        print("❌ Not enough attacking-third samples found; skipping scenario_compare_attacking_third")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor("#2e7d32")

    for ax, sp in zip(axes, samples[:2]):
        draw_pitch(ax)
        x0, y0 = sp["x0"], sp["y0"]
        name = f"Match {sp['match_id']}"
        players = {"origin": sp["player"], "left": sp["receiver"], "right": "Teammate"}

        teammates = []
        if sp.get("team_id") is not None and sp.get("time_start") is not None and sp.get("player_id") is not None:
            teammates = teammate_positions_at_time(
                sp["match_id"],
                sp["team_id"],
                sp["time_start"],
                sp["player_id"],
                name_map,
                max_players=8,
            )
        pass_opts = best_pass_from_teammates(x0, y0, teammates, pass_model, pass_cols, xg_model, xg_cols, max_options=2)
        if not pass_opts:
            pass_opts = best_pass_candidates(x0, y0, pass_model, pass_cols, xg_model, xg_cols, max_options=2)
        shot_ev = shot_xg(xg_model, xg_cols, x0, y0, damp_long=True)
        options = []
        for r in pass_opts:
            label = f"{r['label']} (EV={r['ev']:.3f})"
            xd, yd = r["dest"]
            options.append((label, r["ev"], (xd, yd), r["p_succ"]))

        best_dr = best_dribble_candidate(x0, y0, drib_model, drib_scaler, drib_cols, xg_model, xg_cols)
        dribble_dest = best_dr["dest"]
        dribble_ev = best_dr["ev"]
        p_drib = best_dr["p_keep"]

        best = max(
            options + [("Shoot", shot_ev, None, 1.0), ("Dribble", dribble_ev, dribble_dest, p_drib)],
            key=lambda t: t[1]
        )

        ax.scatter([x0], [y0], color="orange", s=80, zorder=5)

        for label, ev, (xd, yd), p_succ in options:
            ax.arrow(x0, y0, xd - x0, yd - y0, width=0.2, color="tab:blue", alpha=0.7, length_includes_head=True)
            ax.scatter([xd], [yd], color="tab:blue", s=50, zorder=5)
            x_offset = -1.2 if "left" in label.lower() else 1.2
            ax.text(
                xd + x_offset,
                yd + 1.5,
                f"{label}\nEV={ev:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="white",
                bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1),
            )

        ax.arrow(x0, y0, dribble_dest[0] - x0, dribble_dest[1] - y0, width=0.2, color="tab:green", alpha=0.7, length_includes_head=True)
        ax.scatter([dribble_dest[0]], [dribble_dest[1]], color="tab:green", s=50, zorder=5)
        ax.text(
            dribble_dest[0],
            dribble_dest[1] + 2.0,
            f"Dribble ({players['origin']})\nEV={dribble_ev:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="white",
            bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1),
        )

        ax.text(
            x0 + 1.5,
            y0 - 3,
            f"Shot EV={shot_ev:.3f}",
            color="white",
            fontsize=8,
            bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1),
        )

        ax.set_title(
            f"{name} (best: {best[0]} EV={best[1]:.3f})",
            fontsize=11,
            color="white",
            pad=12,
        )
        ax.text(
            0.5,
            -0.08,
            f"{name} ({players['origin']})",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            weight="bold",
            color="white",
        )

    fig.text(
        0.02,
        0.985,
        "Legend: Orange = ball carrier | Blue arrows = pass EV (P* xG@dest) | Green arrows = dribble EV (Pkeep* xG@dest) | Shot EV shown at origin",
        color="white",
        fontsize=9,
        ha="left",
        va="top",
        bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
    )
    fig.suptitle(
        "Attacking-third positions: recommended action compare (pass/dribble/shot EV)",
        fontsize=14,
        color="white",
        y=0.93,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    out_path = results_dir / "scenario_compare_attacking.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


def attack_timeseries(pass_model, pass_cols, xg_model, xg_cols, drib_model, drib_scaler, drib_cols, name_map):
    # Build a short sequence from a real match: two passes then a shot
    match_id = "1149323"
    p1 = sample_pass(match_id, 0, name_map)
    p2 = sample_pass(match_id, 1, name_map)
    sh = sample_shot(match_id, name_map)

    if not (p1 and p2 and sh):
        print("❌ Could not build timeseries from real data; skipping plot")
        return

    # Force a continuous chain: pass -> dribble -> shot
    pass_start = (p1["x0"], p1["y0"])
    pass_end = (p1["x1"], p1["y1"])
    dribble_end = (p2["x1"], p2["y1"])  # use second sample as a dribble destination
    shot_start = dribble_end  # shot taken from where the dribble ended
    actions = [
        {"player": p1["player"], "type": "pass", "start": pass_start, "end": pass_end, "note": f"to {p1['receiver']}"},
        {"player": p1["receiver"], "type": "dribble", "start": pass_end, "end": dribble_end, "note": ""},
        {"player": sh["player"], "type": "shot", "start": shot_start},
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#2e7d32")
    draw_pitch(ax)

    for i, act in enumerate(actions, 1):
        if act["type"] == "pass":
            x0, y0 = act["start"]
            x1, y1 = act["end"]
            p_succ = pass_success(pass_model, pass_cols, x0, y0, x1, y1)
            ev_downstream = shot_xg(xg_model, xg_cols, x1, y1, damp_long=False)
            ev = p_succ * ev_downstream
            ax.arrow(x0, y0, x1 - x0, y1 - y0, width=0.2, color="tab:blue", alpha=0.7, length_includes_head=True)
            ax.scatter([x0], [y0], color="orange", s=60, zorder=5)
            ax.scatter([x1], [y1], color="tab:blue", s=50, zorder=5)
            ax.text(
                x0,
                y0 - 2,
                f"{i}. {act['player']}",
                ha="center",
                va="top",
                fontsize=8,
                color="white",
                bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1),
            )
            mid_y_offset = 2.5 if i % 2 == 0 else -2.5
            ax.text(
                (x0 + x1) / 2,
                (y0 + y1) / 2 + mid_y_offset,
                f"Pass EV={ev:.3f}",
                ha="center",
                fontsize=8,
                color="white",
                bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1),
            )
        elif act["type"] == "dribble":
            x0, y0 = act["start"]
            x1, y1 = act["end"]
            p_keep = dribble_success(drib_model, drib_scaler, drib_cols, x0, y0, x1, y1)
            ev_downstream = shot_xg(xg_model, xg_cols, x1, y1, damp_long=False)
            ev = p_keep * ev_downstream
            ax.arrow(x0, y0, x1 - x0, y1 - y0, width=0.2, color="tab:green", alpha=0.7, length_includes_head=True)
            ax.scatter([x0], [y0], color="orange", s=60, zorder=5)
            ax.scatter([x1], [y1], color="tab:green", s=50, zorder=5)
            ax.text(
                x0,
                y0 - 2,
                f"{i}. {act['player']}",
                ha="center",
                va="top",
                fontsize=8,
                color="white",
                bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1),
            )
            ax.text(
                (x0 + x1) / 2,
                (y0 + y1) / 2 + 2.5,
                f"Dribble EV={ev:.3f}",
                ha="center",
                fontsize=8,
                color="white",
                bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1),
            )
        else:  # shot
            x0, y0 = act["start"]
            xg = shot_xg(xg_model, xg_cols, x0, y0, damp_long=True)
            ax.arrow(x0, y0, 52.5 - x0, -y0, width=0.2, color="red", alpha=0.7, length_includes_head=True)
            ax.scatter([x0], [y0], color="red", s=80, zorder=5)
            ax.text(
                x0,
                y0 - 2,
                f"{i}. {act['player']}",
                ha="center",
                va="top",
                fontsize=8,
                color="white",
                bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1),
            )
            ax.text(
                x0 + 2.5,
                y0 + 2.5,
                f"Shot xG={xg:.3f}",
                fontsize=8,
                color="white",
                bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1),
            )

    # Legend block
    fig.text(
        0.02,
        0.985,
        "Legend: Orange = ball carrier | Blue = pass (EV P* xG@dest) | Green = dribble (EV Pkeep* xG@dest) | Red = shot xG",
        color="white",
        fontsize=9,
        ha="left",
        va="top",
        bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
    )
    ax.set_title(
        "Attacking sequence with EPV-style EV per action",
        fontsize=13,
        color="white",
        pad=24,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    out_path = results_dir / "attack_timeseries.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


def action_chain_goal(pass_model, pass_cols, xg_model, xg_cols, drib_model, drib_scaler, drib_cols, name_map):
    """Three consecutive real events ending in a goal (lead_to_goal=True)."""
    match_id, chain = find_goal_chain(name_map)
    if not chain:
        print("❌ No goal chain found; skipping action_chain_goal")
        return
    if len(chain) < 2:
        print("❌ Goal chain too short; skipping action_chain_goal")
        return

    # Compute actual EV for the first action and a best alternative from the same start
    alt_info = None
    actual_info = None
    first = chain[0]
    x0, y0 = first["start"]
    shot_from_origin = shot_xg(xg_model, xg_cols, x0, y0, damp_long=True)
    best_dr = best_dribble_candidate(x0, y0, drib_model, drib_scaler, drib_cols, xg_model, xg_cols)
    best_pass = best_pass_candidates(x0, y0, pass_model, pass_cols, xg_model, xg_cols, max_options=1)
    candidates = [("Shot", shot_from_origin, None)]
    if best_dr:
        candidates.append(("Dribble", best_dr["ev"], best_dr["dest"]))
    if best_pass:
        candidates.append(("Pass", best_pass[0]["ev"], best_pass[0]["dest"]))
    alt_best = max(candidates, key=lambda t: t[1])

    if first["type"] == "pass" and "end" in first:
        p_succ = pass_success(pass_model, pass_cols, x0, y0, first["end"][0], first["end"][1])
        actual_ev = p_succ * shot_xg(xg_model, xg_cols, first["end"][0], first["end"][1], damp_long=False)
        actual_info = ("Actual pass EV", actual_ev)
    elif first["type"] == "dribble" and "end" in first:
        p_keep = dribble_success(drib_model, drib_scaler, drib_cols, x0, y0, first["end"][0], first["end"][1])
        actual_ev = p_keep * shot_xg(xg_model, xg_cols, first["end"][0], first["end"][1], damp_long=False)
        actual_info = ("Actual dribble EV", actual_ev)
    else:
        actual_info = ("Actual action EV", shot_from_origin)
    alt_info = (f"Best alternative: {alt_best[0]} EV", alt_best[1])

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#2e7d32")
    draw_pitch(ax)

    def draw_action(act, idx):
        if act["type"] == "pass":
            x0, y0 = act["start"]
            x1, y1 = act["end"]
            p_succ = pass_success(pass_model, pass_cols, x0, y0, x1, y1)
            ev_downstream = shot_xg(xg_model, xg_cols, x1, y1, damp_long=False)
            ev = p_succ * ev_downstream
            ax.arrow(x0, y0, x1 - x0, y1 - y0, width=0.2, color="tab:blue", alpha=0.8, length_includes_head=True)
            ax.scatter([x0], [y0], color="orange", s=70, zorder=5)
            ax.scatter([x1], [y1], color="tab:blue", s=55, zorder=6)
            ax.text(x0, y0 - 2, f"{idx}. {act['player']}", ha="center", va="top", fontsize=8, color="white",
                    bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1))
            ax.text(x1, y1 + 2, f"Pass EV={ev:.3f}", ha="center", va="bottom", fontsize=8, color="white",
                    bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1))
            return (x1, y1)
        elif act["type"] == "dribble":
            x0, y0 = act["start"]
            x1, y1 = act["end"]
            p_keep = dribble_success(drib_model, drib_scaler, drib_cols, x0, y0, x1, y1)
            ev_downstream = shot_xg(xg_model, xg_cols, x1, y1, damp_long=False)
            ev = p_keep * ev_downstream
            ax.arrow(x0, y0, x1 - x0, y1 - y0, width=0.2, color="tab:green", alpha=0.8, length_includes_head=True)
            ax.scatter([x0], [y0], color="orange", s=70, zorder=5)
            ax.scatter([x1], [y1], color="tab:green", s=55, zorder=6)
            ax.text(x0, y0 - 2, f"{idx}. {act['player']}", ha="center", va="top", fontsize=8, color="white",
                    bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1))
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 2.2, f"Dribble EV={ev:.3f}", ha="center", fontsize=8, color="white",
                    bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1))
            return (x1, y1)
        else:  # shot (goal)
            x0, y0 = act["start"]
            xg = shot_xg(xg_model, xg_cols, x0, y0, damp_long=False)
            ax.arrow(x0, y0, 52.5 - x0, -y0, width=0.2, color="red", alpha=0.8, length_includes_head=True)
            ax.scatter([x0], [y0], color="red", s=70, zorder=7)
            ax.text(x0, y0 - 2, f"{idx}. {act['player']}", ha="center", va="top", fontsize=8, color="white",
                    bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1))
            ax.text(x0 + 2.5, y0 + 2.5, f"Shot (Goal) xG={xg:.3f}", fontsize=8, color="white",
                    bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1))
            return (x0, y0)

    for i, act in enumerate(chain[-3:], 1):
        draw_action(act, i)

    if actual_info and alt_info:
        ax.text(
            x0 - 5,
            y0 + 6,
            f"{actual_info[0]}={actual_info[1]:.3f}\n{alt_info[0]}={alt_info[1]:.3f}",
            fontsize=8,
            color="white",
            ha="left",
            va="bottom",
            bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=1),
        )

    fig.text(
        0.02,
        0.985,
        "Legend: Orange = ball carrier | Blue = pass EV | Green = dribble EV | Red = shot xG (goal)",
        color="white",
        fontsize=9,
        ha="left",
        va="top",
        bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
    )
    ax.set_title("Three-action chain ending in a goal (best pass/dribble EV)", fontsize=13, color="white", pad=24)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    out_path = results_dir / "action_chain_goal.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


def outside_box_decision(pass_model, pass_cols, xg_model, xg_cols, drib_model, drib_scaler, drib_cols, name_map):
    """Single real situation outside the box: best pass/dribble/shot EV."""
    sample = find_outside_box_pass(name_map, x_min=20, x_max=40, y_abs_max=25)
    if not sample:
        print("❌ No outside-box sample found; skipping outside_box_decision")
        return

    x0, y0 = sample["x"], sample["y"]
    teammates = []
    if sample.get("team_id") is not None and sample.get("time_start") is not None and sample.get("player_id") is not None:
        teammates = teammate_positions_at_time(
            sample["match_id"],
            sample["team_id"],
            sample["time_start"],
            sample["player_id"],
            name_map,
            max_players=8,
        )

    pass_opts = best_pass_from_teammates(x0, y0, teammates, pass_model, pass_cols, xg_model, xg_cols, max_options=1)
    if not pass_opts:
        pass_opts = best_pass_candidates(x0, y0, pass_model, pass_cols, xg_model, xg_cols, max_options=1)
    best_pass = pass_opts[0]

    best_dr = best_dribble_candidate(x0, y0, drib_model, drib_scaler, drib_cols, xg_model, xg_cols)
    shot_ev = shot_xg(xg_model, xg_cols, x0, y0, damp_long=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#2e7d32")
    draw_pitch(ax)

    # Origin
    ax.scatter([x0], [y0], color="orange", s=80, zorder=5)
    ax.text(x0, y0 + 2, f"{sample['player']} ({sample['team_shortname']})", ha="center", va="bottom", fontsize=9, color="white")

    # Pass option
    px, py = best_pass["dest"]
    p_ev = best_pass["ev"]
    ax.arrow(x0, y0, px - x0, py - y0, width=0.2, color="tab:blue", alpha=0.8, length_includes_head=True)
    ax.scatter([px], [py], color="tab:blue", s=50, zorder=5)
    ax.text(px, py + 1.5, f"Best pass\nEV={p_ev:.3f}", ha="center", va="bottom", fontsize=8, color="white",
            bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1))

    # Dribble option
    dx, dy = best_dr["dest"]
    d_ev = best_dr["ev"]
    ax.arrow(x0, y0, dx - x0, dy - y0, width=0.2, color="tab:green", alpha=0.8, length_includes_head=True)
    ax.scatter([dx], [dy], color="tab:green", s=50, zorder=5)
    ax.text(dx, dy + 1.5, f"Best dribble\nEV={d_ev:.3f}", ha="center", va="bottom", fontsize=8, color="white",
            bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1))

    # Shot option
    ax.arrow(x0, y0, 52.5 - x0, -y0, width=0.2, color="red", alpha=0.8, length_includes_head=True)
    ax.text(x0 + 2, y0 - 2, f"Shot xG={shot_ev:.3f}", fontsize=8, color="white",
            bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1))

    fig.text(
        0.02,
        0.985,
        "Legend: Orange = ball carrier | Blue = best pass EV | Green = best dribble EV | Red = shot xG",
        color="white",
        fontsize=9,
        ha="left",
        va="top",
        bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
    )
    ax.set_title("Outside the box: real situation with best pass/dribble/shot EV", fontsize=12, color="white", pad=18)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    out_path = results_dir / "outside_box_decision.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


def outside_box_artur_decision(pass_model, pass_cols, xg_model, xg_cols, drib_model, drib_scaler, drib_cols, name_map):
    """Artur-specific snapshot around 57th minute (real event)."""
    # Hardcoded event identified near 57': match 793315, event 8_645
    sample = get_event("793315", "8_645", player_substr="Artur")
    if not sample:
        print("❌ Artur event not found; skipping outside_box_artur_decision")
        return

    x0, y0 = sample["x"], sample["y"]
    teammates = []
    if sample.get("team_id") is not None and sample.get("time_start") is not None and sample.get("player_id") is not None:
        teammates = teammate_positions_at_time(
            sample["match_id"],
            sample["team_id"],
            sample["time_start"],
            sample["player_id"],
            name_map,
            max_players=8,
        )

    pass_opts = best_pass_from_teammates(x0, y0, teammates, pass_model, pass_cols, xg_model, xg_cols, max_options=1)
    if not pass_opts:
        pass_opts = best_pass_candidates(x0, y0, pass_model, pass_cols, xg_model, xg_cols, max_options=1)
    best_pass = pass_opts[0]

    best_dr = best_dribble_candidate(x0, y0, drib_model, drib_scaler, drib_cols, xg_model, xg_cols)
    shot_ev = shot_xg(xg_model, xg_cols, x0, y0, damp_long=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#2e7d32")
    draw_pitch(ax)

    ax.scatter([x0], [y0], color="orange", s=80, zorder=5)
    ax.text(x0, y0 + 2, f"{sample['player']} ({sample['team_shortname']})\n~{sample['time_start']:.1f}s", ha="center", va="bottom", fontsize=9, color="white")

    px, py = best_pass["dest"]
    p_ev = best_pass["ev"]
    ax.arrow(x0, y0, px - x0, py - y0, width=0.2, color="tab:blue", alpha=0.8, length_includes_head=True)
    ax.scatter([px], [py], color="tab:blue", s=50, zorder=5)
    ax.text(px, py + 1.5, f"Best pass\nEV={p_ev:.3f}", ha="center", va="bottom", fontsize=8, color="white",
            bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1))

    dx, dy = best_dr["dest"]
    d_ev = best_dr["ev"]
    ax.arrow(x0, y0, dx - x0, dy - y0, width=0.2, color="tab:green", alpha=0.8, length_includes_head=True)
    ax.scatter([dx], [dy], color="tab:green", s=50, zorder=5)
    ax.text(dx, dy + 1.5, f"Best dribble\nEV={d_ev:.3f}", ha="center", va="bottom", fontsize=8, color="white",
            bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1))

    ax.arrow(x0, y0, 52.5 - x0, -y0, width=0.2, color="red", alpha=0.8, length_includes_head=True)
    ax.text(x0 + 2, y0 - 2, f"Shot xG={shot_ev:.3f}", fontsize=8, color="white",
            bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1))

    fig.text(
        0.02,
        0.985,
        "Legend: Orange = ball carrier | Blue = best pass EV | Green = best dribble EV | Red = shot xG",
        color="white",
        fontsize=9,
        ha="left",
        va="top",
        bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
    )
    ax.set_title("Artur outside-box decision (~57')", fontsize=12, color="white", pad=18)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    out_path = results_dir / "outside_box_artur_57.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


def _locate_match_files(match_id: str):
    """Find event, tracking, and metadata files for a match."""
    for folder in [BASE / "skillcorner_download", BASE / "more_data"]:
        ev_path = folder / f"{match_id}_dynamic_events.csv"
        tr_path = folder / f"{match_id}_tracking_extrapolated.jsonl"
        if not tr_path.exists():
            tr_path = folder / f"tracking_{match_id}.jsonl"
        meta_path = folder / f"match_{match_id}.json"
        if ev_path.exists() and tr_path.exists():
            return ev_path, tr_path, meta_path if meta_path.exists() else None
    return None, None, None


def _frame_for_event(frames, target_frame_num, target_time):
    """Pick the tracking frame closest to the event (by frame number or timestamp)."""
    idx_by_num = {}
    for idx, fr in enumerate(frames):
        fnum = fr.get("frame")
        if fnum is not None:
            idx_by_num[int(fnum)] = idx
    if target_frame_num is not None and int(target_frame_num) in idx_by_num:
        idx = idx_by_num[int(target_frame_num)]
        return idx, frames[idx].get("player_data") or []
    if target_time is not None:
        best_idx, best_diff = None, float("inf")
        for i, fr in enumerate(frames):
            ts = _ts_to_sec(fr.get("timestamp"))
            if ts is None:
                continue
            diff = abs(ts - target_time)
            if diff < best_diff:
                best_diff = diff
                best_idx = i
        if best_idx is not None:
            return best_idx, frames[best_idx].get("player_data") or []
    for i, fr in enumerate(frames):
        pdlist = fr.get("player_data") or []
        if pdlist:
            return i, pdlist
    return 0, frames[0].get("player_data") or []


def _keeper_ids_from_meta(meta_path: Path):
    """Extract keeper ids/trackable_objects per team from raw match JSON."""
    keepers_by_team = {}
    try:
        raw = json.loads(meta_path.read_text())
        for p in raw.get("players", []):
            role = (p.get("player_role") or {}).get("name", "") or ""
            pos_group = (p.get("player_role") or {}).get("position_group", "") or ""
            if "keeper" in role.lower() or "keeper" in pos_group.lower():
                tid = p.get("team_id")
                if tid is None:
                    continue
                tid = int(tid)
                roster_id = p.get("id")
                track_obj = p.get("trackable_object")
                ids = keepers_by_team.setdefault(tid, set())
                if roster_id is not None:
                    ids.add(int(roster_id))
                if track_obj is not None:
                    try:
                        ids.add(int(track_obj))
                    except Exception:
                        pass
    except Exception:
        return {}
    return keepers_by_team


def _frame_to_period(meta, frame_idx):
    """Map a frame index to period using match_periods if available."""
    try:
        mps = meta.get("match_periods") or []
        for mp in mps:
            sf, ef = mp.get("start_frame"), mp.get("end_frame")
            per = mp.get("period") or mp.get("name")
            if sf is not None and ef is not None and sf <= frame_idx <= ef:
                return int(per) if str(per).isdigit() else None
    except Exception:
        return None
    return None


def _infer_goal_x(team_id, events_df, frame_players, keeper_ids, meta, frames, frame_idx):
    """
    Infer attacking goal x for the given team using multiple checks:
      1) If keeper present in the frame: attack opposite side of keeper.
      2) Else, use historical shots for that team in the current period if available; otherwise all shots.
      3) Fallback: assume normalized data (+52.5).
    """
    # 1) Keeper location in this frame
    if team_id is not None and keeper_ids.get(team_id):
        k_ids = keeper_ids[team_id]
        for p in frame_players:
            pid = p.get("player_id")
            if pid is None:
                continue
            if int(pid) in k_ids and p.get("x") is not None:
                gx = float(p["x"])
                return 52.5 if gx < 0 else -52.5

    # 2) Shot history in events (prefer current period if available)
    if team_id is not None and not events_df.empty and "end_type" in events_df:
        shots = events_df[events_df["end_type"].str.lower() == "shot"]
        shots = shots[shots["team_id"] == team_id]
        if not shots.empty and "x_start" in shots:
            # Filter by current period if we can determine it
            current_period = _frame_to_period(meta, frame_idx) or get_period_safe(frames, frame_idx, meta)
            if current_period is not None and "frame_start" in shots:
                shots = shots.copy()
                shots["frame_start_val"] = shots["frame_start"]
                shots = shots[pd.notna(shots["frame_start_val"])]
                def shot_period(row):
                    try:
                        f = int(row["frame_start_val"])
                        return _frame_to_period(meta, f)
                    except Exception:
                        return None
                shots["period_guess"] = shots.apply(shot_period, axis=1)
                period_shots = shots[shots["period_guess"] == current_period]
                if not period_shots.empty:
                    shots = period_shots
            mean_x = shots["x_start"].dropna().mean()
            if not np.isnan(mean_x):
                return 52.5 if mean_x >= 0 else -52.5

    # 3) Fallback: normalized coordinates
    return 52.5


def _nearest_action(events: pd.DataFrame, target_frame: int | None, target_time: float | None):
    """Return the nearest actionable event (shot/pass/dribble/carry/cross) to a target frame/time."""
    allowed = {"shot", "pass", "dribble", "carry", "cross"}
    if "end_type" not in events:
        return None
    df = events.copy()
    df["etype"] = df["end_type"].str.lower()
    df = df[df["etype"].isin(allowed)]
    df = df[df[["x_start", "y_start"]].notna().all(axis=1)]
    df["frame_val"] = df["frame_start"] if "frame_start" in df else np.nan
    df["time_val"] = df["time_start"].apply(_parse_time)
    df = df[df["time_val"].notna() | df["frame_val"].notna()]
    if df.empty:
        return None

    def dist(row):
        d_frame = abs(row["frame_val"] - target_frame) if target_frame is not None and pd.notna(row["frame_val"]) else np.inf
        d_time = abs(row["time_val"] - target_time) if target_time is not None and pd.notna(row["time_val"]) else np.inf
        return min(d_frame, d_time)

    df["dist"] = df.apply(dist, axis=1)
    df = df.sort_values(["dist", "time_val"]).reset_index(drop=True)
    return df.iloc[0] if not df.empty else None


def render_tracking_play(match_id: str, frame: int | None = None, match_time_str: str | None = None):
    """
    Render a single nearest action around a frame/time using real tracking, pitch control, and skills.

    Inputs:
      - match_id (str)
      - frame (int) or match_time_str ("MM:SS" or seconds as string). Provide one.

    Saves a PNG with the same style as shot_with_tracking_alt.png.
    """
    ev_path, tr_path, meta_path = _locate_match_files(match_id)
    if not ev_path or not tr_path or not meta_path:
        print(f"❌ Missing files for match {match_id} (events/tracking/meta required)")
        return None

    target_time = _parse_time(match_time_str) if match_time_str else None
    target_frame = int(frame) if frame is not None else None

    # Load models and skills
    pass_model, pass_cols = load_model(BASE / "passing_model_improved.pkl")
    xg_model, xg_cols = load_model(BASE / "xg_model_improved.pkl")
    drib_model, drib_scaler, drib_cols = load_dribble(BASE / "models/dribbling_model_full.pkl")
    name_map = load_player_name_map()
    fin_skill_map = load_finishing_skill_map()
    pass_skill_map = load_passing_skill_map()
    drib_skill_map = load_dribbling_skill_map()

    events = pd.read_csv(ev_path, low_memory=False)
    chosen = _nearest_action(events, target_frame, target_time)
    if chosen is None:
        print(f"❌ No actionable events found near the specified time/frame for match {match_id}")
        return None

    x0, y0 = float(chosen["x_start"]), float(chosen["y_start"])
    x1 = float(chosen["x_end"]) if pd.notna(chosen.get("x_end")) else None
    y1 = float(chosen["y_end"]) if pd.notna(chosen.get("y_end")) else None
    etype = chosen["etype"]
    team_id = int(chosen["team_id"]) if pd.notna(chosen.get("team_id")) else None
    player_id = int(chosen["player_id"]) if pd.notna(chosen.get("player_id")) else -1
    player_name = name_map.get(player_id, chosen.get("player_name", f"Player {player_id}"))
    target_frame_num = int(chosen["frame_val"]) if pd.notna(chosen.get("frame_val")) else None
    target_time = chosen["time_val"]

    frames = load_tracking_jsonl(tr_path)
    meta = load_match_meta_robust(meta_path)
    keeper_ids = _keeper_ids_from_meta(meta_path)
    frame_idx, frame_players = _frame_for_event(frames, target_frame_num, target_time)
    pc_runner = PitchControlRunner(meta, frames)
    goal_x = _infer_goal_x(team_id, events, frame_players, keeper_ids, meta, frames, frame_idx)

    roster = {int(pid): int(tid) for pid, tid in meta.get("pid_to_teamid", {}).items()}

    finishing_skill = fin_skill_map.get(player_id, 0.0)
    shot_ev = shot_xg(xg_model, xg_cols, x0, y0, finishing_skill=finishing_skill, damp_long=True, goal_x=goal_x)

    teammates = []
    for p in frame_players:
        pid_p = p.get("player_id")
        if pid_p is None or int(pid_p) == player_id:
            continue
        tid = roster.get(int(pid_p))
        if tid is None or team_id is None or tid != team_id:
            continue
        px, py = p.get("x"), p.get("y")
        if px is None or py is None:
            continue
        teammates.append({"player_id": int(pid_p), "pos": (float(px), float(py))})

    best_pass = None
    if teammates:
        pass_candidates = []
        for tm in teammates:
            px, py = tm["pos"]
            p_succ = pass_success_real(
                pass_model,
                pass_cols,
                x0,
                y0,
                px,
                py,
                frame_players,
                team_id,
                roster,
                pc_runner,
                frame_idx,
                pass_skill_map,
                player_id,
            )
            ev_downstream = shot_xg(xg_model, xg_cols, px, py, finishing_skill=fin_skill_map.get(tm["player_id"], 0.0), damp_long=False, goal_x=goal_x)
            pass_candidates.append({"dest": (px, py), "ev": p_succ * ev_downstream, "p_succ": p_succ})
        pass_candidates.sort(key=lambda r: r["ev"], reverse=True)
        if pass_candidates:
            best_pass = pass_candidates[0]

    dribble_offsets = [(6, 0), (8, 3), (8, -3), (10, 0)] if goal_x > 0 else [(-6, 0), (-8, 3), (-8, -3), (-10, 0)]
    best_dr = None
    for dx, dy in dribble_offsets:
        xd, yd = _clip_pitch(x0 + dx, y0 + dy)
        p_keep = dribble_success_real(
            drib_model,
            drib_scaler,
            drib_cols,
            x0,
            y0,
            xd,
            yd,
            frame_players,
            team_id,
            roster,
            pc_runner,
            frame_idx,
            drib_skill_map,
            player_id,
            goal_x=goal_x,
        )
        ev_downstream = shot_xg(xg_model, xg_cols, xd, yd, finishing_skill=fin_skill_map.get(player_id, 0.0), damp_long=False, goal_x=goal_x)
        ev_val = p_keep * ev_downstream
        if best_dr is None or ev_val > best_dr["ev"]:
            best_dr = {"dest": (xd, yd), "ev": ev_val, "p_keep": p_keep}

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#2e7d32")
    draw_pitch(ax)

    for p in frame_players:
        px, py = p.get("x"), p.get("y")
        pid_p = p.get("player_id")
        if px is None or py is None or pid_p is None:
            continue
        px, py = float(px), float(py)
        tid = roster.get(int(pid_p))
        if tid is None or team_id is None:
            color = "gray"
        elif tid == team_id:
            color = "blue"
        else:
            color = "red"
        size = 70 if int(pid_p) == player_id else 40
        ax.scatter([px], [py], color=color, s=size, alpha=0.8)

    ax.scatter([x0], [y0], color="orange", s=90, zorder=6)
    ax.arrow(x0, y0, goal_x - x0, -y0, width=0.15, color="orange", alpha=0.7, length_includes_head=True)

    if best_pass:
        px, py = best_pass["dest"]
        ax.arrow(x0, y0, px - x0, py - y0, width=0.15, color="tab:blue", alpha=0.7, length_includes_head=True)
    if best_dr:
        dx, dy = best_dr["dest"]
        ax.arrow(x0, y0, dx - x0, dy - y0, width=0.15, color="tab:green", alpha=0.7, length_includes_head=True)

    lines = [
        f"Player: {player_name} ({etype})",
        f"Shot xG={shot_ev:.3f} (fin={finishing_skill:.2f})",
    ]
    if best_pass:
        lines.append(f"Best pass EV={best_pass['ev']:.3f}")
    if best_dr:
        lines.append(f"Best dribble EV={best_dr['ev']:.3f}")
    fig.text(
        0.02,
        0.90,
        "\n".join(lines),
        color="white",
        fontsize=9,
        ha="left",
        va="top",
        bbox=dict(facecolor="black", alpha=0.55, edgecolor="none"),
    )

    fig.text(
        0.02,
        0.985,
        "Legend: Orange = ball carrier | Blue = teammates | Red = opponents | Gray = unknown",
        color="white",
        fontsize=9,
        ha="left",
        va="top",
        bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
    )
    title_time = f"{target_time:.1f}s" if target_time is not None else f"frame {frame_idx}"
    ax.set_title(f"Nearest action with tracking — match {match_id} @ {title_time}", fontsize=13, color="white", pad=18)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    out_path = results_dir / f"{match_id}_tracking.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")
    return out_path


def main():
    import sys

    match_id = sys.argv[1] if len(sys.argv) > 1 else "1276976"
    if len(sys.argv) > 2:
        arg = sys.argv[2]
        if ":" in arg:
            render_tracking_play(match_id, frame=None, match_time_str=arg)
        else:
            try:
                render_tracking_play(match_id, frame=int(arg), match_time_str=None)
            except ValueError:
                render_tracking_play(match_id, frame=None, match_time_str=arg)
    else:
        render_tracking_play(match_id, frame=None, match_time_str=None)


if __name__ == "__main__":
    main()
