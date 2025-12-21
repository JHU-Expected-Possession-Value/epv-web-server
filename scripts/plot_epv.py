"""
Plot EPV snapshot for a given match/time.

Usage (optional CLI):
  python3 scripts/plot_epv.py --match-id 1039803 --time 12:34
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epv_calculator import EPVCalculator
from pitch_control import load_tracking_jsonl


def _parse_time_str(value: str) -> float:
    raw = str(value).strip()
    if not raw:
        raise ValueError("Empty time string")

    parts = raw.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"Unexpected time format: {value}")


def _parse_event_time(value) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return _parse_time_str(str(value))
    except Exception:
        return None


def _attack_dir_from_row(row: pd.Series) -> int:
    side = str(row.get("attacking_side", "")).lower()
    if "right_to_left" in side:
        return -1
    if "left_to_right" in side:
        return 1
    return 1


def _resolve_data_dir(match_id: str) -> Path:
    candidates = [
        ROOT / "more_data",
        ROOT / "skillcorner_download",
        ROOT / "data" / "skillcorner_download",
    ]
    for candidate in candidates:
        if (candidate / f"{match_id}_dynamic_events.csv").exists():
            return candidate
    raise FileNotFoundError(f"Match {match_id} not found in {candidates}")


def _resolve_skill_path(filename: str) -> Path:
    candidates = [ROOT / "data" / filename, ROOT / filename]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing skill file {filename}")


def _build_calc() -> EPVCalculator:
    dribbling_model = ROOT / "models" / "dribbling_model_full.pkl"
    if not dribbling_model.exists():
        dribbling_model = ROOT / "models" / "dribbling_model_proper_split.pkl"

    return EPVCalculator(
        xg_model_path=ROOT / "models" / "xg_model_improved.pkl",
        passing_model_path=ROOT / "models" / "passing_model_improved.pkl",
        dribbling_model_path=dribbling_model,
        xg_skills_path=_resolve_skill_path("player_id_to_finishing_skill.csv"),
        passing_skills_path=_resolve_skill_path("player_id_to_passing_skill.csv"),
        dribbling_skills_path=_resolve_skill_path("player_id_to_skill.csv"),
    )


def _find_action_event(events: pd.DataFrame, target_seconds: float) -> Tuple[pd.Series, float]:
    event_type = events.get("event_type", pd.Series("", index=events.index)).astype(str).str.lower()
    end_type = events.get("end_type", pd.Series("", index=events.index)).astype(str).str.lower()

    carry_raw = events.get("carry", pd.Series(False, index=events.index)).astype(str).str.lower()
    carry = carry_raw.isin(["true", "1", "yes"])

    is_shot = end_type.eq("shot")
    is_pass = end_type.str.contains("pass", na=False)
    is_dribble = event_type.eq("player_possession") & carry

    mask = is_shot | is_pass | is_dribble
    candidates = events[mask].copy()

    for col in ["player_id", "team_id", "frame_start", "x_start", "y_start"]:
        if col in candidates.columns:
            candidates = candidates[candidates[col].notna()]

    if candidates.empty:
        raise RuntimeError("No shot/pass/dribble events found in match")

    if "time_start" in candidates.columns:
        candidates["event_seconds"] = candidates["time_start"].apply(_parse_event_time)
    else:
        candidates["event_seconds"] = np.nan

    if candidates["event_seconds"].isna().all():
        minute = pd.to_numeric(candidates.get("minute_start"), errors="coerce").fillna(0)
        second = pd.to_numeric(candidates.get("second_start"), errors="coerce").fillna(0)
        candidates["event_seconds"] = minute * 60 + second

    candidates = candidates[candidates["event_seconds"].notna()]
    if candidates.empty:
        raise RuntimeError("No events with valid timing found")

    idx = (candidates["event_seconds"] - target_seconds).abs().idxmin()
    row = candidates.loc[idx]
    return row, float(row["event_seconds"])


def _nearest_frame(tracking: Dict[int, dict], frame: int) -> Tuple[int, dict]:
    if frame in tracking:
        return frame, tracking[frame]
    if not tracking:
        raise RuntimeError("No tracking data loaded")
    nearest = min(tracking.keys(), key=lambda k: abs(k - frame))
    return nearest, tracking[nearest]


def _player_xy(frame_data: dict, player_id: int, fallback_x: float, fallback_y: float) -> Tuple[float, float]:
    for p in frame_data.get("player_data", []):
        if p.get("player_id") == player_id:
            return float(p.get("x", fallback_x)), float(p.get("y", fallback_y))
    return float(fallback_x), float(fallback_y)


def _compute_best_pass(calc, x, y, frame, pid, tid, tracking_dict, frame_data):
    return calc.evaluate_best_pass(
        x, y, frame, frame_data, pid, tid, tracking_dict, depth=0, return_dest=True
    )


def _compute_best_dribble(calc, x, y, frame, pid, tid, tracking_dict, frame_data):
    best_q, best_dest = -1e9, None
    for dest_x, dest_y in calc._sample_dribble_destinations(x, y):
        distance = ((dest_x - x) ** 2 + (dest_y - y) ** 2) ** 0.5
        if distance < 1.0:
            continue
        if calc.dribbling_model and getattr(calc.dribbling_model, "is_trained", False):
            p_success = calc._evaluate_dribble_with_model(
                x, y, dest_x, dest_y, frame, frame_data, pid, tid
            )
        else:
            p_success = 0.7 if distance < 5 else 0.5
        v_success = calc.get_epv(dest_x, dest_y, frame + 20, pid, tid, tracking_dict, depth=1)
        v_fail = -0.08
        q = p_success * v_success + (1 - p_success) * v_fail
        if q > best_q:
            best_q, best_dest = q, (dest_x, dest_y)
    return best_q, best_dest


def _sample_dribble_destinations_dir(x: float, y: float, attack_dir: int, n: int = 4):
    destinations = []
    for dist in [3, 5]:
        destinations.append((x + dist * attack_dir, y))
        destinations.append((x + dist * attack_dir, y + 2))
        destinations.append((x + dist * attack_dir, y - 2))
    return destinations[:n]


def _draw_pitch(ax):
    ax.set_xlim(-52.5, 52.5)
    ax.set_ylim(-34, 34)
    ax.add_patch(plt.Rectangle((-52.5, -34), 105, 68, facecolor="#d7f0d2", edgecolor="white", lw=1.2, zorder=0))
    ax.axvline(0, color="white", lw=1.0)
    ax.axhline(0, color="white", lw=0.3, ls=":")
    # Right side boxes + goal
    ax.add_patch(plt.Rectangle((52.5 - 16.5, -20.15), 16.5, 40.3, fill=False, edgecolor="white", lw=1.2))
    ax.add_patch(plt.Rectangle((52.5 - 5.5, -9.16), 5.5, 18.32, fill=False, edgecolor="white", lw=1.2))
    ax.add_patch(plt.Rectangle((52.5, -3.66), 0.5, 7.32, facecolor="white", edgecolor="white"))
    # Left side boxes + goal
    ax.add_patch(plt.Rectangle((-52.5, -20.15), 16.5, 40.3, fill=False, edgecolor="white", lw=1.2))
    ax.add_patch(plt.Rectangle((-52.5, -9.16), 5.5, 18.32, fill=False, edgecolor="white", lw=1.2))
    ax.add_patch(plt.Rectangle((-52.5 - 0.5, -3.66), 0.5, 7.32, facecolor="white", edgecolor="white"))


def plot_epv(match_id: str, time_str: str, data_dir: Path | None = None, results_dir: Path | None = None) -> Path:
    data_dir = data_dir or _resolve_data_dir(match_id)
    results_dir = results_dir or (ROOT / "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    events_path = data_dir / f"{match_id}_dynamic_events.csv"
    tracking_path = data_dir / f"{match_id}_tracking_extrapolated.jsonl"
    match_json = data_dir / f"match_{match_id}.json"

    events = pd.read_csv(events_path, low_memory=False)
    frames = load_tracking_jsonl(tracking_path)
    tracking = {fr["frame"]: fr for fr in frames if isinstance(fr, dict) and "frame" in fr}

    calc = _build_calc()
    calc.set_match_context(match_json, tracking_path, events)

    target_seconds = _parse_time_str(time_str)
    action_row, action_seconds = _find_action_event(events, target_seconds)
    attack_dir = _attack_dir_from_row(action_row)

    frame = int(action_row["frame_start"])
    frame, frame_data = _nearest_frame(tracking, frame)

    pid = int(action_row["player_id"])
    tid = int(action_row["team_id"])
    x_start = float(action_row["x_start"])
    y_start = float(action_row["y_start"])
    x, y = _player_xy(frame_data, pid, x_start, y_start)

    positions = {"own": [], "opp": []}
    roster = calc.current_team_roster or {}
    for p in frame_data.get("player_data", []):
        if p.get("x") is None or p.get("y") is None:
            continue
        player_id = p.get("player_id")
        label = str(player_id) if player_id is not None else ""
        if player_id == pid:
            label = f"*{label}"
        same_team = player_id in roster.get(tid, set()) if player_id is not None else False
        bucket = "own" if same_team else "opp"
        positions[bucket].append({"x": float(p["x"]), "y": float(p["y"]), "label": label, "pid": player_id})

    original_goal_x = calc.goal_x
    calc.goal_x = attack_dir * (calc.pitch_length / 2)

    original_dribble_sampler = calc._sample_dribble_destinations

    calc._sample_dribble_destinations = (
        lambda x0, y0, n=4: _sample_dribble_destinations_dir(x0, y0, attack_dir, n=n)
    )

    try:
        shot_q = calc.evaluate_shoot(x, y, frame, frame_data, pid, tid)
        pass_q, pass_dest = _compute_best_pass(calc, x, y, frame, pid, tid, tracking, frame_data)
        dribble_q, dribble_dest = _compute_best_dribble(calc, x, y, frame, pid, tid, tracking, frame_data)
    finally:
        calc.goal_x = original_goal_x
        calc._sample_dribble_destinations = original_dribble_sampler

    shoot_skill = calc.finishing_skills.get(pid, 0.0)
    pass_skill = calc.passing_skills.get(pid, 0.0)
    dribble_skill = calc.dribbling_skills.get(pid, 0.0)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    _draw_pitch(ax)

    for bucket, color in [("own", "tab:blue"), ("opp", "tab:red")]:
        for p in positions[bucket]:
            size = 90 if p["pid"] == pid else 60
            ax.scatter(p["x"], p["y"], color=color, s=size, alpha=0.9, edgecolor="k", zorder=3)

    goal_x, goal_y = attack_dir * (calc.pitch_length / 2), 0.0
    ax.annotate("", xy=(goal_x, goal_y), xytext=(x, y),
                arrowprops=dict(arrowstyle="->", color="tab:red", lw=2), zorder=2)

    if pass_dest is not None:
        px, py = pass_dest
        ax.annotate("", xy=(px, py), xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", color="tab:blue", lw=2), zorder=2)

    if dribble_dest is not None:
        dx, dy = dribble_dest
        ax.annotate("", xy=(dx, dy), xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", color="tab:green", lw=2, ls="--"), zorder=2)

    best_action = max([("Shot", shot_q), ("Pass", pass_q), ("Dribble", dribble_q)], key=lambda t: t[1])
    ax.set_title(
        f"EPV @ {action_seconds/60:.2f} min (closest to {time_str}) — best: {best_action[0]} ({best_action[1]:.3f})"
    )
    ax.set_xlabel("Pitch X (m)")
    ax.set_ylabel("Pitch Y (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    table_rows = [
        ["Shot", f"{shot_q:.3f}", f"{shoot_skill:.2f}"],
        ["Pass", f"{pass_q:.3f}", f"{pass_skill:.2f}"],
        ["Dribble", f"{dribble_q:.3f}", f"{dribble_skill:.2f}"],
    ]
    table = ax.table(
        cellText=table_rows,
        colLabels=["Action", "EPV", "Skill"],
        colLoc="center",
        cellLoc="center",
        bbox=[0.02, 0.73, 0.26, 0.22],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    safe_time = time_str.replace(":", "-").replace(".", "p")
    out_path = results_dir / f"epv_{match_id}_{safe_time}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-id", required=True, help="Match ID (e.g., 1039803)")
    parser.add_argument("--time", required=True, help="Time in minutes:seconds (e.g., 12:34)")
    args = parser.parse_args()

    out_path = plot_epv(args.match_id, args.time)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
