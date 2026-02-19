import math
import os
import sys
from pathlib import Path
from typing import List, Literal, Optional, Union

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

# Load environment variables from .env file
load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
MODELS_DIR = REPO_ROOT / "models"

# Read EPV_DATA_DIR from environment (required)
EPV_DATA_DIR = os.getenv("EPV_DATA_DIR")
if not EPV_DATA_DIR:
    raise RuntimeError(
        "EPV_DATA_DIR environment variable must be set. "
        "Please set it in your .env file or environment."
    )
DATA_DIR = Path(EPV_DATA_DIR)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

XG_PATH = MODELS_DIR / "xg_model_improved.pkl"
PASSING_PATH = MODELS_DIR / "passing_model_improved.pkl"
DRIBBLING_PATH = MODELS_DIR / "dribbling_model_proper_split.pkl"
FINISHING_SKILLS_PATH = DATA_DIR / "player_id_to_finishing_skill.csv"
PASSING_SKILLS_PATH = DATA_DIR / "player_id_to_passing_skill.csv"
DRIBBLING_SKILLS_PATH = DATA_DIR / "player_id_to_skill.csv"

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
GOAL_X_CENTER = 52.5
GOAL_Y_CENTER = 0.0

app = FastAPI()

# Mount routers
from api.routers import replay

app.include_router(replay.router, prefix="/replay", tags=["replay"])

# --- Player profile registry (from skill CSVs), keyed by profile_id (string), cached at startup ---
_player_registry: Optional[List[dict]] = None
_registry_by_profile_id: Optional[dict] = None


def _load_player_registry() -> List[dict]:
    """Load and merge player profiles from finishing/passing/dribbling skill CSVs. Normalize skills to [0,1]. Keyed by profile_id (string)."""
    global _player_registry, _registry_by_profile_id
    if _player_registry is not None:
        return _player_registry
    import pandas as pd
    out = []
    try:
        fin = pd.read_csv(FINISHING_SKILLS_PATH) if FINISHING_SKILLS_PATH.exists() else None
        pas = pd.read_csv(PASSING_SKILLS_PATH) if PASSING_SKILLS_PATH.exists() else None
        drib = pd.read_csv(DRIBBLING_SKILLS_PATH) if DRIBBLING_SKILLS_PATH.exists() else None
    except Exception:
        _player_registry = []
        _registry_by_profile_id = {}
        return _player_registry
    if fin is None or "player_id" not in fin.columns:
        _player_registry = []
        _registry_by_profile_id = {}
        return _player_registry
    fin_col = "player_finishing_skill" if "player_finishing_skill" in fin.columns else None
    pas_col = "player_passing_skill" if pas is not None and "player_passing_skill" in pas.columns else None
    drib_col = "player_dribbling_skill" if drib is not None and "player_dribbling_skill" in drib.columns else None
    name_col = "fbref_name" if "fbref_name" in fin.columns else ("skillcorner_name" if "skillcorner_name" in fin.columns else None)
    all_ids = set(fin["player_id"].astype(int).tolist())
    if pas is not None and "player_id" in pas.columns:
        all_ids.update(pas["player_id"].astype(int).tolist())
    if drib is not None and "player_id" in drib.columns:
        all_ids.update(drib["player_id"].astype(int).tolist())
    fin_series = fin.set_index("player_id")[fin_col] if fin_col else None
    pas_series = pas.set_index("player_id")[pas_col] if pas is not None and pas_col else None
    drib_series = drib.set_index("player_id")[drib_col] if drib is not None and drib_col else None
    fin_vals = fin_series.dropna().tolist() if fin_series is not None else []
    pas_vals = pas_series.dropna().tolist() if pas_series is not None else []
    drib_vals = drib_series.dropna().tolist() if drib_series is not None else []
    fin_min, fin_max = (min(fin_vals), max(fin_vals)) if fin_vals else (0.0, 1.0)
    pas_min, pas_max = (min(pas_vals), max(pas_vals)) if pas_vals else (0.0, 1.0)
    drib_min, drib_max = (min(drib_vals), max(drib_vals)) if drib_vals else (0.0, 1.0)
    def norm(v, lo, hi):
        if hi <= lo:
            return 0.5
        return max(0.0, min(1.0, (float(v) - lo) / (hi - lo)))
    def safe_get(s: Optional[pd.Series], pid: int, default: float = 0.5) -> float:
        if s is None:
            return default
        try:
            if pid in s.index:
                return float(s.loc[pid])
        except (KeyError, TypeError):
            pass
        return default

    for pid in sorted(all_ids):
        pid = int(pid)
        label = f"Player {pid}"
        if name_col and fin is not None:
            try:
                row = fin.loc[fin["player_id"].astype(int) == pid]
                if not row.empty and name_col in row.columns and pd.notna(row[name_col].iloc[0]):
                    label = str(row[name_col].iloc[0]).strip()
            except Exception:
                pass
        fin_val = safe_get(fin_series, pid)
        pas_val = safe_get(pas_series, pid)
        drib_val = safe_get(drib_series, pid)
        fin_s = norm(fin_val, fin_min, fin_max)
        pas_s = norm(pas_val, pas_min, pas_max)
        drib_s = norm(drib_val, drib_min, drib_max)
        overall = 0.4 * fin_s + 0.3 * pas_s + 0.3 * drib_s
        out.append({
            "player_id": pid,
            "profile_id": str(pid),
            "label": label,
            "display_name": label,
            "position": None,
            "finishing": fin_s,
            "passing": pas_s,
            "dribbling": drib_s,
            "finishing_skill": fin_s,
            "passing_skill": pas_s,
            "dribbling_skill": drib_s,
            "overall_individuality_score": float(overall),
        })
    _player_registry = out
    _registry_by_profile_id = {p["profile_id"]: p for p in out}
    return _player_registry


def _get_registry_by_profile_id() -> dict:
    """Return registry dict keyed by profile_id (string). Cached at startup."""
    global _registry_by_profile_id
    if _registry_by_profile_id is None:
        _load_player_registry()
    return _registry_by_profile_id or {}

# Read ALLOWED_ORIGINS from environment (optional, default includes common localhost variants)
# Supports comma-separated list: "http://localhost:3000,http://127.0.0.1:3000"
allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PlayerProfileResponse(BaseModel):
    profile_id: str
    player_id: int
    label: str
    display_name: Optional[str] = None  # same as label, for backward compatibility
    position: Optional[str] = None
    finishing: float
    passing: float
    dribbling: float
    overall_individuality_score: Optional[float] = None


class SkillMultipliers(BaseModel):
    finishing: Optional[float] = None
    passing: Optional[float] = None
    dribbling: Optional[float] = None


class PlayerIn(BaseModel):
    id: str
    team: Literal["home", "away"]
    x: float
    y: float
    theta: float
    hasBall: bool
    profile_id: Optional[Union[str, int]] = None
    skill_multipliers: Optional[SkillMultipliers] = None


BallOwnerProfile = Literal["average", "elite_passer", "elite_finisher", "elite_dribbler"]

PROFILE_MULTIPLIERS: dict[BallOwnerProfile, tuple[float, float, float]] = {
    "average": (1.0, 1.0, 1.0),
    "elite_finisher": (1.35, 0.95, 0.95),
    "elite_passer": (0.95, 1.35, 0.95),
    "elite_dribbler": (0.95, 0.95, 1.35),
}


class EPVRequest(BaseModel):
    frame: int
    possessionTeam: Literal["home", "away"]
    ballOwnerId: str
    players: list[PlayerIn]
    ballOwnerProfile: BallOwnerProfile = "average"


class BestPassTargetPoint(BaseModel):
    x: float
    y: float


class PassCandidate(BaseModel):
    x: float
    y: float
    score: float


class ExplainShoot(BaseModel):
    base_xg: float
    shot_min_def_dist: float
    shot_blocked: bool
    shot_pressure_multiplier: float


class ExplainPassCandidate(BaseModel):
    receiver_id: int
    receiver_xy: tuple[float, float]
    base: float
    lane_min_def_dist: float
    recv_min_def_dist: float
    risk: float
    adjusted: float


class ExplainPass(BaseModel):
    best_pass_target: Optional[BestPassTargetPoint] = None
    best_pass_risk: float
    best_pass_base: float
    best_pass_adjusted: float
    top_candidates: List[ExplainPassCandidate] = []


class ExplainDribble(BaseModel):
    base_dribble: float
    dribble_min_def_dist: float
    dribble_nearby_defenders: int
    dribble_open_space_m: float
    dribble_pressure_multiplier: float


class ExplainProfile(BaseModel):
    profile_id: Optional[str] = None
    label: Optional[str] = None
    finishing: Optional[float] = None
    passing: Optional[float] = None
    dribbling: Optional[float] = None
    mult_finish: Optional[float] = None
    mult_pass: Optional[float] = None
    mult_dribble: Optional[float] = None
    profile_missing: Optional[bool] = None
    applied_multipliers: Optional[dict] = None  # { "shoot": float, "pass": float, "dribble": float }
    display_name: Optional[str] = None
    position: Optional[str] = None
    overall_individuality_score: Optional[float] = None


class Explain(BaseModel):
    model_config = ConfigDict(serialize_by_alias=True)
    shoot: Optional[ExplainShoot] = None
    pass_: Optional[ExplainPass] = Field(None, alias="pass")
    dribble: Optional[ExplainDribble] = None
    profile: Optional[ExplainProfile] = None


class EPVResponse(BaseModel):
    epv: float
    best_action: Literal["shoot", "pass", "dribble"]
    q_shoot: float
    q_pass: float
    q_dribble: float
    best_action_reason: Optional[str] = None
    best_action_target: Optional[BestPassTargetPoint] = None
    best_pass_target: Optional[BestPassTargetPoint] = None
    pass_candidates: Optional[List[PassCandidate]] = None
    chosen_receiver_id: Optional[str] = None
    chosen_pass_risk: Optional[float] = None
    explain: Optional[Explain] = None


def _string_id_to_int(sid: str, team: Literal["home", "away"]) -> int:
    """Map API string ids (e.g. 'home-1', 'away-3') to integers for EPVCalculator."""
    try:
        parts = sid.rsplit("-", 1)
        num = int(parts[-1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        num = hash(sid) % 100000
    return (1000 + num) if team == "home" else (2000 + num)


def _team_to_int(team: Literal["home", "away"]) -> int:
    return 1 if team == "home" else 2


def _int_to_string_id(payload: EPVRequest, pid_int: int) -> Optional[str]:
    """Resolve calculator player_id (int) to request player id string (e.g. 'home-1')."""
    for p in payload.players:
        if _string_id_to_int(p.id, p.team) == pid_int:
            return p.id
    return None


def _api_to_center_coords(x: float, y: float) -> tuple[float, float]:
    """Convert API pitch coords (0-105 x 0-68) to EPVCalculator center coords (-52.5..52.5 x -34..34)."""
    return (x - GOAL_X_CENTER, y - PITCH_WIDTH / 2)


def _center_to_api_coords(x_center: float, y_center: float) -> tuple[float, float]:
    """Convert EPVCalculator center coords to API pitch coords (0-105 x 0-68)."""
    return (x_center + GOAL_X_CENTER, y_center + PITCH_WIDTH / 2)


def _compute_best_pass_heuristic(payload: EPVRequest) -> tuple[Optional[BestPassTargetPoint], Optional[List[PassCandidate]]]:
    """Heuristic best pass target and top-3 candidates from current layout."""
    owner = next((p for p in payload.players if p.id == payload.ballOwnerId), None)
    if not owner:
        return None, None
    teammates = [
        p for p in payload.players
        if p.team == payload.possessionTeam and p.id != payload.ballOwnerId
    ]
    if not teammates:
        return None, None
    opponents = [p for p in payload.players if p.team != payload.possessionTeam]
    scored = []
    for t in teammates:
        if payload.possessionTeam == "home":
            forward_progress = t.x - owner.x
        else:
            forward_progress = owner.x - t.x
        distance = math.hypot(owner.x - t.x, owner.y - t.y)
        defender_pressure = sum(
            1 for o in opponents
            if math.hypot(o.x - t.x, o.y - t.y) <= 8.0
        )
        score = 0.6 * forward_progress - 0.25 * distance - 0.4 * defender_pressure
        scored.append((t.x, t.y, score))
    scored.sort(key=lambda s: s[2], reverse=True)
    top3 = scored[:3]
    best = scored[0] if scored else None
    best_target = BestPassTargetPoint(x=best[0], y=best[1]) if best else None
    candidates = [PassCandidate(x=x, y=y, score=score) for x, y, score in top3] if top3 else None
    return best_target, candidates


def _profile_by_id(profile_id: Union[str, int]) -> Optional[dict]:
    """Resolve profile_id (string or int) to a registry profile dict, or None. Uses registry keyed by profile_id (string)."""
    by_id = _get_registry_by_profile_id()
    pid_str = str(profile_id).strip() if profile_id is not None else ""
    if not pid_str:
        return None
    return by_id.get(pid_str)


def build_frame_data_and_roster(payload: EPVRequest) -> tuple[dict, dict]:
    """Build tracking frame_data and current_team_roster from request payload. Injects skill_multipliers from profile_id or request."""
    roster: dict[int, set[int]] = {1: set(), 2: set()}
    player_data = []
    ball_x, ball_y = 0.0, 0.0

    for p in payload.players:
        pid_int = _string_id_to_int(p.id, p.team)
        team_id = _team_to_int(p.team)
        roster[team_id].add(pid_int)
        cx, cy = _api_to_center_coords(p.x, p.y)
        theta = getattr(p, "theta", 0.0)
        entry: dict = {
            "player_id": pid_int,
            "x": cx,
            "y": cy,
            "theta": float(theta),
            "is_detected": True,
        }
        skill_multipliers = None
        if getattr(p, "skill_multipliers", None) is not None:
            sm = p.skill_multipliers
            skill_multipliers = {
                "finishing": sm.finishing,
                "passing": sm.passing,
                "dribbling": sm.dribbling,
            }
            skill_multipliers = {k: v for k, v in skill_multipliers.items() if v is not None}
        elif getattr(p, "profile_id", None) is not None:
            pro = _profile_by_id(p.profile_id)
            if pro is not None:
                skill_multipliers = {
                    "finishing": pro.get("finishing"),
                    "passing": pro.get("passing"),
                    "dribbling": pro.get("dribbling"),
                }
            else:
                skill_multipliers = {"finishing": 0.5, "passing": 0.5, "dribbling": 0.5}
        if skill_multipliers:
            entry["skill_multipliers"] = skill_multipliers
        player_data.append(entry)
        if p.hasBall:
            ball_x, ball_y = cx, cy

    frame_data = {
        "player_data": player_data,
        "ball_data": {"x": ball_x, "y": ball_y},
        "period": 1,
    }
    return frame_data, roster


def compute_epv_with_calculator(payload: EPVRequest, calculator) -> EPVResponse:
    owner = next((p for p in payload.players if p.id == payload.ballOwnerId), None)
    if not owner:
        return EPVResponse(
            epv=0.0,
            best_action="pass",
            q_shoot=0.0,
            q_pass=0.0,
            q_dribble=0.0,
            best_action_reason=None,
            best_action_target=None,
            best_pass_target=None,
            pass_candidates=None,
            chosen_receiver_id=None,
            chosen_pass_risk=None,
            explain=None,
        )

    frame_data, roster = build_frame_data_and_roster(payload)
    frame = max(0, payload.frame)
    tracking_dict = {frame: frame_data}

    calculator.current_team_roster = roster
    player_id = _string_id_to_int(owner.id, owner.team)
    team_id = _team_to_int(payload.possessionTeam)
    x_center, y_center = _api_to_center_coords(owner.x, owner.y)

    result = calculator.get_best_action(
        x_center, y_center, frame, player_id, team_id, tracking_dict
    )

    best_action_reason = result.get("best_action_reason")
    best_action_target = None
    t = result.get("best_action_target")
    if t is not None:
        ax, ay = _center_to_api_coords(t["x"], t["y"])
        best_action_target = BestPassTargetPoint(x=ax, y=ay)
    best_pass_target = None
    pass_candidates = None
    chosen_receiver_id = None
    chosen_pass_risk = None
    if result.get("action") == "pass":
        rid = result.get("chosen_pass_receiver_id")
        if rid is not None:
            chosen_receiver_id = _int_to_string_id(payload, rid)
        chosen_pass_risk = result.get("chosen_pass_risk")
    explain = None
    if "explain" in result:
        exp = result["explain"]
        pass_ex = exp.get("pass") if exp else None
        if result["action"] == "pass":
            best_pass_target = best_action_target
            if pass_ex:
                top = pass_ex.get("top_candidates") or []
                pass_candidates = [
                    PassCandidate(
                        x=_center_to_api_coords(c["receiver_xy"][0], c["receiver_xy"][1])[0],
                        y=_center_to_api_coords(c["receiver_xy"][0], c["receiver_xy"][1])[1],
                        score=c["adjusted"],
                    )
                    for c in top
                ]
        shoot_model = None
        if exp and exp.get("shoot"):
            se = exp["shoot"]
            shoot_model = ExplainShoot(
                base_xg=float(se.get("base_xg", 0.0)),
                shot_min_def_dist=float(se.get("shot_min_def_dist", 100.0)),
                shot_blocked=bool(se.get("shot_blocked", False)),
                shot_pressure_multiplier=float(se.get("shot_pressure_multiplier", 1.0)),
            )
        pass_model = None
        if pass_ex:
            t = pass_ex.get("best_pass_target")
            if t is not None:
                px, py = _center_to_api_coords(t["x"], t["y"])
                pass_target = BestPassTargetPoint(x=px, y=py)
            else:
                pass_target = None
            pass_model = ExplainPass(
                best_pass_target=pass_target,
                best_pass_risk=float(pass_ex.get("best_pass_risk", 0.0)),
                best_pass_base=float(pass_ex.get("best_pass_base", 0.0)),
                best_pass_adjusted=float(pass_ex.get("best_pass_adjusted", 0.0)),
                top_candidates=[
                    ExplainPassCandidate(
                        receiver_id=c["receiver_id"],
                        receiver_xy=tuple(c["receiver_xy"]),
                        base=c["base"],
                        lane_min_def_dist=float(c.get("lane_min_def_dist", 0.0)),
                        recv_min_def_dist=float(c.get("recv_min_def_dist", 0.0)),
                        risk=c["risk"],
                        adjusted=c["adjusted"],
                    )
                    for c in (pass_ex.get("top_candidates") or [])
                ],
            )
        dribble_model = None
        if exp and exp.get("dribble"):
            de = exp["dribble"]
            dribble_model = ExplainDribble(
                base_dribble=float(de.get("base_dribble", 0.0)),
                dribble_min_def_dist=float(de.get("dribble_min_def_dist", 50.0)),
                dribble_nearby_defenders=int(de.get("dribble_nearby_defenders", 0)),
                dribble_open_space_m=float(de.get("dribble_open_space_m", 0.0)),
                dribble_pressure_multiplier=float(de.get("dribble_pressure_multiplier", 1.0)),
            )
        profile_model = None
        if exp and exp.get("profile"):
            pe = exp["profile"]
            owner_obj = next((p for p in payload.players if p.id == payload.ballOwnerId), None)
            profile_id = str(owner_obj.profile_id).strip() if owner_obj and getattr(owner_obj, "profile_id", None) is not None else None
            pro = _profile_by_id(profile_id) if profile_id else None
            profile_missing = (profile_id is not None and pro is None)
            label = (pro.get("label") or pro.get("display_name")) if pro else (f"Player {profile_id}" if profile_id else "Average")
            profile_model = ExplainProfile(
                profile_id=profile_id,
                label=label,
                finishing=pe.get("finishing"),
                passing=pe.get("passing"),
                dribbling=pe.get("dribbling"),
                mult_finish=pe.get("mult_finish"),
                mult_pass=pe.get("mult_pass"),
                mult_dribble=pe.get("mult_dribble"),
                profile_missing=profile_missing,
                applied_multipliers=pe.get("applied_multipliers"),
                display_name=(pro.get("display_name") or pro.get("label")) if pro else None,
                position=pro.get("position") if pro else None,
                overall_individuality_score=pro.get("overall_individuality_score") if pro else None,
            )
        if shoot_model or pass_model or dribble_model or profile_model:
            explain = Explain(shoot=shoot_model, pass_=pass_model, dribble=dribble_model, profile=profile_model)

    return EPVResponse(
        epv=float(result["epv"]),
        best_action=result["action"],
        q_shoot=float(result["q_shoot"]),
        q_pass=float(result["q_pass"]),
        q_dribble=float(result["q_dribble"]),
        best_action_reason=best_action_reason,
        best_action_target=best_action_target,
        best_pass_target=best_pass_target,
        pass_candidates=pass_candidates,
        chosen_receiver_id=chosen_receiver_id,
        chosen_pass_risk=chosen_pass_risk,
        explain=explain,
    )


def placeholder_epv(payload: EPVRequest) -> EPVResponse:
    owner = next((p for p in payload.players if p.id == payload.ballOwnerId), None)
    if not owner:
        return EPVResponse(
            epv=0.0,
            best_action="pass",
            q_shoot=0.0,
            q_pass=0.0,
            q_dribble=0.0,
            best_action_reason=None,
            best_action_target=None,
            best_pass_target=None,
            pass_candidates=None,
            chosen_receiver_id=None,
            chosen_pass_risk=None,
            explain=None,
        )
    x, y = owner.x, owner.y
    goal_x = 105.0
    dist = ((goal_x - x) ** 2 + (34 - y) ** 2) ** 0.5
    q_shoot = max(0.0, min(0.5, 0.3 - dist * 0.005))
    q_pass = 0.04 + 0.01 * min(
        sum(1 for p in payload.players if p.team == payload.possessionTeam and p.id != payload.ballOwnerId), 5
    )
    q_pass = min(0.4, q_pass)
    q_dribble = 0.02 + max(
        0,
        3 - sum(1 for p in payload.players if p.team != payload.possessionTeam and ((p.x - x) ** 2 + (p.y - y) ** 2) ** 0.5 < 8),
    ) * 0.01
    q_dribble = min(0.35, q_dribble)
    best = max([("shoot", q_shoot), ("pass", q_pass), ("dribble", q_dribble)], key=lambda t: t[1])
    best_pass_target = None
    pass_candidates = None
    if best[0] == "pass":
        best_pass_target, pass_candidates = _compute_best_pass_heuristic(payload)
    return EPVResponse(
        epv=round(best[1], 4),
        best_action=best[0],
        q_shoot=round(q_shoot, 4),
        q_pass=round(q_pass, 4),
        q_dribble=round(q_dribble, 4),
        best_action_reason=None,
        best_action_target=best_pass_target,
        best_pass_target=best_pass_target,
        pass_candidates=pass_candidates,
        chosen_receiver_id=None,
        chosen_pass_risk=None,
        explain=None,
    )


def apply_profile_to_response(raw: EPVResponse, payload: EPVRequest) -> EPVResponse:
    """Apply ball-owner profile multipliers to q-values and recompute best_action and epv."""
    profile = getattr(payload, "ballOwnerProfile", "average") or "average"
    mult = PROFILE_MULTIPLIERS.get(profile, (1.0, 1.0, 1.0))
    ms, mp, md = mult
    q_shoot = raw.q_shoot * ms
    q_pass = raw.q_pass * mp
    q_dribble = raw.q_dribble * md
    q_values = [("shoot", q_shoot), ("pass", q_pass), ("dribble", q_dribble)]
    best_action_name = max(q_values, key=lambda t: t[1])[0]
    epv = max(q_shoot, q_pass, q_dribble)
    best_pass_target = None
    pass_candidates = None
    if best_action_name == "pass":
        best_pass_target, pass_candidates = _compute_best_pass_heuristic(payload)
    return EPVResponse(
        epv=round(epv, 4),
        best_action=best_action_name,
        q_shoot=round(q_shoot, 4),
        q_pass=round(q_pass, 4),
        q_dribble=round(q_dribble, 4),
        best_action_reason=raw.best_action_reason,
        best_action_target=raw.best_action_target,
        best_pass_target=best_pass_target,
        pass_candidates=pass_candidates,
        chosen_receiver_id=raw.chosen_receiver_id,
        chosen_pass_risk=raw.chosen_pass_risk,
        explain=raw.explain,
    )


_calculator = None
_calculator_error: Optional[str] = None


def _load_calculator() -> None:
    global _calculator, _calculator_error
    if _calculator is not None or _calculator_error is not None:
        return
    try:
        from epv_calculator import EPVCalculator
    except Exception as e:
        _calculator_error = f"Failed to import EPVCalculator: {e}"
        return
    if not XG_PATH.exists():
        _calculator_error = f"xG model not found: {XG_PATH}"
        return
    if not PASSING_PATH.exists():
        _calculator_error = f"Passing model not found: {PASSING_PATH}"
        return
    if not FINISHING_SKILLS_PATH.exists():
        _calculator_error = f"Finishing skills CSV not found: {FINISHING_SKILLS_PATH}"
        return
    if not PASSING_SKILLS_PATH.exists():
        _calculator_error = f"Passing skills CSV not found: {PASSING_SKILLS_PATH}"
        return
    if not DRIBBLING_SKILLS_PATH.exists():
        _calculator_error = f"Dribbling skills CSV not found: {DRIBBLING_SKILLS_PATH}"
        return
    try:
        _calculator = EPVCalculator(
            xg_model_path=XG_PATH,
            passing_model_path=PASSING_PATH,
            dribbling_model_path=DRIBBLING_PATH,
            xg_skills_path=FINISHING_SKILLS_PATH,
            passing_skills_path=PASSING_SKILLS_PATH,
            dribbling_skills_path=DRIBBLING_SKILLS_PATH,
            pitch_length=PITCH_LENGTH,
            pitch_width=PITCH_WIDTH,
        )
    except Exception as e:
        _calculator_error = f"EPVCalculator initialization failed: {e}"
        return


_load_calculator()


@app.get("/players", response_model=List[PlayerProfileResponse])
def get_players() -> List[PlayerProfileResponse]:
    """Return player profile registry from skill CSVs. Skills normalized to [0,1]; overall_individuality_score = 0.4*finishing + 0.3*passing + 0.3*dribbling."""
    registry = _load_player_registry()
    return [
        PlayerProfileResponse(
            profile_id=p.get("profile_id") or str(p["player_id"]),
            player_id=p["player_id"],
            label=p.get("label") or p.get("display_name") or f"Player {p['player_id']}",
            display_name=p.get("label") or p.get("display_name") or f"Player {p['player_id']}",
            position=p.get("position"),
            finishing=p["finishing"],
            passing=p["passing"],
            dribbling=p["dribbling"],
            overall_individuality_score=p.get("overall_individuality_score"),
        )
        for p in registry
    ]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/epv", response_model=EPVResponse)
def epv(payload: EPVRequest) -> EPVResponse:
    if _calculator_error:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "EPVCalculator unavailable",
                "message": _calculator_error,
            },
        )
    if _calculator is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "EPVCalculator unavailable",
                "message": "Models or data failed to load.",
            },
        )
    try:
        raw = compute_epv_with_calculator(payload, _calculator)
        return apply_profile_to_response(raw, payload)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "EPV computation failed",
                "message": str(e),
            },
        )
