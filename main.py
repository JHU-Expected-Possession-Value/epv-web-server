"""FastAPI backend entrypoint.

**PostgreSQL (AWS RDS) at runtime** — request-scoped sessions via `api.db.get_db`:
- **Replay router** (`/replay/*`): `matches`, `teams`, `events`, `frame`, `detection`
  (see `api/routers/replay.py`, `api/services/replay_service.py`).
- **This file**: `/api/tactics/teams`, `/api/tactics/players`, `/api/tactics/roster`,
  `/api/tactics/player-action-heatmap` → `teams`, `players`, `shots`, `passes`, `carries`, `goals`.
- **`POST /replay/recommend`**: single-frame read from `frame`/`detection` + EPV models.

**Not from RDS (by design):**
- **`POST /api/epv`**, **`POST /api/tactics/recommendation`**: EPV **`.pkl`** under `epv-web-server/models/`
  plus optional skill **CSVs** under `epv-web-server/data/` (player individuality).
- **`POST /api/cv/*`**: uploaded images + YOLO weights on the server.

**Local files:** `api/utils/paths.py` (`EPV_DATA_DIR`) is **not** used for website routes; ingestion
uses `fillTables.py` into RDS instead.

Database querying belongs in `api/services/*` where possible; this file mounts routers and tactics.
"""

import math
import os
import sys
from pathlib import Path
from typing import List, Literal, Optional, Union

from dotenv import load_dotenv

# epv-web-server/.env must be loaded before `api.db` is imported: importing `api.db` runs
# `SessionLocal = get_sessionmaker()` which calls `get_engine()` and reads DATABASE_URL / PG*.
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_BACKEND_ROOT / ".env")

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import MetaData, Table, and_, select, text
from sqlalchemy.orm import Session

# from api.db import get_db
from api.services import replay_service

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
MODELS_DIR = REPO_ROOT / "models"

# Local packaged data directory (player skill CSVs).
# This keeps the deployed backend independent of a developer machine path like EPV_DATA_DIR.
DATA_DIR = REPO_ROOT / "data"

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
from api.routers import cv

app.include_router(replay.router, prefix="/replay", tags=["replay"])
app.include_router(cv.router, prefix="/api/cv", tags=["cv"])

# SQLAlchemy table reflection cache for simple read-only queries.
_metadata = MetaData()


def _table(db: Session, name: str) -> Table:
    return Table(name, _metadata, autoload_with=db.get_bind())


def _first_col(t: Table, *names: str):
    for n in names:
        if n in t.c:
            return t.c[n]
    return None

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
    """Build tracking `frame_data` and `current_team_roster` from request payload.

    Coordinate transformation (source of truth for the whole web stack):
    - Tactical board UI uses centered meters: x∈[-52.5, 52.5], y∈[-34, 34]
    - EPVRequest (API schema) uses pitch coords: x∈[0, 105], y∈[0, 68]
    - EPVCalculator consumes centered meters again (its historical convention)
    So we always convert request x/y (API pitch coords) -> centered meters here.

    Attacker/defender identification:
    - `payload.possessionTeam` ("home"|"away") determines the attacking team
    - `current_team_roster` maps team_id -> set[player_id] and is used to split
      teammates vs opponents when computing defender distances / lane pressure.
    """
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
    """Apply ball-owner profile multipliers to q-values for *display*.

    Important: this must NOT re-decide the best action.
    The EPVCalculator already chooses `best_action` using spatial context and
    policy rules (shot pressure/blocks, safe pass override, dribble open space).
    Recomputing `best_action = argmax(q_*)` here can create unrealistic "shoot"
    recommendations by ignoring those contextual rules.
    """
    profile = getattr(payload, "ballOwnerProfile", "average") or "average"
    mult = PROFILE_MULTIPLIERS.get(profile, (1.0, 1.0, 1.0))
    ms, mp, md = mult
    q_shoot = raw.q_shoot * ms
    q_pass = raw.q_pass * mp
    q_dribble = raw.q_dribble * md
    # Preserve the calculator's coherent decision bundle.
    best_action_name = raw.best_action
    epv = raw.epv
    return EPVResponse(
        epv=round(float(epv), 4),
        best_action=best_action_name,
        q_shoot=round(q_shoot, 4),
        q_pass=round(q_pass, 4),
        q_dribble=round(q_dribble, 4),
        best_action_reason=raw.best_action_reason,
        best_action_target=raw.best_action_target,
        best_pass_target=raw.best_pass_target,
        pass_candidates=raw.pass_candidates,
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
        # Tactical-board + replay API calls do not load full match metadata (`match_meta`).
        # EPVCalculator uses `match_meta.home_sides` to decide which goal a team attacks.
        # Without it, `_resolve_goal_x` defaults to attacking the +X goal for *both* teams,
        # which makes away-team shot/dribble/pass values unrealistic.
        #
        # For website usage we treat the abstract "home" side as left→right in period 1,
        # and away as the opposite. (Replay routes also use this 1/2 team_id convention.)
        _calculator.match_meta = {"home_id": 1, "away_id": 2, "home_sides": ["left_to_right"]}
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


@app.get("/api/health/db")
def health_db(db: Session = Depends(get_db)) -> dict:
    """Verify PostgreSQL (e.g. AWS RDS) is reachable using the same pool as all API routes.

    Use after deploy to confirm `DATABASE_URL` / `PG*` env vars point at the intended database.
    Does not cache query results; each call issues `SELECT 1`.
    """
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ok", "database": "connected"}
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={"status": "error", "database": "disconnected", "message": str(e)},
        )


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
        # Final recommendation selection:
        # `EPVCalculator.get_best_action()` already applies the full contextual policy
        # (pressure/blocked-shot penalties, safe-pass override, dribble-open-space context)
        # and returns a single coherent (best_action, epv, q_*) bundle.
        #
        # Avoid any extra post-hoc "take argmax(q_*)" logic here, because it can
        # re-introduce unrealistic shoot recommendations by ignoring those policy rules.
        return compute_epv_with_calculator(payload, _calculator)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "EPV computation failed",
                "message": str(e),
            },
        )


# =========================
# Replay recommendation (arrow only; no resimulation)
# =========================


class ReplayRecommendRequest(BaseModel):
    match_id: str
    moment_id: str
    center_frame: int
    event_type: Optional[str] = None
    team_side: Optional[Literal["home", "away"]] = None
    player_id: Optional[int] = None


class CounterfactualOverlay(BaseModel):
    kind: Literal["lane"]
    from_: dict = Field(..., alias="from")
    to: dict


class ReplayRecommendResponse(BaseModel):
    """Response consumed by the Replay page.

    How arrow data is returned:
    - `overlay` contains a single lane arrow in center coords for the frontend to draw.
    - `recommendation.target_point` is also provided as a direct fallback.
    """

    recommendation: dict
    overlay: Optional[dict] = None
    epv: dict
    decision_frame: Optional[int] = None
    teammate_overlays: Optional[list[dict]] = None
    chosen_target_player_id: Optional[int] = None
    fallback_reason: Optional[str] = None


def _classify_actual_action(event_type: Optional[str], event_subtype: Optional[str] = None) -> Literal["pass", "dribble", "shoot"]:
    """Map the moment's `events.event_type` (and optional subtype) onto one of the three EPV actions.

    Why: the recommendation panel needs to compare "what they actually did" vs the model's
    best action. We don't have a single column that says exactly that, so we infer it from
    `event_type` strings used by the SkillCorner CSV exports.
    """
    blob = " ".join([str(s or "").lower() for s in (event_type, event_subtype)])
    if any(k in blob for k in ("shot", "goal", "miss", "save", "block")):
        return "shoot"
    if any(k in blob for k in ("dribble", "carry", "miscontrol", "take_on", "run_with_ball")):
        return "dribble"
    # Defaults to pass: includes "pass", "failed_pass", "interception"/"tackle"/"recovery"
    # (these are opponent reactions to a pass) and any unclassified action. Pass is the
    # right default in soccer because it's the most common possession-loss path.
    return "pass"


def _q_for_action(epv_response: EPVResponse, action: Literal["pass", "dribble", "shoot"]) -> float:
    if action == "pass":
        return float(epv_response.q_pass)
    if action == "dribble":
        return float(epv_response.q_dribble)
    return float(epv_response.q_shoot)


@app.post("/replay/recommend", response_model=ReplayRecommendResponse)
def replay_recommend(payload: ReplayRecommendRequest, db: Session = Depends(get_db)) -> ReplayRecommendResponse:
    """Return a recommended action arrow for a real replay moment.

    Requirements:
    - Uses AWS-backed tracking/event data (no local files)
    - No resimulation: this returns only an arrow + EPV numbers for the UI

    Data flow (high level):
    1) Read one `frame` worth of `detection` rows from RDS for `(match_id, center_frame)`.
    2) Infer who has the ball (heuristic over detections; caller may override via `player_id`).
    3) Convert that snapshot into an `EPVRequest` and run the existing `EPVCalculator` pipeline.
    4) Map the model's `best_action` into a UI arrow:
       - The arrow starts at the possessor's tracked `(x,y)` in **center** coordinates.
       - The arrow ends at `best_action_target` converted back to center coords when present.
       - If the model does not emit a target point, aim at the defending goal center (shot-like default).
    5) Compare against what the player actually did using the moment's `event_type` so the
       UI can show "actual vs recommended" with both EPV values and a delta.

    The frontend should treat `overlay` as the authoritative geometry for drawing pass/dribble/shot lanes.
    """
    if _calculator_error or _calculator is None:
        raise HTTPException(status_code=503, detail={"error": "EPVCalculator unavailable", "message": _calculator_error or "not loaded"})

    match_id = int(payload.match_id)
    center_frame = int(payload.center_frame)

    # Pull a single frame snapshot from RDS (filtered in SQL by match_id + frame_id).
    frame = replay_service.fetch_single_center_frame(db, match_id=match_id, frame_id=center_frame)
    if frame is None:
        raise HTTPException(status_code=404, detail={"error": "Frame not found", "message": f"No tracking for match {match_id} frame {center_frame}"})

    poss = replay_service.derive_possessor(frame) or {}
    poss_player_id = payload.player_id or poss.get("player_id")
    if poss_player_id is None:
        raise HTTPException(status_code=422, detail={"error": "No possessor", "message": "Could not infer possessor from tracking frame; provide player_id"})

    teams = replay_service.get_match_teams(db, match_id)
    poss_team_id = poss.get("team_id")
    poss_side: Literal["home","away"] = "home"
    if payload.team_side in ("home", "away"):
        poss_side = payload.team_side
    elif poss_team_id is not None and teams.away_team_id is not None and poss_team_id == teams.away_team_id:
        poss_side = "away"

    # Build EPVRequest from the tracking snapshot.
    # Tracking frame uses center coords: x∈[-52.5,52.5], y∈[-34,34].
    players: list[PlayerIn] = []
    for p in frame.get("player_data") or []:
        pid = p.get("player_id")
        if pid is None:
            continue
        # Convert center coords -> API coords expected by EPVRequest (0..105, 0..68).
        x_api, y_api = _center_to_api_coords(float(p["x"]), float(p["y"]))
        side = "home"
        tid = p.get("team_id")
        if tid is not None and teams.away_team_id is not None and int(tid) == int(teams.away_team_id):
            side = "away"
        pro = _profile_by_id(str(int(pid)))
        sm = None
        if pro is not None:
            sm = SkillMultipliers(
                finishing=float(pro.get("finishing", 0.5)),
                passing=float(pro.get("passing", 0.5)),
                dribbling=float(pro.get("dribbling", 0.5)),
            )
        # Orientation (`theta`) is not available in the tracking snapshot we fetch here.
        # We inject a consistent default so dribble context is direction-correct:
        # - home attacks toward +X => theta=0
        # - away attacks toward -X => theta=pi
        theta = 0.0 if side == "home" else math.pi
        players.append(
            PlayerIn(
                id=f"{side}-{int(pid)}",
                team=side,  # type: ignore[arg-type]
                x=float(x_api),
                y=float(y_api),
                theta=float(theta),
                hasBall=(int(pid) == int(poss_player_id) and side == poss_side),
                profile_id=str(int(pid)),
                skill_multipliers=sm,
            )
        )

    owner_id = f"{poss_side}-{int(poss_player_id)}"
    epv_req = EPVRequest(
        frame=center_frame,
        possessionTeam=poss_side,
        ballOwnerId=owner_id,
        players=players,
        ballOwnerProfile="average",
    )
    # Use the calculator's recommendation directly (no post-hoc re-ranking).
    res = compute_epv_with_calculator(epv_req, _calculator)

    # Determine target point in center coords for the arrow.
    if res.best_action_target is not None:
        tx_center, ty_center = _api_to_center_coords(float(res.best_action_target.x), float(res.best_action_target.y))
    else:
        # Default shot target: goal center in center coords.
        tx_center, ty_center = (52.5, 0.0) if poss_side == "home" else (-52.5, 0.0)

    # From point: possessor location from tracking frame (center coords).
    poss_row = next(
        (pp for pp in (frame.get("player_data") or []) if int(pp.get("player_id") or -1) == int(poss_player_id)),
        None,
    )
    if not poss_row:
        raise HTTPException(status_code=422, detail={"error": "Possessor missing", "message": "Possessor not present in tracking detections for this frame"})
    fx, fy = float(poss_row["x"]), float(poss_row["y"])

    overlay = {
        "kind": "lane",
        "from": {"x": fx, "y": fy},
        "to": {"x": float(tx_center), "y": float(ty_center)},
    }

    action = res.best_action

    # Resolve target player id for a recommended pass.
    # `EPVResponse.chosen_receiver_id` is the API string id (e.g. "home-1234").
    # Strip the team prefix to get the integer player_id used in the tracking frame.
    target_player_id_int: Optional[int] = None
    if action == "pass" and res.chosen_receiver_id:
        try:
            target_player_id_int = int(str(res.chosen_receiver_id).rsplit("-", 1)[-1])
        except Exception:
            target_player_id_int = None

    # Look up player names so the UI can say "Pass to Joaquín Pereyra" rather than "pass to id=12345".
    roster = replay_service.get_match_player_index(db, match_id)
    poss_name = (roster.get(int(poss_player_id)) or {}).get("name") if roster else None
    target_name: Optional[str] = None
    if target_player_id_int is not None and roster:
        target_name = (roster.get(int(target_player_id_int)) or {}).get("name")

    # "What they actually did" — derive from the moment's event_type so we can compare.
    actual_action: Literal["pass", "dribble", "shoot"] = _classify_actual_action(payload.event_type)
    epv_recommended = float(res.epv)
    epv_actual = _q_for_action(res, actual_action)
    epv_delta = epv_recommended - epv_actual

    if poss_name:
        actual_phrase = {
            "pass": f"{poss_name} attempted a pass",
            "dribble": f"{poss_name} tried to dribble",
            "shoot": f"{poss_name} took a shot",
        }[actual_action]
    else:
        actual_phrase = {
            "pass": "Player attempted a pass",
            "dribble": "Player tried to dribble",
            "shoot": "Player took a shot",
        }[actual_action]
    if action == "pass" and target_name:
        rec_phrase = f"recommended pass to {target_name}"
    elif action == "shoot":
        rec_phrase = "recommended shot"
    elif action == "dribble":
        rec_phrase = "recommended dribble into space"
    else:
        rec_phrase = f"recommended {action}"
    summary = f"Instead of: {actual_phrase} — {rec_phrase} (ΔEPV {epv_delta:+.3f})"

    return ReplayRecommendResponse(
        recommendation={
            "text": summary,
            "action": action,
            "target_point": {"x": float(tx_center), "y": float(ty_center)},
            "summary": summary,
            "target_player_id": target_player_id_int,
            "target_player_name": target_name,
            "actual_action": actual_action,
            "actual_phrase": actual_phrase,
            "recommended_phrase": rec_phrase,
            "possessor_name": poss_name,
            "epv_delta_est": epv_delta,
        },
        overlay=overlay,
        epv={
            # `epv_original` retained for backward-compat (= EPV of the action they took).
            "epv_original": float(epv_actual),
            "epv_actual": float(epv_actual),
            "epv_recommended": float(epv_recommended),
            "epv_delta": float(epv_delta),
            "actual_action": actual_action,
            "recommended_action": action,
            "q_pass": float(res.q_pass),
            "q_dribble": float(res.q_dribble),
            "q_shoot": float(res.q_shoot),
        },
        decision_frame=center_frame,
        teammate_overlays=None,
        chosen_target_player_id=target_player_id_int,
        fallback_reason=None,
    )


# =========================
# Tactical Board (backend-driven)
# =========================
#
# Frontend responsibilities:
# - Rendering + dragging interactions
# - Formation presets (layout only)
#
# Backend responsibilities (this section):
# - Query MLS 2023 teams/players from AWS RDS (`teams`, `players`)
# - Model inference via EPVCalculator (local .pkl models)
# - Player individuality via skill registry (local skill CSVs)
# - Season heatmaps via AWS tables (`shots`, `passes`, `carries`)


class TacticsTeam(BaseModel):
    team_id: int
    team_name: Optional[str] = None


class TacticsPlayer(BaseModel):
    player_id: int
    name: str
    position: Optional[str] = None
    team_id: Optional[int] = None
    team_name: Optional[str] = None
    # Individuality from local skill registry (normalized to [0,1]).
    pass_skill: float = 0.5
    dribble_skill: float = 0.5
    shot_skill: float = 0.5


class TacticsRosterPlayer(BaseModel):
    """Legacy shape used by the current UI.

    NOTE: `team` is a string; we populate it with the team name.
    """

    player_id: int
    name: str
    team: str
    team_name: str
    position: str
    pass_skill: float
    dribble_skill: float
    shot_skill: float


class TacticsPlayerIn(BaseModel):
    player_id: str
    x: float  # center coords (world): [-52.5, 52.5]
    y: float  # center coords (world): [-34, 34]
    pos: Optional[str] = None


class TacticsBallCarrierIn(BaseModel):
    player_id: str
    x: float
    y: float
    team: Literal["home", "away"]


class TacticsRecommendationRequest(BaseModel):
    ball_carrier: TacticsBallCarrierIn
    home: List[TacticsPlayerIn]
    away: List[TacticsPlayerIn]


class TacticsTarget(BaseModel):
    type: Literal["player", "point", "goal"]
    x: float
    y: float
    player_id: Optional[str] = None


class TacticsExplain(BaseModel):
    pass_risk: float
    nearest_defender_dist: float


class TacticsRecommendationResponse(BaseModel):
    epv: float
    best_action: Literal["pass", "dribble", "shoot"]
    q_pass: float
    q_dribble: float
    q_shoot: float
    target: TacticsTarget
    explain: TacticsExplain


class HeatmapCell(BaseModel):
    col: int
    row: int
    intensity: float


class PlayerHeatmapResponse(BaseModel):
    player_id: int
    player_name: str
    kind: Literal["shots", "passes", "carries", "goals"]
    cols: int
    rows: int
    cells: List[HeatmapCell]
    note: str


def _center_to_api(x_center: float, y_center: float) -> tuple[float, float]:
    """Convert center coords (-52.5..52.5, -34..34) to API coords (0..105, 0..68)."""
    return (x_center + GOAL_X_CENTER, y_center + PITCH_WIDTH / 2)


def _api_to_center(x_api: float, y_api: float) -> tuple[float, float]:
    """Convert API coords (0..105, 0..68) to center coords (-52.5..52.5, -34..34)."""
    return (x_api - GOAL_X_CENTER, y_api - PITCH_WIDTH / 2)


def _skill_profile_for_player_id(player_id: int) -> Optional[dict]:
    # Player individuality is keyed by profile_id == player_id.
    return _profile_by_id(str(player_id))


def _resolve_player_name_from_db(db: Session, player_id: int) -> str:
    players = _table(db, "players")
    pid = _first_col(players, "id")
    pname = _first_col(players, "full_name", "name")
    if pid is None or pname is None:
        return f"Player {player_id}"
    row = db.execute(select(pname).where(pid == int(player_id))).first()
    if row and row[0]:
        return str(row[0]).strip()
    pro = _skill_profile_for_player_id(int(player_id))
    return (pro.get("label") if pro else None) or f"Player {player_id}"


@app.get("/api/tactics/teams", response_model=List[TacticsTeam])
def tactics_teams(db: Session = Depends(get_db)) -> List[TacticsTeam]:
    """MLS teams for Tactical Board dropdowns.

    AWS table: `teams`
    """
    teams = _table(db, "teams")
    tid = _first_col(teams, "id")
    tname = _first_col(teams, "name")
    if tid is None:
        raise HTTPException(status_code=500, detail={"error": "teams table missing id"})
    rows = db.execute(select(tid, tname).order_by(tname if tname is not None else tid)).all()
    return [
        TacticsTeam(team_id=int(r[0]), team_name=str(r[1]).strip() if len(r) > 1 and r[1] is not None else None)
        for r in rows
        if r and r[0] is not None
    ]


@app.get("/api/tactics/players", response_model=List[TacticsPlayer])
def tactics_players(team_id: int = Query(..., ge=1), db: Session = Depends(get_db)) -> List[TacticsPlayer]:
    """Players by team (backend-driven).

    AWS tables: `players`, `teams`
    Individuality: merged from local skill CSV registry.
    """
    players = _table(db, "players")
    teams = _table(db, "teams")

    pid = _first_col(players, "id")
    pname = _first_col(players, "full_name", "name")
    ppos = _first_col(players, "position")
    pteam = _first_col(players, "team_id")
    tid = _first_col(teams, "id")
    tname = _first_col(teams, "name")
    if pid is None or pname is None or pteam is None:
        raise HTTPException(status_code=500, detail={"error": "players table missing required columns"})

    stmt = (
        select(
            pid.label("player_id"),
            pname.label("name"),
            ppos.label("position") if ppos is not None else None,
            pteam.label("team_id"),
            tname.label("team_name") if tname is not None else None,
        )
        .select_from(players.outerjoin(teams, tid == pteam if tid is not None else True))
        .where(pteam == int(team_id))
        .order_by(pname)
    )
    rows = db.execute(stmt).mappings().all()
    out: List[TacticsPlayer] = []
    for r in rows:
        p_id = int(r["player_id"])
        pro = _skill_profile_for_player_id(p_id)
        out.append(
            TacticsPlayer(
                player_id=p_id,
                name=str(r["name"]).strip(),
                position=str(r.get("position")).strip() if r.get("position") is not None else None,
                team_id=int(r.get("team_id")) if r.get("team_id") is not None else None,
                team_name=str(r.get("team_name")).strip() if r.get("team_name") is not None else None,
                pass_skill=float(pro.get("passing", 0.5)) if pro else 0.5,
                dribble_skill=float(pro.get("dribbling", 0.5)) if pro else 0.5,
                shot_skill=float(pro.get("finishing", 0.5)) if pro else 0.5,
            )
        )
    return out


@app.get("/api/tactics/roster", response_model=List[TacticsRosterPlayer])
def tactics_roster(db: Session = Depends(get_db)) -> List[TacticsRosterPlayer]:
    """Legacy roster list used by the current UI.

    AWS tables: `players`, `teams`
    Individuality: merged from local skill CSV registry.
    """
    players = _table(db, "players")
    teams = _table(db, "teams")

    pid = _first_col(players, "id")
    pname = _first_col(players, "full_name", "name")
    ppos = _first_col(players, "position")
    pteam = _first_col(players, "team_id")
    tid = _first_col(teams, "id")
    tname = _first_col(teams, "name")
    if pid is None or pname is None:
        raise HTTPException(status_code=500, detail={"error": "players table missing required columns"})

    stmt = (
        select(
            pid.label("player_id"),
            pname.label("name"),
            ppos.label("position") if ppos is not None else None,
            pteam.label("team_id") if pteam is not None else None,
            tname.label("team_name") if tname is not None else None,
        )
        .select_from(players.outerjoin(teams, tid == pteam if tid is not None and pteam is not None else True))
        .order_by(tname if tname is not None else pid, pname)
    )
    rows = db.execute(stmt).mappings().all()
    out: List[TacticsRosterPlayer] = []
    for r in rows:
        p_id = int(r["player_id"])
        team_name = str(r.get("team_name") or "Team").strip()
        pro = _skill_profile_for_player_id(p_id)
        out.append(
            TacticsRosterPlayer(
                player_id=p_id,
                name=str(r["name"]).strip(),
                team=team_name,
                team_name=team_name,
                position=str(r.get("position") or "").strip(),
                pass_skill=float(pro.get("passing", 0.5)) if pro else 0.5,
                dribble_skill=float(pro.get("dribbling", 0.5)) if pro else 0.5,
                shot_skill=float(pro.get("finishing", 0.5)) if pro else 0.5,
            )
        )
    return out


@app.post("/api/tactics/recommendation", response_model=TacticsRecommendationResponse)
def tactics_recommendation(payload: TacticsRecommendationRequest) -> TacticsRecommendationResponse:
    """EPV-driven recommendation for Tactical Board state.

    Player individuality incorporation:
    - Each selected MLS player is assigned `profile_id == player_id`
    - Skill multipliers are pulled from the local skill CSV registry and injected into the model inputs
    """
    if _calculator_error or _calculator is None:
        raise HTTPException(status_code=503, detail={"error": "EPVCalculator unavailable", "message": _calculator_error or "not loaded"})

    bc = payload.ball_carrier
    players: List[PlayerIn] = []

    def add_player(side: Literal["home", "away"], p: TacticsPlayerIn) -> None:
        ax, ay = _center_to_api(p.x, p.y)
        pid_raw = str(p.player_id)
        pid_int = int(pid_raw) if pid_raw.isdigit() else None
        pro = _skill_profile_for_player_id(pid_int) if pid_int is not None else None
        sm = None
        if pro is not None:
            sm = SkillMultipliers(
                finishing=float(pro.get("finishing", 0.5)),
                passing=float(pro.get("passing", 0.5)),
                dribbling=float(pro.get("dribbling", 0.5)),
            )
        # Tactical board state does not include per-player facing direction.
        # We set a deterministic default aligned with the board convention:
        # - home attacks to +X => theta=0
        # - away attacks to -X => theta=pi
        theta = 0.0 if side == "home" else math.pi
        players.append(
            PlayerIn(
                id=f"{side}-{pid_raw}",
                team=side,
                x=float(ax),
                y=float(ay),
                theta=float(theta),
                hasBall=(pid_raw == str(bc.player_id) and side == bc.team),
                profile_id=pid_raw,
                skill_multipliers=sm,
            )
        )

    for p in payload.home:
        add_player("home", p)
    for p in payload.away:
        add_player("away", p)

    epv_req = EPVRequest(
        frame=0,
        possessionTeam=bc.team,
        ballOwnerId=f"{bc.team}-{bc.player_id}",
        players=players,
        ballOwnerProfile="average",
    )
    # Use the calculator's recommendation directly (no post-hoc re-ranking).
    res = compute_epv_with_calculator(epv_req, _calculator)

    # Backend → frontend target: center coords for arrow drawing.
    if res.best_action == "shoot":
        tx, ty = (52.5, 0.0) if bc.team == "home" else (-52.5, 0.0)
        target = TacticsTarget(type="goal", x=tx, y=ty, player_id=None)
    elif res.best_action_target is not None:
        tx, ty = _api_to_center(float(res.best_action_target.x), float(res.best_action_target.y))
        target = TacticsTarget(type="point", x=float(tx), y=float(ty), player_id=res.chosen_receiver_id)
    else:
        target = TacticsTarget(type="point", x=float(bc.x), y=float(bc.y), player_id=None)

    # Defender-distance shown in the UI should reflect the *board state* directly.
    # Compute nearest opponent to the ball carrier in center coords using the same
    # players the model sees (no placeholders).
    nearest_def = 100.0
    try:
        opp = payload.away if bc.team == "home" else payload.home
        nearest_def = min(
            (math.hypot(float(p.x) - float(bc.x), float(p.y) - float(bc.y)) for p in opp),
            default=100.0,
        )
    except Exception:
        nearest_def = 100.0

    pass_risk = 0.0
    try:
        if res.explain and res.explain.pass_:
            pass_risk = float(res.explain.pass_.best_pass_risk)
    except Exception:
        pass

    return TacticsRecommendationResponse(
        epv=float(res.epv),
        best_action=res.best_action,
        q_pass=float(res.q_pass),
        q_dribble=float(res.q_dribble),
        q_shoot=float(res.q_shoot),
        target=target,
        explain=TacticsExplain(pass_risk=float(pass_risk), nearest_defender_dist=float(nearest_def)),
    )


@app.get("/api/tactics/player-action-heatmap", response_model=PlayerHeatmapResponse)
def player_action_heatmap(
    player_id: int = Query(..., ge=1),
    kind: Literal["shots", "passes", "carries", "goals"] = Query(...),
    team_side: Optional[Literal["home", "away"]] = Query(
        None,
        description="Optional: orient distribution to match board attacking direction (home→right, away→left).",
    ),
    cols: int = Query(12, ge=4, le=40),
    rows: int = Query(8, ge=4, le=30),
    db: Session = Depends(get_db),
) -> PlayerHeatmapResponse:
    """Season action heatmaps for a selected player (MLS 2023).

    AWS tables (filled by `fillTables.py` event split):
    - `shots`, `passes`, `carries`, `goals` — each queried with **WHERE player_id = :id** only
      (no full-table load into Python).
    """
    # DB-backed season event data (AWS RDS):
    # - shots/goals come from the `shots` / `goals` tables (event start locations)
    # - passes come from the `passes` table (pass start locations)
    # - carries come from the `carries` table (carry start locations)
    #
    # This endpoint bins those real event locations into a coarse grid that the
    # frontend renders as a polished "soccer-style" heatmap.
    table_name = {"shots": "shots", "passes": "passes", "carries": "carries", "goals": "goals"}[kind]
    t = _table(db, table_name)
    pid = _first_col(t, "player_id")
    xs = _first_col(t, "x_start", "x")
    ys = _first_col(t, "y_start", "y")
    if pid is None or xs is None or ys is None:
        raise HTTPException(status_code=500, detail={"error": f"{table_name} missing required columns"})

    # Pitch bounds in center coords (same coordinate space as the TacticalBoard UI).
    x_min, x_max = -52.5, 52.5
    y_min, y_max = -34.0, 34.0
    dx = (x_max - x_min) / cols
    dy = (y_max - y_min) / rows

    counts = [[0 for _ in range(cols)] for _ in range(rows)]
    stmt = select(xs, ys).where(pid == int(player_id))

    # Coordinate normalization (source of truth: `EPV_SARG/AWS/goal_heatmap.py` and `carries_heat_map.py`):
    # The DB tables may store any of these conventions depending on ingestion/export:
    # - centered meters: x∈[-52.5,52.5], y∈[-34,34]
    # - pitch coords:    x∈[0,105],     y∈[0,68]
    # - percent coords:  x∈[0,100],     y∈[0,100]
    # - unit coords:     x∈[0,1],       y∈[0,1]
    #
    # We detect the convention from the player's data range, then map into centered meters.
    pts: list[tuple[float, float]] = []
    x_vals: list[float] = []
    y_vals: list[float] = []
    for x_raw, y_raw in db.execute(stmt):
        if x_raw is None or y_raw is None:
            continue
        try:
            xf = float(x_raw)
            yf = float(y_raw)
        except Exception:
            continue
        pts.append((xf, yf))
        x_vals.append(xf)
        y_vals.append(yf)

    if not pts:
        name = _resolve_player_name_from_db(db, int(player_id))
        return PlayerHeatmapResponse(
            player_id=int(player_id),
            player_name=name,
            kind=kind,
            cols=cols,
            rows=rows,
            cells=[],
            note=f"No {kind} locations found for player {player_id}.",
        )

    x_lo, x_hi = min(x_vals), max(x_vals)
    y_lo, y_hi = min(y_vals), max(y_vals)

    def to_center(x: float, y: float) -> tuple[float, float]:
        # Already centered meters
        if x_lo >= -53 and x_hi <= 53 and y_lo >= -35 and y_hi <= 35:
            return (x, y)
        # 0..105 / 0..68 pitch coords
        if x_lo >= 0 and x_hi <= 105.5 and y_lo >= 0 and y_hi <= 68.5:
            return (x - 52.5, y - 34.0)
        # 0..100 percentages
        if x_lo >= 0 and x_hi <= 100.5 and y_lo >= 0 and y_hi <= 100.5:
            return ((x / 100.0) * 105.0 - 52.5, (y / 100.0) * 68.0 - 34.0)
        # 0..1 unit square
        if x_lo >= 0 and x_hi <= 1.01 and y_lo >= 0 and y_hi <= 1.01:
            return (x * 105.0 - 52.5, y * 68.0 - 34.0)
        # Unknown; best-effort assume already centered.
        return (x, y)

    for x_raw, y_raw in pts:
        cx, cy = to_center(x_raw, y_raw)
        # Attacking-direction orientation:
        # - Board convention: home attacks to +X, away attacks to -X.
        # - If caller provides team_side="away", mirror X so the distribution appears on the
        #   same attacking half the away team uses in the board view.
        if team_side == "away":
            cx = -cx
        if cx < x_min or cx > x_max or cy < y_min or cy > y_max:
            continue
        col = int((cx - x_min) / dx)
        row = int((cy - y_min) / dy)
        col = max(0, min(cols - 1, col))
        row = max(0, min(rows - 1, row))
        counts[row][col] += 1

    # Normalization (matches the reference AWS scripts under `EPV_SARG/AWS/*heatmap*.py`):
    # - Create a 2D histogram of counts
    # - Set `vmax` lower than the raw max so hotspots saturate faster (more contrast)
    #   (AWS reference uses `vmax = max(heat) * 0.4`)
    # - Convert each cell to intensity ∈ [0,1] by `min(count / vmax, 1)`
    max_c = max((c for rr in counts for c in rr), default=0) or 1
    vmax = max(1.0, float(max_c) * 0.4)
    cells: List[HeatmapCell] = []
    for r in range(rows):
        for c in range(cols):
            v = counts[r][c]
            if v <= 0:
                continue
            intensity = min(1.0, float(v) / vmax)
            cells.append(HeatmapCell(col=c, row=r, intensity=float(intensity)))

    name = _resolve_player_name_from_db(db, int(player_id))
    return PlayerHeatmapResponse(
        player_id=int(player_id),
        player_name=name,
        kind=kind,
        cols=cols,
        rows=rows,
        cells=cells,
        note=f"{kind} locations aggregated across all games for player {player_id}.",
    )
