import math
import sys
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
MODELS_DIR = REPO_ROOT / "models"
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PlayerIn(BaseModel):
    id: str
    team: Literal["home", "away"]
    x: float
    y: float
    theta: float
    hasBall: bool


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


class EPVResponse(BaseModel):
    epv: float
    best_action: Literal["shoot", "pass", "dribble"]
    q_shoot: float
    q_pass: float
    q_dribble: float
    best_pass_target: Optional[BestPassTargetPoint] = None
    pass_candidates: Optional[List[PassCandidate]] = None


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


def _api_to_center_coords(x: float, y: float) -> tuple[float, float]:
    """Convert API pitch coords (0-105 x 0-68) to EPVCalculator center coords (-52.5..52.5 x -34..34)."""
    return (x - GOAL_X_CENTER, y - PITCH_WIDTH / 2)


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


def build_frame_data_and_roster(payload: EPVRequest) -> tuple[dict, dict]:
    """Build tracking frame_data and current_team_roster from request payload."""
    roster: dict[int, set[int]] = {1: set(), 2: set()}
    player_data = []
    ball_x, ball_y = 0.0, 0.0

    for p in payload.players:
        pid_int = _string_id_to_int(p.id, p.team)
        team_id = _team_to_int(p.team)
        roster[team_id].add(pid_int)
        cx, cy = _api_to_center_coords(p.x, p.y)
        player_data.append({
            "player_id": pid_int,
            "x": cx,
            "y": cy,
            "is_detected": True,
        })
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
            best_pass_target=None,
            pass_candidates=None,
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

    best_pass_target = None
    pass_candidates = None
    if result["action"] == "pass":
        best_pass_target, pass_candidates = _compute_best_pass_heuristic(payload)

    return EPVResponse(
        epv=float(result["epv"]),
        best_action=result["action"],
        q_shoot=float(result["q_shoot"]),
        q_pass=float(result["q_pass"]),
        q_dribble=float(result["q_dribble"]),
        best_pass_target=best_pass_target,
        pass_candidates=pass_candidates,
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
            best_pass_target=None,
            pass_candidates=None,
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
        best_pass_target=best_pass_target,
        pass_candidates=pass_candidates,
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
        best_pass_target=best_pass_target,
        pass_candidates=pass_candidates,
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
