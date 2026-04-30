"""Replay API router (thin).

All routes below use **`Depends(get_db)`** → PostgreSQL (typically AWS RDS) at **request time**.
Data model matches tables filled by `EPV_SARG/AWS/fillTables.py`:

- **`matches`**, **`teams`**: `/matches`
- **`events`** (per-`match_id` SQL): `/moments`, legacy `/{match_id}/moments`
- **`frame`**, **`detection`** (range-filtered by `match_id` + frame bounds): `/tracking_window`,
  `/tracking_window_render`, legacy `/{match_id}/window`

No parallel file-based replay path: do not wire `api.utils.paths` here.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.db import get_db
from api.services import replay_service

router = APIRouter()

# Maximum window size for frame requests
MAX_WINDOW_SIZE = 2000


class MatchListResponse(BaseModel):
    """Response model for list of available matches."""
    match_ids: List[str]


class MomentResponse(BaseModel):
    """Response model for a replay moment with possession-loss focus."""
    match_id: str
    index: int
    event_type: str
    team_name: Optional[str] = None
    opponent_team_name: Optional[str] = None
    player_name: Optional[str] = None
    opponent_player_name: Optional[str] = None
    period: Optional[int] = None
    time_seconds: Optional[float] = None
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    center_frame: Optional[int] = None


class MomentsResponse(BaseModel):
    """Response model for list of moments."""
    moments: List[MomentResponse]


class WindowResponse(BaseModel):
    """Response model for tracking window."""
    match_id: str
    start_frame: int
    end_frame: int
    frames: List[dict]

class ReplayTeamInfo(BaseModel):
    id: Optional[int] = None
    name: Optional[str] = None
    short_name: Optional[str] = None


class ReplayMatch(BaseModel):
    match_id: str
    home_team: ReplayTeamInfo
    away_team: ReplayTeamInfo
    label: str
    # Replayability is determined server-side by checking presence of rows in the
    # DB tables required to preview a match (`events`, `frame`, `detection`).
    replayable: Optional[bool] = None
    replayability_reason: Optional[str] = None


class ReplayMatchesResponse(BaseModel):
    matches: List[ReplayMatch]


class ReplayMoment(BaseModel):
    moment_id: str
    match_id: str
    period: int
    frame_start: int
    frame_end: int
    frame: int
    minute_start: Optional[int] = None
    second_start: Optional[int] = None
    time_label: Optional[str] = None
    team_id: Optional[int] = None
    team_shortname: Optional[str] = None
    player_id: Optional[int] = None
    player_name: Optional[str] = None
    event_type: Optional[str] = None
    event_subtype: Optional[str] = None
    end_type: Optional[str] = None
    pass_outcome: Optional[str] = None
    turnover_type: str


class ReplayMomentsResponse(BaseModel):
    match_id: str
    count: int
    moments: List[ReplayMoment]


class ReplayTrackingWindowResponse(BaseModel):
    match_id: str
    center_frame: int
    start_frame: int
    end_frame: int
    frames: List[dict]


class RenderBall(BaseModel):
    x: float
    y: float
    z: Optional[float] = None


class RenderPlayer(BaseModel):
    id: Optional[int]
    team_id: Optional[int]
    team_side: Optional[str] = None
    name: Optional[str] = None
    x: float
    y: float
    speed: Optional[float] = None


class RenderFrame(BaseModel):
    frame: int
    period: Optional[int] = None
    timestamp: Optional[str] = None
    ball: Optional[RenderBall] = None
    players: List[RenderPlayer]
    derived_possession: Optional[dict] = None
    # Diagnostic fields populated by `build_tracking_render_window`:
    # - `raw_player_count`: number of players the DB returned for this exact frame_id
    #   (BEFORE any forward-fill from neighbouring frames).
    # - `players_filled`: True when this frame's `players` were inherited from another
    #   frame because the raw row set was sparse/empty.
    raw_player_count: Optional[int] = None
    players_filled: Optional[bool] = None


class TrackingRenderWindow(BaseModel):
    match_id: str
    center_frame: int
    effective_center_frame: Optional[int] = None
    start_frame: int
    end_frame: int
    frames: List[RenderFrame]


@router.get("/matches", response_model=ReplayMatchesResponse)
def get_matches(
    replayable_only: int = Query(
        1,
        description=(
            "If 1 (default), only return matches that are replay-previewable. "
            "Replayable requires rows in `events` and tracking rows in `frame`+`detection`."
        ),
    ),
    db: Session = Depends(get_db),
) -> ReplayMatchesResponse:
    try:
        rows = replay_service.list_matches(db)
        if int(replayable_only) == 1:
            rows = [r for r in rows if r.get("replayable")]
        return ReplayMatchesResponse(
            matches=[
                ReplayMatch(
                    match_id=r["match_id"],
                    home_team=ReplayTeamInfo(**r["home_team"]),
                    away_team=ReplayTeamInfo(**r["away_team"]),
                    label=r["label"],
                    replayable=bool(r.get("replayable")) if "replayable" in r else None,
                    replayability_reason=r.get("replayability_reason"),
                )
                for r in rows
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "Failed to list matches", "message": str(e)})




@router.get("/moments", response_model=ReplayMomentsResponse)
def replay_moments(
    match_id: str = Query(..., description="Match ID"),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> ReplayMomentsResponse:
    """Replay moments from RDS.

    AWS table: `events`
    """
    try:
        df = replay_service.fetch_events_df(db, int(match_id))
        count, moments = replay_service.detect_moments(int(match_id), df, limit=limit, offset=offset)
        return ReplayMomentsResponse(
            match_id=str(match_id),
            count=int(count),
            moments=[ReplayMoment(**m) for m in moments],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "Failed to load moments", "message": str(e)})


@router.get("/{match_id}/moments", response_model=MomentsResponse)
def get_moments_legacy(
    match_id: str,
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of moments to return"),
    db: Session = Depends(get_db),
) -> MomentsResponse:
    """Legacy moments endpoint (best-effort compatibility).

    AWS table: `events`
    """
    res = replay_moments(match_id=match_id, limit=limit, offset=0, db=db)
    # Adapt to the old response shape.
    out = [
        MomentResponse(
            match_id=m.match_id,
            index=i,
            event_type=m.event_type or "",
            team_name=m.team_shortname,
            opponent_team_name=None,
            player_name=m.player_name,
            opponent_player_name=None,
            period=m.period,
            time_seconds=None,
            start_frame=m.frame_start,
            end_frame=m.frame_end,
            center_frame=m.frame,
        )
        for i, m in enumerate(res.moments)
    ]
    return MomentsResponse(moments=out)


@router.get("/tracking_window", response_model=ReplayTrackingWindowResponse)
def tracking_window(
    match_id: str = Query(...),
    start_frame: Optional[int] = Query(None, ge=0),
    end_frame: Optional[int] = Query(None, ge=0),
    center_frame: Optional[int] = Query(None, ge=0),
    radius: int = Query(60, ge=1, le=2000),
    db: Session = Depends(get_db),
) -> ReplayTrackingWindowResponse:
    """Fetch raw tracking window from RDS.

    AWS tables: `frame`, `detection`
    """
    if center_frame is not None:
        start_frame = max(0, center_frame - radius)
        end_frame = center_frame + radius
    if start_frame is None or end_frame is None:
        raise HTTPException(status_code=400, detail={"error": "Missing parameters", "message": "Provide center_frame or start_frame/end_frame"})
    if end_frame < start_frame:
        raise HTTPException(status_code=400, detail={"error": "Invalid frame range", "message": "end_frame must be >= start_frame"})
    window_size = end_frame - start_frame + 1
    if window_size > MAX_WINDOW_SIZE:
        raise HTTPException(status_code=400, detail={"error": "Window too large", "message": f"Window size ({window_size}) exceeds maximum ({MAX_WINDOW_SIZE} frames)"})

    frames = replay_service.fetch_tracking_window_raw(db, int(match_id), int(start_frame), int(end_frame))
    return ReplayTrackingWindowResponse(
        match_id=str(match_id),
        center_frame=int(center_frame if center_frame is not None else (start_frame + end_frame) // 2),
        start_frame=int(start_frame),
        end_frame=int(end_frame),
        frames=frames,
    )


@router.get("/tracking_window_render", response_model=TrackingRenderWindow)
def tracking_window_render(
    match_id: str = Query(...),
    center_frame: int = Query(..., ge=0),
    radius: int = Query(60, ge=1, le=2000),
    include_players: int = Query(1, ge=0, le=1),
    max_frames: int = Query(120, ge=1, le=2000),
    db: Session = Depends(get_db),
) -> TrackingRenderWindow:
    """Render-ready tracking window for the web UI (lightweight shape).

    AWS tables: `frame`, `detection`
    """
    payload = replay_service.build_tracking_render_window(
        db,
        match_id=int(match_id),
        center_frame=int(center_frame),
        radius=int(radius),
        include_players=bool(include_players),
        max_frames=int(max_frames),
    )
    return TrackingRenderWindow(**payload)


@router.get("/{match_id}/window", response_model=WindowResponse)
def get_window(
    match_id: str,
    start_frame: Optional[int] = Query(None, ge=0, description="Start frame (inclusive)"),
    end_frame: Optional[int] = Query(None, ge=0, description="End frame (inclusive)"),
    center_frame: Optional[int] = Query(None, ge=0, description="Center frame (alternative to start/end)"),
    radius: int = Query(50, ge=1, le=1000, description="Radius around center_frame (used with center_frame)"),
    db: Session = Depends(get_db),
) -> WindowResponse:
    """Legacy window endpoint.

    AWS tables: `frame`, `detection`
    """
    # Determine start_frame and end_frame
    if center_frame is not None:
        # Mode 2: center_frame + radius
        calculated_start = max(0, center_frame - radius)
        calculated_end = center_frame + radius
        if start_frame is not None or end_frame is not None:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid parameters",
                    "message": "Cannot specify both center_frame and start_frame/end_frame",
                },
            )
        start_frame = calculated_start
        end_frame = calculated_end
    elif start_frame is None or end_frame is None:
        # Mode 1: explicit range required if center_frame not provided
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Missing parameters",
                "message": "Must provide either (start_frame and end_frame) or (center_frame with optional radius)",
            },
        )
    
    # Guardrails
    if end_frame < start_frame:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid frame range",
                "message": f"end_frame ({end_frame}) must be >= start_frame ({start_frame})",
            },
        )
    
    window_size = end_frame - start_frame + 1
    if window_size > MAX_WINDOW_SIZE:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Window too large",
                "message": f"Window size ({window_size}) exceeds maximum ({MAX_WINDOW_SIZE} frames)",
            },
        )
    
    try:
        frames = replay_service.fetch_tracking_window_raw(db, int(match_id), int(start_frame), int(end_frame))
        return WindowResponse(match_id=match_id, start_frame=start_frame, end_frame=end_frame, frames=frames)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "Failed to load window", "message": str(e)})
