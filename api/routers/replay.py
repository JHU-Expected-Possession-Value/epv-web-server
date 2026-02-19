"""Replay router for SkillCorner match data."""

import json
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from api.utils import paths

router = APIRouter()

# Maximum window size for frame requests
MAX_WINDOW_SIZE = 2000

# Assumed frame rate for time-to-frame conversion
FRAMES_PER_SECOND = 25


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


class ColumnMapper:
    """Robust column mapping for different CSV schemas."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.event_type_col = self._find_column(
            ["event_type", "type", "name", "action"]
        )
        self.team_col = self._find_column(
            ["team_name", "team", "possession_team", "teamId", "team_id"]
        )
        self.player_col = self._find_column(
            ["player_name", "player", "playerId", "player_id"]
        )
        self.period_col = self._find_column(
            ["period", "half", "period_id"]
        )
        self.time_col = self._find_column(
            ["time", "timestamp", "time_seconds", "minute", "second", "game_time", "game_clock"]
        )
        self.frame_col = self._find_column(
            ["frame", "frame_id", "start_frame", "end_frame", "startFrame", "endFrame"]
        )
        self.end_frame_col = self._find_column(
            ["end_frame", "endFrame"]
        )
    
    def _find_column(self, candidates: List[str]) -> Optional[str]:
        """Find first matching column from candidates."""
        for candidate in candidates:
            if candidate in self.df.columns:
                return candidate
        return None
    
    def get_value(self, row: pd.Series, col: Optional[str], default=None):
        """Safely get value from row."""
        if col and col in row.index and pd.notna(row[col]):
            return row[col]
        return default


def _is_possession_loss_event(event_type: str) -> bool:
    """Check if event type indicates possession loss."""
    if not event_type:
        return False
    
    event_lower = event_type.lower()
    possession_loss_keywords = [
        "interception", "tackle", "ball_recovery", "dispossessed",
        "miscontrol", "failed_pass", "turnover", "lost", "out",
        "clearance", "block", "save"
    ]
    return any(keyword in event_lower for keyword in possession_loss_keywords)


def _is_shot_event(event_type: str) -> bool:
    """Check if event type is a shot."""
    if not event_type:
        return False
    
    event_lower = event_type.lower()
    shot_keywords = ["shot", "goal", "miss", "save", "block"]
    return any(keyword in event_lower for keyword in shot_keywords)


def _infer_turnover_from_context(
    df: pd.DataFrame,
    idx: int,
    mapper: ColumnMapper,
    current_team: Optional[str]
) -> bool:
    """Infer if this row represents a turnover by checking context."""
    if idx >= len(df) - 1:
        return False
    
    current_row = df.iloc[idx]
    next_row = df.iloc[idx + 1]
    
    # Get event types
    current_event = mapper.get_value(current_row, mapper.event_type_col, "")
    next_event = mapper.get_value(next_row, mapper.event_type_col, "")
    
    current_event_lower = str(current_event).lower()
    next_event_lower = str(next_event).lower()
    
    # Check if current is pass/carry and next is opponent recovery
    if any(x in current_event_lower for x in ["pass", "carry", "dribble"]):
        if any(x in next_event_lower for x in ["ball_recovery", "interception", "tackle"]):
            # Check if team changed
            next_team = mapper.get_value(next_row, mapper.team_col)
            if current_team and next_team and current_team != next_team:
                return True
    
    # Check if possession team changed
    if current_team:
        next_team = mapper.get_value(next_row, mapper.team_col)
        if next_team and current_team != next_team:
            # And current event could lead to turnover
            if any(x in current_event_lower for x in ["pass", "carry", "dribble", "miscontrol"]):
                return True
    
    return False


def _estimate_frame_from_time(time_seconds: Optional[float]) -> Optional[int]:
    """Estimate frame number from time using assumed frame rate."""
    if time_seconds is None or pd.isna(time_seconds):
        return None
    try:
        return int(round(float(time_seconds) * FRAMES_PER_SECOND))
    except (ValueError, TypeError):
        return None


def _calculate_frame_range(
    center_frame: Optional[int],
    start_frame_val: Optional[int],
    end_frame_val: Optional[int],
    time_seconds: Optional[float]
) -> tuple:
    """Calculate start_frame, end_frame, and center_frame."""
    # If we have explicit frame columns, use them
    if start_frame_val is not None and end_frame_val is not None:
        return start_frame_val, end_frame_val, None
    
    # If we have center_frame, calculate range
    if center_frame is not None:
        window_size = 100  # +/- 50 frames
        start = max(0, center_frame - window_size // 2)
        end = center_frame + window_size // 2
        return start, end, center_frame
    
    # If we have time, estimate frame
    if time_seconds is not None:
        estimated_frame = _estimate_frame_from_time(time_seconds)
        if estimated_frame is not None:
            window_size = 100
            start = max(0, estimated_frame - window_size // 2)
            end = estimated_frame + window_size // 2
            return start, end, estimated_frame
    
    return None, None, None


@router.get("/matches", response_model=MatchListResponse)
def get_matches() -> MatchListResponse:
    """Get list of available match IDs from tracking files."""
    try:
        match_ids = paths.list_available_match_ids()
        return MatchListResponse(match_ids=match_ids)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to list matches",
                "message": str(e),
            },
        )


@router.get("/{match_id}/moments", response_model=MomentsResponse)
def get_moments(
    match_id: str,
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of moments to return"),
) -> MomentsResponse:
    """Get candidate replay moments for a match, prioritizing possession-loss moments."""
    try:
        # Get events file path
        events_path = paths.get_events_path(match_id)
        
        if not events_path.exists():
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Events file not found",
                    "message": f"Events file does not exist: {events_path}",
                },
            )
        
        # Load events CSV
        try:
            df = pd.read_csv(events_path)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Failed to read events file",
                    "message": f"Could not parse CSV file {events_path}: {str(e)}",
                },
            )
        
        if df.empty:
            return MomentsResponse(moments=[])
        
        # Create column mapper
        mapper = ColumnMapper(df)
        
        # Extract moments with possession-loss focus
        moments = []
        all_teams = set()
        
        # First pass: collect all team names for opponent inference
        if mapper.team_col:
            all_teams.update(df[mapper.team_col].dropna().unique())
        
        for idx, row in df.iterrows():
            # Get event type
            event_type_raw = mapper.get_value(row, mapper.event_type_col)
            event_type = str(event_type_raw).strip() if event_type_raw else None
            
            if not event_type:
                continue
            
            # Get current team
            current_team = mapper.get_value(row, mapper.team_col)
            if current_team:
                current_team = str(current_team).strip()
            
            # Infer opponent team
            opponent_team = None
            if current_team and len(all_teams) >= 2:
                opponent_team = next((t for t in all_teams if str(t).strip() != current_team), None)
            
            # Check if this is a possession-loss event or shot
            is_possession_loss = _is_possession_loss_event(event_type)
            is_shot = _is_shot_event(event_type)
            
            # If not explicitly a possession-loss event, try to infer
            if not is_possession_loss and not is_shot:
                is_possession_loss = _infer_turnover_from_context(df, idx, mapper, current_team)
            
            # Only include possession-loss events and shots
            if not (is_possession_loss or is_shot):
                continue
            
            # Get time
            time_seconds = mapper.get_value(row, mapper.time_col)
            if time_seconds is not None:
                try:
                    time_seconds = float(time_seconds)
                except (ValueError, TypeError):
                    time_seconds = None
            
            # Skip if no time available
            if time_seconds is None or pd.isna(time_seconds):
                continue
            
            # Get period
            period = mapper.get_value(row, mapper.period_col)
            if period is not None:
                try:
                    period = int(period)
                except (ValueError, TypeError):
                    period = None
            
            # Get frames
            frame_val = mapper.get_value(row, mapper.frame_col)
            center_frame = None
            if frame_val is not None:
                try:
                    center_frame = int(frame_val)
                except (ValueError, TypeError):
                    center_frame = None
            
            start_frame_val = mapper.get_value(row, mapper.frame_col) if mapper.frame_col in ["start_frame", "startFrame"] else None
            end_frame_val = mapper.get_value(row, mapper.end_frame_col)
            
            if start_frame_val is not None:
                try:
                    start_frame_val = int(start_frame_val)
                except (ValueError, TypeError):
                    start_frame_val = None
            
            if end_frame_val is not None:
                try:
                    end_frame_val = int(end_frame_val)
                except (ValueError, TypeError):
                    end_frame_val = None
            
            # Calculate frame range
            start_frame, end_frame, center_frame_calc = _calculate_frame_range(
                center_frame, start_frame_val, end_frame_val, time_seconds
            )
            if center_frame_calc is not None:
                center_frame = center_frame_calc
            
            # Get player names
            player_name = mapper.get_value(row, mapper.player_col)
            if player_name:
                player_name = str(player_name).strip()
            
            # Try to infer opponent player from next row if it's a recovery event
            opponent_player = None
            if idx < len(df) - 1:
                next_row = df.iloc[idx + 1]
                next_event = mapper.get_value(next_row, mapper.event_type_col, "")
                next_event_lower = str(next_event).lower()
                if any(x in next_event_lower for x in ["ball_recovery", "interception", "tackle"]):
                    opponent_player = mapper.get_value(next_row, mapper.player_col)
                    if opponent_player:
                        opponent_player = str(opponent_player).strip()
            
            # Create moment
            moment = MomentResponse(
                match_id=match_id,
                index=int(idx),
                event_type=event_type,
                team_name=current_team,
                opponent_team_name=opponent_team,
                player_name=player_name,
                opponent_player_name=opponent_player,
                period=period,
                time_seconds=time_seconds,
                start_frame=start_frame,
                end_frame=end_frame,
                center_frame=center_frame,
            )
            
            moments.append(moment)
        
        # Sort: possession-loss events first, then shots, then by time
        possession_loss_moments = [m for m in moments if _is_possession_loss_event(m.event_type)]
        shot_moments = [m for m in moments if _is_shot_event(m.event_type)]
        other_moments = [m for m in moments if m not in possession_loss_moments and m not in shot_moments]
        
        # Sort each group by time
        possession_loss_moments.sort(key=lambda m: m.time_seconds or 0)
        shot_moments.sort(key=lambda m: m.time_seconds or 0)
        other_moments.sort(key=lambda m: m.time_seconds or 0)
        
        # Combine: possession-loss first, then shots, then others
        sorted_moments = possession_loss_moments + shot_moments + other_moments
        
        # Apply limit
        limited_moments = sorted_moments[:limit]
        
        return MomentsResponse(moments=limited_moments)
        
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Match not found",
                "message": str(e),
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to load moments",
                "message": str(e),
            },
        )


@router.get("/{match_id}/window", response_model=WindowResponse)
def get_window(
    match_id: str,
    start_frame: Optional[int] = Query(None, ge=0, description="Start frame (inclusive)"),
    end_frame: Optional[int] = Query(None, ge=0, description="End frame (inclusive)"),
    center_frame: Optional[int] = Query(None, ge=0, description="Center frame (alternative to start/end)"),
    radius: int = Query(50, ge=1, le=1000, description="Radius around center_frame (used with center_frame)"),
) -> WindowResponse:
    """Get tracking frames in the specified window. Streams JSONL file.
    
    Supports two modes:
    1. Explicit range: start_frame and end_frame (preferred)
    2. Center + radius: center_frame and radius (backwards compatibility)
    
    At least one mode must be provided.
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
        # Get tracking file path
        tracking_path = paths.get_tracking_path(match_id)
        
        if not tracking_path.exists():
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Tracking file not found",
                    "message": f"Tracking file does not exist: {tracking_path}",
                },
            )
        
        # Stream read JSONL file
        frames = []
        try:
            with open(tracking_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        frame_data = json.loads(line)
                    except json.JSONDecodeError as e:
                        # Skip malformed lines but log
                        continue
                    
                    # Extract frame_id from frame data
                    frame_id = None
                    if isinstance(frame_data, dict):
                        # Try common frame_id fields
                        for key in ["frame_id", "frameId", "frame", "frame_number"]:
                            if key in frame_data:
                                try:
                                    frame_id = int(frame_data[key])
                                    break
                                except (ValueError, TypeError):
                                    pass
                        
                        # If no frame_id found, use line number as fallback
                        if frame_id is None:
                            frame_id = line_num
                        
                        # Check if frame is in range
                        if start_frame <= frame_id <= end_frame:
                            frames.append(frame_data)
                        
                        # Early exit if we've passed the end frame
                        if frame_id > end_frame:
                            break
                    else:
                        # If not a dict, include it anyway (might be valid)
                        frames.append(frame_data)
        
        except IOError as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Failed to read tracking file",
                    "message": f"Could not read file {tracking_path}: {str(e)}",
                },
            )
        
        return WindowResponse(
            match_id=match_id,
            start_frame=start_frame,
            end_frame=end_frame,
            frames=frames,
        )
        
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Match not found",
                "message": str(e),
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to load window",
                "message": str(e),
            },
        )
