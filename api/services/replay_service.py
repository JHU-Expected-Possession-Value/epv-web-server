"""Replay services (DB + lightweight feature logic).

Routers should stay thin: they validate params and return models.
All database access for replay should live here.

Key requirements:
- Tracking queries must be filtered in SQL by match_id + frame range.
- Do not load full `frame` / `detection` tables into memory.

AWS tables used:
- `matches`, `teams`: match list / labels
- `events`: moment candidates (loss-of-possession / shots) and frame ranges
- `frame`, `detection`: tracking windows (match_id + frame range filtered)

Ingestion reference:
- `EPV_SARG/AWS/fillTables.py` is the operational script that populates these tables from
  SkillCorner exports into Postgres (RDS). The replay service assumes that pipeline (or an
  equivalent loader) has already materialized rows for the match you query.

Project goals reflected here:
- AWS-backed replay data (no local EPV_DATA_DIR files)
- Efficient tracking window reads (SQL-filtered; grouped by frame)
- Simple replay recommendation rendering (arrow overlay uses center coords)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Optional

import pandas as pd
from fastapi import HTTPException
from sqlalchemy import MetaData, Table, and_, func, select
from sqlalchemy.orm import Session


@dataclass(frozen=True)
class ReplayTeams:
    home_team_id: Optional[int]
    away_team_id: Optional[int]


_metadata = MetaData()


def _table(db: Session, name: str) -> Table:
    return Table(name, _metadata, autoload_with=db.get_bind())


def _first_col(t: Table, *names: str):
    for n in names:
        if n in t.c:
            return t.c[n]
    return None


def _safe_int(v) -> Optional[int]:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return int(v)
    except Exception:
        return None


def _safe_float(v) -> Optional[float]:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return float(v)
    except Exception:
        return None


def get_frame_bounds(db: Session, match_id: int) -> tuple[Optional[int], Optional[int]]:
    """Return (min_frame_id, max_frame_id) for a match from the `frame` table.

    This is used to clamp moment-derived windows to frames that actually exist in Postgres.
    """
    frame_t = _table(db, "frame")
    f_match = _first_col(frame_t, "match_id")
    f_id = _first_col(frame_t, "id", "frame_id")
    if f_match is None or f_id is None:
        return (None, None)
    row = db.execute(
        select(func.min(f_id).label("min_id"), func.max(f_id).label("max_id")).where(f_match == int(match_id))
    ).mappings().first()
    if not row:
        return (None, None)
    return (_safe_int(row.get("min_id")), _safe_int(row.get("max_id")))


# -----------------------
# Matches
# -----------------------


def list_matches(db: Session) -> list[dict]:
    """Return matches with home/away teams (for frontend dropdown).

    Replayability criteria (used by `/replay/matches` filtering):
    - A match is "replayable" only if it has:
      - at least one row in `events` for that match_id (moments source)
      - at least one row in `frame` and `detection` for that match_id (tracking preview source)

    Why a match may fail preview:
    - `events` loaded but tracking not loaded (common if `fillTables.py` uploaded events but
      `fill_player_tracking()` / tracking ingestion was not run or failed for that match).
    - tracking loaded but no events for that match (rare; moments endpoint will be empty).
    """
    matches = _table(db, "matches")
    teams = _table(db, "teams")

    mid = _first_col(matches, "id", "match_id")
    home_id = _first_col(matches, "home_team_id")
    away_id = _first_col(matches, "away_team_id")

    # `fillTables.py` writes matches with columns:
    #   id, date_time, home_team.id, away_team.id
    # Pandas `to_sql` preserves those dotted column names, so SQLAlchemy reflection
    # will expose them as `"home_team.id"` / `"away_team.id"` — which are NOT caught
    # by the simple `_first_col(..., "home_team_id")` lookup above.
    #
    # To support both schemas (legacy `home_team_id` and the current `home_team.id`),
    # fall back to any column whose name suggests a home/away team FK.
    if home_id is None:
        for c in matches.c:
            n = c.name.lower()
            if "home_team" in n and "id" in n:
                home_id = c
                break
    if away_id is None:
        for c in matches.c:
            n = c.name.lower()
            if "away_team" in n and "id" in n:
                away_id = c
                break

    team_id = _first_col(teams, "id")
    team_name = _first_col(teams, "name")

    if mid is None:
        raise RuntimeError("matches table missing id")
    if team_id is None:
        raise RuntimeError("teams table missing id")

    home_alias = teams.alias("home_t")
    away_alias = teams.alias("away_t")
    stmt = (
        select(
            mid.label("match_id"),
            home_id.label("home_team_id"),
            away_id.label("away_team_id"),
            (home_alias.c[team_name.name] if team_name is not None else None).label("home_team_name"),
            (away_alias.c[team_name.name] if team_name is not None else None).label("away_team_name"),
        )
        .select_from(
            matches.outerjoin(home_alias, home_alias.c[team_id.name] == home_id).outerjoin(
                away_alias, away_alias.c[team_id.name] == away_id
            )
        )
        .order_by(mid.desc())
    )
    rows = db.execute(stmt).mappings().all()

    # Determine replayability from DB presence checks.
    match_ids = [int(r["match_id"]) for r in rows if r.get("match_id") is not None]
    replayability = get_replayability(db, match_ids)

    out = []
    for r in rows:
        m_id = str(r["match_id"])
        hn = r.get("home_team_name")
        an = r.get("away_team_name")
        label = f"{hn or 'Home'} vs {an or 'Away'} ({m_id})" if (hn or an) else m_id
        rep = replayability.get(int(r["match_id"]), {"replayable": False, "reason": "unknown"})
        out.append(
            {
                "match_id": m_id,
                "home_team": {"id": _safe_int(r.get("home_team_id")), "name": hn, "short_name": None},
                "away_team": {"id": _safe_int(r.get("away_team_id")), "name": an, "short_name": None},
                "label": label,
                # Extra fields are safe for clients that ignore them; used to filter/communicate
                # why a match cannot be previewed.
                "replayable": bool(rep.get("replayable", False)),
                "replayability_reason": rep.get("reason"),
            }
        )
    return out


def get_replayability(db: Session, match_ids: list[int]) -> dict[int, dict]:
    """Return replayability flags for the provided match ids.

    DB tables checked:
    - `events`   : must have at least 1 row per match_id (moments exist)
    - `frame`    : must have at least 1 row per match_id (tracking frames exist)
    - `detection`: must have at least 1 row per match_id (player/ball detections exist)
    """
    if not match_ids:
        return {}

    events_t = _table(db, "events")
    frame_t = _table(db, "frame")
    det_t = _table(db, "detection")

    e_mid = _first_col(events_t, "match_id")
    f_mid = _first_col(frame_t, "match_id")
    d_mid = _first_col(det_t, "match_id")
    if e_mid is None or f_mid is None or d_mid is None:
        # If any core column is missing, conservatively mark as non-replayable.
        return {int(mid): {"replayable": False, "reason": "schema_missing"} for mid in match_ids}

    mids = [int(m) for m in match_ids]

    # Use GROUP BY rather than per-match queries.
    events_present = {
        int(r["match_id"])
        for r in db.execute(
            select(e_mid.label("match_id")).where(e_mid.in_(mids)).group_by(e_mid)
        ).mappings().all()
    }
    frames_present = {
        int(r["match_id"])
        for r in db.execute(
            select(f_mid.label("match_id")).where(f_mid.in_(mids)).group_by(f_mid)
        ).mappings().all()
    }
    det_present = {
        int(r["match_id"])
        for r in db.execute(
            select(d_mid.label("match_id")).where(d_mid.in_(mids)).group_by(d_mid)
        ).mappings().all()
    }

    out: dict[int, dict] = {}
    for mid in mids:
        has_events = mid in events_present
        has_frame = mid in frames_present
        has_det = mid in det_present
        replayable = bool(has_events and has_frame and has_det)
        if replayable:
            reason = None
        else:
            missing = []
            if not has_events:
                missing.append("events")
            if not has_frame:
                missing.append("frame")
            if not has_det:
                missing.append("detection")
            reason = f"missing:{','.join(missing)}" if missing else "missing:unknown"
        out[int(mid)] = {"replayable": replayable, "reason": reason}
    return out


def get_match_teams(db: Session, match_id: int) -> ReplayTeams:
    matches = _table(db, "matches")
    mid = _first_col(matches, "id", "match_id")
    home_id = _first_col(matches, "home_team_id")
    away_id = _first_col(matches, "away_team_id")
    # Same schema caveat as `list_matches`: `fillTables.py` may create `"home_team.id"` / `"away_team.id"`.
    if home_id is None:
        for c in matches.c:
            n = c.name.lower()
            if "home_team" in n and "id" in n:
                home_id = c
                break
    if away_id is None:
        for c in matches.c:
            n = c.name.lower()
            if "away_team" in n and "id" in n:
                away_id = c
                break
    if mid is None or home_id is None or away_id is None:
        return ReplayTeams(home_team_id=None, away_team_id=None)
    row = db.execute(select(home_id, away_id).where(mid == int(match_id))).first()
    if not row:
        return ReplayTeams(home_team_id=None, away_team_id=None)
    return ReplayTeams(home_team_id=_safe_int(row[0]), away_team_id=_safe_int(row[1]))


# -----------------------
# Moments (loss-of-possession)
# -----------------------


class _ColumnMapper:
    """Robust column mapping for the `events` table schema variations."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.event_type_col = self._find_column(["event_type", "type", "name", "action"])
        self.team_col = self._find_column(["team_shortname", "team_name", "team", "team_id"])
        self.player_col = self._find_column(["player_name", "player", "player_id"])
        self.player_id_col = self._find_column(["player_id"])
        self.period_col = self._find_column(["period", "half", "period_id"])
        self.time_col = self._find_column(["time_start", "time_seconds", "time", "timestamp"])
        self.minute_col = self._find_column(["minute_start", "minute"])
        self.second_col = self._find_column(["second_start", "second"])
        # SkillCorner dynamic events often contain only a single `frame` column. `fillTables.py`
        # writes the CSV headers verbatim to Postgres, so we treat `frame` as a usable center
        # frame when explicit start/end bounds are missing.
        self.frame_start_col = self._find_column(["frame_start", "start_frame", "startFrame", "frame"])
        self.frame_end_col = self._find_column(["frame_end", "end_frame", "endFrame", "frame"])
        self.frame_col = self._find_column(["frame"])
        self.event_id_col = self._find_column(["event_id"])
        self.event_subtype_col = self._find_column(["event_subtype"])
        self.end_type_col = self._find_column(["end_type"])
        self.pass_outcome_col = self._find_column(["pass_outcome"])
        self.carry_col = self._find_column(["carry"])
        self.team_id_col = self._find_column(["team_id"])

    def _find_column(self, candidates: list[str]) -> Optional[str]:
        for c in candidates:
            if c in self.df.columns:
                return c
        return None

    def get_value(self, row: pd.Series, col: Optional[str], default=None):
        if col and col in row.index and pd.notna(row[col]):
            return row[col]
        return default


def _is_possession_loss_event(event_type: str) -> bool:
    """Keyword classifier for loss-of-possession moments.

    This mirrors the previous file-based logic but is now applied to DB-backed `events` rows.
    """
    if not event_type:
        return False
    s = event_type.lower()
    keywords = [
        "interception",
        "tackle",
        "ball_recovery",
        "dispossessed",
        "miscontrol",
        "failed_pass",
        "turnover",
        "lost",
        "out",
        "clearance",
        "block",
        "save",
    ]
    return any(k in s for k in keywords)

def _is_failed_pass_from_row(row: pd.Series, mapper: _ColumnMapper) -> bool:
    """Detect failed pass turnovers from `end_type` + `pass_outcome` columns.

    `fillTables.py` stores the SkillCorner CSV columns verbatim, so we avoid hardcoding one
    exact value and instead look for common "unsuccessful" markers.
    """
    end_type = str(mapper.get_value(row, mapper.end_type_col, "") or "").lower()
    if "pass" not in end_type:
        return False
    out = str(mapper.get_value(row, mapper.pass_outcome_col, "") or "").lower()
    if not out:
        return False
    # Common variants across providers/exports.
    bad = ["fail", "incomplete", "unsuccess", "out", "lost", "intercept", "blocked"]
    good = ["complete", "success"]
    if any(g in out for g in good):
        return False
    return any(b in out for b in bad) or out not in ("", "nan")


def _is_shot_event(event_type: str) -> bool:
    if not event_type:
        return False
    s = event_type.lower()
    return any(k in s for k in ["shot", "goal", "miss", "save", "block"])


def _infer_turnover_from_context(df: pd.DataFrame, idx: int, mapper: _ColumnMapper, current_team: Optional[str]) -> bool:
    """Fallback inference: pass/carry followed by opponent recovery."""
    if idx >= len(df) - 1:
        return False
    cur = df.iloc[idx]
    nxt = df.iloc[idx + 1]
    cur_ev = str(mapper.get_value(cur, mapper.event_type_col, "")).lower()
    nxt_ev = str(mapper.get_value(nxt, mapper.event_type_col, "")).lower()
    if any(x in cur_ev for x in ["pass", "carry", "dribble"]) and any(
        x in nxt_ev for x in ["ball_recovery", "interception", "tackle"]
    ):
        nxt_team = mapper.get_value(nxt, mapper.team_col)
        if current_team and nxt_team and str(nxt_team).strip() != str(current_team).strip():
            return True
    return False


def fetch_events_df(db: Session, match_id: int) -> pd.DataFrame:
    """Load one match's `events` rows (SQL: **WHERE match_id = :id** only — not the full table).

    NOTE: Moment detection logic mirrors prior file-based behavior but reads RDS `events`.
    """
    events = _table(db, "events")
    mid = _first_col(events, "match_id")
    if mid is None:
        raise RuntimeError("events table missing match_id")

    wanted = [
        "event_id",
        "index",
        "event_type",
        "event_subtype",
        "end_type",
        "pass_outcome",
        "carry",
        "team_id",
        "team_shortname",
        "team_name",
        "player_id",
        "player_name",
        "period",
        "minute_start",
        "second_start",
        "time_start",
        "time_seconds",
        "frame_start",
        "frame_end",
        "start_frame",
        "end_frame",
        "frame",
    ]
    cols = [events.c[n].label(n) for n in wanted if n in events.c]
    if "event_type" not in events.c:
        raise RuntimeError("events table missing event_type")

    stmt = select(*cols).where(mid == int(match_id))
    order_col = _first_col(events, "time_start", "time_seconds", "index", "event_id")
    if order_col is not None:
        stmt = stmt.order_by(order_col)
    rows = db.execute(stmt).mappings().all()
    return pd.DataFrame(rows)


def detect_moments(match_id: int, df: pd.DataFrame, limit: int, offset: int) -> tuple[int, list[dict]]:
    """Detect loss-of-possession and shot moments from DB-backed events.

    How moments are detected:
    - Primary: keyword match on `event_type`
    - Fallback: infer turnovers via next-row context (pass/carry -> opponent recovery)
    """
    if df.empty:
        return 0, []

    mapper = _ColumnMapper(df)
    moments: list[dict] = []

    for idx, row in df.iterrows():
        et_raw = mapper.get_value(row, mapper.event_type_col)
        et = str(et_raw).strip() if et_raw else ""
        # Some SkillCorner event exports encode the semantic in other columns (e.g. end_type,
        # pass_outcome). We allow `event_type` to be empty and still detect turnovers.

        team_label = mapper.get_value(row, mapper.team_col)
        team_label = str(team_label).strip() if team_label is not None else None

        is_loss = _is_possession_loss_event(et) if et else False
        is_shot = _is_shot_event(et) if et else False
        if not is_loss and not is_shot:
            is_loss = _infer_turnover_from_context(df, idx, mapper, team_label)
        if not is_loss and not is_shot:
            is_loss = _is_failed_pass_from_row(row, mapper)
        if not (is_loss or is_shot):
            continue

        fs = _safe_int(mapper.get_value(row, mapper.frame_start_col))
        fe = _safe_int(mapper.get_value(row, mapper.frame_end_col))
        frame_center = _safe_int(mapper.get_value(row, mapper.frame_col))

        # Window selection for website replay:
        # - Prefer explicit (frame_start, frame_end) if present in the table.
        # - If only a single `frame` exists (common for SkillCorner CSV exports), create a small
        #   window around it. This still satisfies "SQL filtered by match_id + frame range".
        if fs is None and fe is None and frame_center is not None:
            pre, post = 45, 75
            fs, fe = max(0, frame_center - pre), frame_center + post
        elif fs is None and fe is not None:
            fs = max(0, int(fe) - 60)
        elif fe is None and fs is not None:
            fe = int(fs) + 60
        if fs is None or fe is None:
            continue

        period = _safe_int(mapper.get_value(row, mapper.period_col)) or 1
        minute = _safe_int(mapper.get_value(row, mapper.minute_col))
        second = _safe_int(mapper.get_value(row, mapper.second_col))
        time_label = f"{minute:02d}:{second:02d}" if minute is not None and second is not None else None

        moment_id_val = mapper.get_value(row, mapper.event_id_col, default=idx)
        moment_id = str(_safe_int(moment_id_val) if _safe_int(moment_id_val) is not None else idx)

        moments.append(
            {
                "moment_id": moment_id,
                "match_id": str(match_id),
                "period": int(period),
                "frame_start": int(fs),
                "frame_end": int(fe),
                "frame": int(fe),
                "minute_start": minute,
                "second_start": second,
                "time_label": time_label,
                "team_id": _safe_int(mapper.get_value(row, mapper.team_id_col)),
                "team_shortname": team_label,
                "player_id": _safe_int(mapper.get_value(row, mapper.player_id_col)),
                "player_name": (str(mapper.get_value(row, mapper.player_col)).strip() if mapper.get_value(row, mapper.player_col) is not None else None),
                "event_type": et,
                "event_subtype": (str(mapper.get_value(row, mapper.event_subtype_col)).strip() if mapper.get_value(row, mapper.event_subtype_col) is not None else None),
                "end_type": (str(mapper.get_value(row, mapper.end_type_col)).strip() if mapper.get_value(row, mapper.end_type_col) is not None else None),
                "pass_outcome": (str(mapper.get_value(row, mapper.pass_outcome_col)).strip() if mapper.get_value(row, mapper.pass_outcome_col) is not None else None),
                "turnover_type": "shot" if is_shot else "possession_loss",
            }
        )

    total = len(moments)
    return total, moments[offset : offset + limit]


# -----------------------
# Tracking windows
# -----------------------


def fetch_tracking_window_raw(db: Session, match_id: int, start_frame: int, end_frame: int) -> list[dict]:
    """Build replay window frames from `frame` + `detection`.

    How replay windows are built:
    - Query `frame` rows for (match_id, frame_id range)
    - Query `detection` rows for the same (match_id, frame_id range)
    - Group detections by frame_id and reconstruct each frame's player+ball payload
    """
    frame_t = _table(db, "frame")
    det_t = _table(db, "detection")

    f_match = _first_col(frame_t, "match_id")
    f_id = _first_col(frame_t, "id", "frame_id")
    f_ts = _first_col(frame_t, "timestamp", "time_stamp", "time_stamp_s", "time")
    if f_match is None or f_id is None:
        raise RuntimeError("frame table missing match_id/id")

    d_match = _first_col(det_t, "match_id")
    d_frame = _first_col(det_t, "frame_id", "frame")
    d_pid = _first_col(det_t, "player_id")
    d_x = _first_col(det_t, "x")
    d_y = _first_col(det_t, "y")
    d_z = _first_col(det_t, "z")
    d_ball = _first_col(det_t, "ball")
    d_team = _first_col(det_t, "team_id", "teamId")
    if d_match is None or d_frame is None:
        raise RuntimeError("detection table missing match_id/frame_id")

    # Clamp requested window to frames that actually exist for this match.
    #
    # Why this matters:
    # - Moments come from the `events` table and may reference a frame index that isn't
    #   present in the `frame` table (e.g. provider export mismatch, partial ingestion).
    # - If the requested window sits entirely outside the available tracking ids, we must
    #   clamp into bounds; otherwise the SQL range returns zero rows and the UI shows
    #   "No frames available for this moment" even though the match has tracking elsewhere.
    fmin, fmax = get_frame_bounds(db, match_id)
    if fmin is not None and fmax is not None:
        # Clamp both ends into [fmin, fmax]
        start_frame = min(max(int(start_frame), int(fmin)), int(fmax))
        end_frame = min(max(int(end_frame), int(fmin)), int(fmax))
        if end_frame < start_frame:
            end_frame = start_frame

    frame_rows = db.execute(
        select(f_id.label("frame"), f_ts.label("timestamp"))
        .where(and_(f_match == int(match_id), f_id >= int(start_frame), f_id <= int(end_frame)))
        .order_by(f_id)
    ).mappings().all()
    if not frame_rows:
        return []

    det_cols = [d_frame.label("frame")]
    if d_pid is not None:
        det_cols.append(d_pid.label("player_id"))
    if d_team is not None:
        det_cols.append(d_team.label("team_id"))
    if d_x is not None:
        det_cols.append(d_x.label("x"))
    if d_y is not None:
        det_cols.append(d_y.label("y"))
    if d_z is not None:
        det_cols.append(d_z.label("z"))
    if d_ball is not None:
        det_cols.append(d_ball.label("ball"))

    det_rows = db.execute(
        select(*det_cols)
        .where(and_(d_match == int(match_id), d_frame >= int(start_frame), d_frame <= int(end_frame)))
        .order_by(d_frame)
    ).mappings().all()

    by_frame: dict[int, list[dict]] = {}
    for r in det_rows:
        fr = _safe_int(r.get("frame"))
        if fr is None:
            continue
        by_frame.setdefault(fr, []).append(dict(r))

    frames_out: list[dict] = []
    for fr in frame_rows:
        fid = _safe_int(fr.get("frame")) or 0
        dets = by_frame.get(fid, [])
        player_data = []
        ball_data = None
        for d in dets:
            is_ball = bool(d.get("ball")) if "ball" in d else (d.get("player_id") == 0)
            if is_ball:
                ball_data = {"x": _safe_float(d.get("x")), "y": _safe_float(d.get("y"))}
                if "z" in d:
                    ball_data["z"] = _safe_float(d.get("z"))
                continue
            player_data.append(
                {
                    "player_id": _safe_int(d.get("player_id")),
                    "team_id": _safe_int(d.get("team_id")),
                    "x": _safe_float(d.get("x")),
                    "y": _safe_float(d.get("y")),
                }
            )
        frames_out.append(
            {
                "frame": fid,
                "timestamp": fr.get("timestamp"),
                "ball_data": ball_data,
                "player_data": player_data,
            }
        )
    return frames_out


def to_center_coords(frames: Iterable[dict]) -> list[dict]:
    """Normalize SkillCorner-style coords to renderer center coords.

    Assumption (SkillCorner):
    - x is 0..105 -> convert to [-52.5..52.5] by subtracting 52.5
    - y is already centered in [-34..34] (pass-through)
    """
    out = []
    for f in frames:
        ball = f.get("ball_data")
        if ball and ball.get("x") is not None:
            ball = {**ball, "x": float(ball["x"]) - 52.5}
        players = []
        for p in f.get("player_data") or []:
            if p.get("x") is None or p.get("y") is None:
                continue
            players.append(
                {
                    "player_id": _safe_int(p.get("player_id")),
                    "team_id": _safe_int(p.get("team_id")),
                    "x": float(p["x"]) - 52.5,
                    "y": float(p["y"]),
                }
            )
        out.append({**f, "ball_data": ball, "player_data": players})
    return out


def derive_possessor(frame: dict, max_dist_m: float = 2.5) -> Optional[dict]:
    """Infer possessor as nearest player to ball within threshold."""
    ball = frame.get("ball_data") or {}
    bx, by = ball.get("x"), ball.get("y")
    if bx is None or by is None:
        return None
    best = None
    best_d2 = max_dist_m * max_dist_m
    for p in frame.get("player_data") or []:
        pid = p.get("player_id")
        if pid is None:
            continue
        dx = float(p["x"]) - float(bx)
        dy = float(p["y"]) - float(by)
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best = p
    if not best:
        return None
    return {"team_id": best.get("team_id"), "player_id": best.get("player_id")}


def build_tracking_render_window(
    db: Session,
    match_id: int,
    center_frame: int,
    radius: int,
    include_players: bool,
    max_frames: int,
) -> dict:
    # Selected moment → tracking window:
    # - Frontend sends `center_frame` (usually event `frame` or `frame_end` from the `events` table)
    # - Backend expands to [center-radius, center+radius]
    # - Backend clamps to the `frame` ids that *actually exist* for this match in Postgres
    #   (because event frames and tracking frames can be misaligned or partially ingested).
    fmin, fmax = get_frame_bounds(db, match_id)
    if fmin is None or fmax is None:
        # Clear signal to UI: this match has no replayable tracking rows.
        raise HTTPException(
            status_code=404,
            detail={
                "error": "No tracking data",
                "message": f"No rows found in `frame` table for match_id={match_id}. This match cannot be replay-previewed.",
            },
        )

    requested_center = int(center_frame)
    clamped_center = min(max(requested_center, int(fmin)), int(fmax))

    start_frame = max(0, int(center_frame) - int(radius))
    end_frame = int(center_frame) + int(radius)
    if end_frame - start_frame + 1 > int(max_frames):
        end_frame = start_frame + int(max_frames) - 1
    # Clamp again using actual tracking bounds (handles the common case where event frames
    # are outside the available tracking range).
    start_frame = min(max(int(start_frame), int(fmin)), int(fmax))
    end_frame = min(max(int(end_frame), int(fmin)), int(fmax))
    if end_frame < start_frame:
        end_frame = start_frame

    raw = fetch_tracking_window_raw(db, match_id, start_frame, end_frame)
    centered = to_center_coords(raw)
    teams = get_match_teams(db, match_id)

    # `effective_center_frame` is what the frontend should "seek" to in the returned frames.
    # Start with the clamped center (so it always sits on an existing tracking id range),
    # then snap to the closest returned frame id if the exact value isn't present.
    effective_center = int(clamped_center)
    if centered:
        # If the requested center frame isn't present (e.g. event frame doesn't align exactly
        # with tracking frame ids), pick the closest available frame for consistent preview.
        ids = [int(f.get("frame") or 0) for f in centered]
        if effective_center not in ids:
            effective_center = min(ids, key=lambda fr: abs(fr - effective_center))

    frames_out = []
    for f in centered:
        poss = derive_possessor(f)
        ball = f.get("ball_data")
        players = []
        if include_players:
            for p in f.get("player_data") or []:
                tid = _safe_int(p.get("team_id"))
                side = None
                if tid is not None and teams.home_team_id is not None and tid == teams.home_team_id:
                    side = "home"
                elif tid is not None and teams.away_team_id is not None and tid == teams.away_team_id:
                    side = "away"
                players.append(
                    {
                        "id": _safe_int(p.get("player_id")),
                        "team_id": tid,
                        "team_side": side,
                        "x": float(p.get("x") or 0.0),
                        "y": float(p.get("y") or 0.0),
                        "speed": None,
                    }
                )

        frames_out.append(
            {
                "frame": int(f.get("frame") or 0),
                "period": None,
                "timestamp": str(f.get("timestamp")) if f.get("timestamp") is not None else None,
                "ball": (
                    {"x": float(ball["x"]), "y": float(ball["y"]), "z": ball.get("z")}
                    if ball and ball.get("x") is not None and ball.get("y") is not None
                    else None
                ),
                "players": players,
                "derived_possession": poss,
            }
        )

    return {
        "match_id": str(match_id),
        # Preserve the user's requested frame for debugging while also returning
        # `effective_center_frame` that is guaranteed to be within tracking bounds.
        "center_frame": int(requested_center),
        "effective_center_frame": int(effective_center),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "frames": frames_out,
    }


def fetch_single_center_frame(db: Session, match_id: int, frame_id: int) -> Optional[dict]:
    raw = fetch_tracking_window_raw(db, match_id, frame_id, frame_id)
    centered = to_center_coords(raw)
    return centered[0] if centered else None

