"""Soccer EPV — Computer Vision Pipeline (Streamlit UI)

Pipeline stages (matching original newapp.py intent):
  1. Upload a match image + roster CSV
  2. YOLO detects all persons (and ball) in the image
  3. User assigns a player name to each detected person via a dropdown
     filtered to the team whose jersey color is the closest match
  4. User clicks to confirm / override ball location
  5. User clicks 4 reference points on a field feature to calibrate
     homography (Center Circle | Penalty Box | Sidelines)
  6. App exports a CSV: player name, team, image-pixel center, and
     field coordinates (meters, origin = center circle = 0,0 with
     +x → right, +y → up on a 105×68 m pitch)

This file is the ONLY file you need to run:
    streamlit run app.py
It imports cv_service (must be in the same directory or on PYTHONPATH).
"""

from __future__ import annotations

import io
import csv
import json
from typing import Optional

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from streamlit_image_coordinates import streamlit_image_coordinates  # pip install streamlit-image-coordinates

import cv_service  # backend module (cv_service.py)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

STAGES = [
    "upload",          # 0 – upload image + roster
    "assign_players",  # 1 – name each detected person
    "ball",            # 2 – confirm ball location
    "calibrate",       # 3 – pick 4 field-feature points
    "results",         # 4 – view & download CSV
]

FEATURE_TYPES = list(cv_service.FEATURE_TEMPLATES.keys())  # Center Circle, Penalty Box, Sideline

# Visual style for overlays
BOX_COLOR_DEFAULT  = (255, 255,   0)   # yellow
BOX_COLOR_SELECTED = (  0, 255,   0)   # green  (player box currently being labelled)
BALL_COLOR         = (255,  80,  80)   # red
POINT_RADIUS       = 8
FONT_SIZE          = 14

# ─────────────────────────────────────────────────────────────────────────────
# Session-state helpers
# ─────────────────────────────────────────────────────────────────────────────

def _init_state() -> None:
    defaults: dict = {
        "stage":            STAGES[0],
        "image_rgb":        None,   # np.ndarray HxWx3 (resized by cv_service)
        "image_pil":        None,   # PIL image for display / drawing
        "roster_rows":      [],     # list[dict] parsed from CSV
        "cv_result":        None,   # dict returned by cv_service.analyze_image
        # assign_players stage
        "player_names":     {},     # index → player name string
        "current_player":   0,      # which bounding box we are naming right now
        # ball stage
        "ball_pixel":       None,   # (x, y) in resized-image pixel space
        # calibrate stage
        "feature_type":     FEATURE_TYPES[0],
        "feature_points":   [],     # list of [x, y] – up to 4
        # results
        "output_csv":       None,   # bytes
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _go(stage: str) -> None:
    st.session_state["stage"] = stage


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _draw_boxes(
    pil_img: Image.Image,
    players: list[dict],
    player_names: dict[int, str],
    current_idx: Optional[int] = None,
    ball_pixel: Optional[tuple[int, int]] = None,
) -> Image.Image:
    """Return a copy of pil_img with bounding boxes, labels, and ball drawn on it."""
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", FONT_SIZE)
    except Exception:
        font = ImageFont.load_default()

    for i, p in enumerate(players):
        color = BOX_COLOR_SELECTED if i == current_idx else BOX_COLOR_DEFAULT
        draw.rectangle([p["x1"], p["y1"], p["x2"], p["y2"]], outline=color, width=2)
        label = player_names.get(i, f"#{i+1} ?")
        draw.text((p["x1"] + 2, p["y1"] + 2), label, fill=color, font=font)

    if ball_pixel:
        bx, by = ball_pixel
        r = POINT_RADIUS
        draw.ellipse([bx - r, by - r, bx + r, by + r], outline=BALL_COLOR, width=3)
        draw.text((bx + r + 2, by - r), "Ball", fill=BALL_COLOR, font=font)

    return img


def _draw_feature_points(
    pil_img: Image.Image,
    points: list[list[float]],
) -> Image.Image:
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", FONT_SIZE)
    except Exception:
        font = ImageFont.load_default()

    colors = [(255, 80, 80), (80, 255, 80), (80, 80, 255), (255, 200, 0)]
    for n, (px, py) in enumerate(points):
        c = colors[n % len(colors)]
        r = POINT_RADIUS
        draw.ellipse([px - r, py - r, px + r, py + r], fill=c, outline=(255, 255, 255), width=2)
        draw.text((px + r + 2, py - r), f"P{n+1}", fill=c, font=font)

    return img


# ─────────────────────────────────────────────────────────────────────────────
# Roster helpers
# ─────────────────────────────────────────────────────────────────────────────

def _team_of_player(row: dict) -> str:
    for k in ("team", "Team", "squad", "Squad"):
        if row.get(k):
            return str(row[k])
    return "Unknown"


def _name_of_player(row: dict) -> str:
    for k in ("name", "Name", "player", "Player", "full_name"):
        if row.get(k):
            return str(row[k])
    # Fall back: join all non-team, non-color keys
    return " ".join(str(v) for k, v in row.items()
                    if k.lower() not in ("team", "squad", "team color", "team_color", "color", "hex"))


def _hex_of_team(roster_rows: list[dict], team: str) -> Optional[str]:
    for r in roster_rows:
        if _team_of_player(r) == team:
            for k in ("Team color", "team_color", "color", "hex", "Color"):
                if r.get(k):
                    return str(r[k])
    return None


def _all_teams(roster_rows: list[dict]) -> list[str]:
    seen: list[str] = []
    for r in roster_rows:
        t = _team_of_player(r)
        if t not in seen:
            seen.append(t)
    return seen


def _players_for_team(roster_rows: list[dict], team: str) -> list[str]:
    names = []
    for r in roster_rows:
        if _team_of_player(r) == team:
            n = _name_of_player(r)
            if n:
                names.append(n)
    return names


def _best_team_for_box(team_guess: Optional[str], all_teams: list[str]) -> str:
    """Return the team name to pre-select in the dropdown."""
    if team_guess and team_guess in all_teams:
        return team_guess
    return all_teams[0] if all_teams else "Unknown"


# ─────────────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────────────

def _build_csv(
    players: list[dict],
    player_names: dict[int, str],
    roster_rows: list[dict],
    ball_pixel: Optional[tuple[int, int]],
    cv_result: dict,
) -> bytes:
    """Build output CSV with player names, teams, pixel coords, and field coords."""
    buf = io.StringIO()
    fieldnames = [
        "name", "number", "team",
        "image_x_px", "image_y_px",
        "field_x_m", "field_y_m",
    ]
    w = csv.DictWriter(buf, fieldnames=fieldnames)
    w.writeheader()

    # Build a name → roster row lookup
    name_to_row: dict[str, dict] = {}
    for r in roster_rows:
        name_to_row[_name_of_player(r)] = r

    for i, p in enumerate(players):
        name = player_names.get(i, "")
        row_data = name_to_row.get(name, {})
        team = _team_of_player(row_data) if row_data else (p.get("team_guess") or "")
        number = row_data.get("number") or row_data.get("Number") or row_data.get("#") or ""
        w.writerow({
            "name":       name,
            "number":     number,
            "team":       team,
            "image_x_px": p["center_x"],
            "image_y_px": p["center_y"],
            "field_x_m":  round(p.get("field_x_m", 0) - 52.5, 3) if p.get("field_x_m") is not None else "",
            "field_y_m":  round(p.get("field_y_m", 0) - 34.0, 3) if p.get("field_y_m") is not None else "",
        })

    # Ball row
    if ball_pixel:
        bx, by = ball_pixel
        # Look up field coords from cv_result ball
        ball_data = cv_result.get("ball") or {}
        fx = ball_data.get("field_x_m")
        fy = ball_data.get("field_y_m")
        w.writerow({
            "name":       "BALL",
            "number":     "",
            "team":       "",
            "image_x_px": bx,
            "image_y_px": by,
            "field_x_m":  round(fx - 52.5, 3) if fx is not None else "",
            "field_y_m":  round(fy - 34.0, 3) if fy is not None else "",
        })

    return buf.getvalue().encode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Stage renderers
# ─────────────────────────────────────────────────────────────────────────────

def _stage_upload() -> None:
    st.header("Step 1 — Upload match image & roster")

    img_file = st.file_uploader("Match image (JPG / PNG)", type=["jpg", "jpeg", "png"])
    csv_file = st.file_uploader(
        "Roster CSV (columns: name, number, team, Team color)",
        type=["csv"],
    )

    if img_file:
        pil = Image.open(img_file).convert("RGB")
        st.image(pil, caption="Uploaded image preview", use_container_width=True)

    if img_file and st.button("Run YOLO detection →", type="primary"):
        with st.spinner("Running YOLOv8 detection…"):
            pil = Image.open(img_file).convert("RGB")
            image_rgb = np.array(pil)
            roster_bytes = csv_file.read() if csv_file else None

            result = cv_service.analyze_image(
                image_rgb=image_rgb,
                roster_csv_bytes=roster_bytes,
            )

        # Resize image to match what cv_service analyzed
        w, h = result["image_width"], result["image_height"]
        image_rgb_resized = cv_service.resize_image_rgb_like_newapp(image_rgb)
        pil_resized = Image.fromarray(image_rgb_resized)

        st.session_state["image_rgb"]    = image_rgb_resized
        st.session_state["image_pil"]    = pil_resized
        st.session_state["cv_result"]    = result
        st.session_state["roster_rows"]  = cv_service._parse_roster_csv(roster_bytes) if roster_bytes else []
        st.session_state["player_names"] = {}
        st.session_state["current_player"] = 0
        st.session_state["ball_pixel"]   = (result["ball"]["x"], result["ball"]["y"]) if result.get("ball") else None
        st.session_state["feature_points"] = []

        n_players = len(result.get("players", []))
        st.success(f"Detected {n_players} player(s)" + (" and the ball." if result.get("ball") else " (ball not detected)."))
        _go("assign_players")
        st.rerun()


def _stage_assign_players() -> None:
    st.header("Step 2 — Assign player names")
    st.caption("For each highlighted bounding box, pick the matching player from the dropdown.")

    result       = st.session_state["cv_result"]
    players      = result.get("players", [])
    roster_rows  = st.session_state["roster_rows"]
    player_names = st.session_state["player_names"]
    idx          = st.session_state["current_player"]
    ball_pixel   = st.session_state["ball_pixel"]

    if not players:
        st.warning("No players detected. Proceed to next step.")
        if st.button("Next →"):
            _go("ball")
            st.rerun()
        return

    # Draw overlay: all boxes, highlight current
    overlay = _draw_boxes(
        st.session_state["image_pil"],
        players,
        player_names,
        current_idx=idx,
        ball_pixel=ball_pixel,
    )
    st.image(overlay, use_container_width=True)

    # Progress
    st.progress((idx) / len(players), text=f"Player {idx + 1} of {len(players)}")

    if idx < len(players):
        p = players[idx]
        all_teams = _all_teams(roster_rows) if roster_rows else []

        col_left, col_right = st.columns([1, 2])
        with col_left:
            # Show crop of current box
            cx1, cy1, cx2, cy2 = p["x1"], p["y1"], p["x2"], p["y2"]
            crop = st.session_state["image_pil"].crop((cx1, cy1, cx2, cy2))
            st.image(crop, caption=f"Detection #{idx + 1}", width=140)

            if p.get("team_guess"):
                st.caption(f"🎽 Jersey color → **{p['team_guess']}**")

        with col_right:
            if roster_rows and all_teams:
                # Team selector, pre-populated from YOLO color guess
                default_team = _best_team_for_box(p.get("team_guess"), all_teams)
                team_choice = st.selectbox(
                    "Team",
                    options=all_teams,
                    index=all_teams.index(default_team),
                    key=f"team_sel_{idx}",
                )
                # Name dropdown filtered to chosen team
                names_for_team = _players_for_team(roster_rows, team_choice)
                if not names_for_team:
                    names_for_team = ["(no players for this team)"]

                # Pre-fill if we already named this player
                prev_name = player_names.get(idx)
                name_idx = names_for_team.index(prev_name) if prev_name in names_for_team else 0

                name_choice = st.selectbox(
                    "Player name",
                    options=names_for_team,
                    index=name_idx,
                    key=f"name_sel_{idx}",
                )
            else:
                st.info("No roster loaded — enter name manually.")
                name_choice = st.text_input(
                    "Player name",
                    value=player_names.get(idx, ""),
                    key=f"name_text_{idx}",
                )

            col_prev, col_skip, col_next = st.columns(3)
            with col_prev:
                if idx > 0 and st.button("← Back"):
                    st.session_state["current_player"] -= 1
                    st.rerun()
            with col_skip:
                if st.button("Skip"):
                    player_names[idx] = "(unknown)"
                    st.session_state["player_names"] = player_names
                    if idx + 1 < len(players):
                        st.session_state["current_player"] += 1
                    else:
                        _go("ball")
                    st.rerun()
            with col_next:
                label = "Next →" if idx + 1 < len(players) else "Done ✓"
                if st.button(label, type="primary"):
                    player_names[idx] = name_choice
                    st.session_state["player_names"] = player_names
                    if idx + 1 < len(players):
                        st.session_state["current_player"] += 1
                        st.rerun()
                    else:
                        _go("ball")
                        st.rerun()
    else:
        st.success("All players assigned!")
        if st.button("Next → Confirm ball location", type="primary"):
            _go("ball")
            st.rerun()


def _stage_ball() -> None:
    st.header("Step 3 — Confirm ball location")
    st.caption(
        "Click on the image to mark the ball's position. "
        "A red circle shows the YOLO-detected location (if found)."
    )

    result      = st.session_state["cv_result"]
    players     = result.get("players", [])
    player_names = st.session_state["player_names"]
    ball_pixel  = st.session_state["ball_pixel"]

    overlay = _draw_boxes(
        st.session_state["image_pil"],
        players,
        player_names,
        ball_pixel=ball_pixel,
    )

    coords = streamlit_image_coordinates(overlay, key="ball_click")

    if coords:
        st.session_state["ball_pixel"] = (int(coords["x"]), int(coords["y"]))
        st.rerun()

    if ball_pixel:
        st.info(f"Ball position: pixel ({ball_pixel[0]}, {ball_pixel[1]})")
    else:
        st.warning("No ball location set yet. Click the image above to mark the ball.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to players"):
            _go("assign_players")
            st.session_state["current_player"] = len(players) - 1
            st.rerun()
    with col2:
        if st.button("Next → Calibrate field →", type="primary"):
            _go("calibrate")
            st.rerun()


def _stage_calibrate() -> None:
    st.header("Step 4 — Field calibration (homography)")
    st.caption(
        "Select which field feature is visible, then click **4 matching points** on the image "
        "in the order shown in the diagram. This maps pixel coordinates to field meters "
        "(origin = center circle, +x → right, +y → up)."
    )

    feature_type   = st.session_state["feature_type"]
    feature_points = st.session_state["feature_points"]
    result         = st.session_state["cv_result"]
    players        = result.get("players", [])
    player_names   = st.session_state["player_names"]
    ball_pixel     = st.session_state["ball_pixel"]

    # Feature type selector
    new_ft = st.selectbox(
        "Field feature visible in image",
        options=FEATURE_TYPES,
        index=FEATURE_TYPES.index(feature_type),
    )
    if new_ft != feature_type:
        st.session_state["feature_type"] = new_ft
        st.session_state["feature_points"] = []
        st.rerun()

    # Show the 4-point guide for the selected feature
    _render_feature_guide(new_ft)

    st.markdown(f"**Points selected: {len(feature_points)} / 4**")
    if feature_points:
        for n, pt in enumerate(feature_points):
            st.caption(f"  P{n+1}: pixel ({int(pt[0])}, {int(pt[1])})")

    # Draw current state
    overlay = _draw_boxes(
        st.session_state["image_pil"],
        players,
        player_names,
        ball_pixel=ball_pixel,
    )
    overlay = _draw_feature_points(overlay, feature_points)

    if len(feature_points) < 4:
        coords = streamlit_image_coordinates(overlay, key="calib_click")
        if coords:
            feature_points.append([float(coords["x"]), float(coords["y"])])
            st.session_state["feature_points"] = feature_points
            st.rerun()
    else:
        st.image(overlay, use_container_width=True)
        st.success("4 points selected!")

    col_reset, col_back, col_next = st.columns(3)
    with col_reset:
        if st.button("🔄 Reset points"):
            st.session_state["feature_points"] = []
            st.rerun()
    with col_back:
        if st.button("← Back to ball"):
            _go("ball")
            st.rerun()
    with col_next:
        can_proceed = len(feature_points) == 4
        if st.button("Compute & Export →", type="primary", disabled=not can_proceed):
            _run_homography_and_export()
            _go("results")
            st.rerun()


def _render_feature_guide(feature_type: str) -> None:
    """Show a small text diagram describing point order for the chosen feature."""
    guides = {
        "Center Circle": (
            "Click in this order:\n"
            "  P1 → Center of the circle (52.5, 34 m)\n"
            "  P2 → Top of circle (52.5, 25 m)\n"
            "  P3 → Bottom of circle (52.5, 43 m)\n"
            "  P4 → Left arc tangent (43.5, 34 m)"
        ),
        "Penalty Box": (
            "Click in this order:\n"
            "  P1 → Top-right corner of box (16.5, 13.84 m)\n"
            "  P2 → Bottom-right corner (16.5, 54.16 m)\n"
            "  P3 → Top-left / goal line top (0, 13.84 m)\n"
            "  P4 → Bottom-left / goal line bottom (0, 54.16 m)"
        ),
        "Sideline": (
            "Click in this order:\n"
            "  P1 → Top-left corner of pitch (0, 0 m)\n"
            "  P2 → Top-right corner (105, 0 m)\n"
            "  P3 → Bottom-right corner (105, 68 m)\n"
            "  P4 → Bottom-left corner (0, 68 m)"
        ),
    }
    st.info(guides.get(feature_type, "Select 4 points in the order matching the feature template."))


def _run_homography_and_export() -> None:
    """Re-run cv_service.analyze_image with homography inputs and build output CSV."""
    result         = st.session_state["cv_result"]
    feature_type   = st.session_state["feature_type"]
    feature_points = st.session_state["feature_points"]
    ball_pixel     = st.session_state["ball_pixel"]
    roster_rows    = st.session_state["roster_rows"]
    player_names   = st.session_state["player_names"]
    image_rgb      = st.session_state["image_rgb"]

    roster_bytes = None
    if roster_rows:
        buf = io.StringIO()
        if roster_rows:
            w = csv.DictWriter(buf, fieldnames=roster_rows[0].keys())
            w.writeheader()
            w.writerows(roster_rows)
        roster_bytes = buf.getvalue().encode("utf-8")

    with st.spinner("Computing homography and projecting coordinates…"):
        full_result = cv_service.analyze_image(
            image_rgb=image_rgb,
            roster_csv_bytes=roster_bytes,
            feature_type=feature_type,
            feature_points=feature_points,
        )

    # Override ball pixel with user-confirmed location and re-project it
    if ball_pixel and full_result.get("homography"):
        H = np.array(full_result["homography"]["H"], dtype=np.float32)
        fx, fy = cv_service.project_to_field(H, float(ball_pixel[0]), float(ball_pixel[1]))
        if full_result.get("ball") is None:
            full_result["ball"] = {}
        full_result["ball"]["x"] = ball_pixel[0]
        full_result["ball"]["y"] = ball_pixel[1]
        full_result["ball"]["field_x_m"] = fx
        full_result["ball"]["field_y_m"] = fy

    st.session_state["cv_result"] = full_result

    csv_bytes = _build_csv(
        players=full_result.get("players", []),
        player_names=player_names,
        roster_rows=roster_rows,
        ball_pixel=ball_pixel,
        cv_result=full_result,
    )
    st.session_state["output_csv"] = csv_bytes


def _stage_results() -> None:
    st.header("Step 5 — Results & Export")

    result       = st.session_state["cv_result"]
    players      = result.get("players", [])
    player_names = st.session_state["player_names"]
    ball_pixel   = st.session_state["ball_pixel"]
    feature_pts  = st.session_state["feature_points"]
    csv_bytes    = st.session_state["output_csv"]

    # Final annotated image
    overlay = _draw_boxes(
        st.session_state["image_pil"],
        players,
        player_names,
        ball_pixel=ball_pixel,
    )
    overlay = _draw_feature_points(overlay, feature_pts)
    st.image(overlay, caption="Final annotated frame", use_container_width=True)

    # Summary table
    st.subheader("Player coordinates")
    import pandas as pd
    rows = []
    name_to_row: dict[str, dict] = {}
    for r in st.session_state["roster_rows"]:
        name_to_row[cv_service._name_of_player(r) if hasattr(cv_service, "_name_of_player") else ""] = r

    for i, p in enumerate(players):
        name = player_names.get(i, "(unknown)")
        team = p.get("team_guess", "")
        rows.append({
            "Name":        name,
            "Team":        team,
            "Image X (px)": p["center_x"],
            "Image Y (px)": p["center_y"],
            "Field X (m)": round(p["field_x_m"] - 52.5, 2) if p.get("field_x_m") is not None else "—",
            "Field Y (m)": round(p["field_y_m"] - 34.0, 2) if p.get("field_y_m") is not None else "—",
        })

    if ball_pixel:
        ball_data = result.get("ball") or {}
        rows.append({
            "Name":        "BALL",
            "Team":        "",
            "Image X (px)": ball_pixel[0],
            "Image Y (px)": ball_pixel[1],
            "Field X (m)": round(ball_data["field_x_m"] - 52.5, 2) if ball_data.get("field_x_m") is not None else "—",
            "Field Y (m)": round(ball_data["field_y_m"] - 34.0, 2) if ball_data.get("field_y_m") is not None else "—",
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Heuristic recommendation
    rec = result.get("recommendation")
    if rec:
        with st.expander("💡 Heuristic spatial recommendation (non-model-based)", expanded=False):
            st.markdown(f"**Action:** {rec['action']}")
            st.markdown(f"**Explanation:** {rec['explanation']}")
            st.progress(rec["strength"], text=f"Confidence indicator: {rec['strength']:.0%}")
            st.caption("⚠️ " + " | ".join(rec.get("limitations", [])))

    # Download
    st.subheader("Download CSV")
    st.caption(
        "Field coordinates use origin = center circle (0, 0). "
        "+X → right side of pitch, +Y → top of pitch (meters on a 105×68 m field)."
    )
    if csv_bytes:
        st.download_button(
            label="⬇️ Download player_coordinates.csv",
            data=csv_bytes,
            file_name="player_coordinates.csv",
            mime="text/csv",
            type="primary",
        )

    if st.button("🔄 Start over"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Soccer EPV — CV Pipeline",
        page_icon="⚽",
        layout="wide",
    )
    st.title("⚽ Soccer EPV — Computer Vision Pipeline")

    _init_state()

    stage = st.session_state["stage"]

    # Sidebar progress indicator
    with st.sidebar:
        st.header("Pipeline stages")
        labels = [
            "1. Upload & detect",
            "2. Assign players",
            "3. Confirm ball",
            "4. Calibrate field",
            "5. Results & export",
        ]
        stage_keys = STAGES
        for i, (lbl, sk) in enumerate(zip(labels, stage_keys)):
            if sk == stage:
                st.markdown(f"**→ {lbl}**")
            elif stage_keys.index(stage) > i:
                st.markdown(f"~~{lbl}~~ ✓")
            else:
                st.markdown(f"&nbsp;&nbsp; {lbl}")

    if stage == "upload":
        _stage_upload()
    elif stage == "assign_players":
        _stage_assign_players()
    elif stage == "ball":
        _stage_ball()
    elif stage == "calibrate":
        _stage_calibrate()
    elif stage == "results":
        _stage_results()
    else:
        st.error(f"Unknown stage: {stage}")


if __name__ == "__main__":
    main()
