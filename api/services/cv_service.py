"""Computer-vision service (backend-only).

Source of truth for behavior: `CV/newapp.py` (Streamlit). This module extracts the same
functions without rewriting the pipeline:

| newapp.py (approx.)        | This module |
|---------------------------|-------------|
| `resize_image` (in UI)    | `resize_image_rgb_like_newapp` — only if width > 800 |
| `detect_objects`          | `detect_objects` — same YOLO labels/conf |
| `project_to_field`        | `project_to_field` — identical OpenCV call |
| `hex_to_bgr`              | `_hex_to_bgr` |
| `get_box_center`          | `_get_box_center` |
| `FEATURE_TEMPLATES`       | `FEATURE_TEMPLATES` |
| roster team-color loop    | `_avg_bgr_in_box` + roster parsing |

UI-only pieces (canvas clicks, session state, CSV download) stay in Streamlit / React.

What input this CV service expects:
- An RGB image (numpy array HxWx3); it is resized like newapp (max width 800) before YOLO.
- Optionally a roster CSV (bytes): columns `team`, `Team color` (same as newapp).
- Optionally `feature_type` + exactly 4 `feature_points` **in the same pixel space as the
  resized analysis image** (width/height returned as `image_width` / `image_height`).

What output shape the frontend should expect:
- `image_width`, `image_height`: dimensions of the analyzed frame (boxes are in this space).
- `players`: bounding boxes, centers, optional `team_guess` / `team_guess_distance`, and when
  homography is computed optional `field_x_m` / `field_y_m` (meters on 105×68 field).
- `ball`: center in image pixels, optional `field_x_m` / `field_y_m` when homography exists.
- `homography`: `H`, `projected_points` (the 4 calibration points in field coords).

Project goals reflected here:
- Heavy processing stays on the backend (YOLO + OpenCV)
- Frontend sends inputs, receives clean JSON, and renders (overlay uses `image_*` for scale)
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, Optional

import numpy as np


FeatureType = Literal["Center Circle", "Penalty Box", "Sideline"]

def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _heuristic_recommendation_from_single_image(out: dict[str, Any]) -> dict[str, Any]:
    """Return a compact "recommended next action" payload for the UI.

    This is intentionally **heuristic**, not model-based.
    From a single broadcast image we typically lack:
    - possession team / ball carrier identity
    - player velocities & orientation
    - full team context (off-screen players)
    - temporal context (what happened before/after)
    Therefore we do NOT pretend to output a true EPV/optimal soccer decision here.

    Visual cues used (when available):
    - presence of ball detection
    - whether homography was computed (gives ball field coordinates on a 105×68 pitch)
    - ball proximity to goal(s) and sidelines (coarse spatial context)
    """
    players = out.get("players") or []
    ball = out.get("ball") or None
    homography = out.get("homography") or None

    has_ball = bool(ball and ball.get("x") is not None and ball.get("y") is not None)
    has_field = bool(
        ball
        and ball.get("field_x_m") is not None
        and ball.get("field_y_m") is not None
        and homography is not None
    )

    # Default: ask the user for the missing calibration step rather than guessing a decision.
    action = "Calibrate field for spatial context"
    explanation = "Add 4 feature points (homography) so detections can be projected to field meters."
    strength = 0.35
    cues = ["detections available"]
    limitations = [
        "Single image lacks time/velocity, so this cannot be EPV/model-based.",
        "Possession team and ball carrier cannot be reliably inferred from one frame.",
    ]

    if has_ball:
        cues.append("ball detected")
        strength += 0.10
    else:
        action = "Verify ball location"
        explanation = "Ball was not detected; consider a clearer frame or manually confirming ball position."
        limitations.append("Without ball location, spatial soccer heuristics are unreliable.")

    if has_field:
        # With field coordinates we can give a *coarse* soccer-action suggestion, but still heuristic.
        bx = float(ball["field_x_m"])
        by = float(ball["field_y_m"])
        cues.append("homography computed (field projection)")
        strength += 0.25

        # Goals in the field template coordinate system.
        left_goal = (0.0, 34.0)
        right_goal = (105.0, 34.0)
        d_left = float(np.hypot(bx - left_goal[0], by - left_goal[1]))
        d_right = float(np.hypot(bx - right_goal[0], by - right_goal[1]))
        d_goal = min(d_left, d_right)
        near_goal_side = "left" if d_left < d_right else "right"

        # Sideline proximity (y in [0,68]).
        d_sideline = min(by, 68.0 - by)

        # Coarse categorization of "danger zone".
        if d_goal <= 22.0 and abs(by - 34.0) <= 18.0:
            action = "Consider a shot (heuristic)"
            explanation = (
                f"Ball projects within ~{d_goal:.0f}m of the {near_goal_side} goal and is relatively central."
            )
            strength += 0.20
            cues.append("ball close to goal")
        elif bx >= 70.0 or bx <= 35.0:
            action = "Look for a pass or carry into space (heuristic)"
            explanation = (
                "Ball projects into an advanced zone but not a clear central shooting pocket from this frame."
            )
            strength += 0.10
            cues.append("ball in advanced zone")
        else:
            action = "Prefer retaining possession via pass (heuristic)"
            explanation = "Ball projects in a middle-zone area; from a single frame, safe progression is more plausible than a low-quality shot."
            strength += 0.05

        if d_sideline <= 6.0:
            cues.append("ball near touchline")
            explanation += " It is also close to the touchline, so options may be constrained—consider passing inside."
            strength += 0.05

        limitations.append(
            "This uses only ball location geometry (no defender pressure, no teammate lanes, no xG/EPV)."
        )
        limitations.append(
            "Attacking direction is unknown; 'near goal' is based on the closest goal in the projected field coordinate system."
        )

    # A small confidence-like scalar for UI. Clamp to [0,1].
    strength = _clamp(strength, 0.0, 0.95)

    return {
        "action": action,
        "explanation": explanation,
        "strength": float(strength),
        "is_model_based": False,
        "cues": cues,
        "limitations": limitations,
        "inputs_used": {
            "ball_detected": has_ball,
            "field_projection_available": has_field,
            "player_detections": int(len(players)),
        },
    }


FEATURE_TEMPLATES: dict[FeatureType, np.ndarray] = {
    # Same templates as `CV/newapp.py` (field coords in meters, 105x68).
    "Center Circle": np.array(
        [[52.5, 34], [52.5, 25], [52.5, 43], [43.5, 34]], dtype=np.float32
    ),
    "Penalty Box": np.array(
        [[16.5, 13.84], [16.5, 54.16], [0, 13.84], [0, 54.16]], dtype=np.float32
    ),
    "Sideline": np.array([[0, 0], [105, 0], [105, 68], [0, 68]], dtype=np.float32),
}


@dataclass(frozen=True)
class CVPlayerDetection:
    x1: int
    y1: int
    x2: int
    y2: int
    center_x: int
    center_y: int
    team_guess: Optional[str] = None
    team_guess_distance: Optional[float] = None


@dataclass(frozen=True)
class CVBallDetection:
    x: int
    y: int


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    hex_color = (hex_color or "").strip()
    if not hex_color.startswith("#"):
        hex_color = f"#{hex_color}"
    rgb = tuple(int(hex_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
    return rgb[::-1]  # BGR


def _get_box_center(box: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def _avg_bgr_in_box(image_bgr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), image_bgr.shape[1] - 1))
    x2 = max(0, min(int(x2), image_bgr.shape[1]))
    y1 = max(0, min(int(y1), image_bgr.shape[0] - 1))
    y2 = max(0, min(int(y2), image_bgr.shape[0]))
    if x2 <= x1 or y2 <= y1:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    crop = image_bgr[y1:y2, x1:x2]
    return np.mean(crop.reshape(-1, 3), axis=0)


def _parse_roster_csv(roster_csv_bytes: bytes) -> list[dict[str, Any]]:
    """Parse a roster CSV similar to the Streamlit app.

    Expected columns (best-effort):
    - team
    - Team color
    """
    text = roster_csv_bytes.decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for r in reader:
        rows.append({k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items()})
    return rows


@lru_cache(maxsize=1)
def _load_yolo_model():
    """Lazy-load YOLO model.

    Reused from `CV/newapp.py` (`YOLO("yolov8m.pt")`); imports stay lazy so the API can
    import without ultralytics installed.
    """
    from ultralytics import YOLO  # type: ignore

    return YOLO("yolov8m.pt")


def resize_image_rgb_like_newapp(image_rgb: np.ndarray, max_width: int = 800) -> np.ndarray:
    """Resize wide images to max_width; keep smaller images unchanged.

    Matches `CV/newapp.py` lines 93–98 (inner `resize_image` in the upload handler), not the
    module-level `resize_image_to_width` which always scales to a fixed width.
    """
    import cv2  # type: ignore

    h, w = image_rgb.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_size = (max_width, int(h * scale))
        return cv2.resize(image_rgb, new_size, interpolation=cv2.INTER_AREA)
    return image_rgb


def project_to_field(H: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """Map image pixel (x, y) to field coordinates using homography H.

    Lifted verbatim from `CV/newapp.py` (`project_to_field`).
    """
    import cv2  # type: ignore

    pt = np.array([[[x, y]]], dtype=np.float32)
    projected = cv2.perspectiveTransform(pt, H)
    return float(projected[0][0][0]), float(projected[0][0][1])


def detect_objects(image_rgb: np.ndarray, conf: float = 0.3) -> tuple[list[tuple[int, int, int, int]], Optional[tuple[int, int]]]:
    """Detect player boxes and ball center from an RGB image.

    Reused from `CV/newapp.py` (same labels: "person", "sports ball").
    """
    model = _load_yolo_model()
    results = model(image_rgb, conf=conf)[0]

    players: list[tuple[int, int, int, int]] = []
    ball: Optional[tuple[int, int]] = None

    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

        if label == "person":
            players.append((x1, y1, x2, y2))
        elif label == "sports ball":
            ball = ((x1 + x2) // 2, (y1 + y2) // 2)

    return players, ball


def _compute_homography_matrix(
    feature_type: FeatureType,
    image_points: list[list[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Same as `CV/newapp.py` homography block (image_pts → field_pts via findHomography)."""
    import cv2  # type: ignore

    if feature_type not in FEATURE_TEMPLATES:
        raise ValueError(f"Unsupported feature_type: {feature_type}")
    if len(image_points) != 4:
        raise ValueError("image_points must contain exactly 4 points")

    image_pts = np.array(image_points, dtype=np.float32)
    field_pts = FEATURE_TEMPLATES[feature_type]
    H, _ = cv2.findHomography(image_pts, field_pts)
    if H is None:
        raise ValueError("Homography computation failed")

    projected = cv2.perspectiveTransform(image_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    return H, projected


def compute_homography(
    feature_type: FeatureType,
    image_points: list[list[float]],
) -> dict[str, Any]:
    """Compute homography mapping image points → field coordinates (JSON-serializable)."""
    H, projected = _compute_homography_matrix(feature_type, image_points)
    return {"H": H.tolist(), "projected_points": projected.tolist()}


def analyze_image(
    image_rgb: np.ndarray,
    roster_csv_bytes: Optional[bytes] = None,
    feature_type: Optional[FeatureType] = None,
    feature_points: Optional[list[list[float]]] = None,
) -> dict[str, Any]:
    """Run CV analysis on a single image and return JSON-serializable results."""
    image_rgb = resize_image_rgb_like_newapp(image_rgb)
    h_an, w_an = int(image_rgb.shape[0]), int(image_rgb.shape[1])

    # Convert RGB→BGR for OpenCV-like operations (color averaging); mirrors newapp BGR crops.
    image_bgr = image_rgb[..., ::-1].copy()

    player_boxes, ball_center = detect_objects(image_rgb)

    roster_rows: list[dict[str, Any]] = []
    if roster_csv_bytes:
        roster_rows = _parse_roster_csv(roster_csv_bytes)

    # Pre-compute team colors if roster provided.
    team_colors: list[tuple[str, tuple[int, int, int]]] = []
    if roster_rows:
        for r in roster_rows:
            team = r.get("team") or r.get("Team") or r.get("squad") or r.get("Squad")
            color = r.get("Team color") or r.get("team_color") or r.get("color")
            if team and color:
                try:
                    team_colors.append((str(team), _hex_to_bgr(str(color))))
                except Exception:
                    continue

    players_out: list[dict[str, Any]] = []
    for (x1, y1, x2, y2) in player_boxes:
        cx, cy = _get_box_center((x1, y1, x2, y2))
        team_guess = None
        team_guess_distance = None

        # Optional: lightweight team scoring by jersey-color distance (reused heuristic).
        if team_colors:
            avg = _avg_bgr_in_box(image_bgr, (x1, y1, x2, y2))
            best_team = None
            best_dist = float("inf")
            for team, bgr in team_colors:
                dist = float(np.linalg.norm(avg - np.array(bgr, dtype=np.float32)))
                if dist < best_dist:
                    best_dist = dist
                    best_team = team
            team_guess = best_team
            team_guess_distance = float(best_dist) if best_team is not None else None

        players_out.append(
            CVPlayerDetection(
                x1=int(x1),
                y1=int(y1),
                x2=int(x2),
                y2=int(y2),
                center_x=int(cx),
                center_y=int(cy),
                team_guess=team_guess,
                team_guess_distance=team_guess_distance,
            ).__dict__
        )

    ball_out: Optional[dict[str, Any]] = None
    if ball_center:
        ball_out = CVBallDetection(x=int(ball_center[0]), y=int(ball_center[1])).__dict__

    out: dict[str, Any] = {
        "image_width": w_an,
        "image_height": h_an,
        "players": players_out,
        "ball": ball_out,
    }

    if feature_type and feature_points:
        H, projected = _compute_homography_matrix(feature_type, feature_points)
        out["homography"] = {"H": H.tolist(), "projected_points": projected.tolist()}
        # Field position (meters) for each detection — `project_to_field` from newapp.
        for p in players_out:
            fx, fy = project_to_field(H, float(p["center_x"]), float(p["center_y"]))
            p["field_x_m"] = fx
            p["field_y_m"] = fy
        if ball_out is not None:
            fx, fy = project_to_field(H, float(ball_out["x"]), float(ball_out["y"]))
            ball_out["field_x_m"] = fx
            ball_out["field_y_m"] = fy
    else:
        out["homography"] = None

    # Recommended next action:
    # This is a **heuristic** helper for the UI, not an EPV/model-based decision.
    # See `_heuristic_recommendation_from_single_image` for cues and limitations.
    out["recommendation"] = _heuristic_recommendation_from_single_image(out)

    return out


def parse_feature_points_json(s: Optional[str]) -> Optional[list[list[float]]]:
    if not s:
        return None
    obj = json.loads(s)
    if not isinstance(obj, list):
        raise ValueError("feature_points must be a JSON list")
    pts: list[list[float]] = []
    for p in obj:
        if not (isinstance(p, list) and len(p) == 2):
            raise ValueError("Each feature point must be [x, y]")
        pts.append([float(p[0]), float(p[1])])
    return pts

