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
zq
What input this CV service expects:
- An RGB image (numpy array HxWx3); it is resized like newapp (max width 800) before YOLO.
- Optionally a roster CSV (bytes): columns `team`, `Team color` (same as newapp).
- Optionally `feature_type` + exactly 4 `feature_points` in the same pixel space as the
  resized analysis image (width/height returned as `image_width` / `image_height`).

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
    """Return a compact "recommended next action" payload for the UI."""
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
        bx = float(ball["field_x_m"])
        by = float(ball["field_y_m"])
        cues.append("homography computed (field projection)")
        strength += 0.25

        left_goal = (0.0, 34.0)
        right_goal = (105.0, 34.0)
        d_left = float(np.hypot(bx - left_goal[0], by - left_goal[1]))
        d_right = float(np.hypot(bx - right_goal[0], by - right_goal[1]))
        d_goal = min(d_left, d_right)
        near_goal_side = "left" if d_left < d_right else "right"

        d_sideline = min(by, 68.0 - by)

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

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from PIL import Image
import numpy as np

app = FastAPI()

# Allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def read_image(file: UploadFile):
    img = Image.open(file.file).convert("RGB")
    return np.array(img)

@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    roster: Optional[UploadFile] = File(None),
):
    image_np = read_image(image)
    roster_bytes = await roster.read() if roster else None

    result = analyze_image(
        image_rgb=image_np,
        roster_csv_bytes=roster_bytes,
    )

    return result


@app.post("/homography")
async def run_homography(
    image: UploadFile = File(...),
    feature_type: str = Form(...),
    feature_points: str = Form(...),  # JSON string
    roster: Optional[UploadFile] = File(None),
):
    image_np = read_image(image)
    roster_bytes = await roster.read() if roster else None
    points = json.loads(feature_points)

    result = analyze_image(
        image_rgb=image_np,
        roster_csv_bytes=roster_bytes,
        feature_type=feature_type,
        feature_points=points,
    )

    return result

@app.post("/compute-homography")
async def compute_homography(
    image: UploadFile = File(...),
    feature_type: str = Form(...),
    feature_points: str = Form(...),
    ball_x: float = Form(...),
    ball_y: float = Form(...),
    roster: Optional[UploadFile] = File(None),
):
    image_np = read_image(image)

    roster_bytes = await roster.read() if roster else None

    points = json.loads(feature_points)

    # Run full analysis with homography
    result = analyze_image(
        image_rgb=image_np,
        roster_csv_bytes=roster_bytes,
        feature_type=feature_type,
        feature_points=points,
    )

    # Project manually selected ball
    if result.get("homography"):
        H = np.array(result["homography"]["H"], dtype=np.float32)

        fx, fy = project_to_field(
            H,
            float(ball_x),
            float(ball_y),
        )

        if result.get("ball") is None:
            result["ball"] = {}

        result["ball"]["x"] = ball_x
        result["ball"]["y"] = ball_y
        result["ball"]["field_x_m"] = fx
        result["ball"]["field_y_m"] = fy

    return result