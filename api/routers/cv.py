"""CV API router (thin) exposing backend-only analysis endpoints.

Behavior matches `CV/newapp.py` processing (see `api.services.cv_service` for the mapping).

How the CV service is called:
- `POST /api/cv/analyze-image` (multipart form)
- Required: `image` (file) — PNG/JPEG decoded to RGB.
- Optional: `roster_csv` — same roster format as Streamlit (`team`, `Team color`).
- Optional: `feature_type` ∈ {Center Circle, Penalty Box, Sideline} and `feature_points`
  as a JSON string of four `[[x,y],...]` pairs in **the analyzed image coordinate system**
  (after backend resize to max width 800; use `image_width` / `image_height` in the response).

What output shape the frontend should expect:
- `image_width`, `image_height` — use these to align overlays with the uploaded image
  (display at this size so bbox pixels match).
- `players`, `ball`, `homography` — see `cv_service.analyze_image` docstring; optional
  `field_x_m` / `field_y_m` on players and ball when homography is returned.

Project goals reflected here:
- Backend-only CV processing (no Streamlit flow)
- Stable JSON output for the web UI to render
"""

from __future__ import annotations

import io
from typing import Any, Literal, Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from api.services import cv_service

router = APIRouter()


@router.post("/analyze-image")
async def analyze_image(
    image: UploadFile = File(..., description="RGB image (png/jpg) to analyze"),
    roster_csv: Optional[UploadFile] = File(None, description="Optional roster CSV with team colors"),
    feature_type: Optional[Literal["Center Circle", "Penalty Box", "Sideline"]] = Form(
        None,
        description="Optional: feature template name for homography",
    ),
    feature_points: Optional[str] = Form(
        None,
        description="Optional: JSON string of 4 points: [[x,y],[x,y],[x,y],[x,y]]",
    ),
) -> dict[str, Any]:
    """Run CV analysis on a single image and return JSON results."""
    try:
        image_bytes = await image.read()
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # type: ignore[name-defined]
        rgb = np.array(pil)
    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": "Invalid image", "message": str(e)})

    roster_bytes = None
    if roster_csv is not None:
        try:
            roster_bytes = await roster_csv.read()
        except Exception as e:
            raise HTTPException(status_code=400, detail={"error": "Invalid roster_csv", "message": str(e)})

    try:
        pts = cv_service.parse_feature_points_json(feature_points)
        result = cv_service.analyze_image(
            rgb,
            roster_csv_bytes=roster_bytes,
            feature_type=feature_type,  # type: ignore[arg-type]
            feature_points=pts,
        )
        return result
    except ModuleNotFoundError as e:
        # If opencv/ultralytics aren't installed in the backend environment.
        raise HTTPException(
            status_code=503,
            detail={
                "error": "CV dependencies missing",
                "message": str(e),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "CV analysis failed", "message": str(e)})

