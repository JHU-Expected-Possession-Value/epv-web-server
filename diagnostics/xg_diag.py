#!/usr/bin/env python3
"""
xG diagnostics: permutation importance + SHAP.

Outputs:
  - diagnostics/xg_permutation_importance.csv
  - diagnostics/xg_permutation_importance.png
  - diagnostics/xg_shap_importance.csv
  - diagnostics/xg_shap_summary.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIAG_DIR = ROOT / "diagnostics"
MPL_CACHE = DIAG_DIR / ".mplcache"
TMP_DIR = DIAG_DIR / ".tmp"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
os.environ.setdefault("TMPDIR", str(TMP_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from xg_model import DampedXGModel

# Import after MPL env is set to avoid matplotlib temp errors
from training import train_xg_model_improved as txg


def _resolve_data_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    for candidate in [ROOT / "more_data", ROOT / "skillcorner_download"]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No data directory found (expected more_data/ or skillcorner_download/)")


def _load_model(model_path: Path):
    with open(model_path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        return obj["model"], obj.get("feature_cols")
    return obj, None


def _default_feature_cols() -> list[str]:
    return [
        "distance_to_goal",
        "angle_to_goal",
        "penalty_area",
        "trajectory_angle",
        "distance_covered",
        "speed_avg",
        "defenders_in_triangle",
        "defenders_within_3m",
        "inside_defensive_shape",
        "last_defensive_line_x",
        "last_defensive_line_height",
        "pitch_control",
        "player_finishing_skill",
    ]


def _sample_rows(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if n is None or n >= len(df):
        return df
    return df.sample(n=n, random_state=seed)


def _plot_importance(df: pd.DataFrame, value_col: str, out_path: Path, title: str):
    plot_df = df.sort_values(value_col, ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(plot_df["feature"], plot_df[value_col], color="#4c78a8")
    ax.set_title(title)
    ax.set_xlabel(value_col)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "models" / "xg_model_improved.pkl"))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--max-matches", type=int, default=30)
    parser.add_argument("--perm-samples", type=int, default=2000)
    parser.add_argument("--perm-repeats", type=int, default=10)
    parser.add_argument("--shap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = _resolve_data_dir(args.data_dir)
    model_path = Path(args.model)

    model, feature_cols = _load_model(model_path)
    if not feature_cols:
        feature_cols = _default_feature_cols()

    print(f"Loading shot features from {data_dir} (max_matches={args.max_matches})...")
    shots_df = txg.extract_shot_features(data_dir, max_matches=args.max_matches)

    for col in feature_cols:
        if col not in shots_df.columns:
            shots_df[col] = 0.0

    X = shots_df[feature_cols].fillna(0).values
    y = shots_df["goal"].values

    # Permutation importance
    perm_df = shots_df[feature_cols].fillna(0)
    perm_sample = _sample_rows(perm_df, args.perm_samples, args.seed)
    perm_y = _sample_rows(shots_df[["goal"]], args.perm_samples, args.seed)["goal"].values

    print(f"Running permutation importance on {len(perm_sample)} shots...")
    perm = permutation_importance(
        model,
        perm_sample.values,
        perm_y,
        scoring="roc_auc",
        n_repeats=args.perm_repeats,
        random_state=args.seed,
        n_jobs=-1,
    )
    perm_importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    perm_csv = DIAG_DIR / "xg_permutation_importance.csv"
    perm_png = DIAG_DIR / "xg_permutation_importance.png"
    perm_importance.to_csv(perm_csv, index=False)
    _plot_importance(perm_importance, "importance_mean", perm_png, "Permutation Importance (ROC AUC)")

    # SHAP
    try:
        import shap
    except ImportError as exc:
        raise SystemExit("shap is not installed. Install with: pip install shap") from exc

    raw_model = model.model if isinstance(model, DampedXGModel) else model
    shap_sample = _sample_rows(perm_df, args.shap_samples, args.seed)

    print(f"Running SHAP on {len(shap_sample)} shots...")
    explainer = shap.TreeExplainer(raw_model)
    shap_values = explainer.shap_values(shap_sample.values)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_vals = shap_values
    if getattr(shap_vals, "ndim", 2) == 3 and shap_vals.shape[-1] > 1:
        shap_vals = shap_vals[:, :, 1]

    shap_mean = np.abs(shap_vals).mean(axis=0)
    shap_importance = pd.DataFrame(
        {"feature": feature_cols, "mean_abs_shap": shap_mean}
    ).sort_values("mean_abs_shap", ascending=False)

    shap_csv = DIAG_DIR / "xg_shap_importance.csv"
    shap_png = DIAG_DIR / "xg_shap_summary.png"
    shap_importance.to_csv(shap_csv, index=False)

    shap.summary_plot(
        shap_vals,
        shap_sample.values,
        feature_names=feature_cols,
        plot_type="bar",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(shap_png, dpi=150)
    plt.close()

    print("\nTop permutation features:")
    print(perm_importance.head(5).to_string(index=False))
    print("\nBottom permutation features:")
    print(perm_importance.tail(5).to_string(index=False))

    print("\nTop SHAP features:")
    print(shap_importance.head(5).to_string(index=False))
    print("\nBottom SHAP features:")
    print(shap_importance.tail(5).to_string(index=False))

    print(f"\nSaved: {perm_csv}")
    print(f"Saved: {perm_png}")
    print(f"Saved: {shap_csv}")
    print(f"Saved: {shap_png}")


if __name__ == "__main__":
    main()
