"""
Generate EPV Heatmap - showing expected possession value across the entire pitch.

This script creates a color-coded heatmap showing the EPV (expected goals)
for different positions on the field given a specific game state/frame.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import pickle
import pandas as pd
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from epv_calculator import EPVCalculator
from pitch_control import load_match_meta_robust, load_tracking_jsonl

# Paths
BASE = Path(__file__).parent.parent
RESULTS_DIR = BASE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Model paths
XG_MODEL_PATH = BASE / "models" / "xg_model_improved.pkl"
PASSING_MODEL_PATH = BASE / "models" / "passing_model_improved.pkl"
DRIBBLING_MODEL_PATH = BASE / "models" / "dribbling_model_proper_split.pkl"
XG_SKILLS_PATH = BASE / "data" / "player_id_to_finishing_skill.csv"
PASSING_SKILLS_PATH = BASE / "data" / "player_id_to_passing_skill.csv"
DRIBBLING_SKILLS_PATH = BASE / "data" / "player_id_to_skill.csv"


def draw_pitch(ax, pitch_length=105, pitch_width=68, center_based=True):
    """Draw a soccer pitch on the given axes.

    Args:
        ax: Matplotlib axes
        pitch_length: Length of pitch in meters
        pitch_width: Width of pitch in meters
        center_based: If True, use center-based coordinates (-52.5 to 52.5)
                     If False, use absolute coordinates (0 to 105)
    """
    if center_based:
        # Use center-based coordinates
        half_length = pitch_length / 2
        half_width = pitch_width / 2

        # Pitch outline
        ax.plot([-half_length, half_length], [-half_width, -half_width], color='white', linewidth=2)
        ax.plot([-half_length, half_length], [half_width, half_width], color='white', linewidth=2)
        ax.plot([-half_length, -half_length], [-half_width, half_width], color='white', linewidth=2)
        ax.plot([half_length, half_length], [-half_width, half_width], color='white', linewidth=2)

        # Centre line
        ax.plot([0, 0], [-half_width, half_width], color='white', linewidth=2)

        # Centre circle
        centre_circle = plt.Circle((0, 0), 9.15, color='white', fill=False, linewidth=2)
        centre_spot = plt.Circle((0, 0), 0.5, color='white')
        ax.add_patch(centre_circle)
        ax.add_patch(centre_spot)

        # Left penalty area (defending left goal at x = -52.5)
        ax.plot([-half_length, -half_length + 16.5], [-20.15, -20.15], color='white', linewidth=2)
        ax.plot([-half_length, -half_length + 16.5], [20.15, 20.15], color='white', linewidth=2)
        ax.plot([-half_length + 16.5, -half_length + 16.5], [-20.15, 20.15], color='white', linewidth=2)

        # Right penalty area (defending right goal at x = 52.5)
        ax.plot([half_length, half_length - 16.5], [-20.15, -20.15], color='white', linewidth=2)
        ax.plot([half_length, half_length - 16.5], [20.15, 20.15], color='white', linewidth=2)
        ax.plot([half_length - 16.5, half_length - 16.5], [-20.15, 20.15], color='white', linewidth=2)

        # Left goal area
        ax.plot([-half_length, -half_length + 5.5], [-9.15, -9.15], color='white', linewidth=2)
        ax.plot([-half_length, -half_length + 5.5], [9.15, 9.15], color='white', linewidth=2)
        ax.plot([-half_length + 5.5, -half_length + 5.5], [-9.15, 9.15], color='white', linewidth=2)

        # Right goal area
        ax.plot([half_length, half_length - 5.5], [-9.15, -9.15], color='white', linewidth=2)
        ax.plot([half_length, half_length - 5.5], [9.15, 9.15], color='white', linewidth=2)
        ax.plot([half_length - 5.5, half_length - 5.5], [-9.15, 9.15], color='white', linewidth=2)

        # Penalty spots
        ax.scatter([-half_length + 11], [0], color='white', s=20, zorder=3)
        ax.scatter([half_length - 11], [0], color='white', s=20, zorder=3)

        # Goals
        ax.plot([-half_length, -half_length], [-3.66, 3.66], color='white', linewidth=4)
        ax.plot([half_length, half_length], [-3.66, 3.66], color='white', linewidth=4)

        ax.set_xlim(-half_length - 5, half_length + 5)
        ax.set_ylim(-half_width - 5, half_width + 5)
    else:
        # Use absolute coordinates (0 to pitch_length)
        # Pitch outline & centre line
        ax.plot([0, 0], [-pitch_width/2, pitch_width/2], color='white', linewidth=2)
        ax.plot([pitch_length, pitch_length], [-pitch_width/2, pitch_width/2], color='white', linewidth=2)
        ax.plot([0, pitch_length], [-pitch_width/2, -pitch_width/2], color='white', linewidth=2)
        ax.plot([0, pitch_length], [pitch_width/2, pitch_width/2], color='white', linewidth=2)
        ax.plot([pitch_length/2, pitch_length/2], [-pitch_width/2, pitch_width/2], color='white', linewidth=2)

        # Centre circle
        centre_circle = plt.Circle((pitch_length/2, 0), 9.15, color='white', fill=False, linewidth=2)
        centre_spot = plt.Circle((pitch_length/2, 0), 0.5, color='white')
        ax.add_patch(centre_circle)
        ax.add_patch(centre_spot)

        # Penalty areas
        # Left penalty area
        ax.plot([0, 16.5], [-20.15, -20.15], color='white', linewidth=2)
        ax.plot([0, 16.5], [20.15, 20.15], color='white', linewidth=2)
        ax.plot([16.5, 16.5], [-20.15, 20.15], color='white', linewidth=2)

        # Right penalty area
        ax.plot([pitch_length, pitch_length-16.5], [-20.15, -20.15], color='white', linewidth=2)
        ax.plot([pitch_length, pitch_length-16.5], [20.15, 20.15], color='white', linewidth=2)
        ax.plot([pitch_length-16.5, pitch_length-16.5], [-20.15, 20.15], color='white', linewidth=2)

        # Goal areas
        # Left goal area
        ax.plot([0, 5.5], [-9.15, -9.15], color='white', linewidth=2)
        ax.plot([0, 5.5], [9.15, 9.15], color='white', linewidth=2)
        ax.plot([5.5, 5.5], [-9.15, 9.15], color='white', linewidth=2)

        # Right goal area
        ax.plot([pitch_length, pitch_length-5.5], [-9.15, -9.15], color='white', linewidth=2)
        ax.plot([pitch_length, pitch_length-5.5], [9.15, 9.15], color='white', linewidth=2)
        ax.plot([pitch_length-5.5, pitch_length-5.5], [-9.15, 9.15], color='white', linewidth=2)

        # Penalty spots
        ax.scatter([11], [0], color='white', s=20, zorder=3)
        ax.scatter([pitch_length-11], [0], color='white', s=20, zorder=3)

        # Goals
        ax.plot([0, 0], [-3.66, 3.66], color='white', linewidth=4)
        ax.plot([pitch_length, pitch_length], [-3.66, 3.66], color='white', linewidth=4)

        ax.set_xlim(-5, pitch_length + 5)
        ax.set_ylim(-pitch_width/2 - 5, pitch_width/2 + 5)

    ax.set_aspect('equal')
    ax.axis('off')


def create_custom_colormap():
    """Create a custom colormap from low (blue) to high (red) EPV."""
    colors = ['#0d47a1', '#1976d2', '#42a5f5', '#90caf9',
              '#fff59d', '#ffee58', '#ffc107', '#ff9800',
              '#ff5722', '#d32f2f']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('epv', colors, N=n_bins)
    return cmap


def generate_epv_heatmap_simple(
    output_path,
    pitch_length=105,
    pitch_width=68,
    grid_resolution=2.0,
    attacking_direction='right'
):
    """
    Generate a simple EPV heatmap using basic xG values (no tracking data required).

    Args:
        output_path: Where to save the heatmap
        pitch_length: Pitch length in meters
        pitch_width: Pitch width in meters
        grid_resolution: Grid cell size in meters
        attacking_direction: 'left' or 'right' (which goal are we attacking)
    """
    print("Generating simple EPV heatmap...")

    # Load xG model
    print("  Loading xG model...")
    with open(XG_MODEL_PATH, 'rb') as f:
        xg_obj = pickle.load(f)
        xg_model = xg_obj['model']
        xg_feature_cols = xg_obj['feature_cols']

    # Load finishing skills (use default if player not found)
    print("  Loading player skills...")
    finishing_skills_df = pd.read_csv(XG_SKILLS_PATH)
    avg_finishing_skill = finishing_skills_df['player_finishing_skill'].mean()

    # Determine goal position (using center-based coordinates)
    if attacking_direction == 'right':
        goal_x = pitch_length / 2  # Right goal at +52.5
        goal_y = 0
    else:
        goal_x = -pitch_length / 2  # Left goal at -52.5
        goal_y = 0

    # Create grid (center-based coordinates)
    print("  Creating grid...")
    x_points = np.arange(-pitch_length/2, pitch_length/2 + grid_resolution, grid_resolution)
    y_points = np.arange(-pitch_width/2, pitch_width/2 + grid_resolution, grid_resolution)

    epv_grid = np.zeros((len(y_points), len(x_points)))

    print(f"  Computing EPV for {len(x_points)} x {len(y_points)} = {len(x_points) * len(y_points)} grid points...")

    # Compute EPV for each grid point
    for i, y in enumerate(y_points):
        for j, x in enumerate(x_points):
            # Calculate xG features
            distance_to_goal = np.sqrt((x - goal_x)**2 + (y - goal_y)**2)

            # Angle to goal (using goal posts)
            goal_post_1 = goal_y - 3.66
            goal_post_2 = goal_y + 3.66

            angle1 = np.arctan2(goal_post_1 - y, goal_x - x)
            angle2 = np.arctan2(goal_post_2 - y, goal_x - x)
            angle_to_goal = abs(np.degrees(angle2 - angle1))

            # Check if in penalty area (using center-based coordinates)
            if attacking_direction == 'right':
                penalty_area = int(x > (pitch_length/2 - 16.5) and abs(y) < 20.15)
            else:
                penalty_area = int(x < (-pitch_length/2 + 16.5) and abs(y) < 20.15)

            # Build feature dict
            feature_dict = {
                'distance_to_goal': distance_to_goal,
                'angle_to_goal': angle_to_goal,
                'penalty_area': penalty_area,
                'trajectory_angle': 0.0,
                'distance_covered': 0.0,
                'speed_avg': 0.0,
                'player_finishing_skill': avg_finishing_skill,
                'defenders_in_triangle': 0,
                'defenders_within_3m': 0,
                'inside_defensive_shape': 0,
                'last_defensive_line_x': pitch_length / 2,
                'last_defensive_line_height': 20.0,
            }

            # Build feature vector
            features = np.array([[feature_dict.get(col, 0.0) for col in xg_feature_cols]])

            # Predict xG
            try:
                xg = xg_model.predict_proba(features)[0, 1]

                # Apply distance penalty for very long shots
                if distance_to_goal > 35:
                    xg *= np.exp(-(distance_to_goal - 35) / 10.0)

                # Apply angle penalty for very tight angles
                if angle_to_goal < 10:
                    xg *= (angle_to_goal / 10.0)

                epv_grid[i, j] = xg
            except:
                epv_grid[i, j] = 0.0

    # Create figure
    print("  Creating visualization...")
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor('#1a472a')
    ax.set_facecolor('#1a472a')

    # Plot heatmap
    cmap = create_custom_colormap()

    # Plot using contourf for smooth gradients (x_points and y_points already center-based)
    X, Y = np.meshgrid(x_points, y_points)

    contour = ax.contourf(X, Y, epv_grid, levels=20,
                          cmap=cmap, alpha=0.7, vmin=0, vmax=0.5)

    # Draw pitch on top (using center-based coordinates)
    draw_pitch(ax, pitch_length, pitch_width, center_based=True)

    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Expected Goals (EPV)', rotation=270, labelpad=20,
                   color='white', fontsize=12, weight='bold')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    # Center coordinates (0, 0) at center of pitch
    ax.set_xlim(-pitch_length/2 - 5, pitch_length/2 + 5)
    ax.set_ylim(-pitch_width/2 - 5, pitch_width/2 + 5)

    # Add title
    direction_text = "→" if attacking_direction == 'right' else "←"
    ax.text(0, pitch_width/2 + 8,
            f'Expected Possession Value (EPV) Heatmap {direction_text}',
            ha='center', va='bottom', color='white',
            fontsize=16, weight='bold')

    # Add subtitle
    ax.text(0, -pitch_width/2 - 8,
            'Higher values (red) indicate better scoring positions | Lower values (blue) indicate lower scoring probability',
            ha='center', va='top', color='white',
            fontsize=10, style='italic', alpha=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='#1a472a', bbox_inches='tight')
    print(f"✅ Heatmap saved to: {output_path}")
    plt.close()


def generate_epv_heatmap_with_tracking(
    match_json_path,
    tracking_path,
    events_csv_path,
    frame_number,
    player_id,
    team_id,
    output_path,
    pitch_length=105,
    pitch_width=68,
    grid_resolution=3.0
):
    """
    Generate EPV heatmap using actual tracking data and EPV calculator.

    Args:
        match_json_path: Path to match metadata JSON
        tracking_path: Path to tracking JSONL file
        events_csv_path: Path to events CSV
        frame_number: Frame to analyze
        player_id: Player with possession
        team_id: Team with possession
        output_path: Where to save the heatmap
        pitch_length: Pitch length in meters
        pitch_width: Pitch width in meters
        grid_resolution: Grid cell size in meters
    """
    print("Generating EPV heatmap with tracking data...")

    # Initialize EPV calculator
    print("  Initializing EPV calculator...")
    epv_calc = EPVCalculator(
        xg_model_path=XG_MODEL_PATH,
        passing_model_path=PASSING_MODEL_PATH,
        dribbling_model_path=DRIBBLING_MODEL_PATH,
        xg_skills_path=XG_SKILLS_PATH,
        passing_skills_path=PASSING_SKILLS_PATH,
        dribbling_skills_path=DRIBBLING_SKILLS_PATH,
        pitch_length=pitch_length,
        pitch_width=pitch_width
    )

    # Load match context
    print("  Loading match context...")
    events_df = pd.read_csv(events_csv_path)
    epv_calc.set_match_context(
        match_json_path=Path(match_json_path),
        tracking_path=Path(tracking_path),
        events_df=events_df
    )

    # Load tracking data into dictionary
    print("  Loading tracking frames...")
    frames = load_tracking_jsonl(Path(tracking_path))
    tracking_dict = {f.get('frame_number', f.get('frame', i)): f for i, f in enumerate(frames)}

    # Create grid (using center-based coordinates: -52.5 to 52.5 for x)
    print("  Creating grid...")
    x_points = np.arange(-pitch_length/2, pitch_length/2 + grid_resolution, grid_resolution)
    y_points = np.arange(-pitch_width/2, pitch_width/2 + grid_resolution, grid_resolution)

    epv_grid = np.zeros((len(y_points), len(x_points)))

    print(f"  Computing EPV for {len(x_points)} x {len(y_points)} = {len(x_points) * len(y_points)} grid points...")
    print(f"  (This may take a few minutes...)")

    # Compute EPV for each grid point
    total_points = len(x_points) * len(y_points)
    computed = 0

    for i, y in enumerate(y_points):
        for j, x in enumerate(x_points):
            try:
                epv = epv_calc.get_epv(
                    x=x,
                    y=y,
                    frame=frame_number,
                    player_id=player_id,
                    team_id=team_id,
                    tracking_dict=tracking_dict,
                    depth=0,
                    max_depth=1  # Reduced depth for speed (immediate actions only)
                )
                epv_grid[i, j] = epv
            except Exception as e:
                epv_grid[i, j] = 0.0

            computed += 1
            if computed % 100 == 0:
                print(f"    Progress: {computed}/{total_points} ({100*computed/total_points:.1f}%)")

    # Create figure
    print("  Creating visualization...")
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor('#1a472a')
    ax.set_facecolor('#1a472a')

    # Plot heatmap
    cmap = create_custom_colormap()

    # Create meshgrid
    X, Y = np.meshgrid(x_points, y_points)

    # Plot using contourf for smooth gradients
    max_epv = np.percentile(epv_grid, 95)  # Use 95th percentile to avoid outliers
    contour = ax.contourf(X, Y, epv_grid, levels=20,
                          cmap=cmap, alpha=0.7, vmin=0, vmax=max(max_epv, 0.3))

    # Draw pitch on top (using center-based coordinates)
    draw_pitch(ax, pitch_length, pitch_width, center_based=True)

    # Plot player positions from tracking data
    frame_data = tracking_dict.get(frame_number, {})
    if 'player_data' in frame_data:
        for player in frame_data['player_data']:
            if player.get('is_detected', False):
                px = player['x'] - pitch_length/2  # Convert to center-based
                py = player['y']

                # Different colors for different teams
                if player['player_id'] in epv_calc.current_team_roster.get(team_id, set()):
                    color = 'cyan'
                    marker = 'o'
                else:
                    color = 'red'
                    marker = 's'

                ax.scatter([px], [py], color=color, s=100,
                          edgecolors='white', linewidths=1.5,
                          marker=marker, zorder=10, alpha=0.8)

    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Expected Goals (EPV)', rotation=270, labelpad=20,
                   color='white', fontsize=12, weight='bold')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    # Set limits
    ax.set_xlim(-pitch_length/2 - 5, pitch_length/2 + 5)
    ax.set_ylim(-pitch_width/2 - 5, pitch_width/2 + 5)

    # Add title
    ax.text(0, pitch_width/2 + 8,
            f'Expected Possession Value (EPV) Heatmap - Frame {frame_number}',
            ha='center', va='bottom', color='white',
            fontsize=16, weight='bold')

    # Add subtitle
    ax.text(0, -pitch_width/2 - 8,
            f'Team {team_id} attacking | Player positions shown: Attacking team (cyan circles), Defending team (red squares)',
            ha='center', va='top', color='white',
            fontsize=9, style='italic', alpha=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='#1a472a', bbox_inches='tight')
    print(f"✅ Heatmap saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    # Generate simple EPV heatmap (no tracking data needed)
    print("\n" + "="*60)
    print("GENERATING SIMPLE EPV HEATMAP")
    print("="*60 + "\n")

    generate_epv_heatmap_simple(
        output_path=RESULTS_DIR / "epv_heatmap_simple.png",
        attacking_direction='right'
    )

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print("\nTo generate a heatmap with actual tracking data, use:")
    print("  generate_epv_heatmap_with_tracking(")
    print("    match_json_path='path/to/match.json',")
    print("    tracking_path='path/to/tracking.jsonl',")
    print("    events_csv_path='path/to/events.csv',")
    print("    frame_number=1000,")
    print("    player_id=123,")
    print("    team_id=1,")
    print("    output_path=RESULTS_DIR / 'epv_heatmap_frame_1000.png'")
    print("  )")
