import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from shapely.geometry import Polygon, LineString, MultiLineString
from shapely.affinity import rotate

# ================== CONFIG ==================
INPUT_CSV = "concave_hull_coords_closed.csv"
OUT_DIR = "Coverage-Path"
TRACK_SPACING_M = 1.0           # 1 m between adjacent passes
ORIENTATION = "auto"            # "auto" or a numeric angle in degrees (0 = tracks along +x)
LINE_MARGIN = 5.0               # extend sweep lines beyond bbox for clean intersections

OUT_WAYPOINTS_CSV = os.path.join(OUT_DIR, "coverage_waypoints_1m.csv")
OUT_PREVIEW_PNG   = os.path.join(OUT_DIR, "coverage_path_preview.png")
OUT_SIM_MP4       = os.path.join(OUT_DIR, "coverage_path_simulation.mp4")
OUT_SIM_GIF       = os.path.join(OUT_DIR, "coverage_path_simulation.gif")

# Animation controls
DRONE_SPEED_MPS = 5.0           # simulated speed along segments (m/s)
FPS = 30                        # frames per second
PAUSE_FRAMES_TURN = 10          # brief pause at ends of rows (frames)
TRAIL_SECONDS = 6               # length of visible trail behind drone (s)
# ============================================

def read_polygon(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["x", "y"]).reset_index(drop=True)
    if len(df) >= 2 and np.isclose(df.loc[0,"x"], df.loc[len(df)-1,"x"]) and np.isclose(df.loc[0,"y"], df.loc[len(df)-1,"y"]):
        df = df.iloc[:-1].reset_index(drop=True)
    poly = Polygon(df[["x","y"]].to_numpy())
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly, df

def auto_orientation_deg(poly: Polygon) -> float:
    mrr = poly.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)[:4]
    # choose the longest edge of the MBR
    edges = [ (coords[i], coords[(i+1)%4]) for i in range(4) ]
    (a,b) = max(edges, key=lambda ab: math.hypot(ab[1][0]-ab[0][0], ab[1][1]-ab[0][1]))
    ang = math.degrees(math.atan2(b[1]-a[1], b[0]-a[0]))
    return ang

def intersect_horizontal_lines(rot_poly: Polygon, spacing=1.0, margin=5.0):
    minx, miny, maxx, maxy = rot_poly.bounds
    y0 = math.floor((miny - 1e-9) / spacing) * spacing
    ys = np.arange(y0, maxy + spacing*0.999, spacing)

    rows = []
    for row_idx, y in enumerate(ys):
        line = LineString([(minx - margin, y), (maxx + margin, y)])
        inter = rot_poly.intersection(line)
        if inter.is_empty:
            continue

        segments = []
        if isinstance(inter, LineString):
            segments = [inter]
        elif isinstance(inter, MultiLineString):
            segments = list(inter.geoms)

        # sort left-to-right
        segments.sort(key=lambda s: (s.bounds[0], s.bounds[1], s.bounds[2], s.bounds[3]))

        # convert segments to traversal points for this row
        row_pts = []
        for s in segments:
            xs, ys_ = s.xy
            start = (xs[0], ys_[0])
            end   = (xs[-1], ys_[-1])
            row_pts.append((start, end))

        # boustrophedon order: alternate direction per row
        forward = (row_idx % 2 == 0)
        ordered_pts = []
        for (start, end) in row_pts:
            a, b = (start, end) if forward else (end, start)
            ordered_pts.extend([a, b])

        if ordered_pts:
            rows.append((y, ordered_pts))
    return rows

def rotate_points(points, angle_deg, origin=(0,0)):
    ox, oy = origin
    c = math.cos(math.radians(angle_deg))
    s = math.sin(math.radians(angle_deg))
    out = []
    for x, y in points:
        xr = ox + (x-ox)*c - (y-oy)*s
        yr = oy + (x-ox)*s + (y-oy)*c
        out.append((xr, yr))
    return out

def flatten_rows_to_waypoints(rows):
    waypoints = []
    last = None
    row_break_indices = []  # indices where a row ends (to pause in animation)
    for _, pts in rows:
        for p in pts:
            if last is None or (abs(p[0]-last[0]) > 1e-9 or abs(p[1]-last[1]) > 1e-9):
                waypoints.append(p)
                last = p
        row_break_indices.append(len(waypoints)-1)
    return waypoints, row_break_indices

def cumulative_lengths(pts):
    L = [0.0]
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i-1][0]
        dy = pts[i][1] - pts[i-1][1]
        L.append(L[-1] + math.hypot(dx, dy))
    return np.array(L)

def resample_polyline(pts, every_m=0.2):
    """Optional: densify for smoother animation; keeps exact endpoints."""
    if len(pts) < 2:
        return pts
    L = cumulative_lengths(pts)
    total = L[-1]
    if total == 0:
        return pts
    new_s = np.arange(0, total, every_m)
    new_s = np.append(new_s, total)

    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])

    x_new = np.interp(new_s, L, xs)
    y_new = np.interp(new_s, L, ys)
    return list(zip(x_new, y_new))

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Build polygon and auto orientation
    poly, df = read_polygon(INPUT_CSV)
    angle = auto_orientation_deg(poly) if ORIENTATION == "auto" else float(ORIENTATION)

    # 2) Rotate polygon to build horizontal sweeps
    rot_poly = rotate(poly, -angle, origin='center', use_radians=False)

    # 3) Intersect with horizontal lines in rotated frame
    rows_rot = intersect_horizontal_lines(rot_poly, spacing=TRACK_SPACING_M, margin=LINE_MARGIN)

    # 4) Flatten rows -> waypoints (rotated frame)
    waypoints_rot, row_breaks_rot = flatten_rows_to_waypoints(rows_rot)

    # 5) Rotate waypoints back
    origin = (poly.centroid.x, poly.centroid.y)
    waypoints = rotate_points(waypoints_rot, angle, origin=origin)

    # 6) Save waypoints
    pd.DataFrame(waypoints, columns=["x", "y"]).to_csv(OUT_WAYPOINTS_CSV, index=False)
    print(f"Saved {len(waypoints)} waypoints → {OUT_WAYPOINTS_CSV}")

    # 7) Static preview
    xs, ys = df["x"].to_numpy(), df["y"].to_numpy()
    wpx, wpy = zip(*waypoints) if waypoints else ([], [])
    plt.figure(figsize=(11,9))
    plt.plot(xs, ys, '-', label="Boundary", linewidth=1.5)
    plt.scatter(wpx, wpy, s=8, label=f"Coverage waypoints ({TRACK_SPACING_M} m)")
    plt.axis('equal'); plt.grid(True); plt.legend()
    plt.title(f"Boustrophedon Coverage Path (spacing={TRACK_SPACING_M} m, angle={angle:.1f}°)")
    plt.savefig(OUT_PREVIEW_PNG, dpi=300)
    plt.close()
    print(f"Preview saved → {OUT_PREVIEW_PNG}")

    # 8) Animation setup (simulate motion)
    # Densify a little for smoother motion
    dense_pts = resample_polyline(waypoints, every_m=max(DRONE_SPEED_MPS/FPS, 0.2))
    if len(dense_pts) < 2:
        print("Not enough waypoints to animate.")
        return

    # Pre-compute pause frames at row ends (map break indices to dense index)
    # We’ll approximate by finding nearest dense index to each break
    dense_arr = np.array(dense_pts)
    breaks_dense_idx = []
    for bi in row_breaks_rot:
        target = np.array(waypoints[bi])
        d2 = np.sum((dense_arr - target)**2, axis=1)
        breaks_dense_idx.append(int(np.argmin(d2)))

    # Build frame sequence with pauses
    frames_idx = []
    for i in range(len(dense_pts)):
        frames_idx.append(i)
        if i in breaks_dense_idx:
            # add pause frames to emphasize turns
            for _ in range(PAUSE_FRAMES_TURN):
                frames_idx.append(i)

    # Figure and artists
    fig, ax = plt.subplots(figsize=(11,9))
    ax.plot(xs, ys, '-', color='#555555', linewidth=1.2, label="Boundary")
    ax.axis('equal'); ax.grid(True)
    ax.set_title("Drone Coverage Path Simulation")
    ax.set_xlabel("X"); ax.set_ylabel("Y")

    # Plot a faint track of all waypoints for context
    ax.plot(wpx, wpy, '--', color='#999999', linewidth=1.0, label="Planned Path")

    # Drone marker and trail
    drone_dot, = ax.plot([], [], 'o', color='red', markersize=6, label="Drone")
    trail_line, = ax.plot([], [], '-', color='red', linewidth=2, alpha=0.7, label="Trail")

    # Legend
    ax.legend(loc="best")

    # Trail length in frames
    trail_len_frames = int(TRAIL_SECONDS * FPS)

    def init():
        drone_dot.set_data([], [])
        trail_line.set_data([], [])
        return drone_dot, trail_line

    def update(frame):
        idx = frames_idx[frame]
        x, y = dense_pts[idx]

        # Set drone position
        drone_dot.set_data([x], [y])

        # Trail from max(0, idx-trail_len_frames) .. idx
        a = max(0, idx - trail_len_frames)
        xs_t = [p[0] for p in dense_pts[a:idx+1]]
        ys_t = [p[1] for p in dense_pts[a:idx+1]]
        trail_line.set_data(xs_t, ys_t)
        return drone_dot, trail_line

    anim = FuncAnimation(
        fig, update, init_func=init,
        frames=len(frames_idx), interval=1000/FPS, blit=True
    )

    # Try saving MP4 (needs ffmpeg). If not available, save GIF.
    try:
        anim.save(OUT_SIM_MP4, fps=FPS)
        print(f"Simulation video saved → {OUT_SIM_MP4}")
    except Exception as e:
        print(f"MP4 save failed ({e}). Saving GIF instead…")
        anim.save(OUT_SIM_GIF, writer=PillowWriter(fps=FPS))
        print(f"Simulation GIF saved → {OUT_SIM_GIF}")

    plt.close(fig)

if __name__ == "__main__":
    main()
