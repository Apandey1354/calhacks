import os
import re
import json
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Convex Hull ----------
def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def convex_hull_monotone_chain(points):
    pts = sorted(set(map(tuple, points)))
    if len(pts) <= 1:
        return pts

    lower = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]

# ---------- Helpers ----------
def _detect_xy_columns(df):
    if {'physical_x_m', 'physical_y_m'}.issubset(df.columns):
        return 'physical_x_m', 'physical_y_m'
    elif {'x', 'y'}.issubset(df.columns):
        return 'x', 'y'
    candidate_pairs = []
    for c in df.columns:
        m = re.match(r'^x_(\d+)$', c)
        if m:
            k = int(m.group(1))
            ycol = f'y_{k}'
            if ycol in df.columns:
                n = df[[c, ycol]].dropna().shape[0]
                candidate_pairs.append((n, c, ycol))
    if candidate_pairs:
        candidate_pairs.sort(reverse=True)
        _, xcol, ycol = candidate_pairs[0]
        return xcol, ycol
    raise ValueError("Could not detect coordinate columns.")

def _shoelace_area(poly):
    if len(poly) < 3:
        return 0.0
    x = np.array([p[0] for p in poly])
    y = np.array([p[1] for p in poly])
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def _perimeter(poly):
    if len(poly) < 2:
        return 0.0
    d = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        d += math.hypot(x2 - x1, y2 - y1)
    return d

# ---------- Main ----------
def build_and_save_convex_hull(
    input_csv,
    out_dir="Processed_images/ConvexHull",
    save_plot=True
):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(input_csv)
    xcol, ycol = _detect_xy_columns(df)

    pts_df = df[[xcol, ycol]].rename(columns={xcol: 'x', ycol: 'y'}).dropna()
    pts_df = pts_df.drop_duplicates(subset=['x', 'y']).reset_index(drop=True)

    if len(pts_df) < 3:
        orig_path = os.path.join(out_dir, "original_points.csv")
        pts_df.to_csv(orig_path, index=False)
        meta = {
            "status": "too_few_points_for_polygon",
            "points": len(pts_df)
        }
        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved original points ({len(pts_df)}). Not enough to form polygon.")
        return

    points = pts_df[['x', 'y']].to_numpy()
    hull = convex_hull_monotone_chain(points)
    hull_df = pd.DataFrame(hull, columns=['x', 'y'])
    hull_closed_df = pd.concat([hull_df, hull_df.iloc[[0]]], ignore_index=True)

    orig_path = os.path.join(out_dir, "original_points.csv")
    hull_path = os.path.join(out_dir, "convex_hull_coords.csv")
    hull_closed_path = os.path.join(out_dir, "convex_hull_coords_closed.csv")

    pts_df.to_csv(orig_path, index=False)
    hull_df.to_csv(hull_path, index=False)
    hull_closed_df.to_csv(hull_closed_path, index=False)

    area = _shoelace_area(hull)
    perim = _perimeter(hull)

    meta = {
        "status": "ok",
        "input_csv": input_csv,
        "points_used": int(len(pts_df)),
        "hull_vertices": int(len(hull)),
        "hull_area": float(area),
        "hull_perimeter": float(perim),
        "outputs": {
            "original_points_csv": orig_path,
            "hull_coords_csv": hull_path,
            "hull_coords_closed_csv": hull_closed_path
        }
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Saved original points -> {orig_path}")
    print(f"✅ Saved hull (open) -> {hull_path}")
    print(f"✅ Saved hull (closed) -> {hull_closed_path}")
    print(f"ℹ️ Vertices: {len(hull)} | Area: {area:.3f} | Perimeter: {perim:.3f}")

    # ---------- Side-by-side Plot ----------
    if save_plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Left: Original points
        axes[0].scatter(pts_df['x'], pts_df['y'], s=6, alpha=0.7)
        axes[0].set_title("Original Points")
        axes[0].set_xlabel("X")
        axes[0].set_ylabel("Y")
        axes[0].grid(True, linestyle="--", alpha=0.3)
        axes[0].set_aspect("equal", adjustable="box")

        # Right: Points + Hull
        axes[1].scatter(pts_df['x'], pts_df['y'], s=6, alpha=0.5, label="Points")
        axes[1].plot(hull_closed_df['x'], hull_closed_df['y'], color="red", linewidth=2, label="Convex Hull")
        axes[1].set_title("Convex Hull Overlay")
        axes[1].set_xlabel("X")
        axes[1].set_ylabel("Y")
        axes[1].grid(True, linestyle="--", alpha=0.3)
        axes[1].set_aspect("equal", adjustable="box")
        axes[1].legend()

        plt.tight_layout()
        plot_path = os.path.join(out_dir, "convex_hull_side_by_side.png")
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.show()
        print(f"🖼️ Saved plot -> {plot_path}")

if __name__ == "__main__":
    build_and_save_convex_hull(
        input_csv="Processed_images/Reflected/cluster_1_reflected.csv",
        out_dir="Processed_images/Reflected/ConvexHull",
        save_plot=True
    )
