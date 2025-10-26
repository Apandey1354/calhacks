import os
import re
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Geometry helpers ----------------
def _dist(a, b):
    return math.hypot(b[0]-a[0], b[1]-a[1])

def _angle(prev, curr, nxt):
    # angle between vector (curr->prev) and (curr->nxt), returned in [0, 2pi)
    v1 = (prev[0]-curr[0], prev[1]-curr[1])
    v2 = (nxt[0]-curr[0], nxt[1]-curr[1])
    ang1 = math.atan2(v1[1], v1[0])
    ang2 = math.atan2(v2[1], v2[0])
    a = (ang2 - ang1) % (2*math.pi)
    return a

def _ccw(a, b, c):
    return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])

def _segments_intersect(a,b,c,d):
    # Proper intersection test (excludes touching at endpoints)
    if (a == c or a == d or b == c or b == d):
        return False
    return _ccw(a,c,d) != _ccw(b,c,d) and _ccw(a,b,c) != _ccw(a,b,d)

def _any_intersection(hull_points, cand):
    # Check if adding segment (hull[-1] -> cand) intersects any prior hull edge
    if len(hull_points) < 2:
        return False
    a = hull_points[-1]
    b = cand
    for i in range(len(hull_points)-2):
        c = hull_points[i]
        d = hull_points[i+1]
        if _segments_intersect(a,b,c,d):
            return True
    return False

# ---------------- Data detection ----------------
def _detect_points(df, merge_all_pairs=True):
    if {'physical_x_m', 'physical_y_m'}.issubset(df.columns):
        pts = df[['physical_x_m','physical_y_m']].dropna().to_numpy()
        return pts
    if {'x','y'}.issubset(df.columns):
        pts = df[['x','y']].dropna().to_numpy()
        return pts

    # Wide format x_i,y_i
    pairs = []
    for c in df.columns:
        m = re.match(r'^x_(\d+)$', c)
        if m:
            i = int(m.group(1))
            ycol = f'y_{i}'
            if ycol in df.columns:
                pair = df[[c, ycol]].dropna().to_numpy()
                if len(pair) > 0:
                    pairs.append(pair)

    if not pairs:
        raise ValueError("Could not detect coordinates. Expected ('physical_x_m','physical_y_m'), ('x','y'), or x_i/y_i pairs.")

    if merge_all_pairs:
        pts = np.vstack(pairs)
        return pts
    else:
        # Use densest pair
        pairs.sort(key=lambda a: -len(a))
        return pairs[0]

# ---------------- KNN / candidate selection ----------------
def _k_nearest(points, idx, k, used_mask=None):
    # brute-force distances
    anchor = points[idx]
    d2 = np.sum((points - anchor)**2, axis=1)
    order = np.argsort(d2)
    # exclude self, and optionally exclude used
    res = []
    for j in order:
        if j == idx:
            continue
        if used_mask is not None and used_mask[j]:
            continue
        res.append(j)
        if len(res) >= k:
            break
    return res

# ---------------- Concave Hull (Moreira & Santos heuristic) ----------------
def concave_hull(points, k_start=3, k_max=30):
    """
    points: (N,2) ndarray of unique points
    Returns a list of (x,y) of the concave hull in CCW order, not closed.
    Will try k from k_start..k_max to find a valid simple polygon.
    """
    pts_unique = np.unique(points, axis=0)
    n = len(pts_unique)
    if n < 3:
        return pts_unique.tolist()

    # start at point with lowest y (then lowest x)
    start_idx = np.lexsort((pts_unique[:,0], pts_unique[:,1]))[0]
    start = tuple(pts_unique[start_idx])

    for k in range(max(3, k_start), max(k_start, k_max)+1):
        # working arrays
        hull = [start]
        used = np.zeros(n, dtype=bool)
        used[start_idx] = True
        curr_idx = start_idx
        # set a fake previous point to define initial heading (point to the left of start)
        prev_pt = (start[0]-1.0, start[1])  # heading roughly +x at the start

        success = True
        for _ in range(2*n + 10):  # safety cap
            # nearest candidate pool (exclude already used to avoid self-crossing cycles)
            cand_ids = _k_nearest(pts_unique, curr_idx, k, used_mask=None)
            # order by rightmost turn (largest angle relative to prev->curr)
            curr_pt = tuple(pts_unique[curr_idx])
            cand_ids_sorted = sorted(
                cand_ids,
                key=lambda j: -_angle(prev_pt, curr_pt, tuple(pts_unique[j]))
            )

            found_next = False
            for j in cand_ids_sorted:
                cand_pt = tuple(pts_unique[j])
                # closing condition: if candidate is start and hull is long enough, try close
                if cand_pt == start and len(hull) >= 3:
                    # check closing segment intersection
                    if not _any_intersection(hull, cand_pt):
                        hull_closed = hull[:]  # not including start twice
                        # validate simple polygon closure
                        return hull_closed
                    else:
                        continue

                # skip if already in hull (prevents tight loops)
                if cand_pt in hull:
                    continue

                # intersection check
                if _any_intersection(hull, cand_pt):
                    continue

                # accept this candidate
                hull.append(cand_pt)
                prev_pt = curr_pt
                curr_idx = j
                used[j] = True
                found_next = True
                break

            if not found_next:
                success = False
                break

        if success and len(hull) >= 3:
            return hull  # fallback, though we should've returned on closure

        # If we couldn't form a simple polygon, try a larger k
        # print(f"[info] k={k} failed, increasing...")

    # As a last resort, return convex hull (monotone chain) to guarantee a polygon
    # --- Convex hull fallback ---
    def _cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    pts = sorted(map(tuple, pts_unique))
    if len(pts) <= 1:
        return pts
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return hull

# ---------------- Main wrapper ----------------
def build_and_save_concave_hull(
    input_csv,
    out_dir="Processed_images/Reflected/ConcaveHull",
    k_start=6,
    k_max=30,
    merge_all_pairs=True,
    save_plot=True
):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(input_csv)
    pts = _detect_points(df, merge_all_pairs=merge_all_pairs)

    # Clean + unique
    pts_df = pd.DataFrame(pts, columns=["x","y"]).dropna().drop_duplicates().reset_index(drop=True)
    pts_arr = pts_df[["x","y"]].to_numpy()

    if len(pts_arr) < 3:
        orig_path = os.path.join(out_dir, "original_points.csv")
        pts_df.to_csv(orig_path, index=False)
        meta = {"status":"too_few_points_for_polygon","points":int(len(pts_df))}
        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved original points ({len(pts_df)}). Not enough to form a polygon.")
        return

    hull = concave_hull(pts_arr, k_start=k_start, k_max=k_max)
    hull_df = pd.DataFrame(hull, columns=["x","y"])
    hull_closed_df = pd.concat([hull_df, hull_df.iloc[[0]]], ignore_index=True)

    # Stats
    def _shoelace_area(poly):
        if len(poly) < 3: return 0.0
        x = np.array([p[0] for p in poly])
        y = np.array([p[1] for p in poly])
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    def _perimeter(poly):
        if len(poly) < 2: return 0.0
        return sum(_dist(poly[i], poly[(i+1)%len(poly)]) for i in range(len(poly)))

    area = float(_shoelace_area(hull))
    perim = float(_perimeter(hull))

    # Save files
    orig_path = os.path.join(out_dir, "original_points.csv")
    hull_path = os.path.join(out_dir, "concave_hull_coords.csv")
    hull_closed_path = os.path.join(out_dir, "concave_hull_coords_closed.csv")
    pts_df.to_csv(orig_path, index=False)
    hull_df.to_csv(hull_path, index=False)
    hull_closed_df.to_csv(hull_closed_path, index=False)

    meta = {
        "status": "ok",
        "input_csv": input_csv,
        "points_used": int(len(pts_df)),
        "hull_vertices": int(len(hull_df)),
        "hull_area": area,
        "hull_perimeter": perim,
        "params": {"k_start": k_start, "k_max": k_max, "merge_all_pairs": merge_all_pairs},
        "outputs": {
            "original_points_csv": orig_path,
            "concave_hull_coords_csv": hull_path,
            "concave_hull_coords_closed_csv": hull_closed_path
        }
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Saved original points -> {orig_path}")
    print(f"✅ Saved concave hull (open) -> {hull_path}")
    print(f"✅ Saved concave hull (closed) -> {hull_closed_path}")
    print(f"ℹ️ k_start={k_start} k_max={k_max} | vertices={len(hull_df)} | area={area:.3f} | perimeter={perim:.3f}")

    # Side-by-side visualization
    if save_plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        # Original
        axes[0].scatter(pts_df["x"], pts_df["y"], s=6, alpha=0.7)
        axes[0].set_title("Original Points")
        axes[0].set_xlabel("X"); axes[0].set_ylabel("Y")
        axes[0].grid(True, linestyle="--", alpha=0.3)
        axes[0].set_aspect("equal", adjustable="box")
        # Overlay hull (RED line)
        axes[1].scatter(pts_df["x"], pts_df["y"], s=5, alpha=0.5, label="Points")
        axes[1].plot(
            hull_closed_df["x"], hull_closed_df["y"],
            linewidth=2, color="red", label="Concave Hull"
        )
        axes[1].set_title("Concave Hull Overlay")
        axes[1].set_xlabel("X"); axes[1].set_ylabel("Y")
        axes[1].grid(True, linestyle="--", alpha=0.3)
        axes[1].set_aspect("equal", adjustable="box")
        axes[1].legend()

        plt.tight_layout()
        plot_path = os.path.join(out_dir, "concave_hull_side_by_side.png")
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.show()
        print(f"🖼️ Saved plot -> {plot_path}")

if __name__ == "__main__":
    # Example: adjust as needed
    build_and_save_concave_hull(
        input_csv="Processed_images/Reflected/cluster_0_reflected.csv",  # or any CSV with coordinates
        out_dir="Processed_images/Reflected/ConcaveHull",
        k_start=100,     # more concave -> smaller k (but risk of failure), smoother -> larger k
        k_max=50,
        merge_all_pairs=True,   # set False to only use the densest x_i,y_i pair in wide files
        save_plot=True
    )
