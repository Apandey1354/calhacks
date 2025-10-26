import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, re

def _find_xy_pairs(df):
    pairs = []
    for c in df.columns:
        m = re.match(r"^x_(\d+)$", c)
        if m:
            i = int(m.group(1))
            y = f"y_{i}"
            if y in df.columns:
                pairs.append((i, c, y))
    pairs.sort(key=lambda t: t[0])
    return pairs

def save_reflected_clusters_and_visualize(
    largest5_csv="Processed_images/largest_5_clusters.csv",
    remaining_csv="Processed_images/remaining_clusters.csv",
    out_dir="Processed_images/Reflected",
    save_plots=True,
    show_plots=True
):
    os.makedirs(out_dir, exist_ok=True)

    # ---------- Top 5 ----------
    if not os.path.exists(largest5_csv):
        raise FileNotFoundError(f"File not found: {largest5_csv}")
    df5 = pd.read_csv(largest5_csv)

    top5_pairs = _find_xy_pairs(df5)
    print(f"Found {len(top5_pairs)} top-5 cluster pairs.")

    # Save + visualize each top-5 cluster
    for idx, xcol, ycol in top5_pairs:
        sub = df5[[xcol, ycol]].dropna().copy()
        sub.columns = ["x", "y"]
        sub["y"] = -sub["y"]  # reflect by X-axis

        out_csv = os.path.join(out_dir, f"cluster_{idx}_reflected.csv")
        sub.to_csv(out_csv, index=False)
        print(f"Saved CSV: {out_csv} (n={len(sub)})")

        # Stats for annotation/log
        xmin, xmax = sub["x"].min(), sub["x"].max()
        ymin, ymax = sub["y"].min(), sub["y"].max()
        print(f"Cluster {idx} bounds (reflected): X[{xmin:.3f}, {xmax:.3f}], Y[{ymin:.3f}, {ymax:.3f}]")

        # Visualization
        fig = plt.figure(figsize=(6, 6))
        plt.scatter(sub["x"], sub["y"], s=6)
        plt.title(f"Largest Cluster {idx} (n={len(sub)}) — Reflected by X-axis")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m) [reflected]")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.gca().set_aspect("equal", adjustable="box")

        if save_plots:
            out_png = os.path.join(out_dir, f"cluster_{idx}_reflected.png")
            plt.savefig(out_png, dpi=200, bbox_inches="tight")
            print(f"Saved plot: {out_png}")
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    # ---------- Remaining ----------
    if not os.path.exists(remaining_csv):
        raise FileNotFoundError(f"File not found: {remaining_csv}")
    rem_df = pd.read_csv(remaining_csv)

    # Reflect all y_i columns in-place copy
    rem_ref = rem_df.copy()
    for c in list(rem_ref.columns):
        m = re.match(r"^y_(\d+)$", c)
        if m:
            rem_ref[c] = -rem_ref[c]

    rem_out_csv = os.path.join(out_dir, "remaining_clusters_reflected.csv")
    rem_ref.to_csv(rem_out_csv, index=False)
    print(f"Saved CSV: {rem_out_csv}")

    # Gather remaining points for visualization
    rem_pairs = _find_xy_pairs(rem_ref)
    all_pts = []
    for idx, xcol, ycol in rem_pairs:
        pts = rem_ref[[xcol, ycol]].dropna().to_numpy()
        if len(pts) > 0:
            all_pts.append((idx, pts))
    total_pts = sum(len(p) for _, p in all_pts)
    print(f"Remaining clusters: {len(all_pts)} found, total points={total_pts}")

    # Plot all remaining clusters together (one window)
    if all_pts:
        fig = plt.figure(figsize=(7, 7))
        for idx, pts in all_pts:
            plt.scatter(pts[:, 0], pts[:, 1], s=5, label=f"Cluster {idx}")
        plt.title(f"All Remaining Clusters — Reflected by X-axis ({len(all_pts)} clusters)")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m) [reflected]")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend(markerscale=2, fontsize=8)
        plt.gca().set_aspect("equal", adjustable="box")

        if save_plots:
            out_png = os.path.join(out_dir, "remaining_clusters_reflected.png")
            plt.savefig(out_png, dpi=200, bbox_inches="tight")
            print(f"Saved plot: {out_png}")
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
    else:
        print("No remaining cluster points to visualize.")

if __name__ == "__main__":
    save_reflected_clusters_and_visualize()
