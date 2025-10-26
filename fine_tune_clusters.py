import pandas as pd
import matplotlib.pyplot as plt
import os

def process_and_plot_clusters(csv_path, output_dir, num_clusters_to_plot=3, min_points_threshold=10):
    print("📥 Loading clustered CSV...")
    df = pd.read_csv(csv_path)

    os.makedirs(output_dir, exist_ok=True)

    total_clusters = df.shape[1] // 2
    clusters_to_plot = min(num_clusters_to_plot, total_clusters)
    cmap = plt.get_cmap("tab20", clusters_to_plot)

    plt.figure(figsize=(14, 10))
    plotted = 0

    for i in range(clusters_to_plot):
        x_col = f'x_{i}'
        y_col = f'y_{i}'

        if x_col not in df.columns or y_col not in df.columns:
            continue

        cluster_data = df[[x_col, y_col]].dropna()

        if len(cluster_data) < min_points_threshold:
            print(f"⏭️ Skipping Cluster {i} (only {len(cluster_data)} points)")
            continue

        # Save to individual CSV
        cluster_csv_path = os.path.join(output_dir, f'cluster_{i}.csv')
        cluster_data.columns = ['x', 'y']
        cluster_data.to_csv(cluster_csv_path, index=False)
        print(f"💾 Saved Cluster {i} to {cluster_csv_path} ({len(cluster_data)} points)")

        # Plot cluster
        plt.scatter(cluster_data['x'], cluster_data['y'] * -1,  # Reflect Y
                    s=10, color=cmap(plotted), label=f'Cluster {i}')
        plotted += 1

    if plotted > 0:
        plt.title(f'Cluster Visualization ({plotted} clusters plotted)', fontsize=16)
        plt.xlabel('Physical X Coordinate (m)')
        plt.ylabel('Reflected Physical Y Coordinate (m)')
        plt.grid(True, alpha=0.3)
        plt.legend(markerscale=2, fontsize=9, loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "selected_clusters_plot.png"), dpi=300)
        plt.show()
    else:
        print("⚠️ No clusters met the criteria to be plotted.")

if __name__ == "__main__":
    csv_input = "Processed_images/clustered_coordinates_hdbscan.csv"
    output_folder = "Processed_images/cluster_exports"
    
    # 🔧 Customize this:
    num_clusters_to_plot = 5         # How many clusters from the beginning you want to process
    min_points_threshold = 15        # Minimum number of points required to save and plot a cluster

    process_and_plot_clusters(csv_input, output_folder, num_clusters_to_plot, min_points_threshold)
