import pandas as pd
import numpy as np
import hdbscan
import os

def cluster_and_save_hdbscan(input_path, output_path):
    print("📦 Loading data...")
    df = pd.read_csv(input_path)
    coords = df[['physical_x_m', 'physical_y_m']].dropna()

    print("🧠 Running HDBSCAN clustering...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
    labels = clusterer.fit_predict(coords)
    coords['cluster'] = labels

    valid_clusters = sorted([c for c in np.unique(labels) if c != -1])  # exclude noise
    print(f"✅ Found {len(valid_clusters)} clusters (excluding noise)")

    # Count cluster sizes
    cluster_sizes = coords[coords['cluster'] != -1]['cluster'].value_counts().sort_values(ascending=False)
    top_5_clusters = cluster_sizes.head(5).index.tolist()
    remaining_clusters = cluster_sizes.iloc[5:].index.tolist()

    def extract_cluster_data(cluster_ids):
        cluster_data = []
        for cluster_id in cluster_ids:
            cluster_points = coords[coords['cluster'] == cluster_id][['physical_x_m', 'physical_y_m']].reset_index(drop=True)
            cluster_data.append(cluster_points)
        max_len = max(len(c) for c in cluster_data)
        for i in range(len(cluster_data)):
            cluster_data[i] = cluster_data[i].reindex(range(max_len))
        result_df = pd.concat(cluster_data, axis=1)
        col_names = []
        for i in range(len(cluster_data)):
            col_names.extend([f"x_{i}", f"y_{i}"])
        result_df.columns = col_names
        return result_df

    # Save top 5 clusters
    top_5_df = extract_cluster_data(top_5_clusters)
    top_5_path = os.path.join(os.path.dirname(output_path), "largest_5_clusters.csv")
    top_5_df.to_csv(top_5_path, index=False)
    print(f"📄 Top 5 clusters saved to: {top_5_path}")

    # Save remaining clusters
    remaining_df = extract_cluster_data(remaining_clusters)
    remaining_path = os.path.join(os.path.dirname(output_path), "remaining_clusters.csv")
    remaining_df.to_csv(remaining_path, index=False)
    print(f"📄 Remaining clusters saved to: {remaining_path}")

    # Save all clusters in a single file (original output)
    all_df = extract_cluster_data(valid_clusters)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    all_df.to_csv(output_path, index=False)
    print(f"📄 All clustered data saved to: {output_path}")

if __name__ == "__main__":
    input_csv = "Processed_images/sampled_coordinates.csv"
    output_csv = "Processed_images/clustered_coordinates_hdbscan_excel.csv"
    cluster_and_save_hdbscan(input_csv, output_csv)
