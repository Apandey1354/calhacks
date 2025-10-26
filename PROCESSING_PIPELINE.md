# Image Processing Pipeline Documentation

## Overview
This document describes the complete processing pipeline for converting `ML_processed_image_dark_green.png` into divided clusters.

## Processing Steps

### 1. Image Conversion to Dark Green (`image_kernel.py`)
**Input:** Original ML processed image  
**Output:** `ML_processed_image_dark_green.png`  
**Process:** Converts all non-black pixels to dark green (0, 100, 0)

### 2. Coordinate Extraction (`grid_maker.py`)
**Input:** `ML_processed_image_dark_green.png`  
**Output:** `Processed_images/green_pixel_coordinates.csv`  
**Process:** 
- Detects green pixels using HSV color space
- Converts pixel coordinates to physical coordinates in meters
- Maps coordinates with (0,0) at image center
- Assumes 500m x 300m physical area

**Key Code:**
```python
# From grid_maker.py
image_path = "Processed_images/ML_processed_image_dark_green.png"
grid_system = CoordinateGridOverlay(image_path, grid_width, grid_height, cell_size)
csv_path = grid_system.save_green_pixels_to_csv("Processed_images/green_pixel_coordinates.csv")
```

### 3. Data Sampling (`coordinate_visualizer.py`)
**Input:** `Processed_images/green_pixel_coordinates.csv`  
**Output:** `Processed_images/sampled_coordinates.csv`  
**Process:**
- Samples 200,000 points for clustering (to reduce computation time)
- Creates visualizations (scatter plots, heatmaps)

**Key Code:**
```python
# From coordinate_visualizer.py
df = pd.read_csv("Processed_images/green_pixel_coordinates.csv")
sample_size = min(200000, len(df))
sampled_df = df.sample(n=sample_size, random_state=42)
sampled_df.to_csv("Processed_images/sampled_coordinates.csv", index=False)
```

### 4. HDBSCAN Clustering (`seperate_clusters.py`)
**Input:** `Processed_images/sampled_coordinates.csv`  
**Output:** 
- `Processed_images/clustered_coordinates_hdbscan_excel.csv` (all clusters)
- `Processed_images/largest_5_clusters.csv` (top 5 clusters)
- `Processed_images/remaining_clusters.csv` (remaining clusters)

**Process:**
- Uses HDBSCAN clustering algorithm with min_cluster_size=5
- Separates top 5 largest clusters from remaining clusters
- Saves clusters in columnar format (x_0, y_0, x_1, y_1, etc.)

**Key Code:**
```python
# From seperate_clusters.py
clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
labels = clusterer.fit_predict(coords)
# Separates into top-5 and remaining clusters
```

### 5. Cluster Separation & Reflection (`seperate_all_clusters.py`)
**Input:** 
- `Processed_images/largest_5_clusters.csv`
- `Processed_images/remaining_clusters.csv`

**Output:**
- `Processed_images/Reflected/cluster_0_reflected.csv` through `cluster_4_reflected.csv`
- Individual cluster visualizations (PNG files)

**Process:**
- Extracts individual clusters from columnar format
- Reflects Y-axis (multiplies Y by -1)
- Saves each cluster as separate CSV file
- Creates individual scatter plot visualizations

**Key Code:**
```python
# From seperate_all_clusters.py
for idx, xcol, ycol in top5_pairs:
    sub = df5[[xcol, ycol]].dropna().copy()
    sub.columns = ["x", "y"]
    sub["y"] = -sub["y"]  # reflect by X-axis
    sub.to_csv(f"Processed_images/Reflected/cluster_{idx}_reflected.csv")
```

### 6. Optional: Fine-Tuning (`fine_tune_clusters.py`)
**Input:** `Processed_images/clustered_coordinates_hdbscan.csv`  
**Output:** `Processed_images/cluster_exports/cluster_0.csv`, etc.

**Process:**
- Extracts top N clusters based on specified criteria
- Applies minimum point threshold filtering
- Creates cluster visualizations

## Complete Data Flow

```
ML_processed_image_dark_green.png
           ↓ (grid_maker.py)
green_pixel_coordinates.csv
           ↓ (coordinate_visualizer.py)
sampled_coordinates.csv
           ↓ (seperate_clusters.py)
├── clustered_coordinates_hdbscan_excel.csv (all clusters)
├── largest_5_clusters.csv
└── remaining_clusters.csv
           ↓ (seperate_all_clusters.py)
Processed_images/Reflected/
├── cluster_0_reflected.csv + PNG
├── cluster_1_reflected.csv + PNG
├── cluster_2_reflected.csv + PNG
├── cluster_3_reflected.csv + PNG
└── cluster_4_reflected.csv + PNG
```

## Post-Clustering: Hull Generation

After clusters are separated into individual CSV files, they go through **hull generation algorithms** to create bounding polygons around the cluster points.

### What Happens After Getting Clusters to Files?

Each cluster CSV file (`cluster_0_reflected.csv` through `cluster_4_reflected.csv`) is processed by several hull algorithms:

#### 1. **Convex Hull** (`convexhull.py`)
- Creates the smallest convex polygon containing all cluster points
- Uses monotone chain algorithm
- **Input:** `cluster_X_reflected.csv`
- **Output:** `Processed_images/Reflected/ConvexHull/`
  - `convex_hull_coords.csv`
  - `convex_hull_coords_closed.csv`
  - `convex_hull_side_by_side.png`
  - `original_points.csv`
  - `metadata.json`

#### 2. **Concave Hull** (`concavehall.py`)
- Creates a tighter-fitting concave polygon around points
- Uses Moreira & Santos algorithm with k-nearest neighbors
- Can be simplified using Douglas-Peucker algorithm (RDP)
- **Input:** `cluster_X_reflected.csv`
- **Output:** `Processed_images/Reflected/ConcaveHull/`
  - `concave_hull_coords.csv`
  - `concave_hull_coords_closed.csv`
  - `concave_hull_simplified.csv`
  - `concave_hull_simplified_closed.csv`
  - `concave_hull_side_by_side.png`
  - `original_points.csv`
  - `metadata.json`

#### 3. **RDP Simplified Hull** (`rdp.py`)
- Same concave hull but with Douglas-Peucker simplification
- Reduces vertex count while maintaining shape
- **Output:** `Processed_images/Reflected/RDP/`

#### 4. **Topology-Safe Hull** (`topologysafe.py`)
- Ensures all original points remain inside the simplified polygon
- Uses coverage checking to prevent point exclusion
- **Output:** `Processed_images/Reflected/topology_safe/`

### Hull Generation Parameters

Each hull algorithm can be configured with:

**Concave Hull Parameters:**
- `k_start`: Starting k for nearest neighbors (default: 6)
- `k_max`: Maximum k value to try (default: 30)
- `merge_all_pairs`: Whether to merge all x_i,y_i pairs

**Simplification Parameters:**
- `simplify`: Enable/disable simplification
- `simplify_epsilon`: Distance threshold for RDP
- `simplify_target_vertices`: Target number of vertices (default: 8)
- `simplify_edge_tolerance`: Distance tolerance for coverage checking (default: 0.05)

### Metadata Files

Each hull output includes a `metadata.json` with:
- Input file reference
- Number of points and vertices
- Hull area and perimeter
- Simplified hull statistics (if applicable)
- Parameter settings
- Output file paths

### Visualization

Each hull algorithm generates a side-by-side plot showing:
- **Left:** Original points with full hull
- **Right:** Simplified hull (if applicable)

## Complete End-to-End Pipeline

```
ML_processed_image_dark_green.png
           ↓ (grid_maker.py)
green_pixel_coordinates.csv
           ↓ (coordinate_visualizer.py)
sampled_coordinates.csv
           ↓ (seperate_clusters.py)
├── clustered_coordinates_hdbscan_excel.csv (all clusters)
├── largest_5_clusters.csv
└── remaining_clusters.csv
           ↓ (seperate_all_clusters.py)
Processed_images/Reflected/
├── cluster_0_reflected.csv + PNG
├── cluster_1_reflected.csv + PNG
├── cluster_2_reflected.csv + PNG
├── cluster_3_reflected.csv + PNG
└── cluster_4_reflected.csv + PNG
           ↓ (hull generation - parallel processing)
├── ConvexHull/         (convexhull.py)
│   ├── convex_hull_coords.csv
│   ├── convex_hull_coords_closed.csv
│   └── visualization PNG
├── ConcaveHull/         (concavehall.py)
│   ├── concave_hull_coords.csv
│   ├── concave_hull_simplified.csv
│   └── visualization PNG
├── RDP/                 (rdp.py)
│   ├── simplified hulls
│   └── visualization PNG
└── topology_safe/        (topologysafe.py)
    ├── simplified hulls
    └── visualization PNG
```

## Summary
The file `ML_processed_image_dark_green.png` passes through these Python scripts to get divided clusters:
1. `grid_maker.py` - Extracts green pixel coordinates
2. `coordinate_visualizer.py` - Samples coordinates for clustering
3. `seperate_clusters.py` - Performs HDBSCAN clustering
4. `seperate_all_clusters.py` - Separates individual clusters
5. **Hull Generation** - Creates bounding polygons around clusters:
   - `convexhull.py` - Convex hull
   - `concavehall.py` - Concave hull with simplification
   - `rdp.py` - RDP simplification
   - `topologysafe.py` - Topology-safe simplification


Finally-sweep algorithm.