import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def create_coordinate_based_visualizations():
    print("📍 Physical Coordinate Visualization")
    print("=" * 40)
    
    df = pd.read_csv("Processed_images/green_pixel_coordinates.csv")
    print(f"✅ Loaded {len(df):,} coordinate points")
    
    os.makedirs("Processed_images", exist_ok=True)

    # === Sample for scatter plot ===
    print("📈 Creating coordinate scatter plot...")
    sample_size = min(200000, len(df))
    sampled_df = df.sample(n=sample_size, random_state=42)

    # Save sampled points to CSV
    sampled_csv_path = "Processed_images/sampled_coordinates.csv"
    sampled_df.to_csv(sampled_csv_path, index=False)
    print(f"💾 Sampled coordinate points saved to {sampled_csv_path}")

    plt.figure(figsize=(16, 10))
    plt.scatter(sampled_df['physical_x_m'], -sampled_df['physical_y_m'], 
                c='darkgreen', alpha=0.6, s=1, edgecolors='none')
    plt.xlabel('Physical X Coordinate (meters)', fontsize=12)
    plt.ylabel('Reflected Physical Y Coordinate (meters)', fontsize=12)
    plt.title(f'Green Pixel Physical Coordinates (Reflected)\n{sample_size:,} sampled points', 
              fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig("Processed_images/coordinate_scatter.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Coordinate scatter plot saved")

    # === Heatmap ===
    print("🔥 Creating coordinate density heatmap...")
    x_bins = np.linspace(df['physical_x_m'].min(), df['physical_x_m'].max(), 100)
    y_bins = np.linspace(-df['physical_y_m'].max(), -df['physical_y_m'].min(), 60)
    
    heatmap, xedges, yedges = np.histogram2d(
        df['physical_x_m'], -df['physical_y_m'], bins=[x_bins, y_bins]
    )

    plt.figure(figsize=(16, 10))
    im = plt.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                    origin='lower', cmap='Greens', aspect='auto')
    plt.colorbar(im, label='Coordinate Point Density')
    plt.xlabel('Physical X Coordinate (meters)', fontsize=12)
    plt.ylabel('Reflected Physical Y Coordinate (meters)', fontsize=12)
    plt.title(f'Coordinate Point Density Heatmap\n{len(df):,} total points', fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("Processed_images/coordinate_density.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Coordinate density heatmap saved")

    # === Distributions ===
    print("📊 Creating coordinate distribution plots...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))

    ax1.hist(df['physical_x_m'], bins=50, color='darkgreen', alpha=0.7, edgecolor='black')
    ax1.set_title('Distribution of Physical X Coordinates', fontsize=14, pad=20)
    ax1.set_xlabel('Physical X Coordinate (meters)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(df['physical_x_m'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["physical_x_m"].mean():.2f}m')
    ax1.legend()

    ax2.hist(-df['physical_y_m'], bins=50, color='darkgreen', alpha=0.7, edgecolor='black')
    ax2.set_title('Distribution of Reflected Y Coordinates', fontsize=14, pad=20)
    ax2.set_xlabel('Reflected Physical Y Coordinate (meters)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axvline((-df['physical_y_m']).mean(), color='red', linestyle='--', 
                label=f'Mean: {(-df["physical_y_m"]).mean():.2f}m')
    ax2.legend()

    # === Sample for 2D plot ===
    sample_2d = df.sample(n=min(5000, len(df)), random_state=42)

    # Save 2D sample points to CSV
    sample_2d_csv_path = "Processed_images/sampled_2d_coordinates.csv"
    sample_2d.to_csv(sample_2d_csv_path, index=False)
    print(f"💾 2D sampled coordinate points saved to {sample_2d_csv_path}")

    ax3.scatter(sample_2d['physical_x_m'], -sample_2d['physical_y_m'], 
                c='darkgreen', alpha=0.6, s=2)
    ax3.set_title(f'2D Coordinate Distribution\n{sample_2d.shape[0]:,} sampled points', 
                  fontsize=14, pad=20)
    ax3.set_xlabel('Physical X Coordinate (meters)', fontsize=12)
    ax3.set_ylabel('Reflected Physical Y Coordinate (meters)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')

    ax4.axis('off')
    stats_text = f"""
    📊 Coordinate Statistics:

    Total Points: {len(df):,}

    X Coordinate:
    • Range: {df['physical_x_m'].min():.2f}m to {df['physical_x_m'].max():.2f}m
    • Mean: {df['physical_x_m'].mean():.2f}m
    • Std Dev: {df['physical_x_m'].std():.2f}m

    Y Coordinate (Reflected):
    • Range: {-df['physical_y_m'].max():.2f}m to {-df['physical_y_m'].min():.2f}m
    • Mean: {(-df['physical_y_m']).mean():.2f}m
    • Std Dev: {(-df['physical_y_m']).std():.2f}m

    Center Point (0,0):
    • Points at center: {len(df[df['is_center'] == True])}
    """
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()
    plt.savefig("Processed_images/coordinate_distributions.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Coordinate distribution plots saved")

if __name__ == "__main__":
    create_coordinate_based_visualizations()
