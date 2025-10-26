import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import csv
import pandas as pd

class CoordinateGridOverlay:
    """
    A class to create coordinate grid overlays on images with (0,0) at center.
    Assumes image represents 500m x 300m physical area.
    Allows coordinate-based pixel analysis and saves green pixel coordinates to CSV.
    """
    
    def __init__(self, image_path, grid_width, grid_height, cell_size=50):
        """
        Initialize the coordinate grid overlay system.
        
        Args:
            image_path (str): Path to the input image
            grid_width (int): Width of the grid in cells
            grid_height (int): Height of the grid in cells
            cell_size (int): Size of each grid cell in pixels
        """
        self.image_path = image_path
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.cell_size = cell_size
        
        # Physical dimensions in meters
        self.physical_width_m = 500.0  # Image represents 500m width
        self.physical_height_m = 300.0  # Image represents 300m height
        
        # Load the image
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Get image dimensions
        self.image_height, self.image_width = self.original_image.shape[:2]
        
        # Calculate meters per pixel
        self.meters_per_pixel_x = self.physical_width_m / self.image_width
        self.meters_per_pixel_y = self.physical_height_m / self.image_height
        
        # Calculate center point (0,0 coordinate)
        self.center_x = self.image_width // 2
        self.center_y = self.image_height // 2
        
        # Calculate grid boundaries
        self.grid_start_x = self.center_x - (grid_width * cell_size) // 2
        self.grid_start_y = self.center_y - (grid_height * cell_size) // 2
        self.grid_end_x = self.grid_start_x + grid_width * cell_size
        self.grid_end_y = self.grid_start_y + grid_height * cell_size
        
        # Create coordinate mapping
        self.coordinate_map = self._create_coordinate_map()
        
        # Green color detection parameters
        self.green_lower = np.array([35, 50, 50])  # HSV lower bound for green
        self.green_upper = np.array([85, 255, 255])  # HSV upper bound for green
        
    def _create_coordinate_map(self):
        """
        Create a mapping of pixel coordinates to grid coordinates.
        
        Returns:
            dict: Mapping of (x, y) pixel coordinates to (grid_x, grid_y) coordinates
        """
        coordinate_map = {}
        
        for grid_x in range(-self.grid_width//2, self.grid_width//2 + 1):
            for grid_y in range(-self.grid_height//2, self.grid_height//2 + 1):
                # Calculate pixel coordinates for this grid cell
                pixel_x = self.center_x + grid_x * self.cell_size
                pixel_y = self.center_y + grid_y * self.cell_size
                
                # Map all pixels in this cell to the grid coordinates
                for dx in range(self.cell_size):
                    for dy in range(self.cell_size):
                        if (pixel_x + dx < self.image_width and 
                            pixel_y + dy < self.image_height):
                            coordinate_map[(pixel_x + dx, pixel_y + dy)] = (grid_x, grid_y)
        
        return coordinate_map
    
    def pixel_to_physical_coordinates(self, pixel_x, pixel_y):
        """
        Convert pixel coordinates to physical coordinates in meters.
        
        Args:
            pixel_x (int): Pixel x coordinate
            pixel_y (int): Pixel y coordinate
            
        Returns:
            tuple: (physical_x_m, physical_y_m) in meters
        """
        # Convert from pixel coordinates to physical coordinates
        # Origin is at top-left, so we need to adjust
        physical_x_m = pixel_x * self.meters_per_pixel_x
        physical_y_m = pixel_y * self.meters_per_pixel_y
        
        # Adjust to center-based coordinate system
        physical_x_m -= (self.image_width * self.meters_per_pixel_x) / 2
        physical_y_m -= (self.image_height * self.meters_per_pixel_y) / 2
        
        return (physical_x_m, physical_y_m)
    
    def is_green_pixel(self, pixel_x, pixel_y):
        """
        Check if a pixel is green using HSV color space.
        
        Args:
            pixel_x (int): Pixel x coordinate
            pixel_y (int): Pixel y coordinate
            
        Returns:
            bool: True if pixel is green, False otherwise
        """
        if (pixel_x < 0 or pixel_x >= self.image_width or 
            pixel_y < 0 or pixel_y >= self.image_height):
            return False
        
        # Get pixel color in BGR
        pixel_color = self.original_image[pixel_y, pixel_x]
        
        # Convert BGR to HSV
        pixel_hsv = cv2.cvtColor(np.uint8([[pixel_color]]), cv2.COLOR_BGR2HSV)
        
        # Check if pixel is within green range
        return (self.green_lower[0] <= pixel_hsv[0, 0, 0] <= self.green_upper[0] and
                self.green_lower[1] <= pixel_hsv[0, 0, 1] <= self.green_upper[1] and
                self.green_lower[2] <= pixel_hsv[0, 0, 2] <= self.green_upper[2])
    
    def find_green_pixels(self):
        """
        Find all green pixels in the image and their coordinates.
        
        Returns:
            list: List of dictionaries containing green pixel information
        """
        green_pixels = []
        
        for y in range(self.image_height):
            for x in range(self.image_width):
                if self.is_green_pixel(x, y):
                    # Get grid coordinates
                    grid_coords = self.get_pixel_coordinates(x, y)
                    
                    # Get physical coordinates in meters
                    physical_coords = self.pixel_to_physical_coordinates(x, y)
                    
                    # Get pixel color
                    pixel_color = self.original_image[y, x]
                    
                    green_pixels.append({
                        'pixel_x': x,
                        'pixel_y': y,
                        'grid_x': grid_coords[0] if grid_coords else None,
                        'grid_y': grid_coords[1] if grid_coords else None,
                        'physical_x_m': round(physical_coords[0], 6),
                        'physical_y_m': round(physical_coords[1], 6),
                        'color_bgr': pixel_color.tolist(),
                        'is_center': (x == self.center_x and y == self.center_y)
                    })
        
        return green_pixels
    
    def save_green_pixels_to_csv(self, output_path="green_pixel_coordinates.csv"):
        """
        Find all green pixels and save their coordinates to a CSV file.
        
        Args:
            output_path (str): Path to save the CSV file
            
        Returns:
            str: Path to the saved CSV file
        """
        print("�� Scanning for green pixels...")
        green_pixels = self.find_green_pixels()
        
        if not green_pixels:
            print("❌ No green pixels found in the image!")
            return None
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save to CSV
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['pixel_x', 'pixel_y', 'grid_x', 'grid_y', 
                         'physical_x_m', 'physical_y_m', 'color_bgr', 'is_center']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for pixel in green_pixels:
                writer.writerow(pixel)
        
        print(f"✅ Found {len(green_pixels)} green pixels")
        print(f"💾 Saved coordinates to: {output_path}")
        
        return output_path
    
    def get_pixel_coordinates(self, pixel_x, pixel_y):
        """
        Get the grid coordinates for a given pixel.
        
        Args:
            pixel_x (int): Pixel x coordinate
            pixel_y (int): Pixel y coordinate
            
        Returns:
            tuple: (grid_x, grid_y) coordinates or None if outside grid
        """
        return self.coordinate_map.get((pixel_x, pixel_y))
    
    def get_pixel_quality(self, pixel_x, pixel_y):
        """
        Analyze pixel quality based on color and position.
        
        Args:
            pixel_x (int): Pixel x coordinate
            pixel_y (int): Pixel y coordinate
            
        Returns:
            dict: Quality metrics for the pixel
        """
        if (pixel_x < 0 or pixel_x >= self.image_width or 
            pixel_y < 0 or pixel_y >= self.image_height):
            return None
        
        # Get pixel color
        pixel_color = self.original_image[pixel_y, pixel_x]
        
        # Get grid coordinates
        grid_coords = self.get_pixel_coordinates(pixel_x, pixel_y)
        
        # Get physical coordinates
        physical_coords = self.pixel_to_physical_coordinates(pixel_x, pixel_y)
        
        # Calculate quality metrics
        brightness = np.mean(pixel_color)
        saturation = np.std(pixel_color)
        
        # Distance from center (0,0)
        distance_from_center = np.sqrt((pixel_x - self.center_x)**2 + 
                                     (pixel_y - self.center_y)**2)
        
        return {
            'pixel_coords': (pixel_x, pixel_y),
            'grid_coords': grid_coords,
            'physical_coords_m': physical_coords,
            'color': pixel_color.tolist(),
            'brightness': float(brightness),
            'saturation': float(saturation),
            'distance_from_center': float(distance_from_center),
            'is_center': (pixel_x == self.center_x and pixel_y == self.center_y),
            'is_green': self.is_green_pixel(pixel_x, pixel_y)
        }
    
    def create_grid_overlay(self, show_coordinates=True, save_path=None):
        """
        Create an image with grid overlay and coordinate system.
        
        Args:
            show_coordinates (bool): Whether to show coordinate labels
            save_path (str): Path to save the overlay image
            
        Returns:
            numpy.ndarray: Image with grid overlay
        """
        # Create a copy of the original image
        overlay_image = self.original_image.copy()
        
        # Draw grid lines
        for i in range(-self.grid_width//2, self.grid_width//2 + 1):
            x = self.center_x + i * self.cell_size
            color = (0, 255, 0) if i == 0 else (128, 128, 128)  # Green for center line
            cv2.line(overlay_image, (x, 0), (x, self.image_height), color, 2)
        
        for i in range(-self.grid_height//2, self.grid_height//2 + 1):
            y = self.center_y + i * self.cell_size
            color = (0, 255, 0) if i == 0 else (128, 128, 128)  # Green for center line
            cv2.line(overlay_image, (0, y), (self.image_width, y), color, 2)
        
        # Draw center point (0,0)
        cv2.circle(overlay_image, (self.center_x, self.center_y), 5, (0, 0, 255), -1)
        cv2.putText(overlay_image, "(0,0)", 
                   (self.center_x + 10, self.center_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Add coordinate labels if requested
        if show_coordinates:
            for grid_x in range(-self.grid_width//2, self.grid_width//2 + 1):
                for grid_y in range(-self.grid_height//2, self.grid_height//2 + 1):
                    if grid_x == 0 and grid_y == 0:
                        continue  # Skip center point as it's already labeled
                    
                    pixel_x = self.center_x + grid_x * self.cell_size + self.cell_size//2
                    pixel_y = self.center_y + grid_y * self.cell_size + self.cell_size//2
                    
                    label = f"({grid_x},{grid_y})"
                    cv2.putText(overlay_image, label, 
                               (pixel_x - 20, pixel_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Save if path provided
        if save_path:
            cv2.imwrite(save_path, overlay_image)
            print(f"Grid overlay saved to: {save_path}")
        
        return overlay_image
    
    def analyze_region_quality(self, grid_x, grid_y):
        """
        Analyze the quality of all pixels in a specific grid region.
        
        Args:
            grid_x (int): Grid x coordinate
            grid_y (int): Grid y coordinate
            
        Returns:
            dict: Quality analysis for the region
        """
        # Calculate pixel boundaries for this grid cell
        start_x = self.center_x + grid_x * self.cell_size
        start_y = self.center_y + grid_y * self.cell_size
        end_x = start_x + self.cell_size
        end_y = start_y + self.cell_size
        
        # Ensure boundaries are within image
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(self.image_width, end_x)
        end_y = min(self.image_height, end_y)
        
        # Extract region
        region = self.original_image[start_y:end_y, start_x:end_x]
        
        if region.size == 0:
            return None
        
        # Calculate quality metrics
        mean_color = np.mean(region, axis=(0, 1))
        std_color = np.std(region, axis=(0, 1))
        brightness = np.mean(mean_color)
        contrast = np.std(std_color)
        
        return {
            'grid_coords': (grid_x, grid_y),
            'pixel_bounds': (start_x, start_y, end_x, end_y),
            'region_size': region.shape,
            'mean_color': mean_color.tolist(),
            'color_std': std_color.tolist(),
            'brightness': float(brightness),
            'contrast': float(contrast),
            'total_pixels': region.shape[0] * region.shape[1]
        }
    
    def find_pixels_by_quality(self, min_brightness=None, max_brightness=None, 
                              min_saturation=None, max_saturation=None,
                              target_color=None, color_tolerance=30):
        """
        Find pixels that match specific quality criteria.
        
        Args:
            min_brightness (float): Minimum brightness threshold
            max_brightness (float): Maximum brightness threshold
            min_saturation (float): Minimum saturation threshold
            max_saturation (float): Maximum saturation threshold
            target_color (list): Target color [B, G, R]
            color_tolerance (int): Color matching tolerance
            
        Returns:
            list: List of pixel coordinates and their qualities
        """
        matching_pixels = []
        
        for y in range(self.image_height):
            for x in range(self.image_width):
                quality = self.get_pixel_quality(x, y)
                if quality is None:
                    continue
                
                # Check brightness criteria
                if min_brightness is not None and quality['brightness'] < min_brightness:
                    continue
                if max_brightness is not None and quality['brightness'] > max_brightness:
                    continue
                
                # Check saturation criteria
                if min_saturation is not None and quality['saturation'] < min_saturation:
                    continue
                if max_saturation is not None and quality['saturation'] > max_saturation:
                    continue
                
                # Check color criteria
                if target_color is not None:
                    color_diff = np.linalg.norm(np.array(quality['color']) - np.array(target_color))
                    if color_diff > color_tolerance:
                        continue
                
                matching_pixels.append(quality)
        
        return matching_pixels
    
    def create_quality_heatmap(self, quality_metric='brightness', save_path=None):
        """
        Create a heatmap showing the distribution of a quality metric across the image.
        
        Args:
            quality_metric (str): 'brightness', 'saturation', or 'distance_from_center'
            save_path (str): Path to save the heatmap
            
        Returns:
            numpy.ndarray: Heatmap image
        """
        heatmap = np.zeros((self.image_height, self.image_width))
        
        for y in range(self.image_height):
            for x in range(self.image_width):
                quality = self.get_pixel_quality(x, y)
                if quality is not None:
                    heatmap[y, x] = quality[quality_metric]
        
        # Normalize heatmap
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Overlay on original image
        overlay = cv2.addWeighted(self.original_image, 0.7, heatmap_colored, 0.3, 0)
        
        if save_path:
            cv2.imwrite(save_path, overlay)
            print(f"Quality heatmap saved to: {save_path}")
        
        return overlay

def main():
    """
    Example usage of the CoordinateGridOverlay system with physical dimensions.
    """
    # Example parameters
    image_path = "Processed_images/ML_processed_image_dark_green.png"  # Change to your image path
    grid_width = 10  # Number of cells horizontally
    grid_height = 8  # Number of cells vertically
    cell_size = 50   # Size of each cell in pixels
    
    try:
        # Create the coordinate grid overlay system
        grid_system = CoordinateGridOverlay(image_path, grid_width, grid_height, cell_size)
        
        print("🌐 Coordinate Grid Overlay System (500m x 300m)")
        print("=" * 60)
        print(f"Image: {image_path}")
        print(f"Image size: {grid_system.image_width} x {grid_system.image_height} pixels")
        print(f"Physical size: {grid_system.physical_width_m} x {grid_system.physical_height_m} meters")
        print(f"Grid size: {grid_width} x {grid_height} cells")
        print(f"Cell size: {cell_size} pixels")
        print(f"Center point (0,0): ({grid_system.center_x}, {grid_system.center_y})")
        print(f"Meters per pixel: X={grid_system.meters_per_pixel_x:.6f}, Y={grid_system.meters_per_pixel_y:.6f}")
        
        # Create and save grid overlay
        overlay_image = grid_system.create_grid_overlay(
            show_coordinates=True,
            save_path="Processed_images/grid_overlay.png"
        )
        
        # Find and save green pixel coordinates
        csv_path = grid_system.save_green_pixels_to_csv("Processed_images/green_pixel_coordinates.csv")
        
        if csv_path:
            # Load and display summary of the CSV data
            df = pd.read_csv(csv_path)
            print(f"\n📊 Green Pixel Analysis Summary:")
            print(f"  Total green pixels found: {len(df)}")
            print(f"  Physical X range: {df['physical_x_m'].min():.2f}m to {df['physical_x_m'].max():.2f}m")
            print(f"  Physical Y range: {df['physical_y_m'].min():.2f}m to {df['physical_y_m'].max():.2f}m")
            print(f"  Grid coordinates range: X={df['grid_x'].min()} to {df['grid_x'].max()}, Y={df['grid_y'].min()} to {df['grid_y'].max()}")
            
            # Show first few entries
            print(f"\n📋 First 5 green pixel coordinates:")
            print(df[['pixel_x', 'pixel_y', 'grid_x', 'grid_y', 'physical_x_m', 'physical_y_m']].head())
        
        # Example: Analyze a specific region
        region_quality = grid_system.analyze_region_quality(1, 1)
        if region_quality:
            print(f"\n📊 Region (1,1) Analysis:")
            print(f"  Mean color: {region_quality['mean_color']}")
            print(f"  Brightness: {region_quality['brightness']:.2f}")
            print(f"  Contrast: {region_quality['contrast']:.2f}")
        
        # Example: Get coordinates for a specific pixel
        pixel_quality = grid_system.get_pixel_quality(100, 100)
        if pixel_quality:
            print(f"\n📍 Pixel (100,100) Analysis:")
            print(f"  Grid coordinates: {pixel_quality['grid_coords']}")
            print(f"  Physical coordinates: {pixel_quality['physical_coords_m']} meters")
            print(f"  Color: {pixel_quality['color']}")
            print(f"  Brightness: {pixel_quality['brightness']:.2f}")
            print(f"  Is green: {pixel_quality['is_green']}")
        
        # Create quality heatmap
        grid_system.create_quality_heatmap(
            quality_metric='brightness',
            save_path="Processed_images/brightness_heatmap.png"
        )
        
        print(f"\n✅ Grid overlay system created successfully!")
        print(f"�� Output files saved in Processed_images/ folder")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()