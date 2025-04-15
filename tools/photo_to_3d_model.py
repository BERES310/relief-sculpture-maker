from PIL import Image
import os
from tqdm import tqdm
from .preprocess_image import preprocess_image
from .generate_depth_map import generate_depth_map
from .depth_map_to_3d_model import depth_map_to_3d_model

def photo_to_3d_model(image_path, output_stl_path, depth_map_path="depth_map.png", preprocessed_path="preprocessed_image.png", output_dir="output_files", width=100.0, height=100.0, depth_scale=10.0, depth_map_width=None, depth_map_height=None, smooth_iterations=3, target_triangle_count=None, thickness=5.0):
    """
    Convert a photo to a solid 3D model (STL) with a flat bottom, trimmed to the shape of the preprocessed image, via depth map, with background removal, smoothing, and optional simplification.
    Dimensions are in millimeters (mm). All outputs are saved in a subfolder.
    
    Parameters:
    - image_path: Path to input photo.
    - output_stl_path: Name of the STL file (e.g., 'output_model.stl').
    - depth_map_path: Name of the depth map file (e.g., 'depth_map.png').
    - preprocessed_path: Name of the preprocessed image file (e.g., 'preprocessed_image.png').
    - output_dir: Subfolder to save all output files (default: 'output_files').
    - width, height: Dimensions of the plane in millimeters.
    - depth_scale: Maximum depth displacement in millimeters (Z-height of top surface).
    - depth_map_width: Desired width of the depth map in pixels (default: input image width).
    - depth_map_height: Desired height of the depth map in pixels (default: input image height).
    - smooth_iterations: Number of Laplacian smoothing iterations.
    - target_triangle_count: Desired number of triangles in the output mesh (default: None, no simplification).
    - thickness: Thickness of the solid body in millimeters (distance from flat bottom to lowest point of top surface).
    """
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    if depth_map_width is None:
        depth_map_width = img_width
    if depth_map_height is None:
        depth_map_height = img_height
    
    # Initialize progress bar for the entire process
    with tqdm(total=100, desc="Generating 3D Model", unit="%") as pbar:
        # Step 1: Preprocess image (10% of total process)
        preprocessed_image_path = preprocess_image(image_path, preprocessed_path, output_dir)
        pbar.update(10)

        # Step 2: Generate depth map (40% of total process)
        generate_depth_map(preprocessed_image_path, depth_map_path, depth_map_width, depth_map_height, output_dir)
        pbar.update(40)

        # Step 3: Convert depth map to 3D model (50% of total process)
        depth_map_to_3d_model(
            os.path.join(output_dir, depth_map_path),
            output_stl_path,
            preprocessed_image_path,
            output_dir,
            width,
            height,
            depth_scale,
            subdivisions_x=depth_map_width,
            subdivisions_y=depth_map_height,
            smooth_iterations=smooth_iterations,
            target_triangle_count=target_triangle_count,
            thickness=thickness,
            progress=pbar
        )