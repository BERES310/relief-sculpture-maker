# Photo to 3D Model Conversion

## Overview
This Python script converts a 2D photo into a solid 3D model (STL format) with a flat bottom, suitable for 3D printing or visualization. It processes the input image by removing the background, generates a depth map using the MiDaS model, and constructs a 3D mesh trimmed to the object's shape. The resulting model is smoothed and can be simplified to a target triangle count.

## Features
- **Background Removal**: Uses the `rembg` library to isolate the object by removing the background.
- **Depth Map Generation**: Employs the MiDaS depth estimation model (`DPT_Large`) to create a depth map, masked by the alpha channel of the preprocessed image.
- **3D Model Creation**: Converts the depth map into a solid 3D mesh with a flat bottom, using the alpha channel to trim the model to the object's shape. A wider base prevents dangling sides.
- **Smoothing and Simplification**: Applies Laplacian smoothing and optional mesh simplification to reduce triangle count.
- **Progress Tracking**: Includes a progress bar (`tqdm`) to monitor the conversion process.
- **Customizable Parameters**: Allows adjustment of model dimensions, depth scale, thickness, and mesh resolution.

## Requirements
To run the script, install the following Python packages:
```
numpy
Pillow
trimesh
torch
torchvision
opencv-python
rembg
open3d
tqdm
scipy
```

You can install them using pip:
```bash
pip install numpy Pillow trimesh torch torchvision opencv-python rembg open3d tqdm scipy
```

**Note**: The `rembg` library may require additional dependencies for background removal, and `torch` requires a compatible CUDA setup for GPU acceleration (optional).

## Usage
1. **Prepare an Input Image**: Provide a JPG or PNG image containing an object with a clear background (e.g., `lion.jpg`).
2. **Run the Script**: Execute the script with the provided main block or call the `photo_to_3d_model` function with custom parameters.

### Example Command
```bash
python script.py
```

The default configuration in the script processes `lion.jpg` and generates:
- A preprocessed image (`preprocessed_image.png`) with the background removed.
- A depth map (`depth_map.png`).
- A 3D model (`output_model.stl`) in the `output_files` directory.

### Function Call
To customize the process, use the `photo_to_3d_model` function:
```python
from script import photo_to_3d_model

photo_to_3d_model(
    image_path="your_image.jpg",
    output_stl_path="model.stl",
    depth_map_path="depth.png",
    preprocessed_path="preprocessed.png",
    output_dir="output",
    width=100.0,           # Model width in mm
    height=100.0,          # Model height in mm
    depth_scale=10.0,      # Max depth displacement in mm
    depth_map_width=None,  # Depth map width (pixels, default: image width)
    depth_map_height=None, # Depth map height (pixels, default: image height)
    smooth_iterations=3,   # Number of smoothing iterations
    target_triangle_count=None, # Target number of triangles (None for no simplification)
    thickness=5.0          # Model thickness in mm
)
```

## Parameters
- `image_path`: Path to the input image (JPG/PNG).
- `output_stl_path`: Name of the output STL file.
- `depth_map_path`: Name of the depth map file.
- `preprocessed_path`: Name of the preprocessed image file.
- `output_dir`: Directory to save output files.
- `width`, `height`: Dimensions of the 3D model in millimeters.
- `depth_scale`: Maximum Z-displacement of the top surface in millimeters.
- `depth_map_width`, `depth_map_height`: Resolution of the depth map in pixels (default: matches input image).
- `smooth_iterations`: Number of Laplacian smoothing iterations for the mesh.
- `target_triangle_count`: Desired number of triangles in the final mesh (optional).
- `thickness`: Distance from the flat bottom to the lowest point of the top surface in millimeters.

## Output
All outputs are saved in the specified `output_dir` (default: `output_files`):
- **Preprocessed Image**: PNG with the background removed (e.g., `preprocessed_image.png`).
- **Depth Map**: Grayscale PNG representing depth (e.g., `depth_map.png`).
- **3D Model**: STL file of the solid model (e.g., `output_model.stl`).

## Notes
- **Units**: All dimensions (`width`, `height`, `depth_scale`, `thickness`) are in millimeters, making the STL file compatible with 3D printing software.
- **Performance**: The script may be computationally intensive, especially with high-resolution images or large `subdivisions_x` and `subdivisions_y`. A GPU is recommended for faster depth map generation.
- **Mesh Quality**: Increase `smooth_iterations` for smoother surfaces or set `target_triangle_count` to reduce file size for simpler models.
- **Alpha Channel**: The preprocessed image's alpha channel ensures the 3D model is trimmed to the object's shape, and a dilated alpha mask creates a wider base for stability.

## Example
For an input image `lion.jpg`, the script:
1. Removes the background, saving `preprocessed_image.png`.
2. Generates a depth map, saving `depth_map.png`.
3. Creates a 10mm x 10mm x 5mm thick STL model (`output_model.stl`) with a maximum depth displacement of 5mm, smoothed over 10 iterations.

## Limitations
- **Depth Accuracy**: MiDaS provides relative depth estimates, which may not be metrically accurate for all objects.
- **Complex Objects**: Objects with intricate details or thin structures may require higher depth map resolution or manual mesh cleanup.
- **Background Removal**: The `rembg` library may struggle with complex backgrounds or low-contrast edges, affecting the alpha mask quality.

## License
This script is provided for educational and personal use. Ensure you have the necessary permissions for the input images and comply with the licenses of the dependencies (e.g., MiDaS, `rembg`).

## Contact
For issues or suggestions, please open an issue on the project repository.
