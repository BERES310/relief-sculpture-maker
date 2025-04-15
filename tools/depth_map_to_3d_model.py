import numpy as np
from PIL import Image
import trimesh
import open3d as o3d
import os
from scipy.ndimage import binary_dilation


def depth_map_to_3d_model(depth_map_path, output_path, preprocessed_path, output_dir="output_files", width=100.0, height=100.0, depth_scale=10.0, subdivisions_x=100, subdivisions_y=100, smooth_iterations=3, target_triangle_count=None, thickness=5.0, progress=None):
    """
    Convert a depth map to a solid 3D model with a flat bottom, trimmed to the shape of the preprocessed image, with a wider base to prevent dangling sides, apply smoothing, simplify if needed, and save as an STL file.
    Dimensions are in millimeters (mm).
    
    Parameters:
    - depth_map_path: Path to the grayscale depth map (PNG).
    - output_path: Name of the STL file (e.g., 'output_model.stl').
    - preprocessed_path: Path to the preprocessed image (with alpha channel).
    - output_dir: Subfolder to save the STL file (default: 'output_files').
    - width, height: Dimensions of the plane in millimeters.
    - depth_scale: Maximum depth displacement in millimeters (Z-height of top surface).
    - subdivisions_x, subdivisions_y: Number of subdivisions in X and Y directions.
    - smooth_iterations: Number of Laplacian smoothing iterations.
    - target_triangle_count: Desired number of triangles in the output mesh (default: None, no simplification).
    - thickness: Thickness of the solid body in millimeters (distance from flat bottom to lowest point of top surface).
    - progress: tqdm progress bar object to update (default: None).
    """
    os.makedirs(output_dir, exist_ok=True)
    output_full_path = os.path.join(output_dir, output_path)
    
    # Load depth map and alpha channel
    depth_img = Image.open(depth_map_path).convert('L')
    depth_array = np.array(depth_img) / 255.0

    img_with_alpha = Image.open(preprocessed_path).convert('RGBA')
    alpha = np.array(img_with_alpha.split()[-1]) / 255.0  # Alpha channel (0 to 1)

    # Create a wider alpha mask for the bottom surface
    alpha_binary = alpha > 0.1
    alpha_bottom = binary_dilation(alpha_binary, iterations=5)  # Dilate to widen the base
    alpha_bottom = alpha_bottom.astype(float)

    img_height, img_width = depth_array.shape

    # Create vertex grid
    x = np.linspace(-width / 2, width / 2, subdivisions_x)
    y = np.linspace(-height / 2, height / 2, subdivisions_y)
    x, y = np.meshgrid(x, y)

    x_pixels = np.linspace(0, img_width - 1, subdivisions_x).astype(int)
    y_pixels = np.linspace(0, img_height - 1, subdivisions_y).astype(int)
    x_pixels, y_pixels = np.meshgrid(x_pixels, y_pixels)
    z_top = depth_array[y_pixels, x_pixels] * depth_scale

    # Apply alpha mask to Z values for the top surface
    alpha_grid = alpha[y_pixels, x_pixels]
    alpha_grid_bottom = alpha_bottom[y_pixels, x_pixels]
    z_top = z_top * (alpha_grid > 0)  # Set Z to 0 where alpha is 0

    # Shift Z so the lowest non-zero point of the top surface is at Z=thickness
    non_zero_z = z_top[z_top > 0]
    if non_zero_z.size > 0:
        z_min = non_zero_z.min()
        z_top = z_top - z_min + thickness
        z_top = z_top * (alpha_grid > 0)  # Ensure Z remains 0 in transparent areas

    # Create top surface vertices
    top_vertices = np.stack([x, y, z_top], axis=-1).reshape(-1, 3)

    # Create bottom surface vertices at Z=0 (flat bottom)
    bottom_vertices = np.stack([x, y, np.zeros_like(z_top)], axis=-1).reshape(-1, 3)

    # Combine vertices (top + bottom)
    all_vertices = np.vstack([top_vertices, bottom_vertices])
    num_vertices_per_layer = subdivisions_x * subdivisions_y

    # Update progress (vertex creation: 5% total)
    if progress:
        progress.update(5)

    # Create faces for the top surface, excluding transparent areas
    top_faces = []
    alpha_threshold = 0.1
    for i in range(subdivisions_y - 1):
        for j in range(subdivisions_x - 1):
            v0 = i * subdivisions_x + j
            v1 = v0 + 1
            v2 = (i + 1) * subdivisions_x + j
            v3 = v2 + 1
            if (alpha_grid[i, j] > alpha_threshold and
                alpha_grid[i, j+1] > alpha_threshold and
                alpha_grid[i+1, j] > alpha_threshold and
                alpha_grid[i+1, j+1] > alpha_threshold):
                top_faces.append([v0, v1, v2])
                top_faces.append([v1, v3, v2])

    # Create faces for the bottom surface, using the wider alpha mask
    bottom_faces = []
    for i in range(subdivisions_y - 1):
        for j in range(subdivisions_x - 1):
            v0 = num_vertices_per_layer + i * subdivisions_x + j
            v1 = v0 + 1
            v2 = num_vertices_per_layer + (i + 1) * subdivisions_x + j
            v3 = v2 + 1
            if (alpha_grid_bottom[i, j] > 0 and
                alpha_grid_bottom[i, j+1] > 0 and
                alpha_grid_bottom[i+1, j] > 0 and
                alpha_grid_bottom[i+1, j+1] > 0):
                bottom_faces.append([v0, v2, v1])
                bottom_faces.append([v1, v2, v3])

    # Update progress (face creation: 7.5% total)
    if progress:
        progress.update(7.5)

    # Find boundary vertices for side faces using the wider bottom alpha mask
    side_faces = []
    # Top edge (y = -height/2)
    for j in range(subdivisions_x - 1):
        top_v0 = j
        top_v1 = j + 1
        bot_v0 = num_vertices_per_layer + j
        bot_v1 = num_vertices_per_layer + j + 1
        if (alpha_grid[0, j] > alpha_threshold and alpha_grid[0, j+1] > alpha_threshold and
            alpha_grid_bottom[0, j] > 0 and alpha_grid_bottom[0, j+1] > 0):
            side_faces.append([top_v0, bot_v0, top_v1])
            side_faces.append([top_v1, bot_v0, bot_v1])

    # Bottom edge (y = height/2)
    for j in range(subdivisions_x - 1):
        top_v0 = (subdivisions_y - 1) * subdivisions_x + j
        top_v1 = top_v0 + 1
        bot_v0 = num_vertices_per_layer + (subdivisions_y - 1) * subdivisions_x + j
        bot_v1 = bot_v0 + 1
        if (alpha_grid[subdivisions_y-1, j] > alpha_threshold and alpha_grid[subdivisions_y-1, j+1] > alpha_threshold and
            alpha_grid_bottom[subdivisions_y-1, j] > 0 and alpha_grid_bottom[subdivisions_y-1, j+1] > 0):
            side_faces.append([top_v0, top_v1, bot_v0])
            side_faces.append([top_v1, bot_v1, bot_v0])

    # Left edge (x = -width/2)
    for i in range(subdivisions_y - 1):
        top_v0 = i * subdivisions_x
        top_v1 = (i + 1) * subdivisions_x
        bot_v0 = num_vertices_per_layer + i * subdivisions_x
        bot_v1 = num_vertices_per_layer + (i + 1) * subdivisions_x
        if (alpha_grid[i, 0] > alpha_threshold and alpha_grid[i+1, 0] > alpha_threshold and
            alpha_grid_bottom[i, 0] > 0 and alpha_grid_bottom[i+1, 0] > 0):
            side_faces.append([top_v0, top_v1, bot_v0])
            side_faces.append([top_v1, bot_v1, bot_v0])

    # Right edge (x = width/2)
    for i in range(subdivisions_y - 1):
        top_v0 = i * subdivisions_x + (subdivisions_x - 1)
        top_v1 = (i + 1) * subdivisions_x + (subdivisions_x - 1)
        bot_v0 = num_vertices_per_layer + i * subdivisions_x + (subdivisions_x - 1)
        bot_v1 = num_vertices_per_layer + (i + 1) * subdivisions_x + (subdivisions_x - 1)
        if (alpha_grid[i, subdivisions_x-1] > alpha_threshold and alpha_grid[i+1, subdivisions_x-1] > alpha_threshold and
            alpha_grid_bottom[i, subdivisions_x-1] > 0 and alpha_grid_bottom[i+1, subdivisions_x-1] > 0):
            side_faces.append([top_v0, bot_v0, top_v1])
            side_faces.append([top_v1, bot_v0, bot_v1])

    # Additional side faces to connect top and bottom boundaries
    for i in range(subdivisions_y - 1):
        for j in range(subdivisions_x - 1):
            # Check transitions from non-zero to zero alpha in top surface
            top_v0 = i * subdivisions_x + j
            top_v1 = top_v0 + 1
            top_v2 = (i + 1) * subdivisions_x + j
            top_v3 = top_v2 + 1
            bot_v0 = num_vertices_per_layer + i * subdivisions_x + j
            bot_v1 = bot_v0 + 1
            bot_v2 = num_vertices_per_layer + (i + 1) * subdivisions_x + j
            bot_v3 = bot_v2 + 1

            # Top edge of quad (check if previous row is transparent or i is at the boundary)
            if (alpha_grid[i, j] > alpha_threshold and alpha_grid[i, j+1] > alpha_threshold and
                alpha_grid_bottom[i, j] > 0 and alpha_grid_bottom[i, j+1] > 0):
                if i == 0 or (alpha_grid[i-1, j] <= alpha_threshold or alpha_grid[i-1, j+1] <= alpha_threshold):
                    side_faces.append([top_v0, bot_v0, top_v1])
                    side_faces.append([top_v1, bot_v0, bot_v1])

            # Bottom edge of quad (check if next row is transparent or i is at the boundary)
            if (alpha_grid[i+1, j] > alpha_threshold and alpha_grid[i+1, j+1] > alpha_threshold and
                alpha_grid_bottom[i+1, j] > 0 and alpha_grid_bottom[i+1, j+1] > 0):
                if i == subdivisions_y-2 or (i+2 < subdivisions_y and (alpha_grid[i+2, j] <= alpha_threshold or alpha_grid[i+2, j+1] <= alpha_threshold)):
                    side_faces.append([top_v2, top_v3, bot_v2])
                    side_faces.append([top_v3, bot_v3, bot_v2])

            # Left edge of quad (check if previous column is transparent or j is at the boundary)
            if (alpha_grid[i, j] > alpha_threshold and alpha_grid[i+1, j] > alpha_threshold and
                alpha_grid_bottom[i, j] > 0 and alpha_grid_bottom[i+1, j] > 0):
                if j == 0 or (alpha_grid[i, j-1] <= alpha_threshold or alpha_grid[i+1, j-1] <= alpha_threshold):
                    side_faces.append([top_v0, top_v2, bot_v0])
                    side_faces.append([top_v2, bot_v2, bot_v0])

            # Right edge of quad (check if next column is transparent or j is at the boundary)
            if (alpha_grid[i, j+1] > alpha_threshold and alpha_grid[i+1, j+1] > alpha_threshold and
                alpha_grid_bottom[i, j+1] > 0 and alpha_grid_bottom[i+1, j+1] > 0):
                if j == subdivisions_x-2 or (j+2 < subdivisions_x and (alpha_grid[i, j+2] <= alpha_threshold or alpha_grid[i+1, j+2] <= alpha_threshold)):
                    side_faces.append([top_v1, bot_v1, top_v3])
                    side_faces.append([top_v3, bot_v1, bot_v3])

    # Combine all faces
    all_faces = np.vstack([top_faces, bottom_faces, side_faces])

    # Create initial mesh
    mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)

    # Convert to Open3D mesh for smoothing and simplification
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(all_vertices),
        triangles=o3d.utility.Vector3iVector(all_faces)
    )

    # Update progress (mesh creation: 2.5% total)
    if progress:
        progress.update(2.5)

    # Apply Laplacian smoothing with progress updates per iteration
    if smooth_iterations > 0:
        increment = 15.0 / smooth_iterations
        for _ in range(smooth_iterations):
            o3d_mesh = o3d_mesh.filter_smooth_laplacian(number_of_iterations=1)
            if progress:
                progress.update(increment)

    # Simplify mesh if target_triangle_count is specified
    if target_triangle_count is not None:
        current_triangle_count = len(o3d_mesh.triangles)
        if target_triangle_count < current_triangle_count:
            num_steps = 10
            target_reduction = current_triangle_count - target_triangle_count
            step_reduction = target_reduction / num_steps
            increment = 10.0 / num_steps
            for i in range(num_steps):
                temp_target = int(current_triangle_count - (i + 1) * step_reduction)
                if temp_target < target_triangle_count:
                    temp_target = target_triangle_count
                o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=temp_target)
                if progress:
                    progress.update(increment)
            print(f"Mesh simplified from {current_triangle_count} to {len(o3d_mesh.triangles)} triangles")
        else:
            print(f"No simplification needed: target {target_triangle_count} >= current {current_triangle_count} triangles")
            if progress:
                progress.update(10)

    # Convert back to trimesh
    final_mesh = trimesh.Trimesh(
        vertices=np.asarray(o3d_mesh.vertices),
        faces=np.asarray(o3d_mesh.triangles)
    )

    # Update progress (conversion: 1.25% total)
    if progress:
        progress.update(1.25)

    # Export final mesh
    final_mesh.export(output_full_path)
    print(f"Solid 3D model saved to {output_full_path} with {len(final_mesh.faces)} triangles")

    # Update progress (export: 1.25% total)
    if progress:
        progress.update(1.25)