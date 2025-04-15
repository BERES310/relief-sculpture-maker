from tools.photo_to_3d_model import photo_to_3d_model

if __name__ == "__main__":
    image_path = "lion.png"
    output_stl_path = "output_model.stl"
    depth_map_path = "depth_map.png"
    preprocessed_path = "preprocessed_image.png"
    output_dir = "output_files"
    photo_to_3d_model(
        image_path,
        output_stl_path,
        depth_map_path,
        preprocessed_path,
        output_dir,
        width=10.0,
        height=10.0,
        depth_scale=5.0,
        depth_map_width=None,
        depth_map_height=None,
        smooth_iterations=10,
        target_triangle_count=None,
        thickness=5.0
    )
    
    print("3D model generation completed.")