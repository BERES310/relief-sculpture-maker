from rembg import remove
import os



def preprocess_image(image_path, output_preprocessed_path, output_dir="output_files"):
    """
    Preprocess the input image by removing the background.
    
    Parameters:
    - image_path: Path to the input photo (JPG/PNG).
    - output_preprocessed_path: Name of the preprocessed image file (e.g., 'preprocessed_image.png').
    - output_dir: Subfolder to save the preprocessed image (default: 'output_files').
    """
    os.makedirs(output_dir, exist_ok=True)
    output_preprocessed_full_path = os.path.join(output_dir, output_preprocessed_path)
    
    with open(image_path, 'rb') as img_file:
        img_data = img_file.read()
    
    img_no_bg = remove(img_data)
    
    with open(output_preprocessed_full_path, 'wb') as out_file:
        out_file.write(img_no_bg)
    print(f"Preprocessed image (background removed) saved to {output_preprocessed_full_path}")

    return output_preprocessed_full_path
