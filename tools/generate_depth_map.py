import numpy as np
from PIL import Image
import torch
import cv2
import torchvision.transforms as transforms
import os


def generate_depth_map(image_path, output_depth_path, depth_map_width=None, depth_map_height=None, output_dir="output_files"):
    """
    Generate a depth map from an input image using MiDaS, masked by the alpha channel of the preprocessed image.
    
    Parameters:
    - image_path: Path to the preprocessed image (with alpha channel).
    - output_depth_path: Name of the depth map file (e.g., 'depth_map.png').
    - depth_map_width: Desired width of the depth map in pixels (default: input image width).
    - depth_map_height: Desired height of the depth map in pixels (default: input image height).
    - output_dir: Subfolder to save the depth map (default: 'output_files').
    """
    os.makedirs(output_dir, exist_ok=True)
    output_depth_full_path = os.path.join(output_dir, output_depth_path)
    
    # Load the preprocessed image to get the alpha channel
    img_with_alpha = Image.open(image_path).convert('RGBA')
    alpha = np.array(img_with_alpha.split()[-1]) / 255.0  # Alpha channel (0 to 1)

    # Load image for MiDaS
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if depth_map_width is None or depth_map_height is None:
        depth_map_height, depth_map_width = img.shape[:2]

    # Load MiDaS model
    model_type = "DPT_Large"
    midas = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True, trust_repo=True)
    midas.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((384, 384)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        depth = midas(img_tensor)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=(depth_map_height, depth_map_width),
            mode="bicubic",
            align_corners=False
        ).squeeze()

    depth = depth.cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min())

    # Mask the depth map with the alpha channel
    depth = depth * alpha  # Set depth to 0 in transparent areas

    # Save the masked depth map
    depth_img = Image.fromarray((depth * 255).astype(np.uint8))
    depth_img.save(output_depth_full_path)
    print(f"Depth map saved to {output_depth_full_path} with resolution {depth_map_width}x{depth_map_height}")

    return depth, alpha
