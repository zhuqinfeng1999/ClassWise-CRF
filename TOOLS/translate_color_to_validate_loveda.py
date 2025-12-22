import os
import numpy as np
from PIL import Image
import glob

# Define the palette mapping RGB colors to class indices
PALETTE = np.array([
    [255, 255, 255],  # Background - 0
    [255, 0, 0],  # Building - 1
    [255, 255, 0],  # Road - 2
    [0, 0, 255],  # Water - 3
    [159, 129, 183],  # Barren - 4
    [0, 255, 0],  # Forest - 5
    [255, 195, 128]  # Agriculture - 6
], dtype=np.uint8)


def rgb_to_label(img, palette):
    """
    Convert an RGB segmentation image to a label image using the provided palette.

    Args:
        img (numpy.ndarray): RGB image array of shape (H, W, 3).
        palette (numpy.ndarray): Array of RGB colors, shape (num_classes, 3).

    Returns:
        numpy.ndarray: Label image of shape (H, W) with class indices.
    """
    H, W, _ = img.shape
    label_img = np.zeros((H, W), dtype=np.uint8)
    for i, color in enumerate(palette):
        # Create a mask where the image matches the palette color
        mask = np.all(img == color, axis=2)
        label_img[mask] = i
    return label_img


# Define input and output directories
input_dir = "/ZQFSSD/crf/OUTPUT/LOVEDATEST/swint_vmambat_opt_weight_e2.5_again"
output_dir = "/ZQFSSD/crf/OUTPUT/LOVEDATEST/swint_vmambat_opt_weight_e2.5_again_nocolor"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get all PNG files from the input directory
png_files = glob.glob(os.path.join(input_dir, "*.png"))

# Process each image
for png_file in png_files:
    # Load the RGB image
    img = Image.open(png_file).convert('RGB')
    img_array = np.array(img)

    # Convert RGB to label image
    label_img = rgb_to_label(img_array, PALETTE)

    # Check for pixels that don't match any palette color (optional safety check)
    unmatched = np.sum(~np.any([np.all(img_array == color, axis=2) for color in PALETTE], axis=0))
    if unmatched > 0:
        print(f"Warning: {unmatched} pixels in {png_file} do not match any palette color.")

    # Convert label array to grayscale image
    label_img_pil = Image.fromarray(label_img, mode='L')

    # Save to output directory with the same filename
    output_path = os.path.join(output_dir, os.path.basename(png_file))
    label_img_pil.save(output_path)
    # print(f"Saved {output_path}")

# Verify the number of processed files matches the expected count
num_files = len(glob.glob(os.path.join(output_dir, "*.png")))
print(f"Total files generated: {num_files}")
if num_files != 1796:
    print(f"Warning: Expected 1796 files, but generated {num_files}.")