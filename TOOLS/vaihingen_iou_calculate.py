import os
import numpy as np
from PIL import Image
import torch

# Define classes and color palette for Vaihingen
CLASSES = ('impervious_surface', 'building', 'low_vegetation', 'tree', 'car', 'clutter')
PALETTE = np.array([[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
                    [255, 255, 0], [255, 0, 0]], dtype=np.uint8)
NUM_CLASSES = len(CLASSES)
IGNORE_LABEL = 255

# Hardcoded directories (replace these with your actual paths)
gt_dir = "/ZQFSSD/crf/DATA/vaihingen/vaihingengt"  # Directory containing ground truth images (.png)
output_dir = "/ZQFSSD/crf/OUTPUT/vaihingen/NPY/swint_vmambat_opt_weight_e3/trial_12"  # Directory containing predicted images (.tif)

# Set device to GPU if available
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Convert palette to a PyTorch tensor on the GPU
PALETTE_TENSOR = torch.tensor(PALETTE, dtype=torch.uint8, device=DEVICE)


def rgb_to_label_gpu(image, palette_tensor):
    """
    Convert an RGB image to a label map using the provided palette on GPU.

    Args:
        image (torch.Tensor): RGB image of shape (H, W, 3)
        palette_tensor (torch.Tensor): Palette tensor of shape (NUM_CLASSES, 3)

    Returns:
        torch.Tensor: Label map of shape (H, W) with class indices or IGNORE_LABEL
    """
    # Expand dimensions for broadcasting
    image = image.unsqueeze(2)  # Shape: (H, W, 1, 3)
    palette = palette_tensor.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, NUM_CLASSES, 3)

    # Check for matches across all channels
    matches = (image == palette).all(dim=3)  # Shape: (H, W, NUM_CLASSES)
    labels = torch.argmax(matches.int(), dim=2)  # Shape: (H, W)

    # Assign IGNORE_LABEL to pixels with no match
    no_match = ~matches.any(dim=2)  # Shape: (H, W)
    labels[no_match] = IGNORE_LABEL
    return labels


def compute_confusion_matrix_gpu(gt_labels, pred_labels, num_classes, ignore_label):
    """
    Compute the confusion matrix for ground truth and predicted labels on GPU.

    Args:
        gt_labels (torch.Tensor): Ground truth label map (H, W)
        pred_labels (torch.Tensor): Predicted label map (H, W)
        num_classes (int): Number of classes
        ignore_label (int): Label to ignore in evaluation

    Returns:
        torch.Tensor: Confusion matrix of shape (num_classes, num_classes)
    """
    # Filter out ignored labels
    valid = gt_labels != ignore_label
    gt_valid = gt_labels[valid]
    pred_valid = pred_labels[valid]

    # Compute indices for confusion matrix
    indices = gt_valid * num_classes + pred_valid

    # Use bincount for efficient counting
    confusion = torch.bincount(indices, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return confusion


def compute_iou(confusion):
    """
    Calculate IoU for each class and mIoU from the confusion matrix.

    Args:
        confusion (np.ndarray): Confusion matrix of shape (NUM_CLASSES, NUM_CLASSES)

    Returns:
        tuple: (iou, miou) where iou is an array of per-class IoUs and miou is the mean IoU
    """
    intersection = np.diag(confusion)
    gt_sum = confusion.sum(axis=1)
    pred_sum = confusion.sum(axis=0)
    union = gt_sum + pred_sum - intersection
    iou = intersection / (union + 1e-10)  # Avoid division by zero
    miou = np.nanmean(iou)  # Mean IoU, ignoring NaN values
    return iou, miou


def calculate_iou_miou(gt_dir, output_dir):
    """
    Calculate IoU and mIoU for all images in the given directories using GPU.

    Args:
        gt_dir (str): Path to directory with ground truth images (.png)
        output_dir (str): Path to directory with predicted images (.tif)

    Returns:
        tuple: (iou, miou) where iou is an array of per-class IoUs and miou is the mean IoU
    """
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.png')])
    pred_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.tif')])
    assert len(gt_files) == len(pred_files), "Number of ground truth and predicted files must match"

    # Initialize total confusion matrix on GPU
    total_confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=DEVICE)

    for gt_file, pred_file in zip(gt_files, pred_files):
        # Map .tif to corresponding .png filename
        expected_gt_file = os.path.splitext(pred_file)[0] + '.png'
        assert gt_file == expected_gt_file, f"File names do not match: {gt_file} vs {expected_gt_file}"

        gt_path = os.path.join(gt_dir, gt_file)
        pred_path = os.path.join(output_dir, pred_file)

        # Load images as PyTorch tensors directly on GPU
        gt_image = torch.from_numpy(np.array(Image.open(gt_path).convert('RGB'))).to(DEVICE)
        pred_image = torch.from_numpy(np.array(Image.open(pred_path).convert('RGB'))).to(DEVICE)

        # Convert RGB images to label maps on GPU
        gt_labels = rgb_to_label_gpu(gt_image, PALETTE_TENSOR)
        pred_labels = rgb_to_label_gpu(pred_image, PALETTE_TENSOR)

        # Compute confusion matrix for this pair on GPU
        conf = compute_confusion_matrix_gpu(gt_labels, pred_labels, NUM_CLASSES, IGNORE_LABEL)
        total_confusion += conf

    # Move total confusion matrix to CPU for IoU calculation
    total_confusion_cpu = total_confusion.cpu().numpy()
    iou, miou = compute_iou(total_confusion_cpu)
    return iou, miou


if __name__ == "__main__":
    # Calculate IoU and mIoU using the specified directories
    iou, miou = calculate_iou_miou(gt_dir, output_dir)

    # Print results
    print("Per-class IoUs:")
    for cls, iou_val in zip(CLASSES, iou):
        print(f"  {cls}: {iou_val:.4f}")
    print(f"mIoU: {miou:.4f}")