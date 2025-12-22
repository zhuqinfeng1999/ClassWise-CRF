import os
import time
import argparse
import numpy as np
from PIL import Image
import torch
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian, create_pairwise_bilateral
import optuna
import torch.multiprocessing as mp
import logging

# Set multiprocessing start method to 'spawn' for compatibility
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

# Configure logging
log_file_path = r"/ZQFSSD/crf/OUTPUT/LOVEDA/NPY/3_opt_weight_e2.5/3_opt_weight_e2.5.log"
log_dir = os.path.dirname(log_file_path)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(log_file_path)])

# 1. Define classes and palette
CLASSES = ('background', 'building', 'road', 'water', 'barren', 'forest', 'agricultural')
PALETTE = np.array([[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255],
                    [159, 129, 183], [0, 255, 0], [255, 195, 128]], dtype=np.uint8)
NUM_CLASSES = len(CLASSES)
IGNORE_LABEL = 255

# 2. Path configuration
net1_dir = r"/ZQFSSD/crf/DATA/loveda/convnexttpre_npy"  # Network 1 predictions (.npy)
net2_dir = r"/ZQFSSD/crf/DATA/loveda/vmambatpre_npy"    # Network 2 predictions (.npy)
net3_dir = r"/ZQFSSD/crf/DATA/loveda/swintpre_npy"      # Network 3 predictions (.npy)
images_dir = r"/ZQFSSD/crf/DATA/loveda/ori"             # Original images
gt_dir = r"/ZQFSSD/crf/DATA/loveda/lovedagt"            # Ground truth
output_dir = r"/ZQFSSD/crf/OUTPUT/LOVEDA/NPY/3_opt_weight_e2.5"  # CRF optimized results

# GPU device and palette tensor
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PALETTE_TENSOR = torch.tensor(PALETTE, dtype=torch.uint8, device=DEVICE)
PALETTE_CODES = (PALETTE_TENSOR[:, 0].int() << 16) | (PALETTE_TENSOR[:, 1].int() << 8) | PALETTE_TENSOR[:, 2].int()

# IoU scores for each class
iou_net1 = np.array([55.58, 66.08, 56.32, 68.61, 32.82, 42.33, 55.88])  # ConvNext-T
iou_net2 = np.array([55.29, 65.08, 56.89, 71.31, 33.68, 40.17, 53.52])  # VMamba-T
iou_net3 = np.array([54.61, 65.37, 55.98, 69.79, 31.69, 44.14, 52.19])  # Swin-T

# Hyperparameter alpha for controlling exponential weighting strength
alpha = 2.5  # Updated to 2.5 as per the output directory

# Compute weights for each class using exponential weighting for three networks
total_weights = iou_net1 ** alpha + iou_net2 ** alpha + iou_net3 ** alpha
weights_net1 = (iou_net1 ** alpha) / total_weights
weights_net2 = (iou_net2 ** alpha) / total_weights
weights_net3 = (iou_net3 ** alpha) / total_weights

# 3. Fuse probabilities from three models (IoU-based exponential weighting)
def fuse_probabilities(net1_prob, net2_prob, net3_prob, weights_net1, weights_net2, weights_net3):
    fused_prob = np.zeros_like(net1_prob)
    for c in range(NUM_CLASSES):
        fused_prob[c] = (weights_net1[c] * net1_prob[c] +
                         weights_net2[c] * net2_prob[c] +
                         weights_net3[c] * net3_prob[c])
    # Normalize to maintain probability distribution
    fused_prob /= fused_prob.sum(axis=0, keepdims=True)
    return fused_prob

# 4. Apply CRF post-processing (CPU)
def apply_crf(image, prob, num_classes, gaussian_sdims, gaussian_compat, bilateral_sdims, bilateral_schan,
              bilateral_compat, iterations=5):
    h, w = image.shape[:2]
    d = dcrf.DenseCRF2D(w, h, num_classes)
    unary = unary_from_softmax(prob)
    d.setUnaryEnergy(unary)
    feats_gaussian = create_pairwise_gaussian(sdims=gaussian_sdims, shape=(h, w))
    d.addPairwiseEnergy(feats_gaussian, compat=gaussian_compat)
    feats_bilateral = create_pairwise_bilateral(sdims=bilateral_sdims, schan=bilateral_schan, img=image, chdim=2)
    d.addPairwiseEnergy(feats_bilateral, compat=bilateral_compat)
    Q = d.inference(iterations)
    return np.argmax(Q, axis=0).reshape((h, w))

# 5. Process a single image (for multithreading)
def process_image(args):
    net1_path, net2_path, net3_path, image_path, output_path, gaussian_sdims, gaussian_compat, bilateral_sdims, bilateral_schan, bilateral_compat, weights_net1, weights_net2, weights_net3 = args
    # Load probabilities from .npy files
    net1_prob = np.load(net1_path)  # Shape: (C, H, W)
    net2_prob = np.load(net2_path)  # Shape: (C, H, W)
    net3_prob = np.load(net3_path)  # Shape: (C, H, W)
    image = np.array(Image.open(image_path).convert('RGB'))

    # Fuse probabilities
    fused_prob = fuse_probabilities(net1_prob, net2_prob, net3_prob, weights_net1, weights_net2, weights_net3)

    # Apply CRF
    crf_result = apply_crf(image, fused_prob, NUM_CLASSES, gaussian_sdims, gaussian_compat,
                           bilateral_sdims, bilateral_schan, bilateral_compat)

    # Save result
    crf_rgb = PALETTE[crf_result]
    crf_img = Image.fromarray(crf_rgb)
    crf_img.save(output_path)

# 6. Compute confusion matrix (GPU-accelerated)
def compute_confusion_matrix(gt_label, pred_label, mask, num_classes):
    valid = mask & (gt_label != IGNORE_LABEL)
    gt_flat = gt_label[valid].long()
    pred_flat = pred_label[valid].long()
    indices = gt_flat * num_classes + pred_flat
    confusion = torch.bincount(indices, minlength=num_classes * num_classes)
    return confusion.view(num_classes, num_classes)

# 7. Compute IoU and mIoU
def compute_iou(confusion):
    intersection = np.diag(confusion)
    gt_sum = confusion.sum(axis=1)
    pred_sum = confusion.sum(axis=0)
    union = gt_sum + pred_sum - intersection
    iou = intersection / (union + 1e-10)
    miou = np.nanmean(iou)
    return iou, miou

# 8. Compute mIoU for directory (GPU-accelerated)
def compute_miou_for_dir(gt_dir, pred_dir):
    gt_files = sorted(f for f in os.listdir(gt_dir) if f.endswith('.png'))
    pred_files = sorted(f for f in os.listdir(pred_dir) if f.endswith('.png'))
    assert len(gt_files) == len(pred_files), "文件数量不匹配"

    total_confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64, device=DEVICE)

    for gt_file, pred_file in zip(gt_files, pred_files):
        gt_path = os.path.join(gt_dir, gt_file)
        pred_path = os.path.join(pred_dir, pred_file)

        gt_image = torch.from_numpy(np.array(Image.open(gt_path).convert('RGB'))).to(DEVICE)
        pred_image = torch.from_numpy(np.array(Image.open(pred_path).convert('RGB'))).to(DEVICE)

        mask = (gt_image != torch.tensor([0, 0, 0], device=DEVICE)).any(dim=-1)
        gt_label = rgb_to_label_gpu(gt_image)
        pred_label = rgb_to_label_gpu(pred_image)

        confusion = compute_confusion_matrix(gt_label, pred_label, mask, NUM_CLASSES)
        total_confusion += confusion

    ious, miou = compute_iou(total_confusion.cpu().numpy())
    return ious, miou

# 9. RGB to label conversion (GPU-accelerated)
def rgb_to_label_gpu(rgb_tensor):
    h, w = rgb_tensor.shape[:2]
    rgb_codes = (rgb_tensor[..., 0].int() << 16) | (rgb_tensor[..., 1].int() << 8) | rgb_tensor[..., 2].int()
    rgb_codes = rgb_codes.view(-1).to(DEVICE)
    matches = (rgb_codes.unsqueeze(1) == PALETTE_CODES.unsqueeze(0))
    label_flat = torch.argmax(matches.int(), dim=1)
    label_flat[~matches.any(dim=1)] = IGNORE_LABEL
    return label_flat.view(h, w)

# 10. Bayesian optimization objective function
def objective(trial):
    gaussian_sdims = (trial.suggest_int('gaussian_sdims_x', 3, 30),
                      trial.suggest_int('gaussian_sdims_y', 3, 30))
    gaussian_compat = trial.suggest_int('gaussian_compat', 1, 3)
    bilateral_sdims = (trial.suggest_int('bilateral_sdims_x', 5, 25),
                       trial.suggest_int('bilateral_sdims_y', 5, 25))
    bilateral_schan = (trial.suggest_int('bilateral_schan_r', 5, 35),
                       trial.suggest_int('bilateral_schan_g', 5, 35),
                       trial.suggest_int('bilateral_schan_b', 5, 35))
    bilateral_compat = trial.suggest_int('bilateral_compat', 1, 5)

    temp_output_dir = os.path.join(output_dir, f"trial_{trial.number}")
    if not os.path.exists(temp_output_dir):
        os.makedirs(temp_output_dir)

    # Prepare multithreading parameters, adding weights_net3
    image_files = [f for f in os.listdir(net1_dir) if f.endswith('.npy')]
    args_list = [
        (os.path.join(net1_dir, f), os.path.join(net2_dir, f), os.path.join(net3_dir, f),
         os.path.join(images_dir, f.replace('.npy', '.png')),
         os.path.join(temp_output_dir, f.replace('.npy', '.png')),
         gaussian_sdims, gaussian_compat, bilateral_sdims, bilateral_schan, bilateral_compat,
         weights_net1, weights_net2, weights_net3)
        for f in image_files
    ]

    # Process images with multithreading
    with mp.Pool(processes=4) as pool:  # Adjust based on your CPU
        pool.map(process_image, args_list)

    # Compute mIoU
    ious, miou = compute_miou_for_dir(gt_dir, temp_output_dir)
    return miou

# 11. Main program
if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    logging.info("\nBest trial:")
    best_trial = study.best_trial
    logging.info(f"  Best mIoU: {best_trial.value:.4f}")
    logging.info("  Best Params:")
    for key, value in best_trial.params.items():
        logging.info(f"    {key}: {value}")