# ClassWise-CRF: Category-Specific Fusion for Enhanced Semantic Segmentation of Remote Sensing Imagery

Code for the paper "ClassWise-CRF: Category-Specific Fusion for Enhanced Semantic Segmentation of Remote Sensing Imagery."  
Accepted by *Neural Networks* (DOI forthcoming).  
arXiv: https://arxiv.org/abs/2504.21491

## Abstract
With the continuous development of visual models such as Convolutional Neural Networks, Vision Transformers, and Vision Mamba, the capabilities of neural networks in semantic segmentation of remote sensing images have seen significant progress. However, these networks exhibit varying performance across different semantic categories, making it challenging to find a single network architecture that excels in all categories. To address this, we propose a result-level category-specific fusion architecture called ClassWise-CRF. This architecture employs a two-stage process: first, it selects expert networks that perform well in specific categories from a pool of candidate networks using a greedy algorithm; second, it integrates the segmentation predictions of these selected networks by adaptively weighting their contributions based on their segmentation performance in each category. Inspired by Conditional Random Field (CRF), the ClassWise-CRF architecture treats the segmentation predictions from multiple networks as confidence vector fields. It leverages segmentation metrics (such as Intersection over Union) from the validation set as priors and employs an exponential weighting strategy to fuse the category-specific confidence scores predicted by each network. This fusion method dynamically adjusts the weights of each network for different categories, achieving category-specific optimization. Building on this, the architecture further optimizes the fused results using unary and pairwise potentials in CRF to ensure spatial consistency and boundary accuracy. To validate the effectiveness of ClassWise-CRF, we conducted experiments on two remote sensing datasets, LoveDA and Vaihingen, using eight classic and advanced semantic segmentation networks. The results show that the ClassWise-CRF architecture significantly improves segmentation performance: on the LoveDA dataset, the mean Intersection over Union (mIoU) metric increased by 1.00% on the validation set and by 0.68% on the test set; on the Vaihingen dataset, the mIoU improved by 0.87% on the validation set and by 0.91% on the test set. These results fully demonstrate the effectiveness and generality of the ClassWise-CRF architecture in semantic segmentation of remote sensing images. The full code is available at https://github.com/zhuqinfeng1999/ClassWise-CRF.

## Installation
**Requirements**
- Ubuntu 20.04
- CUDA 12.4

## Dataset Preparation
LoveDA can be found here: https://github.com/Junjue-Wang/LoveDA

After downloading the dataset, place data and predictions under `DATA/` and keep the folder names consistent with the scripts (or update the paths in the scripts):

```
DATA/
  loveda/
    ori/               # original images
    lovedagt/          # ground truth masks
    convnexttpre_npy/  # ConvNeXt predictions (.npy)
    swintpre_npy/      # Swin predictions (.npy)
    vmambatpre_npy/    # VMamba predictions (.npy)
  vaihingen/
    ori/
    vaihingengt/
    convnexttpre_npy/
    swintpre_npy/
    vmambatpre_npy/
```

Note: many scripts use absolute paths like `/ZQFSSD/crf/DATA/...`; please update those paths to your local dataset location.

## Usage
Run individual fusion scripts from `NPY_WEIGHT_E_FUSE`:

```bash
cd NPY_WEIGHT_E_FUSE
python 3_bayesian_loveda_e2.5.py
```

Run multiple experiments with `BASH.py` (update the script list and log path as needed):

```bash
python BASH.py
```

## Citation
Zhu, Qinfeng, Yunxi Jiang, and Lei Fan. "ClassWise-CRF: Category-Specific Fusion for Enhanced Semantic Segmentation of Remote Sensing Imagery." arXiv preprint arXiv:2504.21491 (2025).

```bibtex
@article{zhu2025classwise,
  title={ClassWise-CRF: Category-Specific Fusion for Enhanced Semantic Segmentation of Remote Sensing Imagery},
  author={Zhu, Qinfeng and Jiang, Yunxi and Fan, Lei},
  journal={arXiv preprint arXiv:2504.21491},
  year={2025}
}
```
