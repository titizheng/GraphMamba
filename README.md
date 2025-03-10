# Dynamic Policy-Driven Adaptive Multi-Instance Learning for Whole Slide Image Classification


[Tingting Zheng](https://scholar.google.com/citations?user=AJ5zl-wAAAAJ&hl=zh-CN), [Hongxun Yao](https://scholar.google.com/citations?user=aOMFNFsAAAAJ), [Sicheng Zhao](https://scholar.google.com/citations?user=LJiQRJIAAAAJ&hl=zh-CN&oi=ao)., 
[Kui jiang](https://scholar.google.com/citations?user=AbOLE9QAAAAJ&hl=en&oi=ao), [Yi Xiao](https://scholar.google.com/citations?user=e3a4aG0AAAAJ).

 


**Abstract:** Multi-instance learning (MIL) has demonstrated promising performance in whole slide image (WSI) analysis. However, existing transformer-based methods struggle back and forth between global representation capability and quadratic complexity, 
particularly when handling millions of instances. Recently, the selective state space model (Mamba) has emerged as a promising alternative for modeling long-range dependencies with linear complexity. Nonetheless, 
WSI remains challenging for Mamba due to its inability to capture complex local tissue and structural patterns, which is crucial for accurate tumor region recognition. To this end, we approach WSI classification from a graph-based perspective 
and present GraphMamba, a novel method that constructs multi-level graphs across instances. GraphMamba involves two key components: intra-group graph mamba (IGM) to grasp instance-level dependencies, and cross-group graph mamba (CGM) for 
exploring group-level relationships. In particular, before aggregating group features into a comprehensive bag representation, CGM utilizes a cross-group feature sampling scheme to extract the most informative features across groups, 
enabling compact and discriminative representations.
Extensive experiments on four datasets demonstrate that GraphMamba outperforms state-of-the-art ACMIL method by 0.5\%, 3.1\%, 2.6\%, and 3.0\% in accuracy on the TCGA BRCA, TCGA Lung, TCGA ESCA, and BRACS datasets.


## Update
- [2025/03/10] Uploading groups and building graph ways

## Pre-requisites:
* Linux (Tested on Ubuntu 18.04)
* NVIDIA GPU (Tested on 3090)

## Dependencies:
```bash
torch
torchvision
numpy
h5py
scipy
scikit-learning
pandas
nystrom_attention
admin_torch
```





The data used for training, validation and testing are expected to be organized as follows:
```bash
DATA_ROOT_DIR/
    ├──DATASET_1_DATA_DIR/
        └── pt_files
                ├── slide_1.pt
                ├── slide_2.pt
                └── ...
        └── h5_files
                ├── slide_1.h5
                ├── slide_2.h5
                └── ...
    ├──DATASET_2_DATA_DIR/
        └── pt_files
                ├── slide_a.pt
                ├── slide_b.pt
                └── ...
        └── h5_files
                ├── slide_i.h5
                ├── slide_ii.h5
                └── ...
    └── ...
```
