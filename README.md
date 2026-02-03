# LSQ-DETR
[![DOI](https://zenodo.org/badge/1144822364.svg)](https://doi.org/10.5281/zenodo.18410583)

Official implementation of LSQ-DETR

> **Associated Manuscript**: "Boosting Tiny Object Detection Performance via Stratified Query Allocation and Weighted Loss Strategies"

> **Submitted to *The Visual Computer* (2026)**

> **Authors**: Xuyang Jia, Junfen Chen,Yi Wang, Bojun Xie

# Introduction
Tiny object detection presents significant challenges in computer vision due to limited pixel footprints and inherent weaknesses in feature representation. This study introduces LSQ-DETR, a novel method leveraging a layered Top-K query strategy and a scale-weighted loss scheme to enhance tiny object detection performance. The layered Top-K query strategy optimizes resource utilization by allocating queries proportionally across feature layers, while the scale-weighted loss prioritizes small objects during training, thereby improving gradient contributions and localization accuracy. Experiments on the AI-TOD and VisDrone datasets demonstrate consistent $mAP$ gains and improved recall for tiny objects without altering model complexity. These results underscore the effectiveness of our approach in addressing the critical challenges of tiny object detection.

# Installation and Get Started
Required environments:

**Python 3.9+**

**PyTorch 2.1+**

**CUDA 11.8+**

**GCC 5+**

**cocoapi-aitod**

Install:
```python
git clone https://github.com/jiaxy89-star/LSQ-DETR.git
cd LSQ-DETR
conda create -n lsqdetr python=3.9 --y
conda activate lsqdetr
bash install.sh
```
# Preparation

Please refer to AI-TOD(https://github.com/Chasel-Tsui/mmdet-aitod) for AI-TOD-v2.0 and AI-TOD-v1.0 dataset.

Download the pretrained model at Google Drive(https://drive.google.com/file/d/1EbmUhcdrv4vKKv0Ad-sZAJIbCFjSNtY3/view?usp=drive_link).

# Training
> Changed the pretrained model path in LSQ.sh
```python
CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/LSQ.sh /path to your dataset
```

# Evaluate
```python
bash scripts/LSQ_eval.sh /path to your dataset /path to your checkpoint
```
# Citation
```bibtex
@article{Jia2026lsq,
  title={Boosting Tiny Object Detection Performance via Stratified Query Allocation and Weighted Loss Strategies},
  author={Xuyang Jia and Junfen Chen and Yi Wang and Bojun Xie},
  journal={The Visual Computer},
  year={2026},
  note={Manuscript submitted for publication}
}
```
