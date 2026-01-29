# LSQ-DETR
[![DOI](https://zenodo.org/badge/1144822364.svg)](https://doi.org/10.5281/zenodo.18410583)

Official implementation of LSQ-DETR

> **Associated Manuscript**: "LSQ-DETR:Layered and Scale-aware Query DETR for Tiny Object Detection"

> **Submitted to *The Visual Computer* (2026)**

> **Authors**: Xuyang Jia, Junfen Chen,Yi Wang, Bojun Xie

# Introduction
Tiny object detection remains a challenging task in computer vision due to the extremely limited pixel footprint and weak discriminative features of such objects. To address these issues, we propose LSQ-DETR, a novel detection framework that integrates a layered Top-K query strategy and a scale-weighted loss scheme to significantly improve tiny object detection performance. The layered Top-K query strategy distributes detection queries proportionally across multi-scale feature layers, thereby optimizing query utilization and enhancing cross-level feature collaboration. Meanwhile, the scale-weighted loss explicitly emphasizes smaller objects during training by adjusting gradient contributions according to object scale, leading to improved localization accuracy and recall. Extensive experiments on the AI-TOD-v2 and VisDrone benchmarks show consistent improvements: our method achieves a +0.4% mAP gain on AI-TOD-v2 and a +0.3% mAP gain on VisDrone, without increasing model complexity or computational overhead.

# Installation and Get Started
Required environments:

**Linux**

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

# Inference
```python
bash scripts/LSQ_eval.sh /path to your dataset /path to your checkpoint
```
# Citation
```bibtex
@article{Jia2026lsq,
  title={LSQ-DETR:Layered and Scale-aware Query DETR for Tiny Object Detection},
  author={Xuyang Jia and Junfen Chen and Yi Wang and Bojun Xie},
  journal={The Visual Computer},
  year={2026},
  note={Manuscript submitted for publication}
}
```
