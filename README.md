# MSTGNet: Multi-Scale Temporal Graph Network for Video-Level Deception Detection from Multimodal Behavioral Landmarks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Dataset](https://img.shields.io/badge/Dataset-Zenodo-blue)](https://doi.org/10.5281/zenodo.17421590)
[![Paper](https://img.shields.io/badge/Paper-Knowledge--Based%20Systems-green)](https://www.journals.elsevier.com/knowledge-based-systems)

> Official implementation of **MSTGNet**, submitted to *Knowledge-Based Systems* (Elsevier), Manuscript No. KNOSYS-D-26-02299.

---

## 📌 Overview

**MSTGNet** is a local-to-meso temporal graph network for video-level deception detection using multimodal behavioral landmarks. It captures multi-scale temporal dynamics from facial landmarks, body pose, iris movement, and audio features to classify deceptive versus truthful behavior in interview-style videos.

Key contributions:
- **Multi-Scale Temporal Graph Network** that models short-to-medium range behavioral dynamics
- **Multimodal fusion** of visual landmarks (face, body, iris) and audio features
- **Video-level evidence aggregation** for robust deception classification
- **Cross-cultural evaluation** on Indonesian (I3D) and English (Real-Life Trial) datasets
- **Reproducible experiments** with full ablation, sensitivity, and statistical robustness analysis

---

## 📂 Repository Structure

```
MSTGNet/
├── 01_Processing_Data.ipynb                          # Data preprocessing pipeline
├── 02_MSTGNet.ipynb                                  # Main MSTGNet model definition
├── 03_B1_MSTGNet_S1_MAIN_CV.ipynb                   # Main cross-validation experiment
├── 03_B2_MSTGNet_S2_BASELINES.ipynb                 # Baseline comparison (ML & DL)
├── 03_B2B_MSTGNet_S5_LANDMARK_QUALITY.ipynb         # Landmark temporal stability analysis
├── 03_B3_MSTGNet_S3_ARCH_ABLATION.ipynb             # Architecture ablation study
├── 03_B4_MSTGNet_S3B_TEMPORAL_SENSITIVITY.ipynb     # Temporal window sensitivity
├── 03_B5_MSTGNet_S4_MODAL_ABLATION.ipynb            # Modality ablation study
├── 03_B6_MSTGNet_S6_CROSS_DATASET.ipynb             # Cross-dataset transfer analysis
├── 03_B7_MSTGNet_S7_MULTI_SEED.ipynb                # Multi-seed robustness (5 seeds)
├── 03_B8_MSTGNet_S10_INTERPRETABILITY.ipynb         # Interpretability & visualization
├── 03_B9_MSTGNet_S8_COMPUTE_COST_S9_CALIBRATION_EXPORT.ipynb  # Compute cost & calibration
├── 03_B10_MSTGNet_S11_EXPORT_PAPER_TABLES.ipynb     # Export all paper tables
├── model_base.py                                     # Base model architecture
├── LICENSE                                           # MIT License
└── README.md                                         # This file
```

---

## 📦 Dataset

The derived **Indonesian I3D Dataset** features used in this study are publicly available:

| Resource | Link |
|----------|------|
| 🗄️ Derived features (landmarks, audio, transcripts, metadata) | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17421590.svg)](https://doi.org/10.5281/zenodo.17421590) |
| 📹 Raw videos | Not publicly released (participant privacy) |

The I3D dataset consists of **1,568 recordings** and **647,871 frames** collected from Indonesian participants in a controlled interview setting.

---

## ⚙️ Installation

### Requirements
- Python 3.8+
- PyTorch 1.12+
- Jupyter Notebook / JupyterLab

### Install Dependencies

```bash
git clone https://github.com/yenidwirahayu/MSTGNet.git
cd MSTGNet
pip install -r requirements.txt
```

> **Note:** `requirements.txt` will be added. Core dependencies include:
> `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`, `tqdm`

---

## 🚀 Usage

### 1. Data Preprocessing
```bash
jupyter notebook 01_Processing_Data.ipynb
```

### 2. Train MSTGNet
```bash
jupyter notebook 02_MSTGNet.ipynb
```

### 3. Run Experiments
Run notebooks in order (`03_B1` → `03_B10`) to reproduce all paper results:

| Notebook | Experiment |
|----------|------------|
| `03_B1` | Main cross-validation (Table 2, 3) |
| `03_B2` | Baseline comparison (Table 2) |
| `03_B2B` | Landmark quality analysis (Section 4.3) |
| `03_B3` | Architecture ablation (Table 3) |
| `03_B4` | Temporal window sensitivity (Section 4.3) |
| `03_B5` | Modality ablation (Table 4) |
| `03_B6` | Cross-dataset transfer (Table 5) |
| `03_B7` | Multi-seed robustness — 5 seeds, Wilcoxon + Holm–Bonferroni (Table 6) |
| `03_B8` | Interpretability & attention visualization (Section 4.8) |
| `03_B9` | Compute cost & calibration (Section 4.7) |
| `03_B10` | Export all paper tables |

---

## 📊 Main Results

### I3D Dataset (Indonesian)
| Model | AUC | F1 |
|-------|-----|----|
| SVM (baseline) | — | — |
| Transformer (baseline) | — | — |
| **MSTGNet (ours)** | **—** | **—** |

> Full results available in the paper and reproducible via `03_B1_MSTGNet_S1_MAIN_CV.ipynb`

---

## 📖 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{mstgnet2026,
  title     = {MSTGNet: Multi-Scale Temporal Graph Network for Video-Level 
               Deception Detection from Multimodal Behavioral Landmarks},
  author    = {Rahayu, Yeni Dwi and Fatichah, Chastine and others},
  journal   = {Knowledge-Based Systems},
  year      = {2026},
  publisher = {Elsevier},
  note      = {Manuscript No. KNOSYS-D-26-02299}
}
```

---

## 👥 Authors

| Name | Affiliation |
|------|-------------|
| Yeni Dwi Rahayu | Institut Teknologi Sepuluh Nopember, Surabaya, Indonesia |
| Chastine Fatichah *(Corresponding)* | Institut Teknologi Sepuluh Nopember, Surabaya, Indonesia |

📧 Corresponding author: chastine@its.ac.id

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

This research was supported by [institution/funding]. The I3D dataset was collected with ethical approval and informed consent from all participants.

---

*This repository supports the reproducibility of results reported in the manuscript submitted to Knowledge-Based Systems (Elsevier).*
