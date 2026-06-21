# AdaMOIC
### Adaptive Multi-Omics Integration for Cancer Subtype Classification

## Overview

AdaMOIC is a deep learning framework for cancer subtype classification using multi-omics data. The model integrates **mRNA**, **CNV**, and **RPPA** omics modalities through adaptive feature fusion and graph-based learning to capture complex biological relationships among cancer patients.

This work is inspired by the DeepMoIC framework and introduces improvements including adaptive fusion mechanisms, enhanced graph representation learning, and robust subtype discrimination.

---

## Key Features

- Multi-omics integration
  - mRNA Expression
  - Copy Number Variation (CNV)
  - Reverse Phase Protein Array (RPPA)

- Autoencoder-based feature extraction
- Attention-based latent feature fusion
- Patient Similarity Network (PSN) construction
- Graph Neural Network (GNN) based classification
- Prototype-based classifier
- Supervised Contrastive Learning
- Probability calibration and subtype refinement

---

## Datasets

The framework is evaluated on TCGA cancer datasets.

| Dataset | Description |
|----------|------------|
| BRCA | Breast Invasive Carcinoma |
| LGG | Lower Grade Glioma |
| KIPAN | Kidney Pan-Cancer |

### Omics Data Used

- mRNA Expression
- CNV
- RPPA

---

## Methodology

### 1. Multi-Omics Feature Extraction

```text
mRNA ──► Autoencoder ──► Latent Features
CNV  ──► Autoencoder ──► Latent Features
RPPA ──► Autoencoder ──► Latent Features
```

### 2. Adaptive Fusion

Latent representations from multiple omics modalities are fused using an attention-based mechanism.

### 3. Patient Similarity Network

Constructed using:

- Cosine Similarity
- K-Nearest Neighbors (KNN)
- Graph Refinement

### 4. Graph Learning

Graph Neural Networks capture inter-patient biological relationships.

### 5. Classification

Prototype-based classifier predicts cancer subtypes.

---

## Installation

```bash
git clone https://github.com/vipulchetan25/AdaMOIC.git
cd AdaMOIC

pip install -r requirements.txt
```

---

## Training

```bash
python train.py
```

---

## Evaluation

```bash
python evaluate.py
```

---

## Experimental Results

The proposed AdaMOIC framework was evaluated on multiple TCGA cancer datasets.

| Dataset | Accuracy (%) |
|----------|-------------|
| BRCA | 87.28 |
| LGG | 73.50 |
| KIPAN | 96.70 |

### BRCA Detailed Metrics

| Metric | Score (%) |
|----------|----------|
| Accuracy | 87.28 |
| Balanced Accuracy | 84.97 |
| Macro F1 Score | 85.26 |


---

### Observations

- KIPAN achieved the highest classification accuracy of **96.70%**, indicating strong separability among kidney cancer subtypes.
- BRCA achieved **87.28%** accuracy despite the challenge of distinguishing similar luminal subtypes.
- LGG achieved **73.50%** accuracy, reflecting the higher complexity and overlap among glioma subtypes.
---
## Technologies Used

- Python
- PyTorch
- PyTorch Geometric
- NumPy
- Pandas
- Scikit-Learn

---

## Future Work

- GraphSAGE-based architecture exploration
- Dynamic graph construction
- Explainable AI
- Survival analysis integration

---

## Authors

**Vipul Chetan**  
B.Tech CSE, IIIT Sri City

**T. Manoj**  
B.Tech CSE, IIIT Sri City

---
## License

This project is intended for academic and research purposes.
