AdaMOIC: Adaptive Multi-Omics Integration for Cancer Subtype Classification
Overview

AdaMOIC is a deep learning framework for cancer subtype classification using multi-omics data. The model integrates mRNA, CNV, and RPPA omics modalities through adaptive feature fusion and graph-based learning to capture complex biological relationships among cancer patients.

This work is inspired by the DeepMoIC framework and introduces several improvements, including adaptive fusion mechanisms, enhanced graph representation learning, and robust subtype discrimination.

Key Features
Multi-omics integration of:
mRNA Expression Data
Copy Number Variation (CNV)
Reverse Phase Protein Array (RPPA)
Autoencoder-based feature extraction for each omics modality.
Adaptive latent feature fusion using attention-based integration.
Patient Similarity Network (PSN) construction using K-Nearest Neighbors and cosine similarity.
Graph Neural Network-based classification.
Prototype-based classifier for improved subtype discrimination.
Supervised Contrastive Learning for enhanced representation quality.
Probability calibration and subtype refinement strategies.
Dataset

The framework is evaluated on TCGA cancer datasets:

Dataset	Description
BRCA	Breast Invasive Carcinoma
LGG	Lower Grade Glioma
KIPAN	Kidney Pan-Cancer Dataset
Omics Data Used
mRNA Expression
CNV
RPPA

All datasets are preprocessed using normalization and feature selection techniques before model training.

Methodology
1. Multi-Omics Feature Extraction

Each omics modality is processed using an independent Autoencoder:

Encoder
Latent Representation
Decoder

The latent embeddings are learned separately for:

mRNA
CNV
RPPA
2. Adaptive Fusion

Latent representations are fused using an attention-based fusion mechanism to generate a unified patient representation.

3. Patient Similarity Network

A Patient Similarity Network (PSN) is constructed using:

Cosine Similarity
K-Nearest Neighbors (KNN)
Graph Refinement
4. Graph Learning

The fused patient graph is processed using Graph Neural Networks to capture inter-patient relationships.

5. Classification

The learned graph representations are used for cancer subtype prediction through a prototype-based classifier.

Project Structure
AdaMOIC/
│
├── data/
│   ├── BRCA/
│   ├── LGG/
│   └── KIPAN/
│
├── models/
│   ├── autoencoder.py
│   ├── graph_model.py
│   └── classifier.py
│
├── utils/
│   ├── preprocessing.py
│   ├── graph_utils.py
│   └── metrics.py
│
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
Installation

Clone the repository:

git clone https://github.com/your-username/AdaMOIC.git
cd AdaMOIC

Create a virtual environment:

python -m venv venv

Activate environment:

Windows:

venv\Scripts\activate

Linux/Mac:

source venv/bin/activate

Install dependencies:

pip install -r requirements.txt
Training

Run model training:

python train.py
Evaluation

Evaluate the trained model:

python evaluate.py
Experimental Results
BRCA
Metric	Score
Accuracy	87.28%
Balanced Accuracy	84.97%
Macro F1 Score	85.26%
Additional Datasets

The framework is also evaluated on:

LGG
KIPAN

Performance varies according to dataset characteristics and subtype complexity.

Technologies Used
Python
PyTorch
PyTorch Geometric
NumPy
Pandas
Scikit-Learn
Future Work
GraphSAGE-based architecture exploration.
Dynamic graph construction.
Improved multi-modal attention mechanisms.
Explainable AI for subtype prediction.
Survival analysis integration.
Citation

If you use this work in your research, please cite:

@misc{adamoic2026,
  title={AdaMOIC: Adaptive Multi-Omics Integration for Cancer Subtype Classification},
  author={Vipul Chetan},
  year={2026}
}
@misc{adamoic2026,
  title={AdaMOIC: Adaptive Multi-Omics Integration for Cancer Subtype Classification},
  author={T.Manoj},
  year={2026}
}
License

This project is released under the MIT License.
