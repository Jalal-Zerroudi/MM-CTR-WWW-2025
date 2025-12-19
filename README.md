# MM-CTR – AUC Boost for Multimodal CTR Prediction (WWW 2025)

This repository contains a fully executed Kaggle notebook developed for the  
**Multimodal Click-Through Rate Prediction (MM-CTR) Challenge – WWW 2025 (EReL@MIR Workshop)**.

📘 Notebook:
- `mmctr-auc-boost-full-notebook.ipynb`

The goal of this work is to **maximize AUC** for **Task 2: Multimodal CTR Prediction** by leveraging **precomputed multimodal item embeddings**.

---

## 🧠 Problem Description

Click-Through Rate (CTR) prediction is a fundamental task in recommender systems.  
Given an item and its multimodal representation (image, text, etc.), the objective is to predict the probability that a user will click on that item.

In the MM-CTR challenge:
- Multimodal representations are provided as **item embeddings**
- The task focuses on **efficient and accurate CTR prediction**
- Evaluation is based on **AUC (Area Under the ROC Curve)**

---

## 🎯 Objective

- Build a strong CTR prediction pipeline
- Exploit multimodal embeddings as structured features
- Optimize ranking quality via **AUC maximization**
- Generate a valid `prediction.csv` submission file

---

## 🗂 Project Structure

```
.
├── mmctr-auc-boost-full-notebook.ipynb
├── prediction.csv
└── README.md
```

---

## ⚙️ Execution Environment

- Platform: **Kaggle Notebook**
- Python: **3.x**
- Hardware: **CPU / GPU (CUDA if available)**

All cells have been successfully executed on Kaggle.

---

## 🧩 Methodology & Code Logic

### 1️⃣ Environment Setup
- Installation of required libraries
- Reproducibility via fixed random seed
- Automatic GPU detection

### 2️⃣ Data Loading
- Training and test data loaded from Parquet files
- Multimodal embeddings loaded from NumPy files

### 3️⃣ Feature Construction
- Each item represented by its embedding vector
- Embeddings used directly as numerical features

### 4️⃣ Model
- Lightweight neural CTR model (MLP)
- Optimized for fast inference and strong AUC

### 5️⃣ Training & Evaluation
- Binary Cross-Entropy loss
- Adam optimizer
- Validation based on AUC

### 6️⃣ Prediction & Submission
- CTR probabilities generated for test set
- Output saved as `prediction.csv`

---

## 📊 Evaluation Metric

**AUC (Area Under the ROC Curve)**
