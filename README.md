# MM-CTR: AUC Boost for Multimodal Click-Through Rate Prediction

This repository contains a fully executed Kaggle notebook:
**`mmctr-auc-boost-full-notebook.ipynb`**, developed for the  
**Multimodal Click-Through Rate Prediction (MM-CTR) Challenge – WWW 2025 (EReL@MIR Workshop)**.

The notebook focuses on **Task 2: Multimodal CTR Prediction**, with the objective of **boosting AUC performance** by effectively leveraging **precomputed multimodal item embeddings**.

---

## 🧠 Project Overview

Click-Through Rate (CTR) prediction is a core problem in recommender systems.  
In the MM-CTR challenge, each item is represented by **multimodal embeddings** (image, text, etc.), and the task is to predict whether a user will click on an item.

This notebook implements a **feature-based CTR pipeline** that:
- Loads multimodal embeddings
- Combines them into a structured feature space
- Trains a high-performance machine learning model
- Optimizes the **AUC (Area Under the ROC Curve)** metric
- Generates a submission-ready `prediction.csv` file

---

## 🎯 Objective

- Improve CTR prediction performance
- Maximize **AUC score**
- Maintain **low-latency inference compatibility**
- Produce a valid submission file for the MM-CTR leaderboard

---

## 🗂 Notebook Structure

The notebook is organized into **logical, sequential steps**:

### 1️⃣ Environment Setup
- Import of required libraries (`numpy`, `pandas`, `sklearn`, etc.)
- Kaggle-compatible execution (no local path dependencies)
- Reproducibility via random seeds

---

### 2️⃣ Data Loading
- Loading of:
  - Training data (labels)
  - Test data
  - Precomputed multimodal embeddings
- Efficient handling of large embedding matrices
- Alignment between item IDs and embedding vectors

**Key idea:**  
Multimodal embeddings are treated as **numerical features**, enabling classical ML models to exploit semantic information.

---

### 3️⃣ Feature Engineering
- Embedding normalization / reshaping
- Feature concatenation
- Optional dimensionality handling
- Construction of final feature matrix `X`

This step bridges **multimodal representation learning (Task 1)** and **CTR prediction (Task 2)**.

---

### 4️⃣ Model Selection
- Use of a **tree-based or linear model optimized for AUC**
- Justification:
  - Handles high-dimensional features
  - Robust to noise
  - Fast inference
  - Strong baseline for CTR tasks

---

### 5️⃣ Training Strategy
- Train/validation split
- AUC as the primary evaluation metric
- Model fitting on multimodal features
- Monitoring of validation performance

---

### 6️⃣ AUC Boosting Logic
- Fine-tuning of model hyperparameters
- Focus on:
  - Reducing overfitting
  - Improving ranking quality
- Emphasis on **relative ordering of predictions**, not raw probabilities

---

### 7️⃣ Prediction
- Inference on the test set
- Generation of CTR probability scores
- Output formatting compliant with challenge rules

---

### 8️⃣ Submission File Generation
- Creation of:
  ```text
  prediction.csv
