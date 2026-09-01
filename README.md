# 🏥 Medical Insurance Cost Prediction: ML & ANFIS

<div align="center">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FdD7s4RYe0YUyPfa36JO0jg-Mlqws-po#scrollTo=8d2e8650-0e50-4584-941b-9191d611ad7b)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)](https://www.kaggle.com/datasets/mirichoi0218/insurance)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)]()
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)]()

</div>

---

## 📌 Project Overview
This project tackles the classic machine learning problem of predicting annual medical insurance costs based on demographic and personal attributes. Medical cost data exhibits highly singular and non-linear behaviors—most notably, the explosive increase in costs when high Body Mass Index (BMI) intersects with smoking habits[cite: 12, 13].

In the healthcare and insurance sectors, absolute accuracy is insufficient; **Interpretability** (the "Glass Box" approach) is critical[cite: 12, 13]. This repository demonstrates a comprehensive comparative analysis between traditional statistical models, deep black-box neural networks, and interpretable fuzzy-logic systems.

---

## 🧠 Methodologies & What We Did

### 1. Robust Data Preprocessing & EDA
- **Data Leakage Guard:** The dataset was split into Training (80%) and Testing (20%) *before* any feature scaling. The `StandardScaler` was strictly fitted only on the training set to prevent information leakage[cite: 12, 13].
- **Feature Encoding:** Applied Label Encoding for binary features (`sex`, `smoker`) and One-Hot Encoding (`drop_first=True`) for multi-class features (`region`) to eliminate multicollinearity[cite: 12, 13, 14].

### 2. Phase 1: Statistical & Interaction Learning
- **Polynomial Regression (d=2):** Based on EDA insights, we explicitly modeled the powerful multiplicative interaction between `BMI` and `Smoker` status. This allowed a simple linear model to warp into a highly flexible surface, capturing the exact data shift[cite: 12, 13, 14].

### 3. Phase 2: Deep Neural Networks (MLP)
- **Deep Pyramidal Architecture:** Designed a deep Multi-Layer Perceptron (MLP) with a narrowing structure `(256-128-64-32)` to hierarchically compress abstract features[cite: 12, 13, 14].
- Optimized with `Adam`, `ReLU` activations, and heavily regularized using **Early Stopping** (on a 10% validation split) to prevent overfitting on the small tabular dataset (1338 rows)[cite: 12, 14].

### 4. Phase 3: Custom ANFIS Implementation (From Scratch)
- **Mathematical Blueprint:** We completely bypassed black-box libraries and implemented the classic 5-layer Takagi-Sugeno **Adaptive Neuro-Fuzzy Inference System (ANFIS)** (Jang, 1993) line-by-line using raw `NumPy`[cite: 12, 13, 14].
- **Hybrid Learning Algorithm:** 
  - *Forward Pass:* Used exact Least Squares Estimation (LSE) to globally optimize the linear consequent parameters[cite: 12, 13, 14].
  - *Backward Pass:* Applied Gradient Descent with custom matrix chain-rule derivatives to update the non-linear premise parameters (Gaussian centers and widths)[cite: 12, 13, 14].
- **Dimensionality Management:** To avoid the "Curse of Dimensionality" and rule explosion (which would generate 4096 rules), we performed strict Feature Selection (Age, BMI, Smoker) and analyzed Rule Saturation (comparing 2, 4, and 8 fuzzy rules)[cite: 12, 13, 14].

---

## 📊 Models & Results
The explicit feature interaction in Polynomial Regression and the deep architecture of the MLP proved highly effective, achieving nearly identical top-tier performance.

| Model Architecture | R² Score | RMSE ($) | MAE ($) |
| :--- | :---: | :---: | :---: |
| **MLP (256-128-64-32)** | **0.8696** | 4499.16 | 2545.79 |
| **Polynomial Regression (d=2)** | **0.8666** | 4551.13 | 2729.50 |
| **MLP (128-64-32)** | 0.8596 | 4668.70 | 2882.57 |
| **MLP (64-32)** | 0.7937 | 5658.85 | 4074.58 |
| **Linear Regression** | 0.7836 | 5796.28 | 4181.19 |
| **ANFIS (8 rules)** | 0.7528 | 6194.96 | 4676.71 |

*(Data derived from the comprehensive testing phase)*[cite: 12, 13, 14]

---

## 📂 Repository Structure
```text
📦 AI-2026-Project-Insurance
 ┣ 📂 code/              # Complete Jupyter notebook implementation
 ┣ 📂 data/              # Dataset placement instructions (insurance.csv)
 ┣ 📂 docs/              # In-depth mathematical report (PDF)
 ┣ 📂 media/             # Presentation slides and output graphs
 ┣ 📜 requirements.txt   # Python dependency list
 ┗ 📜 README.md          # Project overview (this file)
