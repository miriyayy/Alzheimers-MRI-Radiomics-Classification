# 🧠 Alzheimer's Disease Classification via MRI Radiomics


[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Overview

A comprehensive machine learning pipeline for Alzheimer's disease classification using radiomic features extracted from MRI scans. This study systematically evaluates **64 model configurations** (8 feature selection methods × 8 classification algorithms) to identify the optimal approach for early-stage Alzheimer's detection.

### 🎯 Key Highlights

- **Best Performance:** LASSO + Logistic Regression → **ROC-AUC: 0.855** [95% CI: 0.839–0.871]
- **Dataset:** 871 MRI samples (523 control, 348 Alzheimer's)
- **Feature Space:** 10,734+ radiomic features
- **Validation:** 5-fold stratified cross-validation with SMOTE
- **Feature Selection:** Systematic comparison of 8 methods

---

## 🏆 Performance Summary

### Top Performing Models

| Feature Selection | Classifier | ROC-AUC | Accuracy | Sensitivity | Specificity |
|-------------------|-----------|---------|----------|-------------|-------------|
| **LASSO** | **Logistic Regression** | **0.855** | **0.827** | **0.739** | **0.881** |
| LASSO | SVM | 0.849 | 0.821 | 0.698 | 0.899 |
| LASSO | LightGBM | 0.847 | 0.819 | 0.678 | 0.910 |
| LASSO | XGBoost | 0.838 | 0.814 | 0.690 | 0.895 |
| LASSO | GradientBoosting | 0.836 | 0.813 | 0.690 | 0.893 |

### Model Comparison Across Feature Selection Methods

| Method | Best Classifier | ROC-AUC | F1-Score |
|--------|----------------|---------|----------|
| LASSO | Logistic Regression | 0.855 | 0.776 |
| MRMR+LASSO | SVM | 0.794 | 0.743 |
| MRMR | Logistic Regression | 0.791 | 0.754 |
| RF Importance | RandomForest | 0.782 | 0.732 |
| ANOVA | Logistic Regression | 0.769 | 0.724 |

---

## 📊 Dataset Information

### Sample Distribution

```
Total Samples: 871
├─ Control (Class 0): 523 (60.0%)
└─ Alzheimer's (Class 1): 348 (40.0%)

Class Imbalance Ratio: 1.50:1
```

### Feature Categories

- **Total Features:** 10,734
- **Feature Types:** Radiomic features extracted from MRI scans
  - First-order statistics (intensity, histogram)
  - Texture features (GLCM, GLRLM, GLSZM)
  - Shape-based features
  - Wavelet-transformed features

---

## 🛠️ Methodology

### 1. Preprocessing Pipeline

```python
# Class imbalance handling
SMOTE (Synthetic Minority Oversampling Technique)
├─ Original: 523 control, 348 Alzheimer's
└─ After SMOTE: 523 control, 523 Alzheimer's (balanced)

# Feature scaling
StandardScaler (mean=0, std=1)
```

### 2. Feature Selection Methods

Systematic evaluation of **8 feature selection techniques**:

1. **LASSO (L1 Regularization)** ⭐ Best performer
2. **ANOVA F-test**
3. **Chi-Square Test**
4. **Mutual Information**
5. **Wilcoxon Rank-Sum Test**
6. **mRMR (Minimum Redundancy Maximum Relevance)**
7. **Random Forest Importance**
8. **Hybrid MRMR + LASSO**

Each method selected **top 20 features** for model training.

### 3. Classification Algorithms

Benchmarked **8 classifiers** per feature selection method:

- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- Gradient Boosting
- K-Nearest Neighbors (KNN)

**Total Configurations:** 8 feature selection × 8 classifiers = **64 pipelines**

### 4. Validation Strategy

- **5-Fold Stratified Cross-Validation**
- **SMOTE** applied within each fold (prevent data leakage)
- **Bootstrap Confidence Intervals** (1000 iterations)
- **Metrics:** ROC-AUC, Accuracy, Sensitivity, Specificity

---

## 📈 Key Results

### Best Model: LASSO + Logistic Regression

```
ROC-AUC:     0.855 ± 0.021 (95% CI: [0.839, 0.871])
Accuracy:    0.827 ± 0.018
Sensitivity: 0.739 ± 0.032  (Correctly identified Alzheimer's cases)
Specificity: 0.881 ± 0.019  (Correctly identified controls)
```

### Feature Importance Analysis

**Top 5 LASSO-Selected Features:**

1. `original_firstorder_Median` (coef: 0.4823)
2. `wavelet-LHL_glcm_ClusterShade` (coef: -0.3912)
3. `original_glszm_LargeAreaEmphasis` (coef: 0.3654)
4. `wavelet-HLH_firstorder_Skewness` (coef: 0.3201)
5. `original_glcm_Correlation` (coef: -0.2987)

### Progressive Feature Analysis

Performance vs. number of features (LASSO + Gradient Boosting):

| # Features | ROC-AUC | Accuracy | Sensitivity | Specificity |
|-----------|---------|----------|-------------|-------------|
| 1 | 0.724 | 0.701 | 0.592 | 0.775 |
| 5 | 0.789 | 0.768 | 0.681 | 0.826 |
| 10 | 0.821 | 0.799 | 0.715 | 0.857 |
| 15 | 0.836 | 0.813 | 0.732 | 0.869 |
| **20** | **0.841** | **0.817** | **0.741** | **0.872** |

**Insight:** Performance plateaus around 15-20 features, suggesting optimal feature subset size.

---

## 🔬 Technical Stack

**Machine Learning:**  
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-orange?style=flat)
![LightGBM](https://img.shields.io/badge/LightGBM-yellow?style=flat)

**Data Processing:**  
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

**Imbalance Handling:**  
![SMOTE](https://img.shields.io/badge/Imbalanced--learn-blue?style=flat)

**Visualization:**  
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat)

**Optimization:**  
![Optuna](https://img.shields.io/badge/Optuna-purple?style=flat)

---

## 📂 Repository Structure

```
Alzheimer-MRI-Radiomics-Classification/
├── docs/
│   ├── methodology.md           # Detailed methodology
│   ├── results.md               # Comprehensive results
│   └── feature_selection.md     # Feature selection analysis
├── notebooks/
├── visualizations/
│   ├── roc_curves_comparison.png
│   ├── feature_importance_lasso.png
│   ├── metrics_vs_features.png
│   └── confusion_matrix_best_model.png
└── README.md
```

---

## 📸 Visualizations

### ROC Curve with Confidence Intervals

![ROC Curve](visualizations/roc_curves_comparison.png)

---

## 🔍 Research Insights

### Why LASSO Outperformed Other Methods?

1. **Sparsity:** LASSO's L1 regularization naturally performs feature selection, eliminating irrelevant features
2. **Multicollinearity Handling:** Effectively manages highly correlated radiomic features
3. **Generalization:** Prevents overfitting in high-dimensional feature space (10,734 → 20 features)
4. **Interpretability:** Provides clear feature coefficients for clinical interpretation

### Clinical Implications

- **Sensitivity (73.9%):** Correctly identifies ~3 out of 4 Alzheimer's patients
- **Specificity (88.1%):** Low false-positive rate, reducing unnecessary interventions
- **Non-invasive:** Uses standard MRI scans (no contrast agents or invasive procedures)
- **Scalable:** Automated radiomic extraction enables high-throughput screening

---

## 📊 Comparative Analysis

### LASSO vs. Other Feature Selection Methods

| Method | Strength | Limitation |
|--------|----------|------------|
| **LASSO** | Automatic selection, handles multicollinearity | Assumes linear relationships |
| **MRMR** | Balances relevance & redundancy | Computationally expensive |
| **RF Importance** | Captures non-linear interactions | Biased toward high-cardinality features |
| **ANOVA** | Fast, statistically grounded | Assumes normal distribution |
| **Chi²** | Good for categorical relationships | Not suitable for continuous features |

### Classifier Comparison (with LASSO features)

| Classifier | Pros | Cons | ROC-AUC |
|-----------|------|------|---------|
| **Logistic Regression** | Interpretable, fast, probabilistic | Linear decision boundary | **0.855** |
| **SVM** | Effective in high dimensions | Black-box, slow on large datasets | 0.849 |
| **LightGBM** | Fast, handles imbalance | Requires tuning | 0.847 |
| **XGBoost** | Robust, regularization | Memory-intensive | 0.838 |

---

## 🚀 Future Directions

### Immediate Next Steps

1. **External Validation:** Test on independent MRI datasets (ADNI, OASIS)
2. **Multiclass Classification:** Extend to MCI (Mild Cognitive Impairment) detection
3. **Deep Learning Comparison:** Benchmark against CNN-based approaches
4. **Feature Stability Analysis:** Bootstrap-based feature selection stability

### Long-term Goals

- **Clinical Trial Integration:** Prospective validation in hospital settings
- **Multi-modal Fusion:** Combine MRI radiomics with PET, CSF biomarkers
- **Explainability:** SHAP/LIME analysis for model interpretability
- **Publication:** Peer-reviewed journal submission (2026)

---

## 🔒 Code Availability

**Note:** Full source code is available upon reasonable request. This repository contains:

- ✅ Detailed methodology documentation
- ✅ Performance metrics and visualizations
- ✅ Sanitized demonstration notebooks
- ✅ Feature selection comparison results
- ❌ Proprietary clinical MRI data

For research collaboration inquiries: [mutlumiraysude34@gmail.com](mailto:mutlumiraysude34@gmail.com)



## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Biruni University Research Center (BAMER)
- Supervisor Dr.Ramin Rasi for clinical domain expertise
- MRI radiomics extraction pipeline developers

---

## 📊 Project Status

🟢 **Active Research** | Abstract approved for conference submission  
📅 **Last Updated:** April 2026  
🎯 **Next Milestone:** External validation on ADNI dataset

---

⭐ **If you find this research useful, please consider starring this repository!**
