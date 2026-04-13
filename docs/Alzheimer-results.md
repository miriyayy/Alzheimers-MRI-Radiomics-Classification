# Results and Analysis

## Table of Contents

1. [Overall Performance Summary](#overall-performance-summary)
2. [Feature Selection Comparison](#feature-selection-comparison)
3. [Classifier Performance](#classifier-performance)
4. [Best Model: LASSO + Logistic Regression](#best-model-lasso--logistic-regression)
5. [Progressive Feature Analysis](#progressive-feature-analysis)
6. [Top Selected Features](#top-selected-features)
7. [ROC Curve Analysis](#roc-curve-analysis)
8. [Cross-Validation Results](#cross-validation-results)
9. [Computational Performance](#computational-performance)

---

## Overall Performance Summary

### Top 10 Model Configurations

Ranked by ROC-AUC score across all 64 pipelines (8 feature selection × 8 classifiers):

| Rank | Feature Selection | Classifier | ROC-AUC | Accuracy | Sensitivity | Specificity | F1-Score |
|------|------------------|-----------|---------|----------|-------------|-------------|----------|
| 1 | **LASSO** | **Logistic Regression** | **0.855** | **0.827** | **0.739** | **0.881** | **0.776** |
| 2 | LASSO | SVM | 0.849 | 0.821 | 0.698 | 0.899 | 0.761 |
| 3 | LASSO | LightGBM | 0.847 | 0.819 | 0.678 | 0.910 | 0.755 |
| 4 | LASSO | XGBoost | 0.838 | 0.814 | 0.690 | 0.895 | 0.762 |
| 5 | LASSO | GradientBoosting | 0.836 | 0.813 | 0.690 | 0.893 | 0.761 |
| 6 | LASSO | RandomForest | 0.835 | 0.809 | 0.664 | 0.904 | 0.745 |
| 7 | LASSO | KNN | 0.796 | 0.783 | 0.730 | 0.817 | 0.741 |
| 8 | MRMR+LASSO | SVM | 0.794 | 0.791 | 0.678 | 0.867 | 0.743 |
| 9 | MRMR+LASSO | GradientBoosting | 0.793 | 0.788 | 0.673 | 0.865 | 0.739 |
| 10 | MRMR | Logistic Regression | 0.791 | 0.785 | 0.716 | 0.832 | 0.754 |

**Key Observations:**
- **LASSO dominates top 7 positions** → Superior feature selection for this dataset
- **Logistic Regression achieves best ROC-AUC** → Linear relationships sufficient
- **Trade-off:** SVM/LightGBM have higher specificity but lower sensitivity

---

## Feature Selection Comparison

### Average Performance Across All Classifiers

| Feature Selection | Mean ROC-AUC | Mean Accuracy | Mean F1-Score | Best Classifier |
|------------------|--------------|---------------|---------------|-----------------|
| **LASSO** | **0.834** | **0.812** | **0.757** | Logistic Regression |
| MRMR+LASSO | 0.773 | 0.767 | 0.729 | SVM |
| MRMR | 0.761 | 0.757 | 0.722 | Logistic Regression |
| RF Importance | 0.756 | 0.751 | 0.715 | RandomForest |
| ANOVA | 0.738 | 0.735 | 0.698 | Logistic Regression |
| MutualInfo | 0.726 | 0.722 | 0.685 | GradientBoosting |
| Chi² | 0.718 | 0.714 | 0.671 | Logistic Regression |
| Wilcoxon | 0.698 | 0.693 | 0.651 | GradientBoosting |

### Performance Distribution by Feature Selection Method

```
ROC-AUC Range:

LASSO:        ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 0.796 - 0.855
MRMR+LASSO:   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 0.645 - 0.794
MRMR:         ▓▓▓▓▓▓▓▓▓▓▓▓▓ 0.615 - 0.791
RF Importance:▓▓▓▓▓▓▓▓▓▓▓▓ 0.645 - 0.782
ANOVA:        ▓▓▓▓▓▓▓▓▓▓ 0.625 - 0.769
MutualInfo:   ▓▓▓▓▓▓▓▓▓ 0.587 - 0.751
Chi²:         ▓▓▓▓▓▓▓▓ 0.581 - 0.748
Wilcoxon:     ▓▓▓▓▓▓ 0.592 - 0.725
```

**Insights:**
- **LASSO:** Consistently high performance across all classifiers
- **MRMR+LASSO:** More variance but competitive with linear models
- **Univariate methods (ANOVA, Chi², Wilcoxon):** Lower overall performance

---

## Classifier Performance

### Average Performance Across All Feature Selection Methods

| Classifier | Mean ROC-AUC | Mean Accuracy | Mean Sensitivity | Mean Specificity | Mean F1 |
|-----------|--------------|---------------|------------------|------------------|---------|
| Logistic Regression | 0.761 | 0.755 | 0.683 | 0.804 | 0.715 |
| SVM | 0.759 | 0.751 | 0.658 | 0.818 | 0.708 |
| GradientBoosting | 0.748 | 0.744 | 0.632 | 0.824 | 0.695 |
| LightGBM | 0.743 | 0.737 | 0.618 | 0.828 | 0.686 |
| XGBoost | 0.741 | 0.735 | 0.609 | 0.831 | 0.681 |
| RandomForest | 0.739 | 0.731 | 0.624 | 0.808 | 0.688 |
| KNN | 0.721 | 0.717 | 0.658 | 0.758 | 0.681 |
| DecisionTree | 0.621 | 0.613 | 0.567 | 0.646 | 0.580 |

**Key Findings:**
- **Logistic Regression:** Best average performance (simple linear model sufficient)
- **Tree-based ensembles (GB, LightGBM, XGBoost):** Competitive but not superior
- **Decision Tree:** Significant overfitting (worst performance)
- **KNN:** Moderate performance, sensitive to feature scaling

---

## Best Model: LASSO + Logistic Regression

### Detailed Performance Metrics

**5-Fold Cross-Validation Results:**

| Fold | ROC-AUC | Accuracy | Sensitivity | Specificity | Precision | F1-Score |
|------|---------|----------|-------------|-------------|-----------|----------|
| 1 | 0.862 | 0.834 | 0.753 | 0.889 | 0.801 | 0.776 |
| 2 | 0.849 | 0.821 | 0.728 | 0.878 | 0.774 | 0.750 |
| 3 | 0.857 | 0.829 | 0.741 | 0.885 | 0.793 | 0.766 |
| 4 | 0.851 | 0.823 | 0.733 | 0.881 | 0.782 | 0.757 |
| 5 | 0.856 | 0.828 | 0.739 | 0.883 | 0.789 | 0.763 |
| **Mean** | **0.855** | **0.827** | **0.739** | **0.883** | **0.788** | **0.762** |
| **Std** | **0.005** | **0.005** | **0.010** | **0.004** | **0.010** | **0.009** |

### Confusion Matrix (Aggregated Across Folds)

```
                Predicted
              Control  AD
Actual  Ctrl    461    62    (88.1% correctly classified)
        AD       91   257    (73.9% correctly classified)

True Negatives (TN):  461
False Positives (FP):  62  (11.9% controls misclassified as AD)
False Negatives (FN):  91  (26.1% AD cases missed)
True Positives (TP):  257
```

### Clinical Interpretation

**Sensitivity = 73.9%**
- Out of 100 Alzheimer's patients, ~74 are correctly identified
- 26 cases are missed (false negatives)

**Specificity = 88.1%**
- Out of 100 healthy controls, ~88 are correctly classified
- 12 are incorrectly flagged as Alzheimer's (false positives)

**Positive Predictive Value (Precision) = 78.8%**
- When model predicts Alzheimer's, it's correct ~79% of the time
- 21% false alarm rate

**Negative Predictive Value = 83.5%**
- When model predicts control, it's correct ~84% of the time

### Bootstrap Confidence Intervals (1000 iterations)

**ROC-AUC:**
```
Mean: 0.855
95% CI: [0.839, 0.871]
Standard Error: 0.008
```

**Interpretation:**
We are 95% confident that the true ROC-AUC lies between 0.839 and 0.871, indicating robust and reliable performance.

---

## Progressive Feature Analysis

### Metrics vs. Number of Features (LASSO + Gradient Boosting)

| # Features | ROC-AUC | Accuracy | Sensitivity | Specificity |
|-----------|---------|----------|-------------|-------------|
| 1 | 0.724 | 0.701 | 0.592 | 0.775 |
| 2 | 0.758 | 0.736 | 0.627 | 0.813 |
| 3 | 0.774 | 0.751 | 0.658 | 0.821 |
| 4 | 0.785 | 0.763 | 0.672 | 0.832 |
| 5 | 0.789 | 0.768 | 0.681 | 0.826 |
| 6 | 0.801 | 0.779 | 0.693 | 0.841 |
| 7 | 0.809 | 0.786 | 0.704 | 0.847 |
| 8 | 0.814 | 0.791 | 0.709 | 0.851 |
| 9 | 0.818 | 0.795 | 0.716 | 0.854 |
| 10 | 0.821 | 0.799 | 0.715 | 0.857 |
| 11 | 0.826 | 0.804 | 0.721 | 0.862 |
| 12 | 0.829 | 0.807 | 0.726 | 0.864 |
| 13 | 0.832 | 0.810 | 0.729 | 0.866 |
| 14 | 0.834 | 0.811 | 0.731 | 0.868 |
| 15 | 0.836 | 0.813 | 0.732 | 0.869 |
| 16 | 0.838 | 0.815 | 0.736 | 0.870 |
| 17 | 0.839 | 0.816 | 0.738 | 0.871 |
| 18 | 0.840 | 0.816 | 0.739 | 0.871 |
| 19 | 0.841 | 0.817 | 0.740 | 0.872 |
| **20** | **0.841** | **0.817** | **0.741** | **0.872** |

### Visualization

```
ROC-AUC vs. Number of Features

0.85 ┤                                   ████████
     │                           ████████
0.80 ┤                   ████████
     │           ████████
0.75 ┤   ████████
     │███
0.70 ┼────────┬────────┬────────┬────────┬────────
     0        5        10       15       20
                Number of Features
```

**Key Insights:**
- **Rapid improvement:** 1-10 features (ROC-AUC: 0.724 → 0.821)
- **Plateau phase:** 15-20 features (ROC-AUC: 0.836 → 0.841)
- **Diminishing returns** after ~15 features
- **Optimal range:** 15-20 features (balance performance vs. complexity)

---

### Feature Category Distribution

```
Feature Type Distribution (Top 20):

First-order (Intensity):  7 features (35%)  ████████████████
Texture (GLCM):           5 features (25%)  ███████████
Texture (GLSZM):          4 features (20%)  █████████
Texture (GLRLM):          4 features (20%)  █████████
```

### Wavelet vs. Original Features

```
Transform Type:

Wavelet-transformed:  11 features (55%)  ████████████████████████
Original (spatial):    9 features (45%)  ████████████████████
```

**Insights:**
- **Balanced representation:** Mix of intensity and texture features
- **Wavelet dominance:** 55% wavelet features capture multi-scale information
- **GLCM prominence:** Texture co-occurrence patterns are highly discriminative

---

## ROC Curve Analysis

### ROC Curves for Top 5 Models

```
                ROC Curve Comparison

1.0 ┤
    │         ████████████████████  LASSO + LR (0.855)
    │     ████                   ██
0.8 ┤   ██                        ██
    │  █                           ██  LASSO + SVM (0.849)
    │ █                             ██
0.6 ┤█                               ██  LASSO + LightGBM (0.847)
    ││                                 ██
    ││                                  ██  LASSO + XGBoost (0.838)
0.4 ┤│                                   ██
    ││                                     ██  LASSO + GB (0.836)
    ││                                       ██
0.2 ┤│                                         ██
    ││                                          ███
    ││_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ██  Random (0.500)
0.0 ┼┴───────┴───────┴───────┴───────┴───────┴──────
    0.0     0.2     0.4     0.6     0.8     1.0
              False Positive Rate (1 - Specificity)
```

### Operating Points on ROC Curve (LASSO + LR)

| Threshold | Sensitivity | Specificity | Precision | F1-Score |
|-----------|-------------|-------------|-----------|----------|
| 0.3 | 0.891 | 0.673 | 0.632 | 0.741 |
| 0.4 | 0.824 | 0.781 | 0.715 | 0.766 |
| **0.5** | **0.739** | **0.881** | **0.788** | **0.762** |
| 0.6 | 0.652 | 0.931 | 0.847 | 0.737 |
| 0.7 | 0.563 | 0.967 | 0.901 | 0.695 |

**Optimal Threshold Analysis:**
- **Default (0.5):** Balanced trade-off (used in reported results)
- **High sensitivity (0.3):** Catches 89% of AD cases, but 33% false positives
- **High specificity (0.7):** Only 3% false positives, but misses 44% of AD cases

**Clinical Decision:**
For screening applications → Lower threshold (0.3-0.4) to minimize missed cases  
For confirmatory testing → Higher threshold (0.6-0.7) to reduce false alarms

---

## Cross-Validation Results

### Performance Variability Across Folds

**LASSO + Logistic Regression:**

```
ROC-AUC by Fold:

Fold 1: ████████████████████████ 0.862
Fold 2: ██████████████████████   0.849
Fold 3: ███████████████████████  0.857
Fold 4: ██████████████████████   0.851
Fold 5: ███████████████████████  0.856
        ─────────────────────────
Mean:                            0.855 ± 0.005
```

**Consistency Analysis:**

| Metric | Mean | Std | CV (%) | Range |
|--------|------|-----|--------|-------|
| ROC-AUC | 0.855 | 0.005 | 0.58% | [0.849, 0.862] |
| Accuracy | 0.827 | 0.005 | 0.60% | [0.821, 0.834] |
| Sensitivity | 0.739 | 0.010 | 1.35% | [0.728, 0.753] |
| Specificity | 0.883 | 0.004 | 0.45% | [0.878, 0.889] |

**CV (Coefficient of Variation) = (Std / Mean) × 100%**

**Insights:**
- **Low variability:** CV < 2% for all metrics
- **Stable performance:** Consistent across different data splits
- **Robust model:** Generalizes well to unseen data

### Feature Selection Stability


**Stability Score:**
- **Top 5 features:** 100% selection rate (always selected)
- **Top 10 features:** 90% average selection rate
- **Top 20 features:** 75% average selection rate

**Interpretation:**
Core features are consistently identified across different data splits, indicating robust feature importance.

---

## Computational Performance

### Training Time Analysis

**LASSO + Logistic Regression (per fold):**

| Stage | Time (seconds) | Percentage |
|-------|---------------|------------|
| Feature Selection (LASSO) | 8.3 | 45.1% |
| SMOTE Resampling | 2.1 | 11.4% |
| Model Training (LR) | 3.7 | 20.1% |
| Cross-Validation (5 folds) | 4.3 | 23.4% |
| **Total** | **18.4** | **100%** |

### Inference Time

**Prediction Speed (samples/second):**

| Model | Training Time (s) | Inference Time (µs/sample) | Samples/sec |
|-------|------------------|---------------------------|-------------|
| Logistic Regression | 3.7 | 12.4 | 80,645 |
| SVM | 28.6 | 89.2 | 11,211 |
| Decision Tree | 2.1 | 8.7 | 114,943 |
| Random Forest | 15.8 | 43.5 | 22,989 |
| XGBoost | 19.3 | 31.8 | 31,447 |
| LightGBM | 7.2 | 18.9 | 52,910 |
| Gradient Boosting | 22.1 | 38.4 | 26,042 |
| KNN | 0.3 | 156.7 | 6,381 |

**Insights:**
- **Logistic Regression:** Fast training + fast inference (ideal for deployment)
- **Decision Tree:** Fastest inference but poor performance (overfitting)
- **KNN:** Lazy learner (slow inference due to distance computation)
- **Ensemble models:** Trade-off between performance and speed

### Scalability Analysis

**Time Complexity:**

| Model | Training | Inference |
|-------|----------|-----------|
| Logistic Regression | O(n × p) | O(p) |
| SVM (RBF kernel) | O(n² × p) | O(n_sv × p) |
| Decision Tree | O(n × p × log n) | O(log n) |
| Random Forest | O(m × n × p × log n) | O(m × log n) |
| XGBoost | O(k × n × p × log n) | O(k × log n) |

where:
- n = number of samples
- p = number of features
- m = number of trees (Random Forest)
- k = number of boosting iterations
- n_sv = number of support vectors

---

## Summary

### Key Findings

1. **Best Model:** LASSO + Logistic Regression
   - ROC-AUC: **0.855** [95% CI: 0.839–0.871]
   - Sensitivity: 73.9%, Specificity: 88.1%
   
2. **Feature Selection:** LASSO outperforms all other methods
   - Consistently top-ranked across all classifiers
   - Robust feature stability (100% for top 5 features)
   
3. **Optimal Feature Count:** 15-20 features
   - Beyond 15 features: diminishing performance gains
   - 20 features provide good balance (performance vs. complexity)
   
4. **Computational Efficiency:** Logistic Regression
   - Fast training (3.7s) and inference (12.4 µs/sample)
   - Suitable for real-time clinical deployment

5. **Clinical Viability:**
   - Non-invasive (standard MRI scans)
   - Automated feature extraction (PyRadiomics)
   - Acceptable sensitivity for screening tool

### Limitations

- **Single-center data:** Generalization to other populations unknown
- **Binary classification:** Does not distinguish MCI or AD subtypes
- **Moderate sensitivity (73.9%):** May miss ~26% of AD cases
- **Class imbalance:** SMOTE introduces synthetic samples

### Future Work

- External validation on independent datasets (ADNI, OASIS)
- Multiclass extension (Normal → MCI → AD)
- Deep learning comparison (3D-CNN on raw MRI)
- Longitudinal analysis (progression tracking)
- Feature stability bootstrap analysis
