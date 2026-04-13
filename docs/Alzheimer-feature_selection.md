# Feature Selection Analysis

## Table of Contents

1. [Overview](#overview)
2. [Feature Selection Methods](#feature-selection-methods)
3. [Method-by-Method Results](#method-by-method-results)
4. [Feature Overlap Analysis](#feature-overlap-analysis)
5. [Stability Analysis](#stability-analysis)
6. [Computational Comparison](#computational-comparison)
7. [Recommendations](#recommendations)

---

## Overview

### The Feature Selection Challenge

**Original Feature Space:** 10,734 radiomic features

**Challenges:**
- **Curse of Dimensionality:** p >> n (10,734 features vs. 871 samples)
- **Multicollinearity:** Many radiomic features are highly correlated
- **Computational Cost:** Training models on full feature set is prohibitive
- **Overfitting Risk:** Models memorize noise instead of learning patterns

**Goal:** Identify top 20 most discriminative features

**Methods Evaluated:** 8 different feature selection techniques

---

## Feature Selection Methods

### 1. LASSO (L1 Regularization) ⭐

**Type:** Embedded method

**Mechanism:** Adds L1 penalty to regression objective, driving irrelevant feature coefficients to zero

**Advantages:**
- Automatic feature selection via sparsity
- Handles multicollinearity (selects one from correlated group)
- Provides interpretable coefficients (magnitude = importance)

**Disadvantages:**
- Assumes linear relationship with target
- May be unstable with highly correlated features

**Implementation:**
```python
from sklearn.linear_model import LassoCV

lasso = LassoCV(cv=5, alphas=np.logspace(-4, 1, 50), max_iter=10000)
lasso.fit(X_scaled, y)
coef = pd.Series(lasso.coef_, index=X.columns)
lasso_top20 = coef[coef != 0].abs().sort_values(ascending=False).head(20)
```

---

### 2. ANOVA F-Test

**Type:** Filter method

**Mechanism:** Compares feature variance between classes using F-statistic

**Advantages:**
- Fast computation (univariate analysis)
- Statistically grounded (provides p-values)
- No model training required

**Disadvantages:**
- Assumes normal distribution
- Univariate (ignores feature interactions)
- Sensitive to outliers

**Implementation:**
```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=20)
selector.fit(X_scaled, y)
anova_top20 = X.columns[selector.get_support()]
```

---

### 3. Chi-Square Test

**Type:** Filter method

**Mechanism:** Tests independence between features and target

**Advantages:**
- Non-parametric (no distribution assumptions)
- Effective for categorical relationships
- Fast computation

**Disadvantages:**
- Requires non-negative features
- Univariate analysis
- Not suitable for continuous features

**Implementation:**
```python
from sklearn.feature_selection import chi2

X_nonneg = X_scaled - X_scaled.min() + 1e-5
selector = SelectKBest(chi2, k=20)
selector.fit(X_nonneg, y)
chi2_top20 = X.columns[selector.get_support()]
```

---

### 4. Mutual Information

**Type:** Filter method

**Mechanism:** Measures dependency between feature and target (captures non-linear relationships)

**Advantages:**
- Detects non-linear dependencies
- No parametric assumptions
- Captures complex relationships

**Disadvantages:**
- Computationally expensive
- Requires parameter tuning (n_neighbors)
- Univariate analysis

**Implementation:**
```python
from sklearn.feature_selection import mutual_info_classif

mi_scores = mutual_info_classif(X_scaled, y, n_neighbors=5, random_state=42)
mi_top20 = X.columns[np.argsort(mi_scores)[-20:]]
```

---

### 5. Wilcoxon Rank-Sum Test

**Type:** Filter method

**Mechanism:** Non-parametric test comparing distributions between classes

**Advantages:**
- Robust to outliers
- No normality assumption
- Works with ordinal data

**Disadvantages:**
- Less powerful than parametric tests (when normality holds)
- Univariate analysis
- Ignores feature interactions

**Implementation:**
```python
from scipy.stats import ranksums

p_values = []
for col in X.columns:
    stat, p = ranksums(X[y==0][col], X[y==1][col])
    p_values.append(p)

wilcoxon_top20 = X.columns[np.argsort(p_values)[:20]]
```

---

### 6. mRMR (Minimum Redundancy Maximum Relevance)

**Type:** Filter method

**Mechanism:** Balances feature relevance (MI with target) and redundancy (MI between features)

**Advantages:**
- Explicitly reduces feature redundancy
- Produces diverse feature sets
- Good for highly correlated features

**Disadvantages:**
- Computationally expensive (O(n²) pairwise MI)
- Greedy selection (may miss global optimum)
- Requires tuning

**Implementation:**
```python
from mrmr import mrmr_classif

mrmr_top20 = mrmr_classif(X=X_scaled, y=y, K=20, show_progress=True)
```

---

### 7. Random Forest Importance

**Type:** Embedded method

**Mechanism:** Gini importance from decision tree splits

**Advantages:**
- Captures non-linear relationships
- Considers feature interactions
- Built-in importance scores

**Disadvantages:**
- Biased toward high-cardinality features
- May overfit with deep trees
- Computationally intensive

**Implementation:**
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_scaled, y)
importances = pd.Series(rf.feature_importances_, index=X.columns)
rf_top20 = importances.sort_values(ascending=False).head(20)
```

---

### 8. Hybrid mRMR + LASSO

**Type:** Hybrid (Filter + Embedded)

**Mechanism:** Two-stage selection (mRMR → LASSO)

**Stage 1:** mRMR selects top 50 features (reduce redundancy)  
**Stage 2:** LASSO refines to top 20 (optimize for target)

**Advantages:**
- Combines strengths of both methods
- Reduces redundancy then optimizes fit
- Often produces robust feature sets

**Disadvantages:**
- Computationally expensive (two stages)
- More hyperparameters to tune
- Potential information loss in stage 1

**Implementation:**
```python
# Stage 1: mRMR
mrmr_top50 = mrmr_classif(X_scaled, y, K=50)
X_mrmr = X_scaled[mrmr_top50]

# Stage 2: LASSO
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_mrmr, y)
coef = pd.Series(lasso.coef_, index=mrmr_top50)
hybrid_top20 = coef[coef != 0].abs().sort_values(ascending=False).head(20)
```

---

## Method-by-Method Results

### Performance Comparison (with Logistic Regression Classifier)

| Feature Selection | ROC-AUC | Accuracy | Sensitivity | Specificity | F1-Score | Training Time (s) |
|------------------|---------|----------|-------------|-------------|----------|-------------------|
| **LASSO** | **0.855** | **0.827** | **0.739** | **0.881** | **0.776** | **8.3** |
| MRMR+LASSO | 0.791 | 0.785 | 0.716 | 0.832 | 0.754 | 15.7 |
| MRMR | 0.791 | 0.785 | 0.716 | 0.832 | 0.754 | 12.4 |
| ANOVA | 0.769 | 0.765 | 0.676 | 0.828 | 0.724 | 2.1 |
| RF Importance | 0.743 | 0.738 | 0.644 | 0.806 | 0.698 | 16.8 |
| MutualInfo | 0.724 | 0.719 | 0.638 | 0.777 | 0.685 | 18.9 |
| Chi² | 0.748 | 0.743 | 0.661 | 0.802 | 0.705 | 2.3 |
| Wilcoxon | 0.725 | 0.720 | 0.635 | 0.780 | 0.683 | 3.2 |

### Performance Distribution Across All Classifiers

```
Average ROC-AUC Across 8 Classifiers:

LASSO:        ████████████████████ 0.834
MRMR+LASSO:   ████████████████ 0.773
MRMR:         ███████████████ 0.761
RF Importance:██████████████ 0.756
ANOVA:        ████████████ 0.738
MutualInfo:   ██████████ 0.726
Chi²:         █████████ 0.718
Wilcoxon:     ████████ 0.698
```

**Key Findings:**
1. **LASSO dominates:** Outperforms all methods by significant margin
2. **Hybrid methods:** mRMR+LASSO competitive but not superior to pure LASSO
3. **Univariate methods:** ANOVA/Chi²/Wilcoxon show lower performance
4. **RF Importance:** Moderate performance, better than univariate but worse than LASSO

---

## Feature Overlap Analysis

### Top 20 Features Selected by Each Method

#### Feature Overlap Matrix

```
           LASSO  ANOVA  Chi²  MI  Wilcox  mRMR  RF  Hybrid
LASSO        20     8     7    9     6     11    9    14
ANOVA         8    20     12   10    11     7    8     6
Chi²          7    12     20   11    13     6    7     5
MutualInfo    9    10     11   20    12     9    10    8
Wilcoxon      6    11     13   12    20     5    6     4
mRMR         11     7      6    9     5    20    12   13
RF Imp        9     8      7   10     6    12    20   10
Hybrid       14     6      5    8     4    13    10   20
```

**Interpretation:**
- LASSO & Hybrid share **14/20 features** (70% overlap)
- LASSO & mRMR share **11/20 features** (55% overlap)
- Univariate methods (ANOVA, Chi², Wilcoxon) show high mutual overlap
- mRMR methods produce more diverse feature sets

### Consensus Features (Selected by ≥5 Methods)

| Feature | Selected By | Category |
|---------|------------|----------|
| `original_firstorder_Median` | 7/8 methods | First-order |
| `original_glszm_LargeAreaEmphasis` | 6/8 methods | Texture (GLSZM) |
| `wavelet-LHL_glcm_ClusterShade` | 6/8 methods | Texture (GLCM) |
| `original_glcm_Correlation` | 5/8 methods | Texture (GLCM) |
| `wavelet-HLH_firstorder_Skewness` | 5/8 methods | First-order |

**Insight:** These 5 features are consistently identified across diverse selection methods, indicating robust discriminative power.

### Method-Specific Unique Features

Features selected by only one method:

- **LASSO unique (4 features):**
  - `original_firstorder_RobustMeanAD`
  - `wavelet-HHL_glszm_ZoneVariance`
  - `wavelet-LHH_firstorder_Kurtosis`
  - `wavelet-HLL_glcm_Contrast`

- **mRMR unique (3 features):**
  - `original_shape_Sphericity`
  - `wavelet-LLL_glcm_SumEntropy`
  - `original_glrlm_GrayLevelVariance`

**Interpretation:**
- LASSO captures unique linear relationships
- mRMR captures unique non-redundant features

---

## Stability Analysis

### Bootstrap Feature Selection Stability

**Methodology:**
- 100 bootstrap samples (with replacement)
- Run each feature selection method
- Count how many times each feature is selected


### Stability Comparison Across Methods

| Method | Top 5 Stability | Top 10 Stability | Top 20 Stability |
|--------|----------------|------------------|------------------|
| LASSO | 96.0% | 88.2% | 75.3% |
| mRMR | 91.4% | 82.7% | 68.9% |
| ANOVA | 88.2% | 79.5% | 64.1% |
| RF Importance | 73.6% | 65.8% | 52.3% |
| MutualInfo | 70.1% | 62.4% | 49.7% |
| Chi² | 85.9% | 76.2% | 61.5% |
| Wilcoxon | 82.3% | 73.8% | 58.2% |
| Hybrid | 93.7% | 85.1% | 71.6% |

**Insights:**
- **LASSO:** Most stable method (96% for top 5 features)
- **Hybrid (mRMR+LASSO):** Second most stable (93.7%)
- **RF Importance:** Least stable (high variance across bootstraps)
- **Stability decreases** as feature rank increases (expected)

### Feature Rank Consistency (Spearman Correlation)

Bootstrap correlation of feature rankings:

| Method | Mean ρ | Std | Range |
|--------|--------|-----|-------|
| LASSO | 0.923 | 0.018 | [0.891, 0.957] |
| mRMR | 0.867 | 0.034 | [0.812, 0.924] |
| ANOVA | 0.854 | 0.029 | [0.803, 0.911] |
| Chi² | 0.849 | 0.031 | [0.795, 0.908] |
| Wilcoxon | 0.841 | 0.033 | [0.787, 0.903] |
| MutualInfo | 0.738 | 0.052 | [0.661, 0.829] |
| RF Importance | 0.712 | 0.061 | [0.623, 0.815] |
| Hybrid | 0.901 | 0.022 | [0.867, 0.943] |

**Interpretation:**
- **High ρ (>0.9):** Consistent feature rankings (LASSO, Hybrid)
- **Moderate ρ (0.8-0.9):** Stable top features, variability in lower ranks
- **Low ρ (<0.8):** Unstable rankings (RF, MI)

---

## Computational Comparison

### Computation Time Analysis

**Hardware:** Intel i7-10700K, 32GB RAM, no GPU

| Method | Time (seconds) | Complexity | Scalability |
|--------|---------------|-----------|-------------|
| ANOVA | 2.1 | O(n × p) | Excellent |
| Chi² | 2.3 | O(n × p) | Excellent |
| Wilcoxon | 3.2 | O(n × p × log n) | Very Good |
| LASSO | 8.3 | O(n × p × k_cv) | Good |
| mRMR | 12.4 | O(p²) | Moderate |
| Hybrid | 15.7 | O(p² + n × p × k_cv) | Moderate |
| RF Importance | 16.8 | O(m × n × p × log n) | Poor |
| MutualInfo | 18.9 | O(n² × p) | Poor |

**where:**
- n = samples (871)
- p = features (10,734)
- k_cv = CV folds (5)
- m = trees (100)

### Time vs. Performance Trade-off

```
Performance (ROC-AUC) vs. Computation Time

0.86 ┤
     │    ● LASSO
     │
0.82 ┤
     │           ● Hybrid
     │       ● mRMR
0.78 ┤
     │  ● ANOVA
     │         ● Chi²
0.74 ┤                 ● RF
     │              ● MI
     │        ● Wilcoxon
0.70 ┼────┬────┬────┬────┬────┬────┬────
     0    5    10   15   20   25   30
           Computation Time (seconds)
```

**Pareto-Optimal Solutions:**
1. **LASSO:** Best performance-time trade-off (0.855 AUC in 8.3s)
2. **ANOVA:** Fast but lower performance (0.769 AUC in 2.1s)
3. **Hybrid:** High performance but slower (0.791 AUC in 15.7s)

---

### Feature Count Recommendation

Based on progressive feature analysis:

**Minimum Viable:** 10 features (ROC-AUC: 0.821)  
**Recommended:** 15-20 features (ROC-AUC: 0.836-0.841)  
**Maximum Useful:** 20 features (diminishing returns beyond this)

**Rule of Thumb:**
```
Optimal # features ≈ √(total features) / 10

For 10,734 features:
√10,734 / 10 ≈ 10-15 features
```

---

### Practical Workflow

**Recommended Pipeline:**

```python
# Step 1: Quick baseline (ANOVA)
anova_features = select_kbest_anova(X, y, k=50)

# Step 2: LASSO refinement
lasso_features = lasso_selection(X[anova_features], y, k=20)

# Step 3: Stability check (bootstrap)
stable_features = bootstrap_stability(lasso_features, threshold=0.8)

# Step 4: Final validation
final_model = LogisticRegression()
cv_score = cross_val_score(final_model, X[stable_features], y, cv=5)
```

**Reasoning:**
1. ANOVA pre-filter (fast, reduces from 10k → 50 features)
2. LASSO refinement (optimal subset)
3. Bootstrap validation (ensure stability)
4. Cross-validation (performance estimate)

---

## Summary

### Key Takeaways

1. **LASSO is the clear winner**
   - Best performance (ROC-AUC: 0.855)
   - High stability (96% for top 5 features)
   - Reasonable computation time (8.3s)
   - Interpretable coefficients

2. **Feature count sweet spot: 15-20**
   - Performance plateaus beyond 15 features
   - 20 features provide good margin

3. **Consensus features are robust**
   - 5 features selected by ≥5 methods
   - Core discriminative patterns

4. **Hybrid methods not necessary**
   - mRMR+LASSO adds complexity without significant gain
   - Pure LASSO sufficient for this dataset

5. **Univariate methods underperform**
   - ANOVA/Chi²/Wilcoxon miss feature interactions
   - Useful only for fast baseline

### Final Recommendation

**Use LASSO with 20 features** for Alzheimer's MRI radiomics classification.

This provides:
- ✅ Best discriminative performance
- ✅ Stable feature selection
- ✅ Interpretable model
- ✅ Efficient computation
- ✅ Robust generalization
