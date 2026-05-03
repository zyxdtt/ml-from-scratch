# AdaBoost Implementation from Scratch

A C++ implementation of the AdaBoost (Adaptive Boosting) algorithm based on **Chapter 8 of "Statistical Learning Methods" by Li Hang**. The model uses decision stumps (single-level decision trees) as weak classifiers.

---

## Algorithm Overview

AdaBoost combines multiple weak classifiers through weighted voting. Each iteration adjusts sample weights to focus on previously misclassified examples.

### Mathematical Formulation

Given training data $(x_i, y_i)$ with $y_i \in \{-1, +1\}$:

1. Initialize weights: $w_i^{(1)} = \frac{1}{N}$

2. For $t = 1, 2, ..., T$:
   - Train weak classifier $h_t(x)$ minimizing weighted error:
     $$\epsilon_t = \frac{\sum_{i=1}^N w_i^{(t)} \cdot \mathbb{I}(y_i \neq h_t(x_i))}{\sum_{i=1}^N w_i^{(t)}}$$
   - Compute classifier weight:
     $$\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)$$
   - Update sample weights:
     $$w_i^{(t+1)} = w_i^{(t)} \cdot \exp(-\alpha_t y_i h_t(x_i))$$
   - Normalize weights to sum to 1

3. Final classifier:
   $$H(x) = \text{sign}\left(\sum_{t=1}^T \alpha_t h_t(x)\right)$$

### Weak Classifier: Decision Stump

A decision stump is a single-split classifier:

- If `x_j <= theta`, predict `-1`
- Otherwise, predict `+1`

where `j` is the feature dimension and `theta` is the threshold.

---


### Core Data Structures

```cpp
struct machine {
    double a;           // Alpha weight
    int dimension;      // Feature index to split on
    int division;       // Index into precomputed threshold list
};

class AdaBoost {
    vector<double> weights;           // Sample weights
    vector<machine> classify_machine; // Ensemble of weak classifiers
    vector<Point> division_list;      // Precomputed thresholds per feature
};
```

### Key Methods

| Method | Description |
|--------|-------------|
| `fit(X_train, y_train, T)` | Train AdaBoost with at most T weak classifiers |
| `predict(X_test)` | Return predicted labels (-1 or +1) |
| `accuracy(X_test, y_test)` | Compute classification accuracy |

### Termination Conditions

1. Perfect classification on training set ($\epsilon_t = 0$)
2. Classifier error rate $\epsilon_t \geq 0.5$ (no better than random)
3. Maximum iterations $T$ reached

---

## Experiments

### Dataset 1: Breast Cancer (scikit-learn)

| Property | Value |
|----------|-------|
| Samples | 569 |
| Features | 30 (numeric, cell nucleus measurements) |
| Classes | Benign (+1), Malignant (-1) |
| Train/Test split | 80%/20% |

**Results:**
```
epoch: 1, train accuracy: 0.920879
epoch: 5, train accuracy: 0.964835
epoch: 10, train accuracy: 0.984615
epoch: 15, train accuracy: 0.995604
epoch: 20, train accuracy: 0.995604
epoch: 25, train accuracy: 1.000000 (stopped)

Final Results:
Train accuracy: 1.0000
Test accuracy: 0.9649
```

### Dataset 2: IMDB Sentiment (using raw word IDs)

| Property | Value |
|----------|-------|
| Samples | 10,000 (5,000 train, 5,000 test) |
| Features | 500 (sequence positions, each storing word ID) |
| Classes | Positive (+1), Negative (-1) |

**Important Note:** Features are raw word IDs (0-4999), not Bag-of-Words. This violates the numerical ordering assumption of decision stumps.

**Results:**
```
epoch: 50, train accuracy: 0.6196
Final Test accuracy: 0.5252  (barely above random 50%)
Training time: ~29 minutes
```

---

## Limitations

### 1. Decision Stump Limitation

Decision stumps only split on a single feature dimension with a single threshold. This cannot capture:
- Feature interactions (e.g., XOR patterns)
- Complex decision boundaries without many boosting rounds

### 2. Feature Scale Assumption

Decision stumps use numerical comparisons ($\leq \theta$). This fails for:
- Categorical features without meaningful order
- Word IDs (as shown in IMDB experiment)

### 3. Sensitive to Noisy Labels

Exponential loss heavily penalizes misclassified samples. Outliers or label noise can dominate weight updates.

### 4. IMDB Failure Analysis

The poor performance on IMDB (52.5% accuracy) is **not a bug in implementation** but a feature mismatch:

| Issue | Explanation |
|-------|-------------|
| Feature meaning | Word ID 123 vs 456 - comparison has no semantic meaning |
| Position locking | Same word at different positions are different features |
| Missing Bag-of-Words | Should use "word appears in document" not "word at position i" |

With proper Bag-of-Words feature engineering, AdaBoost achieves 85-88% on IMDB.

### 5. Computational Complexity

Training complexity: $O(T \cdot d \cdot N \cdot \log N)$ where:
- $T$ = number of weak classifiers
- $d$ = feature dimensions
- $N$ = training samples

For IMDB: $T=50, d=500, N=5000$ → ~29 minutes

For Breast Cancer: $T=50, d=30, N=455$ → < 1 second

---

## Compilation & Usage

```bash
# Generate datasets
python prepare_cancer.py
python prepare_imdb.py

# Compile
g++ -o test_cancer test_cancer.cpp -std=c++17 -O2

# Run
./test_cancer
```

---

## Dependencies

- C++17 compiler
- Python 3.8+ with: `numpy`, `scikit-learn`, `keras` (for IMDB)

---

## References

- Li Hang. "Statistical Learning Methods", Chapter 8
- Freund, Y., & Schapire, R. E. (1997). "A decision-theoretic generalization of on-line learning and an application to boosting"
```
