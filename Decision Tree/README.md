# Decision Tree Implementation (CART Algorithm)

A pure C++ implementation of CART (Classification and Regression Tree) algorithm from scratch, based on **Li Hang's "Statistical Learning Methods" Chapter 5**.

## Algorithm

### Regression Tree

The regression tree minimizes squared error at each split:

1. **Traverse all features and sample points** to find optimal split `(j, s)`, where `j` is feature index and `s` is split value
2. **Partition input space**: `R₁(j,s) = {x | x^(j) ≤ s}` and `R₂(j,s) = {x | x^(j) > s}`
3. **Minimize**: `min_{j,s} [min_{c₁} Σ(yᵢ - c₁)² + min_{c₂} Σ(yᵢ - c₂)²]`
4. **Set output values**: `c₁ = avg(yᵢ | xᵢ ∈ R₁)`, `c₂ = avg(yᵢ | xᵢ ∈ R₂)`
5. **Recursively split** until stopping criteria met

**Stopping criteria:**
- Node contains only 1 sample
- Variance falls below threshold `lowest_limit`

### Classification Tree

The classification tree minimizes Gini impurity at each split:

1. **Traverse all features and unique feature values** to find optimal split
2. **Calculate Gini impurity** for candidate split:
   - `Gini(D) = 1 - Σ (|Cₖ|/|D|)²` where `Cₖ` is subset of class `k`
   - `Gini(D, A) = (|D₁|/|D|)·Gini(D₁) + (|D₂|/|D|)·Gini(D₂)`
3. **Select split** that minimizes weighted Gini impurity
4. **Recursively split** until stopping criteria met

**Stopping criteria:**
- Node contains fewer than `lowest_samples` samples
- Gini impurity falls below threshold `lowest_Gini`

### Optimization Features
- **Target normalization**: Prevents numerical overflow when computing squared errors
- **Defensive programming**: Handles edge cases where no valid split exists
- **Move semantics**: Uses `std::move` to avoid unnecessary vector copies

## Datasets

### California Housing (Regression)
- **Source**: sklearn.datasets.fetch_california_housing
- **Samples**: 5,000 training, 1,000 testing
- **Features**: 8 (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude)
- **Target**: House price (in $100,000 units)

### Digits (Classification)
- **Source**: sklearn.datasets.load_digits
- **Samples**: 1,437 training, 360 testing
- **Features**: 64 (8×8 grayscale pixel values)
- **Classes**: 10 (digits 0-9)

## Results

### Regression Tree (California Housing)

| Threshold | RMSE  | R²     | Time (ms) |
|-----------|-------|--------|-----------|
| 0.050     | 0.910 | 0.354  | 2424      |
| **0.020** | **0.695** | **0.623** | **5053** |
| 0.010     | 0.706 | 0.612  | 5892      |
| 0.005     | 0.751 | 0.560  | 6149      |

**Best parameter: `limit = 0.020`**  
Achieves RMSE = 0.695 and R² = 0.623

### Classification Tree (Digits)

| Min Samples | Gini | Accuracy | F1 Score | Time (ms) |
|-------------|------|----------|----------|-----------|
| 2           | 0.10 | 78.89%   | 0.7918   | 214       |
| 5           | 0.10 | 79.17%   | 0.7940   | 209       |
| **10**      | **0.10** | **80.00%** | **0.8007** | **199** |
| 20          | 0.10 | 79.44%   | 0.7964   | 181       |

**Best parameters: `samples = 10, Gini = 0.10`**  
Achieves Accuracy = 80.00% and Weighted F1 = 0.8007

## Hyperparameter Tuning

### Regression Tree: Variance Threshold (`limit`)

| limit | Effect |
|-------|--------|
| > 0.05 | Underfitting (stops too early, high bias) |
| 0.02 | **Optimal** (balances bias-variance tradeoff) |
| < 0.01 | Overfitting (grows too deep, high variance) |

**Finding**: Stricter thresholds (>0.05) cause underfitting, while extremely small thresholds (<0.005) lead to overfitting with degraded test performance.

### Classification Tree

| Parameter | Optimal Value | Effect |
|-----------|---------------|--------|
| `samples` | 10 | Controls minimum leaf size; prevents overfitting to small groups |
| `Gini`    | 0.10 | Balances purity requirement; lower values risk overfitting |

**Finding**: Moderate values (`samples=10`, `Gini=0.10`) outperform both aggressive pruning (`samples=20`) and minimal pruning (`samples=2`).

## Usage

#include "Decision_Tree.h"

// Regression
Decision_Tree_Regression reg;
reg.fit(X_train, y_train, 0.02);  // variance threshold = 0.02
vector<double> predictions = reg.predict(X_test);
double rmse = reg.RMSE(X_test, y_test);

// Classification
Decision_Tree_Classifier clf;
clf.fit(X_train, y_train, 10, 0.10);  // min_samples=10, Gini=0.10
vector<int> predictions = clf.predict(X_test);
double f1 = clf.weighted_F1(X_test, y_test);

## Compilation

g++ -std=c++17 -O2 -o test test_decision_tree.cpp
./test

## Reference

Li Hang. (2012). *Statistical Learning Methods*. Tsinghua University Press. Chapter 5: Decision Trees.
