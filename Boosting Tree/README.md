# Gradient Boosting Tree Regression (from Scratch)

A C++ implementation of the Gradient Boosting Tree algorithm for regression tasks, based on **Chapter 8 of "Statistical Learning Methods" by Li Hang**.

---

## Algorithm Overview

Gradient Boosting Tree builds an ensemble of weak learners (decision stumps) in a stage-wise fashion. Each new learner corrects the errors of the previous ensemble by fitting the residual (the difference between the true value and current prediction).

### Mathematical Formulation

Given training data $\{(x_i, y_i)\}_{i=1}^N$ with $y_i \in \mathbb{R}$:

1. Initialize residual $r_i^{(0)} = y_i$ for $i = 1, \dots, N$

2. For $t = 1, 2, \dots, T$:
   - Fit a weak learner $h_t(x)$ to the current residual $\{r_i^{(t-1)}\}$
   - Choose split point that minimizes squared error:
     $$\min_{j,s} \left[ \sum_{x_i^{(j)} \le s} (r_i - c_1)^2 + \sum_{x_i^{(j)} > s} (r_i - c_2)^2 \right]$$
   - Update the residual:
     $$r_i^{(t)} = r_i^{(t-1)} - h_t(x_i)$$

3. Final prediction:
   $$F(x) = \sum_{t=1}^T h_t(x)$$

### Weak Learner: Decision Stump

A decision stump is a single-split regression tree:

- If `x[j] <= threshold`, predict `c1`
- Otherwise, predict `c2`

where `c1` and `c2` are the mean residuals in the left/right regions.

---

## Implementation Highlights

| Component | Description |
|-----------|-------------|
| **Precomputed thresholds** | Candidate split points are midpoints between consecutive feature values |
| **Prefix sums** | Cumulative sums of residuals enable O(1) region mean calculation |
| **Sliding window update** | `term1` and `term2` maintain MSE incrementally as split point moves |
| **Early stopping** | Training stops when MSE falls below `eps` |
| **Efficiency** | O($d \cdot N$) per iteration, where $d$ = feature count, $N$ = sample count |

---

## Usage

### Compilation

```bash
g++ -o test_california test_california.cpp -std=c++17 -O2
```

### Training

```cpp
Boosting_Tree model;
model.fit(X_train, y_train,      // training data
          T = 50,                // number of weak learners
          eps = 1e-4,            // early stopping threshold
          verbose = true);       // print progress
```

### Prediction

```cpp
vector<double> predictions = model.predict(X_test);
```

---

## Test Dataset: California Housing

| Property | Value |
|----------|-------|
| Samples | 20,640 |
| Features | 8 (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude) |
| Target | Median house value (in $100,000) |
| Train/Test split | 80% / 20% |

### Results

| Metric | Value |
|--------|-------|
| Train MSE | 0.437 |
| Test MSE | 0.458 |
| Train RMSE | 0.661 |
| Test RMSE | 0.677 |

**Loss curve (50 iterations):**

```
epoch 1, loss: 0.9130
epoch 2, loss: 0.8226
epoch 3, loss: 0.7371
...
epoch 23, loss: 0.4371
epoch 24-50, loss: 0.4374 (converged)
```

The model converges after ~24 iterations, achieving a final test RMSE of **0.6769**.

---

## Key Features

- ✅ Pure C++17 implementation, no external dependencies
- ✅ Efficient O(N) split search using prefix sums
- ✅ Decision stumps as weak learners
- ✅ Residual fitting (Gradient Boosting)
- ✅ Early stopping based on MSE threshold

---

## Reference

Li Hang. "Statistical Learning Methods", Chapter 8. Boosting.
