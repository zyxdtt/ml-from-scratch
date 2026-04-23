# Multi-class Logistic Regression

A C++ implementation of multi-class logistic regression using softmax and gradient descent, following Chapter 6 of *Statistical Learning Methods* by Li Hang.

## Algorithm

The model learns class-conditional probabilities using the softmax function:

$$P(Y = k | X = x) = \frac{\exp(w_k \cdot x)}{\sum_{j=1}^K \exp(w_j \cdot x)}$$

Parameters are estimated via maximum likelihood. The negative log-likelihood (cross-entropy loss) is minimized using batch gradient descent:

$$w_k \leftarrow w_k + \eta \sum_{i=1}^N \left( \mathbb{1}(y_i = k) - P(Y = k | X = x_i) \right) x_i$$

where $\eta$ is the learning rate and $K$ is the number of classes.

**Optimizations:**
- Max-score subtraction in softmax for numerical stability
- $O(KND)$ time complexity by precomputing class scores per sample
- Bias term integrated via feature augmentation ($x \leftarrow [x, 1.0]$)

## Usage

```cpp
#include "Logistic_Regression.hpp"

// Prepare data
vector<Point> X_train = ...;  // N × D features
vector<int> y_train = ...;    // N labels (0 to K-1)

// Train model
Logistic_Regression model;
model.fit(X_train, y_train, 
          0.3,    // learning_rate (default: 0.3)
          1000,   // max_iter (default: 1000)
          1e-3);  // convergence threshold (default: 1e-3)

// Predict
vector<int> y_pred = model.predict(X_test);

// Evaluate
double f1 = model.weighted_F1(X_test, y_test);
```

## Experiments

**Dataset:** UCI Digits (8×8 handwritten digit images, 10 classes, 64 features)

| Metric | Result |
|--------|--------|
| Training samples | 1,437 |
| Test samples | 360 |
| Training accuracy | 99.30% |
| Test accuracy | 93.89% |
| Training time | ~4.8 seconds |

Training converged within 500 iterations with learning rate 0.3.

## Implementation Notes

- Weights initialized to zero
- Early stopping when max gradient magnitude < threshold
- No regularization applied
- Features should be standardized before training for optimal performance
