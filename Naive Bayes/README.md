# Naive Bayes Classifier

## Algorithm

Implementation of the Naive Bayes classifier with Laplace smoothing, following the algorithm described in Chapter 4 of "Statistical Learning Methods" by Li Hang.

### Training Process

1. **Calculate prior probabilities** with Laplace smoothing:
   ```
   P(Y = y_k) = (count(Y = y_k) + 1) / (N + K)
   ```
   where N is total training samples, K is number of classes.

2. **Calculate conditional probabilities** with Laplace smoothing for each feature:
   ```
   P(X^(j) = x | Y = y_k) = (count(X^(j) = x, Y = y_k) + 1) / (count(Y = y_k) + S_j)
   ```
   where S_j is the number of unique values for feature j.

3. **Store probability tables** for efficient prediction.

### Prediction Process

For a test sample, apply Bayes' rule with conditional independence assumption:
```
y = argmax_{y_k} P(Y = y_k) * ∏_{j=1}^{n} P(X^(j) = x^(j) | Y = y_k)
```

Laplace smoothing ensures no zero probabilities for unseen feature values.

## References

Li Hang. "Statistical Learning Methods" (2nd Edition). Chapter 4: Naive Bayes Method. Tsinghua University Press, 2019.

## Dataset

**Digits Dataset** (MNIST-like handwritten digits)
- Total samples: 1,797
- Features: 64 (8×8 pixel grid)
- Feature values: 0-255 (discretized from original 0-16 grayscale)
- Classes: 10 (digits 0-9)
- Training set: 1,437 samples (80%)
- Test set: 360 samples (20%)

## Experimental Results

| Metric | Value |
|--------|-------|
| Accuracy | 91.94% |
| Weighted F1 | 0.9198 |
| Training time | 13 ms |
| Prediction time (360 samples) | 22 ms |
| Rejection rate | 0% |

### Per-class Accuracy

| Class | Accuracy |
|-------|----------|
| 0 | 96.7% |
| 1 | 82.9% |
| 2 | 88.2% |
| 3 | 94.9% |
| 4 | 91.7% |
| 5 | 89.2% |
| 6 | 93.5% |
| 7 | 100.0% |
| 8 | 93.9% |
| 9 | 90.0% |

### Confusion Matrix Highlights
- Most misclassifications occur between visually similar digits (e.g., 1 vs 2, 1 vs 7, 7 vs 9)
- Class 7 achieves perfect classification
