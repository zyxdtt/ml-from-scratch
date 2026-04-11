# Perceptron

A from-scratch implementation of the classic Perceptron algorithm in C++17, following Chapter 1 of *Statistical Learning Methods* by Li Hang.

## Algorithm Overview

The Perceptron is a binary linear classifier that finds a separating hyperplane $w \cdot x + b = 0$.

**Parameter updates (SGD):**
$$w \leftarrow w + \eta y_i x_i$$
$$b \leftarrow b + \eta y_i$$

where $\eta$ is the learning rate and $(x_i, y_i)$ is a misclassified sample with $y_i \in \{-1, +1\}$.

**Convergence:** Guaranteed in finite steps if and only if the data is linearly separable (Novikoff, 1962).

## References

- Li, Hang. *Statistical Learning Methods*. Tsinghua University Press, 2012. Chapter 1.
- Rosenblatt, F. (1958). The Perceptron: A probabilistic model for information storage and organization in the brain. *Psychological Review*, 65(6), 386-408.
- Novikoff, A. B. (1962). On convergence proofs for perceptrons. *Symposium on the Mathematical Theory of Automata*, 12, 615-622.

## Test Results

PERCEPTRON TEST SUITE

Test 1: AND Logic Gate (linearly separable)
----------------------------------------------
Training log: Converged after 2 epochs.
✓ Converged successfully
True labels:  1 -1 -1 -1
Predictions:  1 -1 -1 -1
✓ All predictions correct
Accuracy: 100%
Learned weights: 0.5 0.5
Learned bias: -0.5

Test 2: XOR Logic Gate (linearly inseparable)
----------------------------------------------
Training log: Model not converged after 10 epochs. You can use continual_fit() method.
✓ Correctly failed to converge (XOR is linearly inseparable)
Accuracy: 50%
✓ Accuracy < 100% as expected

Test 3: Continual Training
----------------------------------------------
Initial training: Model not converged after 3 epochs. You can use continual_fit() method.
Continual training: Converged after 4 additional epochs.
Final predictions: 1 1 -1

Test 4: Prediction Interface
----------------------------------------------
Test samples:  [1.5,1.5] [2.5,2.5]
Predictions:   1 -1

Test 5: Evaluation Metrics
----------------------------------------------
Accuracy:  50%
Precision: 0%
Recall:    0%
F1 Score:  0%

Classification Report:
Accuracy:  50%
Precision: 0%
Recall:    0%
F1 Score:  0%

Test 6: Edge Cases and Error Handling
----------------------------------------------
Empty data: Error: Empty training data.
✓ Predict before fit returns empty vector
