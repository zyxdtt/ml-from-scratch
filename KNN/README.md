# KNN - k-Nearest Neighbor Classifier

A C++ implementation of k-Nearest Neighbor with automatic strategy selection (brute force vs. kd-tree), based on Chapter 3 of Li Hang's *Statistical Learning Methods*.

## Algorithm

### k-Nearest Neighbor

Given a training set $T = \{(x_1, y_1), (x_2, y_2), ..., (x_N, y_N)\}$ where $x_i \in \mathbb{R}^n$ and $y_i \in \mathcal{Y}$, for a test point $x$:

1. Find the $k$ training points closest to $x$ under Euclidean distance
2. Predict the label by majority vote among these $k$ neighbors

### kd-Tree Construction

Recursively partition the training set:

- At depth $d$, split on dimension $d \bmod n$
- Choose the median point as the split node
- Points with coordinate $\le$ median go to left subtree, $>$ to right
- Each node stores one training point

### kd-Tree Search

Given query point $x$ and a max-heap maintaining the $k$ closest points:

1. Traverse down the tree to a leaf, always going to the child that contains $x$
2. At each node, compute distance and update the heap
3. Backtrack: at each internal node, check if the other side may contain closer points
4. Condition for searching the other side: `(x[axis] - node.point[axis])² < current_kth_distance²`
5. Recursively search the other side if condition holds

## Strategy Selection

| Condition | Strategy |
|-----------|----------|
| `dimension > 50` | Brute force |
| `dimension ≤ 50` and `n_samples / dimension > 500` | kd-tree |
| `dimension ≤ 50` and `n_samples / dimension ≤ 500` | Brute force |

The decision is made automatically during `fit()`.

## Usage

```cpp
KNN knn;
knn.fit(X_train, y_train, k);
vector<int> predictions = knn.predict(X_test);
double acc = knn.accuracy(X_test, y_test);
double f1 = knn.weighted_F1(X_test, y_test);
```

## API

| Method | Description |
|--------|-------------|
| `KNN(int dim = 0)` | Constructor. `dim` auto-detected if 0 |
| `fit(X_train, y_train, k = 3)` | Train the model |
| `predict(X_test)` | Return predicted labels |
| `accuracy(X, y)` | Compute accuracy score |
| `weighted_F1(X, y)` | Compute weighted F1 score |
