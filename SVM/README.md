# SVM Implementation from Scratch

A C++ implementation of Support Vector Machine using the SMO (Sequential Minimal Optimization) algorithm, based on the principles described in "Statistical Learning Methods" (Chapter 7) by Li Hang.

## Algorithm Overview

The core algorithm follows the standard SMO approach:

1. **Problem Formulation**: Solve the dual problem of SVM
   ```
   max Σαᵢ - ½ ΣαᵢαⱼyᵢyⱼK(xᵢ,xⱼ)
   s.t. 0 ≤ αᵢ ≤ C, Σαᵢyᵢ = 0
   ```

2. **KKT Conditions**: The algorithm checks three cases for each sample
   - αᵢ = 0      → yᵢ·g(xᵢ) ≥ 1
   - 0 < αᵢ < C  → yᵢ·g(xᵢ) = 1
   - αᵢ = C      → yᵢ·g(xᵢ) ≤ 1

3. **SMO Heuristics**:
   - First variable: Find sample violating KKT conditions
   - Second variable: Maximize |E₁ - E₂| for largest step size
   - Multi-layer fallback: non-bound → all samples

4. **Update Rules**:
   - Compute unclipped α₂_new, then clip to [L, H]
   - Update α₁ using linear constraint
   - Update bias b based on new α values
   - Maintain error cache E for efficiency

## Class Design

```
SVM
├── Model Parameters
│   ├── alpha[]    - Lagrange multipliers
│   ├── B          - Bias term
│   └── C          - Penalty parameter
├── Kernel Functions
│   ├── Linear     (via polynomial with degree=1)
│   └── RBF        (K(x,z) = exp(-γ·||x-z||²))
├── SMO Core
│   ├── KKT checking
│   ├── α pair selection
│   ├── Clip operation
│   └── Error cache update
└── Prediction
    └── sign(ΣαᵢyᵢK(xᵢ, x) + b)
```

## Dataset Preparation

### Synthetic Circles Data (Non-linear Separable)

```python
from sklearn.datasets import make_circles
import numpy as np

X, y = make_circles(n_samples=500, factor=0.5, noise=0.05, random_state=42)
y = np.where(y == 0, -1, 1)

# Save as libsvm format
with open('circles_train.libsvm', 'w') as f:
    for i in range(400):
        f.write(f"{y[i]} 1:{X[i,0]} 2:{X[i,1]}\n")

with open('circles_test.libsvm', 'w') as f:
    for i in range(400, 500):
        f.write(f"{y[i]} 1:{X[i,0]} 2:{X[i,1]}\n")
```

### 2D Linear Separable Data

```python
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=400, centers=2, n_features=2,
                  cluster_std=0.8, random_state=42)
y = np.where(y == 0, -1, 1)

with open('linear_train.libsvm', 'w') as f:
    for i in range(300):
        f.write(f"{y[i]} 1:{X[i,0]} 2:{X[i,1]}\n")
```

## Test Results

### Circles Dataset (400 train / 100 test, 2D, non-linear separable)

| Kernel | Train Acc | Test Acc | Time (ms) |
|--------|-----------|----------|-----------|
| Linear | 56.75%    | 53%      | 23        |
| RBF (γ=10) | 100%   | 100%     | 303       |

*Linear kernel cannot separate concentric circles; RBF kernel achieves perfect classification.*

## Usage

### Basic Example

```cpp
#include "SVM.hpp"
#include <vector>
#include <fstream>

using namespace std;

vector<Point> load_data(const string& filename, vector<int>& labels) {
    vector<Point> X;
    ifstream file(filename);
    string line;
    
    while (getline(file, line)) {
        if (line.empty()) continue;
        
        istringstream iss(line);
        int label;
        iss >> label;
        labels.push_back(label);
        
        Point x(2, 0.0);
        string token;
        while (iss >> token) {
            size_t colon = token.find(':');
            if (colon != string::npos) {
                int idx = stoi(token.substr(0, colon)) - 1;
                double val = stod(token.substr(colon + 1));
                if (idx < 2) x[idx] = val;
            }
        }
        X.push_back(x);
    }
    return X;
}

int main() {
    vector<int> y_train, y_test;
    vector<Point> X_train = load_data("circles_train.libsvm", y_train);
    vector<Point> X_test = load_data("circles_test.libsvm", y_test);
    
    SVM svm;
    svm.fit(X_train, y_train, 
            C = 1.0,           // penalty parameter
            KKT_eps = 1e-3,    // KKT tolerance
            STEP_eps = 1e-3,   // minimum step size
            "Gaussian",        // kernel type: "Gaussian" or "Poly"
            gamma = 10.0,      // kernel parameter (γ for RBF)
            max_iter = 100);   // maximum iterations
    
    vector<int> y_pred = svm.predict(X_test);
    
    double acc = svm.weighted_F1(X_test, y_test);
    
    return 0;
}
```

### Important Notes

1. **Feature Scaling**: SVM is sensitive to feature scales. Normalize data before training:
   ```cpp
   for (int j = 0; j < dim; j++) {
       double max_val = 0;
       for (auto& x : X) max_val = max(max_val, x[j]);
       if (max_val > 0) {
           for (auto& x : X) x[j] /= max_val;
       }
   }
   ```

2. **Kernel Selection**:
   - Use **linear kernel** for high-dimensional sparse data
   - Use **RBF kernel** for low-dimensional non-linear data

3. **Parameter Tuning**:
   - Small γ (e.g., 0.1) → smooth boundary, risk of underfitting
   - Large γ (e.g., 10) → complex boundary, risk of overfitting
   - Large C → less tolerance for misclassification

## References

- Li Hang. "Statistical Learning Methods", Chapter 7
- Platt, J. "Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines" (1998)
