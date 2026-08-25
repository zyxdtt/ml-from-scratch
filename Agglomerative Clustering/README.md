# Agglomerative Clustering

A C++ implementation of agglomerative hierarchical clustering with multiple distance metrics and linkage criteria.

## Features

- **Distance Metrics**: Minkowski, Mahalanobis, Correlation, Cosine
- **Linkage Criteria**: Min (single-linkage), Max (complete-linkage), Average, Centroid
- **Union-Find**: Disjoint-set data structure with path compression and union by rank
- **Eigen Integration**: Efficient matrix operations for Mahalanobis distance

## Dependencies

- C++11 or later
- Eigen3 library

## Installation

```bash
sudo apt-get install libeigen3-dev
```

Include the header in your project:

```cpp
#include "Agg_clustering.hpp"
```

## Usage

```cpp
#include "Agg_clustering.hpp"
#include <iostream>

int main() {
    // Sample data: 5 points, 3 dimensions
    vector<point> data = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0},
        {1.5, 2.5, 3.5},
        {4.5, 5.5, 6.5}
    };
    
    int n = data.size();           // number of samples
    int num_clusters = 2;          // desired number of clusters
    string distance = "Minkowski"; // or "Mahalanobis", "correlation", "cosine"
    string linkage = "min";        // or "max", "average", "centroid"
    int p = 2;                     // Minkowski parameter (default: 2 = Euclidean)
    
    agglomerative clusterer(data, n, num_clusters, distance, linkage, p);
    vector<vector<int>> clusters = clusterer.train();
    
    // Print results
    for (int i = 0; i < clusters.size(); ++i) {
        cout << "Cluster " << i << ": ";
        for (int idx : clusters[i]) {
            cout << idx << " ";
        }
        cout << endl;
    }
    
    return 0;
}
```

## API Reference

### Constructor

```cpp
agglomerative(
    const vector<point>& xx,    // Input data matrix (n x dim)
    int nn,                     // Number of samples
    int numm,                   // Target number of clusters
    string d_type = "Minkowski",// Distance metric
    string c_type = "min",      // Linkage criterion
    int pp = 2                  // Minkowski parameter
)
```

### Methods

| Method | Description |
|--------|-------------|
| `train()` | Execute clustering algorithm, returns vector of clusters |

### Distance Metrics

| Metric | Description |
|--------|-------------|
| `Minkowski` | Generalized distance with parameter p (p=2: Euclidean) |
| `Mahalanobis` | Distance accounting for feature correlations |
| `correlation` | 1 - Pearson correlation coefficient |
| `cosine` | 1 - Cosine similarity |

### Linkage Criteria

| Criterion | Description |
|-----------|-------------|
| `min` | Single-linkage (minimum distance between clusters) |
| `max` | Complete-linkage (maximum distance between clusters) |
| `average` | Average-linkage (mean distance between all pairs) |
| `centroid` | Distance between cluster centroids |

## Algorithm

The implementation uses the standard agglomerative hierarchical clustering approach:

1. Initialize each point as its own cluster
2. Compute pairwise distance matrix
3. Repeatedly merge closest clusters based on linkage criterion
4. Continue until target number of clusters is reached
5. Return final cluster assignments

## Notes

- Mahalanobis distance requires `n > dim` (more samples than dimensions)
- For correlation and cosine metrics, the code internally converts to distances (1 - similarity)
- Union-Find structure maintains cluster membership efficiently during merging

## License

MIT
