# K-Means Clustering

C++ implementation of K-Means algorithm from Li Hang's "Statistical Learning Methods" (Chapter 15).

## Algorithm Overview

K-Means is an unsupervised learning algorithm that partitions n data points into k clusters. Each point is assigned to the cluster with the nearest centroid, and centroids are iteratively updated until convergence.

### Steps:
1. Initialize k cluster centroids randomly
2. Assign each point to the nearest centroid
3. Update centroids by computing the mean of points in each cluster
4. Repeat steps 2-3 until cluster assignments stabilize

## ⚠️ Critical Limitation: Local Optimum Only

**K-Means does not guarantee global optimum — it only converges to a local optimum.**

Due to random centroid initialization, the algorithm may converge to different solutions across multiple runs. Some runs produce optimal clustering, while others get trapped in suboptimal configurations (e.g., merged clusters or empty clusters).

### Experimental Evidence

Tested on 12 sample points with 3 well-separated natural clusters:

**Dataset:**
- Cluster A: (0,0), (0,1), (1,0), (1,1)
- Cluster B: (10,11), (11,11), (11,10), (10,10)
- Cluster C: (-5,-6), (-6,-5), (-5,-5), (-6,-6)

**Results from 5 runs with random initialization:**

| Run | Cluster Assignments | Status |
|-----|-------------------|--------|
| 1   | {4,5,6,7} {8,9,10,11} {0,1,2,3} | ✅ Optimal |
| 2   | {0,1,2,3} {4,5,6,7} {8,9,10,11} | ✅ Optimal |
| 3   | {0,1,2,3,8,9,10,11} {4,5,6,7} {} | ❌ Merged clusters, empty cluster |
| 4   | {4,5,6,7,8,9,10,11} {0,1,2,3} {} | ❌ Merged clusters, empty cluster |
| 5   | {0,1,2,3,4,5,6,7} {8,9,10,11} {} | ❌ Merged clusters, empty cluster |

**Observations:**
- Runs 1-2: Successfully identified all 3 natural clusters
- Runs 3-5: Failed — two clusters merged into one, leaving an empty cluster
- Success rate: ~40% with random initialization

### Why This Happens

Poor initialization can place centroids too close to each other, causing them to compete for the same cluster while other regions remain unassigned. The algorithm then converges to a suboptimal local minimum.

### Recommended Solution

**Run K-Means multiple times (e.g., 10-20 times) with different initializations and select the result with the lowest Within-Cluster Sum of Squares (WCSS).**

```cpp
// Select best result from multiple runs
float best_wcss = INF;
for (int run = 0; run < 10; run++) {
    reinitialize_centroids();
    auto result = train(x);
    float wcss = compute_wcss(x, result);
    if (wcss < best_wcss) {
        best_wcss = wcss;
        best_result = result;
    }
}
Alternative: K-Means++
Use K-Means++ initialization to spread initial centroids farther apart, reducing the chance of poor local optima.

Build
bash
g++ -std=c++17 -O2 main.cpp -o kmeans
License
MIT
