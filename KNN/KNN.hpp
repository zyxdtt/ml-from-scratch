//2026 4 13
#pragma once
#include <vector>
#include <queue>
#include <algorithm>
#include <string>
#include <utility>
#include <unordered_map>

using namespace std;
using Point = vector<double>;

// K-D tree node structure
struct Node {
	Point point;      // Data point stored at this node
	int left;         // Index of left child, -1 if none
	int right;        // Index of right child, -1 if none
	int label;        // Class label of the point
	Node() :left(-1), right(-1) {}
};

class KNN {
private:

	vector<Point> features;          // Training features (reordered for kd-tree)
	vector<int> classfications;      // Training labels
	vector<Node> kd_tree;            // K-D tree nodes

	int dimension;       // Feature space dimension
	int K;               // Number of nearest neighbors
	int number_of_train; // Size of training set
	int root_idx;        // Root node index of kd-tree

	bool is_fitted;          // Whether model has been trained
	bool use_linear_search;  // If true use brute force, otherwise use kd-tree

	// Compute squared Euclidean distance (avoid sqrt for efficiency)
	double get_distance(const Point& pointA,
		const Point& pointB) const {
		double dis = 0;
		for (int i = 0; i < dimension; i++) dis += (pointA[i] - pointB[i]) * (pointA[i] - pointB[i]);
		return dis;
	}

	// Brute force k-nearest neighbor search
	vector<int> linear_search(const vector<Point>& X_test) const {
		int number_of_tests = X_test.size();
		vector<int> y_test(number_of_tests);
		for (int i = 0; i < number_of_tests; i++) {
			// Compute distances to all training points
			vector<pair<double, int>> temp(number_of_train);
			for (int j = 0; j < number_of_train; j++) {
				temp[j] = { get_distance(X_test[i], features[j]),j };
			}
			// Partial sort to find K smallest distances
			nth_element(temp.begin(), temp.begin() + K, temp.end());
			// Vote among K nearest neighbors
			unordered_map<int, int> cnt;
			for (int j = 0; j < K; j++) cnt[classfications[temp[j].second]]++;
			int max_cnt = 0, max_classifications = -1;
			for (auto [classify, count] : cnt) {
				if (count > max_cnt) {
					max_cnt = count;
					max_classifications = classify;
				}
			}
			y_test[i] = max_classifications;
		}
		return y_test;
	}

	// Recursively build kd-tree from features[l, r)
	// Returns the index of the root node of this subtree
	int build_kd_tree(int cur_dim, int l, int r) {
		if (l >= r) return -1;
		int mid = (l + r) / 2;
		// Partition around median in current dimension
		nth_element(features.begin() + l, features.begin() + mid, features.begin() + r,
			[cur_dim](const Point& a, const Point& b) {
				return a[cur_dim] < b[cur_dim];
			});
		// Recursively build left and right subtrees
		int left = build_kd_tree((cur_dim + 1) % dimension, l, mid);
		int right = build_kd_tree((cur_dim + 1) % dimension, mid + 1, r);
		kd_tree[mid].left = left;
		kd_tree[mid].right = right;
		return mid;
	}

	// Recursive kd-tree search for a single query point
	void tree_search(const Point& point,
		int cur_dim, int node,
		priority_queue<pair<double, int>>& top_k) const {
		if (node == -1) return;
		// Compute distance to current node
		double distance = get_distance(kd_tree[node].point, point);
		// Maintain max-heap of K closest points
		if (top_k.size() < K) {
			top_k.emplace(distance, kd_tree[node].label);
		}
		else if (distance < top_k.top().first) {
			top_k.pop();
			top_k.emplace(distance, kd_tree[node].label);
		}
		// Determine which side the query point belongs to
		bool use_left = true;
		if (point[cur_dim] > kd_tree[node].point[cur_dim]) use_left = false;
		int next_node = kd_tree[node].left;
		if (!use_left) next_node = kd_tree[node].right;
		// Search the primary side first
		tree_search(point, (cur_dim + 1) % dimension, next_node, top_k);
		// Check if we need to search the other side
		// Condition: splitting plane intersects the search ball
		double axis_dis = point[cur_dim] - kd_tree[node].point[cur_dim];
		if (top_k.size() < K || axis_dis * axis_dis <= top_k.top().first) {
			if (use_left) next_node = kd_tree[node].right;
			else next_node = kd_tree[node].left;
			tree_search(point, (cur_dim + 1) % dimension, next_node, top_k);
		}
	}

	// Batch kd-tree search for multiple query points
	vector<int> tree_search(const vector<Point>& X_test) const {
		int number_of_tests = X_test.size();
		vector<int> y_test(number_of_tests);
		priority_queue<pair<double, int>> top_k;  // Max-heap of (distance, label)
		for (int i = 0; i < number_of_tests; i++) {
			// Find K nearest neighbors
			tree_search(X_test[i], 0, root_idx, top_k);
			// Vote among neighbors
			unordered_map<int, int> cnt;
			while (!top_k.empty()) {
				auto [distance, label] = top_k.top();
				top_k.pop();
				cnt[label]++;
			}
			// Select label with majority vote
			int max_cnt = 0, best_label = -1;
			for (auto [label, count] : cnt) {
				if (count > max_cnt) {
					max_cnt = count;
					best_label = label;
				}
			}
			y_test[i] = best_label;
		}
		return y_test;
	}

	// Compute F1 score for a single class
	double F1_in_one_label(const vector<int>& pred,
		const vector<int>& y, int label) const {
		int tp = 0, fp = 0, fn = 0;
		for (int i = 0; i < y.size(); i++) {
			if (pred[i] == label && y[i] == label) tp++;
			if (pred[i] == label && y[i] != label) fp++;
			if (pred[i] != label && y[i] == label) fn++;
		}
		return 2.0 * tp / (2.0 * tp + fp + fn);
	}

public:

	~KNN() = default;

	// Constructor: optionally specify feature dimension
	KNN(int dim = 0) :dimension(dim), is_fitted(false),
		number_of_train(0), use_linear_search(true) {}

	// Train the model
	void fit(const vector<Point>& X_train,
		const vector<int>& y_train,
		int k = 3) {
		is_fitted = true;
		K = k;
		if (dimension == 0) dimension = X_train[0].size();
		number_of_train = y_train.size();
		features = X_train;
		// Strategy selection: dimension > 50 -> brute force
		// Otherwise use kd-tree only if samples/dimension ratio > 500
		if (dimension > 50) use_linear_search = true;
		else {
			if (number_of_train / dimension > 500) use_linear_search = false;
			else use_linear_search = true;
		}
		if (use_linear_search) {
			classfications = y_train;
			return;
		}
		else {
			kd_tree.resize(number_of_train);
			for (int i = 0; i < number_of_train; i++) {
				kd_tree[i].point = X_train[i];
				kd_tree[i].label = y_train[i];
			}
			root_idx = build_kd_tree(0, 0, number_of_train);
		}
	}

	// Predict labels for test samples
	vector<int> predict(const vector<Point>& X_test) const {
		vector<int> y_test;
		if (!is_fitted) return y_test;
		if (use_linear_search) return linear_search(X_test);
		else return tree_search(X_test);
	}

	// Compute accuracy score
	double accuracy(const vector<Point>& X, const vector<int>& y) const {
		auto y_pred = predict(X);
		if (y_pred.empty() || y_pred.size() != y.size()) return 0.0;
		int correct = 0;
		for (size_t i = 0; i < y.size(); i++) {
			if (y_pred[i] == y[i]) correct++;
		}
		return static_cast<double>(correct) / y.size();
	}

	// Compute weighted F1 score (macro-averaged by class frequency)
	double weighted_F1(const vector<Point>& X, const vector<int>& y) const {
		auto y_pred = predict(X);
		if (y_pred.empty() || y_pred.size() != y.size()) return 0.0;
		int y_size = y.size();
		unordered_map<int, int> classify;
		for (int label : y) classify[label]++;
		double weighted_f1 = 0;
		for (auto [label, count] : classify) {
			weighted_f1 += (count * 1.0 / y_size) * F1_in_one_label(y_pred, y, label);
		}
		return weighted_f1;
	}

};