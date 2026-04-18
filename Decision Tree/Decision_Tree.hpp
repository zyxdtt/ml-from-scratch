//2026 4 18
#pragma once
#include <vector>
#include <numeric>
#include <algorithm>
#include <utility>
#include <unordered_set>
#include <unordered_map>
#include <cmath>

using namespace std;
using Point = vector<double>;

// Node structure for regression tree
struct Node {
	int dimension;      // Feature index used for splitting
	double split;       // Threshold value for splitting
	int left;           // Index of left child node
	int right;          // Index of right child node
	bool is_leaf;       // Whether this is a leaf node
	double y_val;       // Predicted value for leaf nodes
};

// CART Regression Tree Implementation
// Reference: Li Hang, "Statistical Learning Methods", Chapter 5
class Decision_Tree_Regression {
private:
	int number_of_dimension;      // Number of features in training data
	int number_of_train;          // Number of training samples
	int root_idx;                 // Index of root node in the tree vector
	double ratio;                 // Scaling factor for normalization
	double lowest_limit;          // Variance threshold for stopping split
	bool is_fitted;               // Flag indicating if model has been trained
	vector<Node> regression_tree; // Container storing all tree nodes

	// Recursively build regression tree using CART algorithm
	// X_train: training features
	// y_train: training targets (normalized)
	// choice_space: indices of samples in current node
	// var: variance of samples in current node
	// Returns: index of created node
	int build_regression_tree(const vector<Point>& X_train,
		const vector<double>& y_train,
		vector<int> choice_space,
		double var) {

		// Stop splitting if only one sample remains or variance is below threshold
		if (choice_space.size() == 1 || var <= lowest_limit) {
			// Calculate mean value for leaf node prediction
			double mean_y = 0;
			for (int idx : choice_space) mean_y += y_train[idx];
			mean_y /= choice_space.size();
			regression_tree.emplace_back(-1,
				-1, -1, -1, true, mean_y);
			return regression_tree.size() - 1;
		}

		double min_S = 1e15;           // Minimum squared error
		pair<int, double> best_js;     // Best (feature, split_value)
		vector<int> left_choice;        // Sample indices for left child
		vector<int> right_choice;       // Sample indices for right child
		double left_var;                // Variance of left child
		double right_var;               // Variance of right child

		// Search for best split point across all samples and features
		for (int choice : choice_space) {
			for (int dimension = 0; dimension < number_of_dimension; dimension++) {
				double S = 0;
				double c1 = 0, c2 = 0;
				vector<int> c1_set, c2_set;
				double split = X_train[choice][dimension];

				// Partition samples based on split value
				for (int idx : choice_space) {
					double val = X_train[idx][dimension];
					if (val <= split) {
						c1_set.push_back(idx);
						c1 += y_train[idx];
					}
					else {
						c2_set.push_back(idx);
						c2 += y_train[idx];
					}
				}

				// Skip if split doesn't partition the data
				if (c1_set.empty() || c2_set.empty()) continue;

				// Calculate mean values for each partition
				c1 /= c1_set.size() * 1.0;
				c2 /= c2_set.size() * 1.0;

				// Calculate squared errors
				double c1_square_sum = 0;
				double c2_square_sum = 0;
				for (int c1_idx : c1_set) {
					double diff = c1 - y_train[c1_idx];
					c1_square_sum += diff * diff;
				}
				for (int c2_idx : c2_set) {
					double diff = c2 - y_train[c2_idx];
					c2_square_sum += diff * diff;
				}
				S = c1_square_sum + c2_square_sum;

				// Update best split if current is better
				if (S < min_S) {
					min_S = S;
					best_js = { dimension,split };
					left_var = c1_square_sum / (c1_set.size() * 1.0);
					right_var = c2_square_sum / (c2_set.size() * 1.0);
					left_choice = move(c1_set);
					right_choice = move(c2_set);
				}
			}
		}

		// Defensive: if no valid split found, create leaf node
		if (min_S == 1e15) {
			double mean_y = 0;
			for (int idx : choice_space) mean_y += y_train[idx];
			mean_y /= choice_space.size();
			regression_tree.emplace_back(-1, -1, -1, -1, true, mean_y);
			return regression_tree.size() - 1;
		}

		// Create internal node and recursively build children
		regression_tree.emplace_back(best_js.first, best_js.second, -1, -1, false, -1);
		int idx = regression_tree.size() - 1;
		regression_tree[idx].left = build_regression_tree(X_train,
			y_train, left_choice, left_var);
		regression_tree[idx].right = build_regression_tree(X_train,
			y_train, right_choice, right_var);
		return idx;
	}

	// Recursively traverse tree to make prediction for a single sample
	double search_tree(const Point& point, int from) const {
		if (regression_tree[from].is_leaf)
			return regression_tree[from].y_val;
		if (point[regression_tree[from].dimension] <= regression_tree[from].split)
			return search_tree(point, regression_tree[from].left);
		else
			return search_tree(point, regression_tree[from].right);
	}

public:
	~Decision_Tree_Regression() = default;
	Decision_Tree_Regression() = default;

	// Train the regression tree
	// X_train: feature matrix, each element is a sample point
	// y_train: target values
	// limit: variance threshold for early stopping (default 0.1)
	void fit(const vector<Point>& X_train,
		const vector<double>& y_train,
		double limit = 0.1) {
		is_fitted = true;
		lowest_limit = limit;
		number_of_train = y_train.size();
		number_of_dimension = X_train[0].size();

		// Normalize target values to prevent numerical overflow
		ratio = *max_element(y_train.begin(), y_train.end());
		if (ratio == 0) ratio = 1.0;
		auto y_train_ratio = y_train;
		for (auto& y : y_train_ratio) y /= ratio;

		// Initialize all sample indices
		vector<int> choice_space(number_of_train);
		iota(choice_space.begin(), choice_space.end(), 0);

		// Calculate initial variance
		double ave = accumulate(y_train_ratio.begin(), y_train_ratio.end(), 0.0) / number_of_train * 1.0;
		double var = 0;
		for (double y : y_train_ratio) var += (y - ave) * (y - ave);
		var /= number_of_train;

		// Build tree recursively
		root_idx = build_regression_tree(X_train, y_train_ratio, choice_space, var);
	}

	// Predict target values for test samples
	vector<double> predict(const vector<Point>& X_test) const {
		vector<double> y_test;
		if (!is_fitted) return y_test;
		size_t number_of_test = X_test.size();
		y_test.resize(number_of_test);
		for (int i = 0; i < number_of_test; i++) {
			y_test[i] = search_tree(X_test[i], root_idx);
			y_test[i] *= ratio;  // Denormalize predictions
		}
		return y_test;
	}

	// Calculate Root Mean Square Error
	double RMSE(const vector<Point>& X, const vector<double>& y) const {
		auto y_pred = predict(X);
		double square_sum = 0;
		for (int i = 0; i < y.size(); i++)
			square_sum += (y[i] - y_pred[i]) * (y[i] - y_pred[i]);
		square_sum /= y.size() * 1.0;
		return sqrt(square_sum);
	}
};

// Node structure for classification tree
struct node {
	int dimension;       // Feature index used for splitting
	double split;        // Feature value for splitting (equality test)
	int left;            // Index of left child node
	int right;           // Index of right child node
	bool is_leaf;        // Whether this is a leaf node
	int classification;  // Predicted class for leaf nodes
};

// CART Classification Tree Implementation
// Reference: Li Hang, "Statistical Learning Methods", Chapter 5
class Decision_Tree_Classifier {
private:
	int number_of_dimension;               // Number of features in training data
	int number_of_train;                   // Number of training samples
	int root_idx;                          // Index of root node in the tree vector
	int lowest_samples;                    // Minimum samples required to split
	double lowest_Gini;                    // Gini impurity threshold for stopping
	bool is_fitted;                        // Flag indicating if model has been trained
	vector<node> classification_tree;      // Container storing all tree nodes
	vector<unordered_set<double>> feature_graph;  // Unique feature values per dimension

	// Recursively build classification tree using CART algorithm
	// X_train: training features
	// y_train: training labels
	// choice_space: indices of samples in current node
	// Returns: index of created node
	int build_tree(const vector<Point>& X_train,
		const vector<int>& y_train,
		vector<int> choice_space) {

		int number_of_choice = choice_space.size();

		// Stop if sample count below threshold
		if (number_of_choice < lowest_samples) {
			// Find majority class
			unordered_map<int, int> class_count;
			for (int idx : choice_space) class_count[y_train[idx]]++;
			int majority_class = choice_space[0];
			int max_count = 0;
			for (auto& [cls, cnt] : class_count) {
				if (cnt > max_count) {
					max_count = cnt;
					majority_class = cls;
				}
			}
			classification_tree.emplace_back(-1,
				-1, -1, -1, true, majority_class);
			return classification_tree.size() - 1;
		}

		double min_Gini = 1e15;        // Minimum Gini impurity
		pair<int, double> best_js;     // Best (feature, split_value)
		vector<int> left_choice;        // Sample indices for left child (feature == split)
		vector<int> right_choice;       // Sample indices for right child (feature != split)

		// Search for best split across all features and their unique values
		for (int dimension = 0; dimension < number_of_dimension; dimension++) {
			for (auto feature : feature_graph[dimension]) {
				vector<int> all_features;  // Samples with feature == split value
				vector<int> another;       // Samples with feature != split value

				// Partition samples
				for (int choice : choice_space) {
					if (X_train[choice][dimension] == feature)
						all_features.push_back(choice);
					else another.push_back(choice);
				}

				// Skip if split doesn't partition the data
				if (all_features.empty() || another.empty()) continue;

				int number_of_features = all_features.size();
				int number_of_another = another.size();
				unordered_map<int, int> cnt;

				// Calculate Gini impurity for left child
				for (int feature : all_features) {
					cnt[y_train[feature]]++;
				}
				double left_Gini = 1.0;
				double right_Gini = 1.0;
				for (auto [feature, num] : cnt)
					left_Gini -= (num * 1.0 / number_of_features) * (num * 1.0 / number_of_features);

				// Calculate Gini impurity for right child
				cnt.clear();
				for (int feature : another)
					cnt[y_train[feature]]++;
				for (auto [feature, num] : cnt)
					right_Gini -= (num * 1.0 / number_of_another) * (num * 1.0 / number_of_another);

				// Weighted average of Gini impurities
				double last_Gini = left_Gini * (number_of_features * 1.0 / number_of_choice);
				last_Gini += right_Gini * (number_of_another * 1.0 / number_of_choice);

				// Update best split if current is better
				if (last_Gini < min_Gini) {
					min_Gini = last_Gini;
					best_js = { dimension,feature };
					left_choice = all_features;
					right_choice = another;
				}
			}
		}

		// Defensive: if no valid split found, create leaf node
		if (min_Gini == 1e15) {
			unordered_map<int, int> class_count;
			for (int idx : choice_space) class_count[y_train[idx]]++;
			int majority_class = choice_space[0];
			int max_count = 0;
			for (auto& [cls, cnt] : class_count) {
				if (cnt > max_count) {
					max_count = cnt;
					majority_class = cls;
				}
			}
			classification_tree.emplace_back(-1, -1, -1, -1, true, majority_class);
			return classification_tree.size() - 1;
		}

		// Stop if Gini impurity is below threshold
		if (min_Gini < lowest_Gini) {
			unordered_map<int, int> class_count;
			for (int idx : choice_space) class_count[y_train[idx]]++;
			int majority_class = choice_space[0];
			int max_count = 0;
			for (auto& [cls, cnt] : class_count) {
				if (cnt > max_count) {
					max_count = cnt;
					majority_class = cls;
				}
			}
			classification_tree.emplace_back(-1,
				-1, -1, -1, true, majority_class);
			return classification_tree.size() - 1;
		}

		// Create internal node and recursively build children
		classification_tree.emplace_back(best_js.first, best_js.second, -1, -1, false, -1);
		int idx = classification_tree.size() - 1;
		classification_tree[idx].left = build_tree(X_train,
			y_train, left_choice);
		classification_tree[idx].right = build_tree(X_train,
			y_train, right_choice);
		return idx;
	}

	// Recursively traverse tree to make prediction for a single sample
	int search_tree(const Point& point, int from) const {
		if (classification_tree[from].is_leaf)
			return classification_tree[from].classification;
		// Left child: feature equals split value, Right child: otherwise
		if (point[classification_tree[from].dimension] == classification_tree[from].split)
			return search_tree(point, classification_tree[from].left);
		else
			return search_tree(point, classification_tree[from].right);
	}

	// Calculate F1 score for a specific class label
	double F1_in_one_label(const vector<int>& pred,
		const vector<int>& y, int label) const {
		int tp = 0, fp = 0, fn = 0;
		for (int i = 0; i < y.size(); i++) {
			if (pred[i] == label && y[i] == label) tp++;  // True positive
			if (pred[i] == label && y[i] != label) fp++;  // False positive
			if (pred[i] != label && y[i] == label) fn++;  // False negative
		}
		return 2.0 * tp / (2.0 * tp + fp + fn);
	}

public:
	~Decision_Tree_Classifier() = default;
	Decision_Tree_Classifier() = default;

	// Train the classification tree
	// X_train: feature matrix, each element is a sample point
	// y_train: class labels
	// samples: minimum samples required to split a node (default 3)
	// Gini: Gini impurity threshold for early stopping (default 0.1)
	void fit(const vector<Point>& X_train,
		const vector<int>& y_train,
		int samples = 3,
		double Gini = 0.1) {
		is_fitted = true;
		lowest_samples = samples;
		lowest_Gini = Gini;
		number_of_train = y_train.size();
		number_of_dimension = X_train[0].size();

		// Initialize all sample indices
		vector<int> choice_space(number_of_train);
		iota(choice_space.begin(), choice_space.end(), 0);

		// Build feature graph: collect all unique values for each feature
		feature_graph.resize(number_of_dimension);
		for (const auto& point : X_train) {
			for (int i = 0; i < number_of_dimension; i++) {
				feature_graph[i].insert(point[i]);
			}
		}

		// Build tree recursively
		root_idx = build_tree(X_train, y_train, choice_space);
	}

	// Predict class labels for test samples
	vector<int> predict(const vector<Point>& X_test) const {
		vector<int> y_test;
		if (!is_fitted) return y_test;
		size_t number_of_test = X_test.size();
		y_test.resize(number_of_test);
		for (int i = 0; i < number_of_test; i++) {
			y_test[i] = search_tree(X_test[i], root_idx);
		}
		return y_test;
	}

	// Calculate weighted F1 score (weighted by class frequency)
	double weighted_F1(const vector<Point>& X, const vector<int>& y) const {
		auto y_pred = predict(X);
		if (y_pred.empty() || y_pred.size() != y.size()) return 0.0;
		int y_size = y.size();

		// Count frequency of each class in ground truth
		unordered_map<int, int> classify;
		for (int label : y) classify[label]++;

		// Calculate weighted average of F1 scores
		double weighted_f1 = 0;
		for (auto [label, count] : classify) {
			weighted_f1 += (count * 1.0 / y_size) * F1_in_one_label(y_pred, y, label);
		}
		return weighted_f1;
	}
};