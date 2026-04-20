#pragma once
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <cmath>

using namespace std;
using Point = vector<double>;

class Logistic_Regression {
private:
	// Model weights: mapping from class label to weight vector (including bias)
	unordered_map<int, Point> W;

	// Model parameters
	int number_of_classifications;
	int number_of_dimension;
	int number_of_train;
	bool is_fitted;

	// Compute softmax probability for a specific class
	// Uses max-score subtraction for numerical stability
	double softmax(const unordered_map<int, double>& scores, int key) const {
		double max_score = -numeric_limits<double>::infinity();
		for (const auto& [_, s] : scores) {
			if (s > max_score) max_score = s;
		}

		double denominator = 0.0;
		for (const auto& [_, s] : scores) {
			denominator += exp(s - max_score);
		}

		return exp(scores.at(key) - max_score) / denominator;
	}

	// Compute dot product between two vectors
	double dot_product(const Point& p1, const Point& p2) const {
		double product_sum = 0;
		for (int i = 0; i < p1.size(); i++)
			product_sum += p1[i] * p2[i];
		return product_sum;
	}

	// Compute Euclidean norm (magnitude) of a vector
	double magnitude(const Point& point) const {
		double product_sum = 0.0;
		for (const auto& p : point) product_sum += p * p;
		return sqrt(product_sum);
	}

	// Calculate F1 score for a single class
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
	~Logistic_Regression() = default;

	Logistic_Regression() :is_fitted(false) {}

	// Train the multi-class logistic regression model using gradient descent
	void fit(const vector<Point>& X_train_org,
		const vector<int>& y_train,
		double learning_rate = 0.3,
		int max_iter = 1000,
		double limit = 1e-3) {

		is_fitted = true;

		// Add bias term by extending feature dimension by 1
		number_of_dimension = X_train_org[0].size() + 1;
		number_of_train = y_train.size();

		// Initialize weight vectors for each class with zeros
		for (int y : y_train) {
			if (!W.count(y)) W[y] = Point(number_of_dimension, 0);
		}
		number_of_classifications = W.size();

		// Extend training samples with bias term (constant 1.0)
		auto X_train = X_train_org;
		for (auto& X : X_train) X.push_back(1.0);

		// Gradient descent iterations
		while (max_iter--) {
			unordered_map<int, Point> k_grad;

			// Accumulate gradients over all training samples
			for (int i = 0; i < number_of_train; i++) {
				// Compute raw scores (dot products) for all classes
				unordered_map<int, double> product;
				for (const auto& [classification, W_k] : W) {
					product[classification] = dot_product(W_k, X_train[i]);
				}

				// Update gradients for each class
				for (const auto& [classification, _] : W) {
					Point grad(number_of_dimension, 0.0);
					double y_i = (y_train[i] == classification) ? 1.0 : 0.0;
					double error = (y_i - softmax(product, classification));

					for (int dimension = 0; dimension < number_of_dimension; dimension++) {
						grad[dimension] = error * X_train[i][dimension];
					}

					// Accumulate gradient for current class
					auto temp = k_grad[classification];
					temp.resize(number_of_dimension + 1);
					for (int dimension = 0; dimension < number_of_dimension; dimension++) {
						temp[dimension] += grad[dimension];
					}
					k_grad[classification] = move(temp);
				}
			}

			// Average gradients over training set
			for (auto& [_, grad] : k_grad) {
				for (auto& grad_i : grad)
					grad_i /= number_of_train * 1.0;
			}

			// Check convergence: stop if max gradient magnitude is below threshold
			double max_magnitude = 0.0;
			for (auto& [_, grad] : k_grad) {
				max_magnitude = max(max_magnitude, magnitude(grad));
			}
			if (max_magnitude < limit) return;

			// Apply learning rate to gradients
			for (auto& [_, grad] : k_grad) {
				for (auto& grad_i : grad)
					grad_i *= learning_rate;
			}

			// Update model weights
			for (auto& [classification, W_k] : W) {
				for (int dimension = 0; dimension < number_of_dimension; dimension++) {
					W_k[dimension] += k_grad[classification][dimension];
				}
			}
		}
	}

	// Predict class labels for test samples
	vector<int> predict(const vector<Point>& X_test_org) const {
		vector<int> y_test;
		if (!is_fitted) return y_test;

		int number_of_tests = X_test_org.size();
		y_test.resize(number_of_tests);

		// Extend test samples with bias term (constant 1.0)
		auto X_test = X_test_org;
		for (auto& X : X_test) X.push_back(1.0);

		// Predict class with highest score for each sample
		for (int i = 0; i < number_of_tests; i++) {
			unordered_map<int, double> pro_distribution;
			for (const auto& [classification, W_k] : W) {
				pro_distribution[classification] = dot_product(W_k, X_test[i]);
			}

			int best_classfication = -1;
			double max_probabilities = -numeric_limits<double>::infinity();
			for (const auto& [classification, probability] : pro_distribution) {
				if (probability > max_probabilities) {
					max_probabilities = probability;
					best_classfication = classification;
				}
			}
			y_test[i] = best_classfication;
		}
		return y_test;
	}

	// Compute weighted F1 score across all classes
	double weighted_F1(const vector<Point>& X, const vector<int>& y) const {
		auto y_pred = predict(X);
		if (y_pred.empty() || y_pred.size() != y.size()) return 0.0;

		int y_size = y.size();

		// Count samples per class
		unordered_map<int, int> classify;
		for (int label : y) classify[label]++;

		// Weighted average of per-class F1 scores
		double weighted_f1 = 0;
		for (auto [label, count] : classify) {
			weighted_f1 += (count * 1.0 / y_size) * F1_in_one_label(y_pred, y, label);
		}
		return weighted_f1;
	}
};