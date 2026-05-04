// 2026 5 4
#pragma once

#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>

using namespace std;
using Point = vector<double>;

// Weak learner: a decision stump that splits on a single feature
struct machine {
    int dimension;      // Feature index to split on
    int division;       // Index into division_list (precomputed threshold)
    double left;        // Prediction for samples <= threshold
    double right;       // Prediction for samples > threshold
};

class Boosting_Tree {
private:
    vector<machine> regression_machine;   // Ensemble of weak learners
    vector<Point> division_list;          // Precomputed candidate thresholds per feature
    int number_of_train;                  // Number of training samples
    int number_of_dimension;              // Number of features
    bool is_fitted;                       // Model training flag

    // Ensemble prediction: sum of all weak learners' outputs
    double F(const Point& X) const {
        double result = 0.0;
        for (auto [dimension, division, left, right] : regression_machine) {
            double threshold = division_list[dimension][division];
            if (X[dimension] <= threshold)
                result += left;
            else
                result += right;
        }
        return result;
    }

    // Prediction of the t-th weak learner (index starts at 0)
    double f(const Point& X, int machine_idx) const {
        auto [dimension, division, left, right] = regression_machine[machine_idx];
        double threshold = division_list[dimension][division];
        if (X[dimension] <= threshold)
            return left;
        else
            return right;
    }

public:
    Boosting_Tree() : is_fitted(false) {
        // Dummy placeholder to simplify indexing (index 0 unused)
        regression_machine.emplace_back(0, 0, 0, 0);
    }

    ~Boosting_Tree() = default;

    // Fit gradient boosting tree regression model
    // X_train: training features (each sample is a Point)
    // y_train: target values
    // T: number of boosting iterations (weak learners)
    // eps: early stopping threshold for MSE
    // verbose: print progress if true
    void fit(const vector<Point>& X_train,
        const vector<double>& y_train,
        int T = 50,
        double eps = 1e-1,
        bool verbose = true) {

        is_fitted = true;
        number_of_train = y_train.size();
        number_of_dimension = X_train[0].size();

        // Precompute candidate split thresholds for each feature
        // Initialize with (N-1) slots per feature
        division_list.resize(number_of_dimension, Point(number_of_train - 1));
        vector<vector<int>> y_seq(number_of_dimension, vector<int>(number_of_train));

        for (int dim = 0; dim < number_of_dimension; ++dim) {
            // Sort samples by current feature, store original indices
            vector<pair<double, int>> temp(number_of_train);
            for (int i = 0; i < number_of_train; ++i) {
                temp[i].first = X_train[i][dim];
                temp[i].second = i;
            }
            sort(temp.begin(), temp.end());

            // Store mapping: sorted position -> original index
            for (int i = 0; i < number_of_train; ++i) {
                y_seq[dim][i] = temp[i].second;
            }

            // Compute midpoints between consecutive unique feature values
            for (int i = 0; i < number_of_train - 1; ++i) {
                division_list[dim][i] = (temp[i].first + temp[i + 1].first) / 2.0;
            }
        }

        // Initialize residuals with the original target values
        auto residual = y_train;

        // Main boosting loop
        for (int iter = 1; iter <= T; ++iter) {
            double min_mse = numeric_limits<double>::infinity();
            int best_dim = 0, best_div = 0;
            double best_left = 0.0, best_right = 0.0;

            // Search for the best weak learner (decision stump) across all features
            for (int dim = 0; dim < number_of_dimension; ++dim) {
                // Prefix sums of residuals in sorted order
                vector<double> prefix_sum(number_of_train, 0.0);
                prefix_sum[0] = residual[y_seq[dim][0]];
                for (int i = 1; i < number_of_train; ++i) {
                    prefix_sum[i] = prefix_sum[i - 1] + residual[y_seq[dim][i]];
                }

                // Precompute s2_n[i] = (prefix_sum[i-1])^2 / i
                // Used for efficient left subtree MSE update
                vector<double> s2_n(number_of_train + 1, 0.0);
                for (int i = 1; i <= number_of_train; ++i) {
                    s2_n[i] = prefix_sum[i - 1] * prefix_sum[i - 1] / (double)i;
                }

                // Precompute S2_n[i] = (suffix_sum from i)^2 / (N - i)
                // Used for efficient right subtree MSE update
                vector<double> S2_n(number_of_train + 1, 0.0);
                double suffix_sum = 0.0;
                for (int i = number_of_train - 1; i >= 0; --i) {
                    suffix_sum += residual[y_seq[dim][i]];
                    S2_n[i] = suffix_sum * suffix_sum / (double)(number_of_train - i);
                }

                // Initialize term2 as total sum of squared residuals
                // We will subtract contributions as we move split point
                double term1 = 0.0;
                double term2 = -s2_n[number_of_train];
                for (int i = 0; i < number_of_train; ++i) {
                    term2 += residual[i] * residual[i];
                }

                // Slide the split point from left to right
                for (int div = 0; div < number_of_train - 1; ++div) {
                    double left_pred = prefix_sum[div] / (double)(div + 1);
                    double right_pred = (prefix_sum[number_of_train - 1] - prefix_sum[div]) /
                        (double)(number_of_train - div - 1);

                    // Update term1 and term2 using precomputed tables
                    // Note: r_{k+1}^2 cancels out mathematically, so we omit it
                    term1 += -s2_n[div + 1] + s2_n[div];
                    term2 -= -S2_n[div] + S2_n[div + 1];

                    double total_mse = term1 + term2;

                    if (total_mse < min_mse) {
                        min_mse = total_mse;
                        best_dim = dim;
                        best_div = div;
                        best_left = left_pred;
                        best_right = right_pred;
                    }
                }
            }

            // Store the best weak learner found
            regression_machine.emplace_back(best_dim, best_div, best_left, best_right);

            // Update residuals: subtract the current weak learner's predictions
            int learner_idx = regression_machine.size() - 1;
            for (int i = 0; i < number_of_train; ++i) {
                residual[i] -= f(X_train[i], learner_idx);
            }

            // Compute current mean squared error (for monitoring and early stopping)
            double loss = 0.0;
            for (double r : residual) {
                loss += r * r;
            }
            loss /= (double)number_of_train;

            if (verbose) {
                cout << "epoch " << iter << ", loss: " << loss << endl;
            }

            // Early stopping if MSE is below threshold
            if (loss <= eps) return;
        }
    }

    // Predict on test samples
    vector<double> predict(const vector<Point>& X_test) const {
        if (!is_fitted) return {};

        int n_test = X_test.size();
        vector<double> predictions(n_test);
        for (int i = 0; i < n_test; ++i) {
            predictions[i] = F(X_test[i]);
        }
        return predictions;
    }
};