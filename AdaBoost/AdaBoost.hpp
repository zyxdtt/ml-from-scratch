//2026 5 3
#pragma once
#include <vector>
#include <utility>
#include <unordered_map>
#include <algorithm>
#include <cmath>

using namespace std;
using Point = vector<double>;  // A single data point represented as a vector of feature values

// Structure to store a weak classifier (decision stump)
struct machine {
    double a;           // Alpha weight of this classifier in the final ensemble
    int dimension;      // Which feature dimension to split on
    int division;       // Index into division_list (not the actual threshold value)
};

class AdaBoost {
private:
    vector<double> weights;              // Sample weights for the current boosting round
    vector<machine> classify_machine;    // Ensemble of weak classifiers collected so far
    int number_of_train;                 // Number of training samples
    int number_of_dimension;             // Number of features per sample
    int number_of_machine;               // Maximum number of weak classifiers (T parameter)
    bool is_fitted;                      // Flag indicating whether the model has been trained
    vector<Point> division_list;         // Precomputed candidate thresholds for each feature dimension

    // Prediction of the i-th weak classifier on a single sample x
    int g(const Point& x, int i) const {
        // Retrieve the actual threshold value using the stored dimension and division index
        double threshold = division_list[classify_machine[i].dimension][classify_machine[i].division];
        if (x[classify_machine[i].dimension] <= threshold) return -1;
        else return 1;
    }

    // Final ensemble prediction (weighted majority vote) on a single sample x
    int G(const Point& x) const {
        double final_decision = 0.0;
        // Sum over all weak classifiers: alpha * prediction
        for (auto [am, dimension, division] : classify_machine) {
            double decision;
            double threshold = division_list[dimension][division];
            if (x[dimension] <= threshold) decision = -1;
            else decision = 1;
            final_decision += am * decision;
        }
        // Return -1 if total <= 0, otherwise +1
        return (final_decision <= 0) ? -1 : 1;
    }

public:
    AdaBoost() = default;
    ~AdaBoost() = default;

    // Train the AdaBoost model
    // X_train: training feature matrix (each Point is a sample)
    // y_train: training labels, must be +1 or -1
    // T: maximum number of weak classifiers (iterations)
    // verbose: print progress if true
    void fit(const vector<Point>& X_train,
        const vector<int>& y_train,
        const int T = 50,
        bool verbose = true) {

        is_fitted = true;
        number_of_train = y_train.size();
        number_of_dimension = X_train[0].size();

        // Initialize sample weights uniformly
        weights.resize(number_of_train, 1.0 / number_of_train);
        number_of_machine = T;

        // Precompute candidate split thresholds for each feature dimension
        // For each dimension, take the midpoints between consecutive sorted unique feature values
        division_list.assign(number_of_dimension, Point(number_of_train - 1));
        for (int dimension = 0; dimension < number_of_dimension; dimension++) {
            vector<double> temp(number_of_train);
            for (int i = 0; i < number_of_train; i++) {
                temp[i] = X_train[i][dimension];
            }
            sort(temp.begin(), temp.end());
            for (int i = 0; i < number_of_train - 1; i++) {
                double mid = (temp[i] + temp[i + 1]) / 2.0;
                division_list[dimension][i] = mid;
            }
        }

        // Main AdaBoost loop
        for (int machine = 1; machine <= number_of_machine; machine++) {
            double max_abs_loss = 0.0;   // Tracks the maximum distance from 0.5 (best classifier)
            int best_dim = 0;            // Best feature dimension found
            int best_div = 0;            // Best division index found
            double best_em = 0.0;        // Weighted error rate of the best classifier

            // --- Step 1: Find the best weak classifier (decision stump) ---
            // Iterate over all dimensions and all candidate thresholds
            for (int dimension = 0; dimension < number_of_dimension; dimension++) {
                for (int division = 0; division < number_of_train - 1; division++) {
                    double em = 0.0;   // Weighted error for this (dimension, threshold) pair
                    double threshold = division_list[dimension][division];

                    // Compute weighted classification error
                    for (int i = 0; i < number_of_train; i++) {
                        // Misclassification condition:
                        // - If feature <= threshold, predict -1. Error when true label != -1
                        if (X_train[i][dimension] <= threshold && y_train[i] != -1) {
                            em += weights[i];
                        }
                        // - If feature > threshold, predict +1. Error when true label != +1
                        if (X_train[i][dimension] > threshold && y_train[i] != 1) {
                            em += weights[i];
                        }
                    }

                    // Select the classifier whose error rate is farthest from 0.5
                    // (i.e., the most informative one, far from random guessing)
                    if (abs(em - 0.5) >= max_abs_loss) {
                        max_abs_loss = abs(em - 0.5);
                        best_em = em;
                        best_dim = dimension;
                        best_div = division;
                    }
                }
            }

            // --- Step 2: Check termination conditions ---
            if (best_em == 0.0) {
                // Perfect classifier found (zero error) - can stop training
                classify_machine.emplace_back(1.0, best_dim, best_div);
                return;
            }
            else if (abs(best_em - 1.0) < 1e-6) {
                // Completely wrong classifier (100% error) - invert it and stop
                classify_machine.emplace_back(-1.0, best_dim, best_div);
                return;
            }
            else if (abs(best_em - 0.5) < 1e-6) {
                // Classifier is no better than random guessing - cannot improve further
                classify_machine.emplace_back(1.0, best_dim, best_div);
                return;
            }
            else {
                // --- Step 3: Compute alpha (weight of this weak classifier) ---
                // alpha = 0.5 * ln((1 - error) / error)
                double am = 0.5 * log((1 - best_em) / best_em);
                classify_machine.emplace_back(am, best_dim, best_div);

                // --- Step 4: Update sample weights for the next round ---
                // new_weight = old_weight * exp(-alpha * y * h(x))
                vector<double> new_weights(number_of_train);
                double zm = 0.0;   // Normalization factor (sum of new weights)
                for (int train = 0; train < number_of_train; train++) {
                    // Use the most recently added classifier (index = machine-1)
                    double exp_loss = exp(-am * y_train[train] * g(X_train[train], machine - 1));
                    new_weights[train] = weights[train] * exp_loss;
                    zm += new_weights[train];
                }
                // Normalize so that weights sum to 1
                for (auto& w : new_weights) w /= zm;
                weights = move(new_weights);
            }

            // --- Step 5: Check training accuracy for early stopping ---
            double acc = accuracy(X_train, y_train);
            if (verbose) {
                cout << "epoch: " << machine << ", train accuracy: " << acc << endl;
            }
            // Stop if we have perfect classification on the training set
            if (acc == 1.0) return;
        }
    }

    // Predict labels for test samples
    vector<int> predict(const vector<Point>& X_test) const {
        vector<int> y_test;
        if (!is_fitted) return y_test;   // Model not trained yet
        int number_of_test = X_test.size();
        y_test.resize(number_of_test);
        for (int i = 0; i < number_of_test; i++) {
            y_test[i] = G(X_test[i]);    // Use ensemble prediction
        }
        return y_test;
    }

    // Compute accuracy (fraction of correct predictions)
    double accuracy(const vector<Point>& X_test,
        const vector<int>& y_test) const {
        vector<int> pred = move(predict(X_test));
        int number_of_test = y_test.size();
        int correct = 0;
        for (int i = 0; i < number_of_test; i++) {
            if (pred[i] == y_test[i]) correct++;
        }
        return correct * 1.0 / number_of_test;
    }
};