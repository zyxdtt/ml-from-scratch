//2026 4 11
#pragma once
#include <vector>
#include <string>
#include <sstream>

using namespace std;

class perceptron {
private:
    vector<double> W;
    double B;
    int dimension;
    bool is_fitted;

    double dot_product(const vector<double>& a, const vector<double>& b) const {
        double product = 0;
        for (size_t i = 0; i < a.size(); i++) product += a[i] * b[i];
        return product;
    }

public:
    perceptron(int dim = 0) : B(0), is_fitted(false), dimension(dim) {
        if (dim > 0) W.resize(dim, 0);
    }

    ~perceptron() = default;

    string fit(const vector<vector<double>>& X_train,
        const vector<int>& y_train,
        int max_iter = 1000,
        double learning_rate = 0.5) {
        int number_of_train = y_train.size();
        if (number_of_train == 0 || X_train.empty()) {
            return "Error: Empty training data.";
        }
        if (dimension == 0) {
            dimension = X_train[0].size();
            W.assign(dimension, 0.0);
        }
        W.assign(dimension, 0.0);
        B = 0.0;
        is_fitted = true;
        int original_max_iter = max_iter;
        while (max_iter--) {
            bool converged = true;
            for (int i = 0; i < number_of_train; i++) {
                if (y_train[i] * (dot_product(X_train[i], W) + B) <= 0) {
                    converged = false;
                    for (int d = 0; d < dimension; d++) {
                        W[d] += learning_rate * y_train[i] * X_train[i][d];
                    }
                    B += learning_rate * y_train[i];
                }
            }
            if (converged) {
                return "Converged after " + to_string(original_max_iter - max_iter) + " epochs.";
            }
        }
        return "Model not converged after " + to_string(original_max_iter) + " epochs. You can use continual_fit() method.";
    }

    string continual_fit(const vector<vector<double>>& X_train,
        const vector<int>& y_train,
        int max_iter = 1000,
        double learning_rate = 0.5) {
        if (!is_fitted) {
            return "Error: Model not fitted yet. Call fit() first.";
        }
        int number_of_train = y_train.size();
        if (number_of_train == 0 || X_train.empty()) {
            return "Error: Empty training data.";
        }
        if ((int)X_train[0].size() != dimension) {
            return "Error: Dimension mismatch.";
        }
        int original_max_iter = max_iter;
        while (max_iter--) {
            bool converged = true;
            for (int i = 0; i < number_of_train; i++) {
                if (y_train[i] * (dot_product(X_train[i], W) + B) <= 0) {
                    converged = false;
                    for (int d = 0; d < dimension; d++) {
                        W[d] += learning_rate * y_train[i] * X_train[i][d];
                    }
                    B += learning_rate * y_train[i];  
                }
            }
            if (converged) {
                return "Converged after " + to_string(original_max_iter - max_iter) + " additional epochs.";
            }
        }
        return "Model still not converged after " + to_string(original_max_iter) + " additional epochs.";
    }

    vector<int> predict(const vector<vector<double>>& X_test) const {
        vector<int> y_test;
        if (!is_fitted) {
            return y_test;  
        }
        if (X_test.empty()) {
            return y_test;
        }
        if ((int)X_test[0].size() != dimension) {
            return y_test;
        }
        int number_of_tests = X_test.size();
        y_test.reserve(number_of_tests);
        for (int i = 0; i < number_of_tests; i++) {
            double score = dot_product(W, X_test[i]) + B;
            y_test.push_back(score > 0 ? 1 : -1);
        }
        return y_test;
    }

    double accuracy(const vector<vector<double>>& X, const vector<int>& y) const {
        auto y_pred = predict(X);
        if (y_pred.empty() || y_pred.size() != y.size()) return 0.0;
        int correct = 0;
        for (size_t i = 0; i < y.size(); i++) {
            if (y_pred[i] == y[i]) correct++;
        }
        return static_cast<double>(correct) / y.size();
    }

    double precision(const vector<vector<double>>& X, const vector<int>& y) const {
        auto y_pred = predict(X);
        if (y_pred.empty() || y_pred.size() != y.size()) return 0.0;
        int tp = 0, tpfp = 0;
        for (size_t i = 0; i < y.size(); i++) {
            if (y_pred[i] == 1) {  
                tpfp++;
                if (y[i] == 1) tp++;
            }
        }
        if (tpfp == 0) return 0.0;
        return static_cast<double>(tp) / tpfp;
    }

    double recall(const vector<vector<double>>& X, const vector<int>& y) const {
        auto y_pred = predict(X);
        if (y_pred.empty() || y_pred.size() != y.size()) return 0.0;
        int tp = 0, tpfn = 0;
        for (size_t i = 0; i < y.size(); i++) {
            if (y[i] == 1) {  
                tpfn++;
                if (y_pred[i] == 1) tp++;
            }
        }
        if (tpfn == 0) return 0.0;
        return static_cast<double>(tp) / tpfn;
    }

    double f1(const vector<vector<double>>& X, const vector<int>& y) const {
        double p = precision(X, y);
        double r = recall(X, y);
        if (p + r == 0.0) return 0.0;
        return 2.0 * p * r / (p + r);
    }

    string classification_report(const vector<vector<double>>& X,
        const vector<int>& y) const {
        double acc = accuracy(X, y);
        double prec = precision(X, y);
        double rec = recall(X, y);
        double f1_val = f1(X, y);
        stringstream oss;
        oss << "Accuracy:  " << acc * 100 << "%\n";
        oss << "Precision: " << prec * 100 << "%\n";
        oss << "Recall:    " << rec * 100 << "%\n";
        oss << "F1 Score:  " << f1_val * 100 << "%\n";
        return oss.str();
    }

    const vector<double>& weights() const { return W; }
    double bias() const { return B; }
    bool fitted() const { return is_fitted; }
};