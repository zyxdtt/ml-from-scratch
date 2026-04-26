// SVM.hpp
// A complete implementation of SVM using SMO algorithm
// Supports linear kernel (via polynomial with degree=1) and RBF kernel

#pragma once
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>
#include <unordered_map>

using namespace std;
using Point = vector<double>;

class SVM {
private:
    // Model parameters
    Point alpha;                    // Lagrange multipliers, size = N
    vector<Point> X_copy;           // Copy of training data
    vector<int> y_copy;             // Copy of training labels
    double B;                       // Bias term
    double C;                       // Penalty parameter for soft margin
    int number_of_train;            // Number of training samples
    int number_of_dimension;        // Feature dimension
    double k;                       // Kernel parameter (gamma for RBF, degree for poly)
    bool is_fitted;                 // Flag indicating if model is trained
    bool use_polynomial;            // Flag for polynomial kernel
    bool use_gaussian;              // Flag for RBF kernel
    double eps;                     // Numerical tolerance

    // Compute dot product of two vectors
    double dot_product(const Point& t1, const Point& t2) const {
        double product_sum = 0.0;
        for (int i = 0; i < (int)t1.size(); i++)
            product_sum += t1[i] * t2[i];
        return product_sum;
    }

    // Compute squared Euclidean distance between two vectors
    double distance(const Point& t1, const Point& t2) const {
        double dis = 0.0;
        for (int i = 0; i < (int)t1.size(); i++)
            dis += (t1[i] - t2[i]) * (t1[i] - t2[i]);
        return dis;
    }

    // Polynomial kernel: K(x,z) = (x·z + 1)^p
    double polynomial_kernel(const Point& x, const Point& z, double p) const {
        return pow(dot_product(x, z) + 1.0, p);
    }

    // RBF (Gaussian) kernel: K(x,z) = exp(-γ * ||x-z||^2)
    double gaussian_kernel(const Point& x, const Point& z, double gamma) const {
        return exp(-gamma * distance(x, z));
    }

    // Dispatch kernel function based on type
    double kernel(const Point& x, const Point& z, double k_param) const {
        if (use_gaussian) return gaussian_kernel(x, z, k_param);
        else return polynomial_kernel(x, z, k_param);
    }

    // Decision function value before sign: g(x) = Σ α_i y_i K(x_i, x) + b
    double g(const Point& x, double k_param) const {
        double sum = 0.0;
        for (int i = 0; i < number_of_train; i++) {
            sum += alpha[i] * y_copy[i] * kernel(x, X_copy[i], k_param);
        }
        return sum + B;
    }

    // Calculate eta = K11 + K22 - 2*K12 for SMO update
    double ita(int a1, int a2, double k_param) const {
        double k11 = kernel(X_copy[a1], X_copy[a1], k_param);
        double k22 = kernel(X_copy[a2], X_copy[a2], k_param);
        double k12 = kernel(X_copy[a1], X_copy[a2], k_param);
        return k11 + k22 - 2 * k12;
    }

    // Clip alpha2 to feasible region [L, H]
    double clip(double a2_new, int a1, int a2) const {
        double L, H;
        if (y_copy[a1] == y_copy[a2]) {
            L = max(0.0, alpha[a2] + alpha[a1] - C);
            H = min(C, alpha[a2] + alpha[a1]);
        }
        else {
            L = max(0.0, alpha[a2] - alpha[a1]);
            H = min(C, C + alpha[a2] - alpha[a1]);
        }
        if (a2_new >= L && a2_new <= H) return a2_new;
        else if (a2_new > H) return H;
        else return L;
    }

    // Compute F1 score for a single class
    double F1_in_one_label(const vector<int>& pred, const vector<int>& y, int label) const {
        int tp = 0, fp = 0, fn = 0;
        for (int i = 0; i < (int)y.size(); i++) {
            if (pred[i] == label && y[i] == label) tp++;
            if (pred[i] == label && y[i] != label) fp++;
            if (pred[i] != label && y[i] == label) fn++;
        }
        return 2.0 * tp / (2.0 * tp + fp + fn);
    }

public:
    ~SVM() = default;

    // Constructor
    SVM() : B(0), use_polynomial(false), use_gaussian(false), eps(1e-6) {}

    // Train SVM model using SMO algorithm
    void fit(const vector<Point>& X_train,
        const vector<int>& y_train,
        double C = 1.0,
        double KKT_eps = 1e-3,
        double STEP_eps = 1e-3,
        string kernel_use = "Gaussian",
        double k_param = 1.0,
        int max_iter = 100) {

        is_fitted = true;

        // Set kernel type
        if (kernel_use == "Gaussian") use_gaussian = true;
        else if (kernel_use == "Poly") use_polynomial = true;
        else return;

        // Initialize training data
        number_of_train = (int)y_train.size();
        if (number_of_train == 0) return;
        number_of_dimension = (int)X_train[0].size();
        this->C = C;
        this->k = k_param;

        // Initialize error cache: E[i] = g(x_i) - y_i
        Point E(number_of_train);
        for (int i = 0; i < number_of_train; i++) {
            E[i] = -y_train[i];
        }

        X_copy = X_train;
        y_copy = y_train;
        alpha.assign(number_of_train, 0.0);
        B = 0.0;

        // Main SMO loop
        for (int iter = 1; iter <= max_iter; iter++) {
            bool any_update = false;

            // Level 1: Scan non-bound samples (0 < alpha < C)
            for (int i = 0; i < number_of_train; i++) {
                if (!(alpha[i] > 0 && alpha[i] < C)) continue;

                double g_xi = g(X_train[i], k);
                double yg = y_train[i] * g_xi;
                bool violate = false;

                // Check KKT conditions
                if (alpha[i] < KKT_eps && yg < 1.0 - KKT_eps) violate = true;
                else if (alpha[i] > KKT_eps && alpha[i] < C - KKT_eps && fabs(yg - 1.0) > KKT_eps) violate = true;
                else if (alpha[i] > C - KKT_eps && yg > 1.0 + KKT_eps) violate = true;

                if (!violate) continue;

                // Select alpha2: maximize |E1 - E2|
                int i2 = -1;
                double max_diff = -1.0;
                for (int j = 0; j < number_of_train; j++) {
                    if (j == i) continue;
                    double diff = fabs(E[i] - E[j]);
                    if (diff > max_diff) {
                        max_diff = diff;
                        i2 = j;
                    }
                }
                if (i2 == -1) continue;

                double eta = ita(i, i2, k);
                if (eta <= 1e-12) continue;

                double a2_new = alpha[i2] + y_train[i2] * (E[i] - E[i2]) / eta;
                a2_new = clip(a2_new, i, i2);

                // If step too small, try fallback strategies
                if (fabs(alpha[i2] - a2_new) < STEP_eps) {
                    // Level 2: Scan non-bound samples
                    bool succeeded = false;
                    for (int j = 0; j < number_of_train; j++) {
                        if (j == i) continue;
                        if (!(alpha[j] > 0 && alpha[j] < C)) continue;

                        double eta2 = ita(i, j, k);
                        if (eta2 <= 1e-12) continue;

                        double a2t = alpha[j] + y_train[j] * (E[i] - E[j]) / eta2;
                        a2t = clip(a2t, i, j);

                        if (fabs(alpha[j] - a2t) >= STEP_eps) {
                            double k12 = kernel(X_train[i], X_train[j], k);
                            double k11 = kernel(X_train[i], X_train[i], k);
                            double k22 = kernel(X_train[j], X_train[j], k);
                            double a1_new = alpha[i] + y_train[i] * y_train[j] * (alpha[j] - a2t);

                            double b1_new = -y_train[i] * k11 * (a1_new - alpha[i]);
                            b1_new -= y_train[j] * k12 * (a2t - alpha[j]);
                            b1_new += B - E[i];

                            double b2_new = -y_train[i] * k12 * (a1_new - alpha[i]);
                            b2_new -= y_train[j] * k22 * (a2t - alpha[j]);
                            b2_new += B - E[j];

                            if (a1_new > 0 && a1_new < C) B = b1_new;
                            else if (a2t > 0 && a2t < C) B = b2_new;
                            else B = (b1_new + b2_new) / 2.0;

                            alpha[i] = a1_new;
                            alpha[j] = a2t;

                            for (int idx = 0; idx < number_of_train; idx++)
                                E[idx] = g(X_train[idx], k) - y_train[idx];

                            succeeded = true;
                            any_update = true;
                            break;
                        }
                    }
                    if (succeeded) continue;

                    // Level 3: Scan all samples
                    for (int j = 0; j < number_of_train; j++) {
                        if (j == i) continue;

                        double eta2 = ita(i, j, k);
                        if (eta2 <= 1e-12) continue;

                        double a2t = alpha[j] + y_train[j] * (E[i] - E[j]) / eta2;
                        a2t = clip(a2t, i, j);

                        if (fabs(alpha[j] - a2t) >= STEP_eps) {
                            double k12 = kernel(X_train[i], X_train[j], k);
                            double k11 = kernel(X_train[i], X_train[i], k);
                            double k22 = kernel(X_train[j], X_train[j], k);
                            double a1_new = alpha[i] + y_train[i] * y_train[j] * (alpha[j] - a2t);

                            double b1_new = -y_train[i] * k11 * (a1_new - alpha[i]);
                            b1_new -= y_train[j] * k12 * (a2t - alpha[j]);
                            b1_new += B - E[i];

                            double b2_new = -y_train[i] * k12 * (a1_new - alpha[i]);
                            b2_new -= y_train[j] * k22 * (a2t - alpha[j]);
                            b2_new += B - E[j];

                            if (a1_new > 0 && a1_new < C) B = b1_new;
                            else if (a2t > 0 && a2t < C) B = b2_new;
                            else B = (b1_new + b2_new) / 2.0;

                            alpha[i] = a1_new;
                            alpha[j] = a2t;

                            for (int idx = 0; idx < number_of_train; idx++)
                                E[idx] = g(X_train[idx], k) - y_train[idx];

                            any_update = true;
                            break;
                        }
                    }
                }
                else {
                    // Level 1 succeeded, update directly
                    double k12 = kernel(X_train[i], X_train[i2], k);
                    double k11 = kernel(X_train[i], X_train[i], k);
                    double k22 = kernel(X_train[i2], X_train[i2], k);
                    double a1_new = alpha[i] + y_train[i] * y_train[i2] * (alpha[i2] - a2_new);

                    double b1_new = -y_train[i] * k11 * (a1_new - alpha[i]);
                    b1_new -= y_train[i2] * k12 * (a2_new - alpha[i2]);
                    b1_new += B - E[i];

                    double b2_new = -y_train[i] * k12 * (a1_new - alpha[i]);
                    b2_new -= y_train[i2] * k22 * (a2_new - alpha[i2]);
                    b2_new += B - E[i2];

                    if (a1_new > 0 && a1_new < C) B = b1_new;
                    else if (a2_new > 0 && a2_new < C) B = b2_new;
                    else B = (b1_new + b2_new) / 2.0;

                    alpha[i] = a1_new;
                    alpha[i2] = a2_new;

                    for (int idx = 0; idx < number_of_train; idx++)
                        E[idx] = g(X_train[idx], k) - y_train[idx];

                    any_update = true;
                }
            }

            // If no non-bound updates found, scan all samples (including bounds)
            if (!any_update) {
                for (int i = 0; i < number_of_train; i++) {
                    double g_xi = g(X_train[i], k);
                    double yg = y_train[i] * g_xi;
                    bool violate = false;

                    if (alpha[i] < KKT_eps && yg < 1.0 - KKT_eps) violate = true;
                    else if (alpha[i] > KKT_eps && alpha[i] < C - KKT_eps && fabs(yg - 1.0) > KKT_eps) violate = true;
                    else if (alpha[i] > C - KKT_eps && yg > 1.0 + KKT_eps) violate = true;

                    if (!violate) continue;

                    // Try to find a good alpha2
                    int i2 = -1;
                    double max_diff = -1.0;
                    for (int j = 0; j < number_of_train; j++) {
                        if (j == i) continue;
                        double diff = fabs(E[i] - E[j]);
                        if (diff > max_diff) {
                            max_diff = diff;
                            i2 = j;
                        }
                    }
                    if (i2 == -1) continue;

                    double eta = ita(i, i2, k);
                    if (eta <= 1e-12) continue;

                    double a2_new = alpha[i2] + y_train[i2] * (E[i] - E[i2]) / eta;
                    a2_new = clip(a2_new, i, i2);

                    if (fabs(alpha[i2] - a2_new) < STEP_eps) continue;

                    double k12 = kernel(X_train[i], X_train[i2], k);
                    double k11 = kernel(X_train[i], X_train[i], k);
                    double k22 = kernel(X_train[i2], X_train[i2], k);
                    double a1_new = alpha[i] + y_train[i] * y_train[i2] * (alpha[i2] - a2_new);

                    double b1_new = -y_train[i] * k11 * (a1_new - alpha[i]);
                    b1_new -= y_train[i2] * k12 * (a2_new - alpha[i2]);
                    b1_new += B - E[i];

                    double b2_new = -y_train[i] * k12 * (a1_new - alpha[i]);
                    b2_new -= y_train[i2] * k22 * (a2_new - alpha[i2]);
                    b2_new += B - E[i2];

                    if (a1_new > 0 && a1_new < C) B = b1_new;
                    else if (a2_new > 0 && a2_new < C) B = b2_new;
                    else B = (b1_new + b2_new) / 2.0;

                    alpha[i] = a1_new;
                    alpha[i2] = a2_new;

                    for (int idx = 0; idx < number_of_train; idx++)
                        E[idx] = g(X_train[idx], k) - y_train[idx];

                    any_update = true;
                    break;
                }
            }

            // Stop if no updates were made in this iteration
            if (!any_update) break;
        }
    }

    // Predict labels for test samples
    vector<int> predict(const vector<Point>& X_test) const {
        vector<int> y_test;
        if (!is_fitted) return y_test;

        int number_of_test = (int)X_test.size();
        y_test.resize(number_of_test);

        for (int i = 0; i < number_of_test; i++) {
            double val = g(X_test[i], k);
            y_test[i] = (val <= 0) ? -1 : 1;
        }
        return y_test;
    }

    // Compute weighted F1 score
    double weighted_F1(const vector<Point>& X, const vector<int>& y) const {
        vector<int> y_pred = predict(X);
        if (y_pred.empty() || y_pred.size() != y.size()) return 0.0;

        int y_size = (int)y.size();

        // Count samples per class
        unordered_map<int, int> classify;
        for (int label : y) classify[label]++;

        // Weighted average of per-class F1 scores
        double weighted_f1 = 0.0;
        for (const auto& pair : classify) {
            int label = pair.first;
            int count = pair.second;
            weighted_f1 += (count * 1.0 / y_size) * F1_in_one_label(y_pred, y, label);
        }
        return weighted_f1;
    }
};