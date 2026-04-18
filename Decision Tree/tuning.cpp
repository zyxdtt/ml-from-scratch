// test_decision_tree_tuned.cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>
#include "Decision_Tree.hpp"

using namespace std;

// Helper function to read X CSV files
vector<Point> read_X_csv(const string& filename) {
    vector<Point> data;
    ifstream file(filename);
    string line;

    while (getline(file, line)) {
        if (line.empty()) continue;
        Point point;
        stringstream ss(line);
        string value;

        while (getline(ss, value, ',')) {
            point.push_back(stod(value));
        }
        data.push_back(point);
    }

    return data;
}

// Helper function to read y CSV files for regression
vector<double> read_y_csv_regression(const string& filename) {
    vector<double> data;
    ifstream file(filename);
    string line;

    getline(file, line);  // Only one line
    stringstream ss(line);
    string value;

    while (getline(ss, value, ',')) {
        data.push_back(stod(value));
    }

    return data;
}

// Helper function to read y CSV files for classification
vector<int> read_y_csv_classification(const string& filename) {
    vector<int> data;
    ifstream file(filename);
    string line;

    getline(file, line);  // Only one line
    stringstream ss(line);
    string value;

    while (getline(ss, value, ',')) {
        data.push_back(stoi(value));
    }

    return data;
}

void test_regression_tree() {
    cout << "\n========================================" << endl;
    cout << "Testing Regression Tree - California Housing Dataset" << endl;
    cout << "========================================" << endl;

    // Load data
    cout << "Loading data..." << endl;
    auto X_train = read_X_csv("data/california_train_X.csv");
    auto y_train = read_y_csv_regression("data/california_train_y.csv");
    auto X_test = read_X_csv("data/california_test_X.csv");
    auto y_test = read_y_csv_regression("data/california_test_y.csv");

    cout << "Training set: " << X_train.size() << " samples, "
        << X_train[0].size() << " features" << endl;
    cout << "Test set: " << X_test.size() << " samples" << endl;

    // Train
    Decision_Tree_Regression tree;

    cout << "\nTraining..." << endl;
    auto start = chrono::high_resolution_clock::now();

    tree.fit(X_train, y_train, 0.01);  // variance threshold = 0.01

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Training completed! Time: " << duration.count() << " ms" << endl;

    // Predict
    cout << "\nPredicting..." << endl;
    start = chrono::high_resolution_clock::now();

    auto y_pred = tree.predict(X_test);

    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Prediction completed! Time: " << duration.count() << " ms" << endl;

    // Evaluate
    double mse = 0, mae = 0;
    for (size_t i = 0; i < y_test.size(); i++) {
        double diff = y_test[i] - y_pred[i];
        mse += diff * diff;
        mae += abs(diff);
    }
    mse /= y_test.size();
    mae /= y_test.size();
    double rmse = sqrt(mse);

    cout << "\nEvaluation Results:" << endl;
    cout << fixed << setprecision(6);
    cout << "  MSE:  " << mse << endl;
    cout << "  RMSE: " << rmse << endl;
    cout << "  MAE:  " << mae << endl;

    // Calculate R²
    double y_mean = 0;
    for (double y : y_test) y_mean += y;
    y_mean /= y_test.size();

    double ss_tot = 0, ss_res = 0;
    for (size_t i = 0; i < y_test.size(); i++) {
        ss_tot += (y_test[i] - y_mean) * (y_test[i] - y_mean);
        ss_res += (y_test[i] - y_pred[i]) * (y_test[i] - y_pred[i]);
    }
    double r2 = 1 - ss_res / ss_tot;
    cout << "  R²:   " << r2 << endl;
}

void test_classification_tree() {
    cout << "\n========================================" << endl;
    cout << "Testing Classification Tree - Digits Dataset" << endl;
    cout << "========================================" << endl;

    // Load data
    cout << "Loading data..." << endl;
    auto X_train = read_X_csv("data/digits_train_X.csv");
    auto y_train = read_y_csv_classification("data/digits_train_y.csv");
    auto X_test = read_X_csv("data/digits_test_X.csv");
    auto y_test = read_y_csv_classification("data/digits_test_y.csv");

    cout << "Training set: " << X_train.size() << " samples, "
        << X_train[0].size() << " features" << endl;
    cout << "Test set: " << X_test.size() << " samples" << endl;

    // Train
    Decision_Tree_Classifier tree;

    cout << "\nTraining..." << endl;
    auto start = chrono::high_resolution_clock::now();

    tree.fit(X_train, y_train, 5, 0.01);  // min samples = 5, Gini threshold = 0.01

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Training completed! Time: " << duration.count() << " ms" << endl;

    // Predict
    cout << "\nPredicting..." << endl;
    start = chrono::high_resolution_clock::now();

    auto y_pred = tree.predict(X_test);

    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Prediction completed! Time: " << duration.count() << " ms" << endl;

    // Evaluate
    int correct = 0;
    for (size_t i = 0; i < y_test.size(); i++) {
        if (y_test[i] == y_pred[i]) correct++;
    }
    double accuracy = correct * 100.0 / y_test.size();

    cout << "\nEvaluation Results:" << endl;
    cout << fixed << setprecision(4);
    cout << "  Accuracy: " << accuracy << "% (" << correct << "/" << y_test.size() << ")" << endl;

    // Use built-in weighted F1 score
    double weighted_f1 = tree.weighted_F1(X_test, y_test);
    cout << "  Weighted F1: " << weighted_f1 << endl;
}


void tune_regression_tree(const vector<Point>& X_train,
    const vector<double>& y_train,
    const vector<Point>& X_test,
    const vector<double>& y_test) {
    cout << "\n========================================" << endl;
    cout << "Tuning Regression Tree Parameters" << endl;
    cout << "========================================" << endl;

    vector<double> thresholds = { 0.05, 0.02, 0.01, 0.005, 0.002, 0.001 };

    cout << "\nThreshold |   RMSE   |   MAE    |   R²     | Time(ms)" << endl;
    cout << "----------|----------|----------|----------|---------" << endl;

    double best_rmse = 1e10;
    double best_threshold = 0.01;

    for (double threshold : thresholds) {
        Decision_Tree_Regression tree;

        auto start = chrono::high_resolution_clock::now();
        tree.fit(X_train, y_train, threshold);
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

        auto y_pred = tree.predict(X_test);

        // Calculate metrics
        double mse = 0, mae = 0;
        for (size_t i = 0; i < y_test.size(); i++) {
            double diff = y_test[i] - y_pred[i];
            mse += diff * diff;
            mae += abs(diff);
        }
        mse /= y_test.size();
        mae /= y_test.size();
        double rmse = sqrt(mse);

        // Calculate R²
        double y_mean = 0;
        for (double y : y_test) y_mean += y;
        y_mean /= y_test.size();

        double ss_tot = 0, ss_res = 0;
        for (size_t i = 0; i < y_test.size(); i++) {
            ss_tot += (y_test[i] - y_mean) * (y_test[i] - y_mean);
            ss_res += (y_test[i] - y_pred[i]) * (y_test[i] - y_pred[i]);
        }
        double r2 = 1 - ss_res / ss_tot;

        cout << fixed << setprecision(3);
        cout << setw(8) << threshold << " | "
            << setw(8) << setprecision(4) << rmse << " | "
            << setw(8) << mae << " | "
            << setw(8) << r2 << " | "
            << setw(7) << duration.count() << endl;

        if (rmse < best_rmse) {
            best_rmse = rmse;
            best_threshold = threshold;
        }
    }

    cout << "\nBest threshold: " << best_threshold
        << " (RMSE = " << best_rmse << ")" << endl;
}

void tune_classification_tree(const vector<Point>& X_train,
    const vector<int>& y_train,
    const vector<Point>& X_test,
    const vector<int>& y_test) {
    cout << "\n========================================" << endl;
    cout << "Tuning Classification Tree Parameters" << endl;
    cout << "========================================" << endl;

    vector<int> min_samples_list = { 2, 3, 5, 10, 20 };
    vector<double> gini_thresholds = { 0.1, 0.05, 0.01, 0.005, 0.001 };

    cout << "\nMin Samples | Gini | Accuracy |  F1 Score | Time(ms)" << endl;
    cout << "------------|------|----------|-----------|---------" << endl;

    double best_accuracy = 0;
    pair<int, double> best_params = { 5, 0.01 };

    for (int min_samples : min_samples_list) {
        for (double gini_thresh : gini_thresholds) {
            Decision_Tree_Classifier tree;

            auto start = chrono::high_resolution_clock::now();
            tree.fit(X_train, y_train, min_samples, gini_thresh);
            auto end = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

            auto y_pred = tree.predict(X_test);

            int correct = 0;
            for (size_t i = 0; i < y_test.size(); i++) {
                if (y_test[i] == y_pred[i]) correct++;
            }
            double accuracy = correct * 100.0 / y_test.size();
            double f1 = tree.weighted_F1(X_test, y_test);

            cout << setw(10) << min_samples << " | "
                << setw(4) << fixed << setprecision(3) << gini_thresh << " | "
                << setw(7) << setprecision(2) << accuracy << "% | "
                << setw(8) << setprecision(4) << f1 << " | "
                << setw(7) << duration.count() << endl;

            if (accuracy > best_accuracy) {
                best_accuracy = accuracy;
                best_params = { min_samples, gini_thresh };
            }
        }
    }

    cout << "\nBest parameters: min_samples = " << best_params.first
        << ", Gini threshold = " << best_params.second
        << " (Accuracy = " << best_accuracy << "%)" << endl;
}

int main() {
    cout << "========================================" << endl;
    cout << "   Decision Tree Hyperparameter Tuning" << endl;
    cout << "========================================" << endl;

    // Load regression data
    cout << "\nLoading California Housing dataset..." << endl;
    auto X_train_reg = read_X_csv("data/california_train_X.csv");
    auto y_train_reg = read_y_csv_regression("data/california_train_y.csv");
    auto X_test_reg = read_X_csv("data/california_test_X.csv");
    auto y_test_reg = read_y_csv_regression("data/california_test_y.csv");

    tune_regression_tree(X_train_reg, y_train_reg, X_test_reg, y_test_reg);

    // Load classification data
    cout << "\nLoading Digits dataset..." << endl;
    auto X_train_clf = read_X_csv("data/digits_train_X.csv");
    auto y_train_clf = read_y_csv_classification("data/digits_train_y.csv");
    auto X_test_clf = read_X_csv("data/digits_test_X.csv");
    auto y_test_clf = read_y_csv_classification("data/digits_test_y.csv");

    tune_classification_tree(X_train_clf, y_train_clf, X_test_clf, y_test_clf);

    cout << "\n========================================" << endl;
    cout << "Tuning completed!" << endl;
    cout << "========================================" << endl;

    return 0;
}