// test_decision_tree.cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <cmath>
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

int main() {
    cout << "========================================" << endl;
    cout << "   Decision Tree Algorithm Test" << endl;
    cout << "========================================" << endl;

    try {
        test_regression_tree();
        test_classification_tree();

        cout << "\n========================================" << endl;
        cout << "All tests completed!" << endl;
        cout << "========================================" << endl;
    }
    catch (const exception& e) {
        cerr << "\nError: " << e.what() << endl;
        return 1;
    }

    return 0;
}