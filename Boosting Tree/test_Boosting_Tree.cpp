// test_california.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include "Boosting_Tree.hpp"

using namespace std;

vector<Point> load_data(const string& filename, vector<double>& targets) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return {};
    }

    int n_samples, n_features;
    file >> n_samples >> n_features;

    cout << "Loading " << n_samples << " samples, " << n_features << " features" << endl;

    vector<Point> data(n_samples, Point(n_features));
    targets.resize(n_samples);

    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_features; j++) {
            file >> data[i][j];
        }
        file >> targets[i];
    }

    return data;
}

int main() {
    cout << "========================================" << endl;
    cout << "Gradient Boosting Tree Test" << endl;
    cout << "California Housing Dataset" << endl;
    cout << "========================================" << endl;

    auto start = chrono::high_resolution_clock::now();

    vector<double> y_train, y_test;
    vector<Point> X_train = load_data("california_train.txt", y_train);
    vector<Point> X_test = load_data("california_test.txt", y_test);

    if (X_train.empty() || X_test.empty()) {
        cerr << "Failed to load data" << endl;
        return 1;
    }

    cout << "\nTrain data: " << X_train.size() << " samples, "
        << X_train[0].size() << " features" << endl;
    cout << "Test data: " << X_test.size() << " samples" << endl;

    // Display first 3 samples
    cout << "\nFirst 3 training samples (first 3 features):" << endl;
    for (int i = 0; i < min(3, (int)X_train.size()); i++) {
        cout << "  Sample " << i << ": ";
        for (int j = 0; j < min(3, (int)X_train[i].size()); j++) {
            cout << X_train[i][j] << " ";
        }
        cout << "... target=" << y_train[i] << endl;
    }

    auto load_end = chrono::high_resolution_clock::now();
    auto load_duration = chrono::duration_cast<chrono::seconds>(load_end - start);
    cout << "\nLoad time: " << load_duration.count() << " seconds" << endl;

    cout << "\nStarting Gradient Boosting training..." << endl;
    auto train_start = chrono::high_resolution_clock::now();

    Boosting_Tree model;
    model.fit(X_train, y_train, 50, 1e-4, true);

    auto train_end = chrono::high_resolution_clock::now();
    auto train_duration = chrono::duration_cast<chrono::seconds>(train_end - train_start);
    cout << "\nTraining time: " << train_duration.count() << " seconds" << endl;

    auto predict_start = chrono::high_resolution_clock::now();

    vector<double> train_pred = model.predict(X_train);
    vector<double> test_pred = model.predict(X_test);

    auto predict_end = chrono::high_resolution_clock::now();
    auto predict_duration = chrono::duration_cast<chrono::seconds>(predict_end - predict_start);
    cout << "Prediction time: " << predict_duration.count() << " seconds" << endl;

    // Calculate MSE
    double train_mse = 0.0, test_mse = 0.0;
    for (size_t i = 0; i < train_pred.size(); i++) {
        double diff = train_pred[i] - y_train[i];
        train_mse += diff * diff;
    }
    train_mse /= train_pred.size();

    for (size_t i = 0; i < test_pred.size(); i++) {
        double diff = test_pred[i] - y_test[i];
        test_mse += diff * diff;
    }
    test_mse /= test_pred.size();

    cout << "\n========================================" << endl;
    cout << "Final Results:" << endl;
    cout << "========================================" << endl;
    cout << "Train MSE: " << train_mse << endl;
    cout << "Test MSE: " << test_mse << endl;
    cout << "Train RMSE: " << sqrt(train_mse) << endl;
    cout << "Test RMSE: " << sqrt(test_mse) << endl;
    cout << "========================================" << endl;

    return 0;
}