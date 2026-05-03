// test_AdaBoost.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include "AdaBoost.hpp"

using namespace std;

vector<Point> load_data(const string& filename, vector<int>& labels) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return {};
    }

    int n_samples, n_features;
    file >> n_samples >> n_features;

    cout << "Loading " << n_samples << " samples, " << n_features << " features" << endl;

    vector<Point> data(n_samples, Point(n_features));
    labels.resize(n_samples);

    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_features; j++) {
            file >> data[i][j];
        }
        file >> labels[i];
    }

    return data;
}

int main() {
    cout << "========================================" << endl;
    cout << "AdaBoost Test with Breast Cancer Dataset" << endl;
    cout << "========================================" << endl;

    auto start = chrono::high_resolution_clock::now();

    vector<int> y_train, y_test;
    vector<Point> X_train = load_data("cancer_train.txt", y_train);
    vector<Point> X_test = load_data("cancer_test.txt", y_test);

    if (X_train.empty() || X_test.empty()) {
        cerr << "Failed to load data" << endl;
        return 1;
    }

    cout << "\nTrain data: " << X_train.size() << " samples, "
        << X_train[0].size() << " features" << endl;
    cout << "Test data: " << X_test.size() << " samples, "
        << X_test[0].size() << " features" << endl;

    // 显示前3个样本（用于验证）
    cout << "\nFirst 3 training samples (first 5 features):" << endl;
    for (int i = 0; i < min(3, (int)X_train.size()); i++) {
        cout << "  Sample " << i << ": ";
        for (int j = 0; j < min(5, (int)X_train[i].size()); j++) {
            cout << X_train[i][j] << " ";
        }
        cout << "... label=" << y_train[i] << endl;
    }

    auto load_end = chrono::high_resolution_clock::now();
    auto load_duration = chrono::duration_cast<chrono::seconds>(load_end - start);
    cout << "\nLoad time: " << load_duration.count() << " seconds" << endl;

    cout << "\nStarting AdaBoost training..." << endl;
    auto train_start = chrono::high_resolution_clock::now();

    // 训练 AdaBoost（T=50 应该就够了）
    AdaBoost model;
    model.fit(X_train, y_train, 50, true);

    auto train_end = chrono::high_resolution_clock::now();
    auto train_duration = chrono::duration_cast<chrono::seconds>(train_end - train_start);
    cout << "\nTraining time: " << train_duration.count() << " seconds" << endl;

    auto test_start = chrono::high_resolution_clock::now();

    double train_acc = model.accuracy(X_train, y_train);
    double test_acc = model.accuracy(X_test, y_test);

    auto test_end = chrono::high_resolution_clock::now();
    auto test_duration = chrono::duration_cast<chrono::seconds>(test_end - test_start);

    cout << "Testing time: " << test_duration.count() << " seconds" << endl;

    cout << "========================================" << endl;
    cout << "Final Results:" << endl;
    cout << "Train accuracy: " << train_acc << endl;
    cout << "Test accuracy: " << test_acc << endl;
    cout << "========================================" << endl;

    return 0;
}