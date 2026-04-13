#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <unordered_set>
#include "KNN.hpp"

using namespace std;

bool load_digits(const string& filename, vector<Point>& X, vector<int>& y) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Cannot open file " << filename << endl;
        return false;
    }

    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        string val;
        vector<double> point;
        int label;

        getline(ss, val, ',');
        label = stoi(val);

        while (getline(ss, val, ',')) {
            point.push_back(stod(val));
        }

        if (X.empty()) {
            cout << "Feature dimension: " << point.size() << endl;
        }

        y.push_back(label);
        X.push_back(point);
    }

    file.close();
    return true;
}

void shuffle_data(vector<Point>& X, vector<int>& y) {
    random_device rd;
    mt19937 gen(rd());

    vector<size_t> indices(X.size());
    for (size_t i = 0; i < indices.size(); i++) indices[i] = i;
    shuffle(indices.begin(), indices.end(), gen);

    vector<Point> X_shuffled(X.size());
    vector<int> y_shuffled(y.size());

    for (size_t i = 0; i < indices.size(); i++) {
        X_shuffled[i] = X[indices[i]];
        y_shuffled[i] = y[indices[i]];
    }

    X = move(X_shuffled);
    y = move(y_shuffled);
}

double compute_accuracy(const vector<int>& pred, const vector<int>& truth) {
    if (pred.size() != truth.size()) return 0.0;
    int correct = 0;
    for (size_t i = 0; i < pred.size(); i++) {
        if (pred[i] == truth[i]) correct++;
    }
    return static_cast<double>(correct) / pred.size();
}

int main() {
    cout << "========================================" << endl;
    cout << "KNN Test on Digits Dataset" << endl;
    cout << "========================================" << endl << endl;

    vector<Point> X;
    vector<int> y;

    if (!load_digits("digits.csv", X, y)) {
        return 1;
    }

    cout << "Total samples: " << X.size() << endl;
    cout << "Feature dimension: " << X[0].size() << endl;

    unordered_set<int> unique_labels(y.begin(), y.end());
    cout << "Number of classes: " << unique_labels.size() << " (0-" << unique_labels.size() - 1 << ")" << endl << endl;

    shuffle_data(X, y);

    size_t train_size = static_cast<size_t>(X.size() * 0.8);
    size_t test_size = X.size() - train_size;

    vector<Point> X_train(X.begin(), X.begin() + train_size);
    vector<Point> X_test(X.begin() + train_size, X.end());
    vector<int> y_train(y.begin(), y.begin() + train_size);
    vector<int> y_test(y.begin() + train_size, y.end());

    cout << "Training set size: " << train_size << endl;
    cout << "Test set size: " << test_size << endl << endl;

    cout << "=== Performance Comparison for Different K Values ===" << endl;
    cout << "K\tAccuracy\tF1 Score\tTime(ms)" << endl;
    cout << "------------------------------------------------" << endl;

    vector<int> k_values = { 1, 3, 5, 7, 9, 11, 15 };

    for (int k : k_values) {
        KNN knn;

        auto start = chrono::high_resolution_clock::now();
        knn.fit(X_train, y_train, k);
        auto y_pred = knn.predict(X_test);
        auto end = chrono::high_resolution_clock::now();

        double acc = compute_accuracy(y_pred, y_test);
        double f1 = knn.weighted_F1(X_test, y_test);
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

        cout << k << "\t" << acc << "\t\t" << f1 << "\t\t" << duration.count() << endl;
    }

    cout << endl << "=== Detailed Report (K=3) ===" << endl;
    cout << "------------------------------------------------" << endl;

    KNN knn;
    knn.fit(X_train, y_train, 3);
    auto y_pred = knn.predict(X_test);

    double acc = compute_accuracy(y_pred, y_test);
    double f1 = knn.weighted_F1(X_test, y_test);

    cout << "Accuracy:  " << acc << endl;
    cout << "Weighted F1: " << f1 << endl << endl;

    cout << "=== First 20 Test Sample Predictions ===" << endl;
    cout << "Idx\tTrue\tPred\tStatus" << endl;
    cout << "------------------------------------------------" << endl;

    int display_count = min(20, (int)X_test.size());
    int error_count = 0;

    for (int i = 0; i < display_count; i++) {
        string status = (y_pred[i] == y_test[i]) ? "OK" : "ERR";
        if (y_pred[i] != y_test[i]) error_count++;
        cout << i << "\t" << y_test[i] << "\t" << y_pred[i] << "\t" << status << endl;
    }

    if (error_count > 0) {
        cout << endl << "Errors in first 20 samples: " << error_count << endl;
    }

    cout << endl << "=== Per-Class Accuracy ===" << endl;
    cout << "Class\tTotal\tCorrect\tAccuracy" << endl;
    cout << "------------------------------------------------" << endl;

    for (int label = 0; label < 10; label++) {
        int total = 0;
        int correct = 0;
        for (size_t i = 0; i < X_test.size(); i++) {
            if (y_test[i] == label) {
                total++;
                if (y_pred[i] == label) correct++;
            }
        }
        if (total > 0) {
            double class_acc = static_cast<double>(correct) / total;
            cout << label << "\t" << total << "\t" << correct << "\t" << class_acc << endl;
        }
    }

    cout << endl << "Test completed!" << endl;

    return 0;
}