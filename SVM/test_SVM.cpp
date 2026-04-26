// test_svm_circles.cpp
#include "SVM.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>

using namespace std;

struct DataPoint {
    vector<double> x;
    int y;
};

bool load_libsvm(const string& filename, vector<DataPoint>& data) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Cannot open file: " << filename << endl;
        return false;
    }

    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;

        istringstream iss(line);
        DataPoint dp;
        iss >> dp.y;

        dp.x.resize(2, 0.0);  
        string token;
        while (iss >> token) {
            size_t colon = token.find(':');
            if (colon != string::npos) {
                int idx = stoi(token.substr(0, colon)) - 1;
                double val = stod(token.substr(colon + 1));
                if (idx >= 0 && idx < 2) {
                    dp.x[idx] = val;
                }
            }
        }
        data.push_back(dp);
    }

    cout << "Loaded " << data.size() << " samples from " << filename << endl;
    return true;
}

double accuracy(const vector<int>& pred, const vector<int>& truth) {
    if (pred.empty() || pred.size() != truth.size()) return 0.0;
    int correct = 0;
    for (size_t i = 0; i < pred.size(); i++) {
        if (pred[i] == truth[i]) correct++;
    }
    return (double)correct / pred.size();
}

int main() {
    cout << "\n========================================" << endl;
    cout << "   SVM on Circles Dataset (Non-linear)" << endl;
    cout << "========================================\n" << endl;


    vector<DataPoint> train_data, test_data;
    if (!load_libsvm("circles_train.libsvm", train_data)) return 1;
    if (!load_libsvm("circles_test.libsvm", test_data)) return 1;

    vector<Point> X_train, X_test;
    vector<int> y_train, y_test;

    for (const auto& dp : train_data) {
        X_train.push_back(dp.x);
        y_train.push_back(dp.y);
    }
    for (const auto& dp : test_data) {
        X_test.push_back(dp.x);
        y_test.push_back(dp.y);
    }

    cout << "\nDataset info:" << endl;
    cout << "  Train samples: " << X_train.size() << endl;
    cout << "  Test samples: " << X_test.size() << endl;
    cout << "  Feature dimension: 2" << endl;

    SVM svm_linear, svm_rbf;

    cout << "\n=== 1. Linear Kernel (Polynomial degree=1) ===" << endl;
    auto start = chrono::high_resolution_clock::now();
    svm_linear.fit(X_train, y_train, 1.0, 1e-3, 1e-3, "Poly", 1.0, 200);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Training time: " << duration.count() << " ms" << endl;

    auto y_pred_train_linear = svm_linear.predict(X_train);
    auto y_pred_test_linear = svm_linear.predict(X_test);

    double train_acc_linear = accuracy(y_pred_train_linear, y_train);
    double test_acc_linear = accuracy(y_pred_test_linear, y_test);

    cout << "Train accuracy: " << train_acc_linear * 100 << "%" << endl;
    cout << "Test accuracy: " << test_acc_linear * 100 << "%" << endl;

    cout << "\n=== 2. RBF Kernel (gamma=10.0) ===" << endl;
    start = chrono::high_resolution_clock::now();
    svm_rbf.fit(X_train, y_train, 1.0, 1e-3, 1e-3, "Gaussian", 10.0, 200);
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Training time: " << duration.count() << " ms" << endl;

    auto y_pred_train_rbf = svm_rbf.predict(X_train);
    auto y_pred_test_rbf = svm_rbf.predict(X_test);

    double train_acc_rbf = accuracy(y_pred_train_rbf, y_train);
    double test_acc_rbf = accuracy(y_pred_test_rbf, y_test);

    cout << "Train accuracy: " << train_acc_rbf * 100 << "%" << endl;
    cout << "Test accuracy: " << test_acc_rbf * 100 << "%" << endl;

    cout << "\n=== Summary ===" << endl;
    cout << "Linear Kernel - Test accuracy: " << test_acc_linear * 100 << "%" << endl;
    cout << "RBF Kernel   - Test accuracy: " << test_acc_rbf * 100 << "%" << endl;

    if (test_acc_rbf > test_acc_linear + 0.05) {
        cout << "\n[OK] RBF kernel is significantly better than linear kernel." << endl;
        cout << "     This means RBF successfully handles non-linear data." << endl;
    }
    else if (test_acc_rbf > test_acc_linear) {
        cout << "\n[OK] RBF kernel is slightly better than linear kernel." << endl;
        cout << "     SVM implementation is correct." << endl;
    }
    else {
        cout << "\n[WARN] RBF kernel performance is not ideal." << endl;
        cout << "       Try adjusting gamma parameter or check implementation." << endl;
    }

    cout << "\n=== First 10 test predictions (RBF) ===" << endl;
    for (int i = 0; i < min(10, (int)y_pred_test_rbf.size()); i++) {
        cout << "Sample " << i + 1 << ": Predict=" << (y_pred_test_rbf[i] == 1 ? "POS" : "NEG")
            << " | Truth=" << (y_test[i] == 1 ? "POS" : "NEG");
        if (y_pred_test_rbf[i] == y_test[i]) {
            cout << " [OK]" << endl;
        }
        else {
            cout << " [ERR]" << endl;
        }
    }

    cout << "\n========================================" << endl;
    cout << "   Test completed!" << endl;
    cout << "========================================\n" << endl;

    return 0;
}