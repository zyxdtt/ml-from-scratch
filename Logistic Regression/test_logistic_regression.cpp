// test_logistic_regression.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include "Logistic_Regression.hpp"  

using namespace std;
using Point = vector<double>;

vector<Point> load_X(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file) {
        cerr << "Cannot open file: " << filename << endl;
        exit(1);
    }

    int rows, cols;
    file.read(reinterpret_cast<char*>(&rows), sizeof(int));
    file.read(reinterpret_cast<char*>(&cols), sizeof(int));

    vector<Point> X(rows, Point(cols));
    for (int i = 0; i < rows; i++) {
        file.read(reinterpret_cast<char*>(X[i].data()), cols * sizeof(double));
    }

    file.close();
    return X;
}

vector<int> load_y(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file) {
        cerr << "Cannot open file: " << filename << endl;
        exit(1);
    }

    int size;
    file.read(reinterpret_cast<char*>(&size), sizeof(int));

    vector<int> y(size);
    file.read(reinterpret_cast<char*>(y.data()), size * sizeof(int));

    file.close();
    return y;
}

double accuracy(const vector<int>& y_true, const vector<int>& y_pred) {
    int correct = 0;
    for (size_t i = 0; i < y_true.size(); i++) {
        if (y_true[i] == y_pred[i]) correct++;
    }
    return 100.0 * correct / y_true.size();
}

int main() {
    cout << "=== Testing Logistic Regression on Digits Dataset ===" << endl;

    cout << "\n1. Loading data..." << endl;
    auto X_train = load_X("digits_train_X.bin");
    auto y_train = load_y("digits_train_y.bin");
    auto X_test = load_X("digits_test_X.bin");
    auto y_test = load_y("digits_test_y.bin");

    cout << "   Training samples: " << X_train.size() << endl;
    cout << "   Test samples: " << X_test.size() << endl;
    cout << "   Features: " << X_train[0].size() << endl;

    cout << "\n2. Training model..." << endl;
    Logistic_Regression model;

    auto start = chrono::high_resolution_clock::now();
    model.fit(X_train, y_train, 0.3, 500, 1e-3);
    auto end = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "   Training time: " << duration.count() << " ms" << endl;

    cout << "\n3. Predicting..." << endl;
    auto y_pred_train = model.predict(X_train);
    auto y_pred_test = model.predict(X_test);


    cout << "\n4. Results:" << endl;
    double train_acc = accuracy(y_train, y_pred_train);
    double test_acc = accuracy(y_test, y_pred_test);

    cout << "   Training accuracy: " << train_acc << "%" << endl;
    cout << "   Test accuracy: " << test_acc << "%" << endl;

    cout << "\n5. Sample predictions (first 10 test samples):" << endl;
    cout << "   True:     ";
    for (int i = 0; i < min(10, (int)y_test.size()); i++) {
        cout << y_test[i] << " ";
    }
    cout << "\n   Predicted: ";
    for (int i = 0; i < min(10, (int)y_pred_test.size()); i++) {
        cout << y_pred_test[i] << " ";
    }
    cout << endl;

    return 0;
}