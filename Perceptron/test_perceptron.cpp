// test_perceptron.cpp
// Tests for Perceptron implementation
// Following Chapter 1 of "Statistical Learning Methods" by Li Hang

#include "perceptron.hpp"
#include <iostream>
#include <vector>

using namespace std;

void test_and_gate() {
    cout << "Test 1: AND Logic Gate (linearly separable)" << endl;
    cout << "----------------------------------------------" << endl;

    vector<vector<double>> X = {
        {1, 1},
        {1, -1},
        {-1, 1},
        {-1, -1}
    };
    vector<int> y = { 1, -1, -1, -1 };  // AND: only (1,1) is positive

    perceptron model;
    string log = model.fit(X, y, 100, 0.5);
    cout << "Training log: " << log << endl;

    // Check convergence
    if (log.find("Converged") != string::npos) {
        cout << "yes Converged successfully" << endl;
    }
    else {
        cout << "no Failed to converge" << endl;
    }

    // Predictions
    auto y_pred = model.predict(X);
    cout << "True labels:  ";
    for (int val : y) cout << val << " ";
    cout << endl;
    cout << "Predictions:  ";
    for (int val : y_pred) cout << val << " ";
    cout << endl;

    // Check if all predictions match
    bool all_correct = true;
    for (size_t i = 0; i < y.size(); i++) {
        if (y_pred[i] != y[i]) {
            all_correct = false;
            break;
        }
    }
    cout << (all_correct ? "yes" : "no") << " All predictions correct" << endl;

    // Accuracy
    double acc = model.accuracy(X, y);
    cout << "Accuracy: " << acc * 100 << "%" << endl;

    // Learned parameters
    cout << "Learned weights: ";
    for (double w : model.weights()) cout << w << " ";
    cout << endl;
    cout << "Learned bias: " << model.bias() << endl;

    cout << endl;
}

void test_xor_gate() {
    cout << "Test 2: XOR Logic Gate (linearly inseparable)" << endl;
    cout << "----------------------------------------------" << endl;

    vector<vector<double>> X = {
        {1, 1},
        {1, -1},
        {-1, 1},
        {-1, -1}
    };
    vector<int> y = { 1, -1, -1, 1 };  // XOR: (1,1) and (-1,-1) are positive

    perceptron model;
    string log = model.fit(X, y, 10, 0.5);
    cout << "Training log: " << log << endl;

    // Check non-convergence (correct behavior for XOR)
    if (log.find("not converged") != string::npos) {
        cout << "yes Correctly failed to converge (XOR is linearly inseparable)" << endl;
    }
    else {
        cout << "no Unexpected convergence" << endl;
    }

    // Accuracy should be less than 100%
    double acc = model.accuracy(X, y);
    cout << "Accuracy: " << acc * 100 << "%" << endl;

    if (acc < 1.0) {
        cout << "yes Accuracy < 100% as expected" << endl;
    }
    else {
        cout << "no Unexpected perfect accuracy" << endl;
    }

    cout << endl;
}

void test_continual_training() {
    cout << "Test 3: Continual Training" << endl;
    cout << "----------------------------------------------" << endl;

    vector<vector<double>> X = {
        {1, 2},
        {2, 3},
        {3, 4}
    };
    vector<int> y = { 1, 1, -1 };

    perceptron model;

    // Initial training with few iterations
    string log1 = model.fit(X, y, 3, 0.1);
    cout << "Initial training: " << log1 << endl;

    // Continue training
    string log2 = model.continual_fit(X, y, 20, 0.1);
    cout << "Continual training: " << log2 << endl;

    auto y_pred = model.predict(X);
    cout << "Final predictions: ";
    for (int p : y_pred) cout << p << " ";
    cout << endl;

    cout << endl;
}

void test_prediction_interface() {
    cout << "Test 4: Prediction Interface" << endl;
    cout << "----------------------------------------------" << endl;

    vector<vector<double>> X_train = {
        {1, 1},
        {2, 2},
        {3, 3}
    };
    vector<int> y_train = { 1, 1, -1 };

    perceptron model;
    model.fit(X_train, y_train, 50, 0.1);

    vector<vector<double>> X_test = { {1.5, 1.5}, {2.5, 2.5} };
    auto y_pred = model.predict(X_test);

    cout << "Test samples:  [" << X_test[0][0] << "," << X_test[0][1] << "] ";
    cout << "[" << X_test[1][0] << "," << X_test[1][1] << "]" << endl;
    cout << "Predictions:   ";
    for (int p : y_pred) cout << p << " ";
    cout << endl;

    cout << endl;
}

void test_evaluation_metrics() {
    cout << "Test 5: Evaluation Metrics" << endl;
    cout << "----------------------------------------------" << endl;

    vector<vector<double>> X = {
        {1, 0},
        {0, 1},
        {2, 2},
        {-1, -1}
    };
    vector<int> y = { 1, 1, -1, -1 };

    perceptron model;
    model.fit(X, y, 100, 0.5);

    double acc = model.accuracy(X, y);
    double prec = model.precision(X, y);
    double rec = model.recall(X, y);
    double f1_val = model.f1(X, y);

    cout << "Accuracy:  " << acc * 100 << "%" << endl;
    cout << "Precision: " << prec * 100 << "%" << endl;
    cout << "Recall:    " << rec * 100 << "%" << endl;
    cout << "F1 Score:  " << f1_val * 100 << "%" << endl;

    cout << "\nClassification Report:" << endl;
    cout << model.classification_report(X, y);

    cout << endl;
}

void test_edge_cases() {
    cout << "Test 6: Edge Cases and Error Handling" << endl;
    cout << "----------------------------------------------" << endl;

    // Empty data
    {
        perceptron model;
        vector<vector<double>> X_empty;
        vector<int> y_empty;
        string log = model.fit(X_empty, y_empty);
        cout << "Empty data: " << log << endl;
    }

    // Predict before fit
    {
        perceptron model;
        vector<vector<double>> X = { {1, 2} };
        auto y_pred = model.predict(X);
        if (y_pred.empty()) {
            cout << "yes Predict before fit returns empty vector" << endl;
        }
    }

    cout << endl;
}

int main() {
    cout << "==============================================" << endl;
    cout << "     PERCEPTRON TEST SUITE" << endl;
    cout << "==============================================" << endl << endl;

    test_and_gate();
    test_xor_gate();
    test_continual_training();
    test_prediction_interface();
    test_evaluation_metrics();
    test_edge_cases();

    cout << "==============================================" << endl;
    cout << "     ALL TESTS COMPLETED" << endl;
    cout << "==============================================" << endl;

    return 0;
}