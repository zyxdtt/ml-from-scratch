#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <unordered_set>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include "Naive_Bayes.hpp"

using namespace std;

vector<Point> discretize_to_255(const vector<Point>& X) {
    vector<Point> X_discrete = X;
    if (X.empty() || X[0].empty()) return X_discrete;

    size_t num_samples = X.size();
    size_t num_features = X[0].size();

    vector<double> min_vals(num_features, X[0][0]);
    vector<double> max_vals(num_features, X[0][0]);

    for (size_t f = 0; f < num_features; f++) {
        for (size_t i = 0; i < num_samples; i++) {
            min_vals[f] = min(min_vals[f], X[i][f]);
            max_vals[f] = max(max_vals[f], X[i][f]);
        }
    }

    for (size_t i = 0; i < num_samples; i++) {
        for (size_t f = 0; f < num_features; f++) {
            if (max_vals[f] == min_vals[f]) {
                X_discrete[i][f] = 0;
            }
            else {
                double normalized = (X[i][f] - min_vals[f]) / (max_vals[f] - min_vals[f]);
                X_discrete[i][f] = static_cast<int>(normalized * 255);
                X_discrete[i][f] = min(255.0, max(0.0, X_discrete[i][f]));
            }
        }
    }

    return X_discrete;
}

vector<Point> round_to_int(const vector<Point>& X) {
    vector<Point> X_discrete = X;
    for (auto& sample : X_discrete) {
        for (auto& val : sample) {
            val = static_cast<int>(round(val));
        }
    }
    return X_discrete;
}

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

void print_confusion_matrix(const vector<int>& pred, const vector<int>& truth, int num_classes = 10) {
    vector<vector<int>> confusion(num_classes, vector<int>(num_classes, 0));

    for (size_t i = 0; i < pred.size(); i++) {
        if (pred[i] != -1) {  
            confusion[truth[i]][pred[i]]++;
        }
    }

    cout << "\n=== Confusion Matrix ===" << endl;
    cout << "     ";
    for (int i = 0; i < num_classes; i++) {
        cout << setw(4) << i;
    }
    cout << endl;
    cout << "     " << string(num_classes * 4, '-') << endl;

    for (int i = 0; i < num_classes; i++) {
        cout << setw(3) << i << " |";
        for (int j = 0; j < num_classes; j++) {
            cout << setw(4) << confusion[i][j];
        }
        cout << endl;
    }
}

int main() {
    cout << "========================================" << endl;
    cout << "Naive Bayes Test on Digits Dataset" << endl;
    cout << "========================================" << endl << endl;

    vector<Point> X;
    vector<int> y;

    if (!load_digits("digits.csv", X, y)) {
        return 1;
    }

    cout << "Original data:" << endl;
    cout << "  Total samples: " << X.size() << endl;
    cout << "  Feature dimension: " << X[0].size() << endl;
    cout << "  Feature value range: 0-" << *max_element(X[0].begin(), X[0].end()) << endl;
    cout << "\n  Sample pixel values (first 10 features of first sample):" << endl;
    cout << "  ";
    for (int i = 0; i < min(10, (int)X[0].size()); i++) {
        cout << X[0][i] << " ";
    }
    cout << endl;

    cout << "\nDiscretizing features to 0-255..." << endl;
    vector<Point> X_discrete = discretize_to_255(X);

    cout << "  Discretized sample values (first 10 features):" << endl;
    cout << "  ";
    for (int i = 0; i < min(10, (int)X_discrete[0].size()); i++) {
        cout << X_discrete[0][i] << " ";
    }
    cout << endl;

    unordered_set<int> unique_labels(y.begin(), y.end());
    cout << "\nNumber of classes: " << unique_labels.size() << " (0-" << unique_labels.size() - 1 << ")" << endl << endl;

    shuffle_data(X_discrete, y);

    size_t train_size = static_cast<size_t>(X_discrete.size() * 0.8);
    size_t test_size = X_discrete.size() - train_size;

    vector<Point> X_train(X_discrete.begin(), X_discrete.begin() + train_size);
    vector<Point> X_test(X_discrete.begin() + train_size, X_discrete.end());
    vector<int> y_train(y.begin(), y.begin() + train_size);
    vector<int> y_test(y.begin() + train_size, y.end());

    cout << "Data split:" << endl;
    cout << "  Training set size: " << train_size << endl;
    cout << "  Test set size: " << test_size << endl << endl;

    cout << "=== Naive Bayes Performance ===" << endl;
    cout << "------------------------------------------------" << endl;

    Naive_Bayes nb;
    auto start = chrono::high_resolution_clock::now();
    nb.fit(X_train, y_train);
    auto end = chrono::high_resolution_clock::now();
    auto train_time = chrono::duration_cast<chrono::milliseconds>(end - start);

    start = chrono::high_resolution_clock::now();
    auto y_pred = nb.predict(X_test);
    end = chrono::high_resolution_clock::now();
    auto predict_time = chrono::duration_cast<chrono::milliseconds>(end - start);

    double acc = nb.accuracy(X_test, y_test);
    double f1 = nb.weighted_F1(X_test, y_test);

    cout << "Training time:   " << train_time.count() << " ms" << endl;
    cout << "Prediction time: " << predict_time.count() << " ms" << endl;
    cout << "Accuracy:        " << fixed << setprecision(4) << acc << endl;
    cout << "Weighted F1:     " << fixed << setprecision(4) << f1 << endl;

    int rejected_count = 0;
    int correct_count = 0;
    int wrong_count = 0;

    for (size_t i = 0; i < y_pred.size(); i++) {
        if (y_pred[i] == -1) {
            rejected_count++;
        }
        else if (y_pred[i] == y_test[i]) {
            correct_count++;
        }
        else {
            wrong_count++;
        }
    }

    cout << "\nPrediction statistics:" << endl;
    cout << "  Correct:   " << correct_count << " (" << fixed << setprecision(2)
        << (correct_count * 100.0 / test_size) << "%)" << endl;
    cout << "  Wrong:     " << wrong_count << " ("
        << (wrong_count * 100.0 / test_size) << "%)" << endl;
    cout << "  Rejected:  " << rejected_count << " ("
        << (rejected_count * 100.0 / test_size) << "%)" << endl;

    cout << "\n=== First 30 Test Sample Predictions ===" << endl;
    cout << "Idx\tTrue\tPred\tStatus" << endl;
    cout << "------------------------------------------------" << endl;

    int display_count = min(30, (int)X_test.size());
    int display_errors = 0;
    int display_rejects = 0;

    for (int i = 0; i < display_count; i++) {
        string status;
        if (y_pred[i] == -1) {
            status = "REJ";
            display_rejects++;
        }
        else if (y_pred[i] == y_test[i]) {
            status = "OK";
        }
        else {
            status = "ERR";
            display_errors++;
        }
        cout << i << "\t" << y_test[i] << "\t" << y_pred[i] << "\t" << status << endl;
    }

    if (display_errors > 0 || display_rejects > 0) {
        cout << endl << "First " << display_count << " samples summary:" << endl;
        cout << "  Errors:   " << display_errors << endl;
        cout << "  Rejected: " << display_rejects << endl;
    }

    print_confusion_matrix(y_pred, y_test);

    cout << "\n=== Per-Class Performance ===" << endl;
    cout << "Class\tTotal\tCorrect\tRejected\tAcc(accepted)\tAcc(all)" << endl;
    cout << "------------------------------------------------" << endl;

    for (int label = 0; label < 10; label++) {
        int total = 0;
        int correct = 0;
        int rejected = 0;

        for (size_t i = 0; i < X_test.size(); i++) {
            if (y_test[i] == label) {
                total++;
                if (y_pred[i] == -1) {
                    rejected++;
                }
                else if (y_pred[i] == label) {
                    correct++;
                }
            }
        }

        if (total > 0) {
            double acc_accepted = (total - rejected > 0) ?
                static_cast<double>(correct) / (total - rejected) : 0.0;
            double acc_all = static_cast<double>(correct) / total;

            cout << label << "\t" << total << "\t" << correct << "\t"
                << rejected << "\t\t" << fixed << setprecision(3) << acc_accepted
                << "\t\t" << acc_all << endl;
        }
    }

    cout << "\n=== Feature Coverage Analysis ===" << endl;
    cout << "------------------------------------------------" << endl;

    size_t total_unique_values = 0;
    size_t max_unique_values = 0;

    for (size_t f = 0; f < X_train[0].size(); f++) {
        unordered_set<int> unique_vals;
        for (size_t i = 0; i < X_train.size(); i++) {
            unique_vals.insert(static_cast<int>(X_train[i][f]));
        }
        total_unique_values += unique_vals.size();
        max_unique_values = max(max_unique_values, unique_vals.size());
    }

    double avg_unique = static_cast<double>(total_unique_values) / X_train[0].size();
    cout << "Training set feature coverage:" << endl;
    cout << "  Average unique values per feature: " << fixed << setprecision(1) << avg_unique << endl;
    cout << "  Maximum unique values per feature: " << max_unique_values << endl;
    cout << "  Feature dimension: " << X_train[0].size() << endl;

    cout << endl << "Test completed!" << endl;

    return 0;
}