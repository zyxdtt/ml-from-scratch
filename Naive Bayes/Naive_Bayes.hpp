#pragma once
#include <vector>
#include <unordered_map>
#include <utility>

using namespace std;
using Point = vector<double>;

class Naive_Bayes {
private:
	unordered_map<int,double> y_possibilitys;
	vector<unordered_map<int, unordered_map<int, double>>> Xy_possibilitys;
	unordered_map<int, int> y_number;
	vector<unordered_map<int, unordered_map<int,int>>> Xy_number;
	vector<int> X_features_number;
	size_t number_of_train;
	size_t number_of_features;
	bool is_fitted;
	double F1_in_one_label(const vector<int>& pred,
		const vector<int>& y, int label) const {
		int tp = 0, fp = 0, fn = 0;
		for (int i = 0; i < y.size(); i++) {
			if (pred[i] == label && y[i] == label) tp++;
			if (pred[i] == label && y[i] != label) fp++;
			if (pred[i] != label && y[i] == label) fn++;
		}
		return 2.0 * tp / (2.0 * tp + fp + fn);
	}
public:
	~Naive_Bayes() = default;
	Naive_Bayes() {}
	void fit(const vector<Point>& X_train,
		const vector<int>& y_train) {
		is_fitted = true;
		number_of_train = y_train.size();
		number_of_features = X_train[0].size();
		Xy_number.resize(number_of_features);
		Xy_possibilitys.resize(number_of_features);
		X_features_number.resize(number_of_features);
		for (int i = 0; i < number_of_train; i++) {
			y_number[y_train[i]]++;
			for (int j = 0; j < number_of_features; j++) {
				Xy_number[j][y_train[i]][X_train[i][j]]++;
			}
		}
		for (int j = 0; j < number_of_features; j++) {
			unordered_map<int, int> cnt;
			for (int i = 0; i < number_of_train; i++) {
				cnt[X_train[i][j]]++;
			}
			X_features_number[j] = cnt.size();
		}
		for (auto [y_class, num] : y_number) {
			double possibility = (num * 1.0 + 1) / (number_of_train + y_number.size() * 1.0);
			y_possibilitys[y_class] = possibility;
		}
		for (auto [y_class, y_num] : y_number) {
			for (int feature = 0; feature < number_of_features; feature++) {
				for (auto [Xy_class, Xy_num] : Xy_number[feature][y_class]) {
					double possibility = (Xy_num * 1.0 + 1.0) / (y_num + X_features_number[feature] * 1.0);
					Xy_possibilitys[feature][y_class][Xy_class] = possibility;
				}
			}
		}
	}
	vector<int> predict(const vector<Point>& X_test) { 
		size_t number_of_tests = X_test.size();
		vector<int> y_test(number_of_tests, -1);
		if (!is_fitted) return y_test;
		for (int i = 0; i < number_of_tests; i++) {
			unordered_map<int,double> choice;
			for (auto [y_class, y_possibility] : y_possibilitys) {
				double possibility = y_possibility;
				for (int feature = 0; feature < number_of_features; feature++) {
					int targ_X = X_test[i][feature];
					double result= 1.0 / (y_number[y_class] + X_features_number[feature]);
					if (Xy_possibilitys[feature][y_class].count(targ_X))
						result = Xy_possibilitys[feature][y_class][targ_X];
					possibility *= result;
				}
				choice[y_class] = possibility;
			}
			double max_possibility = 0;
			int class_max = -1;
			for (auto [y_class, possibility] : choice) {
				if (possibility > max_possibility) {
					max_possibility = possibility;
					class_max = y_class;
				}
			}
			y_test[i] = class_max;
		}
		return y_test;
	}

	double accuracy(const vector<Point>& X, const vector<int>& y) {
		auto y_pred = predict(X);
		if (y_pred.empty() || y_pred.size() != y.size()) return 0.0;
		int correct = 0;
		for (size_t i = 0; i < y.size(); i++) {
			if (y_pred[i] == y[i]) correct++;
		}
		return static_cast<double>(correct) / y.size();
	}

	double weighted_F1(const vector<Point>& X, const vector<int>& y) {
		auto y_pred = predict(X);
		if (y_pred.empty() || y_pred.size() != y.size()) return 0.0;
		int y_size = y.size();
		unordered_map<int, int> classify;
		for (int label : y) classify[label]++;
		double weighted_f1 = 0;
		for (auto [label, count] : classify) {
			weighted_f1 += (count * 1.0 / y_size) * F1_in_one_label(y_pred, y, label);
		}
		return weighted_f1;
	}
};