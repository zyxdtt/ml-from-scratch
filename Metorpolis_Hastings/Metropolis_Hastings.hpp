#pragma once
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>

using namespace std;
using point = vector<double>;
using mat = vector<point>;
using func = double(*)(point);

class MH {
private:
	mt19937 gen{ random_device()() };
	point x;
	point next_x;
	int dim;
	normal_distribution<double> norm;
	uniform_real_distribution<double> uni;
	int m, n;
	func get;

	point sam() {
		point temp(dim);
		generate(temp.begin(), temp.end(), [this]() {return norm(gen); });//assume L=I
		return temp;
	}

	inline double alpha(double now, double next) {
		return min(1.0, next / now);
	}

	void step() {
		auto temp = sam();
		for (int j = 0; j < dim; j++) next_x[j] = x[j] + temp[j];
		double now = get(x);
		double next = get(next_x);
		double prob = alpha(now, next);
		double acc = uni(gen);
		if (acc < prob) {
			x = move(next_x);
			next_x.resize(dim);
		}
	}
public:

	MH(int d, int mm, int nn) :dim(d), norm(0, 1),
		m(mm), n(nn), uni(0, 1),
		x(d), next_x(d) {
		generate(x.begin(), x.end(), [this]() {return uni(gen); });
	}

	vector<point> sample(func f) {
		vector<point> result;
		get = f;
		for (int i = 0; i < n; i++) {
			step();
			if (i >= m) result.push_back(x);
		}
		return result;
	}
};