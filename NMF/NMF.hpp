#pragma once
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>

using namespace std;
using point = vector<double>;
using mat = vector<point>;

class NMF {
private:
	int k;
	mt19937 gen;
	uniform_real_distribution<double> dis;

	mat matmul(const mat& a, const mat& b) {
		int aa = a.size(), bb = b[0].size(), cc = a[0].size();
		mat re(aa, point(bb));
		for (int i = 0; i < aa; i++) {
			for (int j = 0; j < bb; j++) {
				double val = 0;
				for (int k = 0; k < cc; k++) val += a[i][k] * b[k][j];
				re[i][j] = val;
			}
		}
		return move(re);
	}

	mat transpose(const mat& a) {
		mat b(a[0].size(), point(a.size()));
		for (int i = 0; i < a.size(); i++)
			for (int j = 0; j < a[0].size(); j++)
				b[j][i] = a[i][j];
		return move(b);
	}

	void gui(mat& u, mat& v, const mat& x) {
		for (int i = 0; i < k; i++) {
			double num = 0;
			for (int j = 0; j < x.size(); j++) num += pow(u[j][i], 2);
			num = sqrt(num);
			for (int j = 0; j < x.size(); j++) u[j][i] /= num;
			for (int j = 0; j < x[0].size(); j++) v[i][j] *= num;
		}
	}
public:
	NMF(int kk) :k(kk), gen{ random_device()() }, dis(0.0, 1.0) {}
	pair<mat, mat> train(const mat& x, int iter = 100) {
		mat u(x.size(), point(k)), v(k, point(x[0].size()));
		for (auto& xx : u) generate(xx.begin(), xx.end(), [&]() {return dis(gen); });
		for (auto& xx : v) generate(xx.begin(), xx.end(), [&]() {return dis(gen); });
		for (int i = 0; i < iter; i++) {
			gui(u, v, x);
			mat v_trans = transpose(v);
			mat zi = matmul(x, v_trans);
			mat mu = matmul(matmul(u, v), v_trans);
			for (int j = 0; j < u.size(); j++)
				for (int k = 0; k < u[0].size(); k++) u[j][k] *= zi[j][k] / mu[j][k];
			mat u_trans = transpose(u);
			mat zii = matmul(u_trans, x);
			mat muu = matmul(matmul(u_trans, u), v);
			for (int j = 0; j < v.size(); j++)
				for (int k = 0; k < v[0].size(); k++) v[j][k] *= zii[j][k] / muu[j][k];
		}
		return make_pair(move(u), move(v));
	}
};