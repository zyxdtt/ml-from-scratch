#pragma once//2026 09 06

#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <numeric>

using namespace std;
using point = vector<double>;
using mat = vector<point>;
using tensor = vector<mat>;

class PLSA {
private:
	int ci, ben, ti;
	mt19937 gen;
	uniform_real_distribution<double> dis;
public:
	PLSA(int cii, int benn, int tii) :ci(cii),
		ben(benn), ti(tii), gen(random_device()()), dis(0.0, 1.0) {}

	pair<mat, mat> train(const mat& x, int iter = 100) {
		mat citi(ci, point(ti)), tiben(ti, point(ben));
		for (auto& xx : citi) generate(xx.begin(), xx.end(), [&]() {return dis(gen); });
		for (auto& xx : tiben) generate(xx.begin(), xx.end(), [&]() {return dis(gen); });
		//EM
		tensor prob(ti, mat(ci, point(ben)));
		point temp(ti);
		point citemp(ci);
		for (int i = 0; i < iter; i++) {
			//E
			for (int j = 0; j < ci; j++) {
				for (int k = 0; k < ben; k++) {
					for (int l = 0; l < ti; l++) temp[l] = citi[j][l] * tiben[l][k];
					double sum = accumulate(temp.begin(), temp.end(), 0.0);
					if (sum < 1e-9) sum = 1e-9;
					for (int l = 0; l < ti; l++) prob[l][j][k] = temp[l] / sum;
				}
			}
			//M
			for (int j = 0; j < ti; j++) {
				for (int k = 0; k < ci; k++) {
					double sum = 0;
					for (int l = 0; l < ben; l++) sum += x[k][l] * prob[j][k][l];
					citemp[k] = sum;
				}
				double sum = accumulate(citemp.begin(), citemp.end(), 0.0);
				if (sum < 1e-9) sum = 1e-9;
				for (int k = 0; k < ci; k++) citi[k][j] = citemp[k] / sum;
			}
			for (int j = 0; j < ben; j++) {
				for (int k = 0; k < ti; k++) {
					double sum = 0; double pin = 0;
					for (int l = 0; l < ci; l++) {
						sum += x[l][j] * prob[k][l][j];
						pin += x[l][j];
					}
					if (pin < 1e-9) pin = 1e-9;
					tiben[k][j] = sum / pin;
				}
			}
		}
		return make_pair(move(citi), move(tiben));
	}
};