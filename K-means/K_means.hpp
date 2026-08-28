#pragma once
#include <vector>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <random>
#include <cmath>
#include <limits>

using namespace std;
using point = vector<float>;
using mat = vector<point>;

class ff {
private:
	vector<int> a;
	int cnt;
public:
	ff(int n) :a(n), cnt(n) {
		iota(a.begin(), a.end(), 0);
	}
	int root(int n) {
		if (a[n] == n) return n;
		return a[n] = root(a[n]);
	}
	void bing(int i, int j) {
		int ri = root(i), rj = root(j);
		if (ri != rj) {
			a[ri] = rj;
			cnt--;
		}
	}
	bool is(int i, int j) {
		return root(i) == root(j);
	}
	int count() const { return cnt; }
	void reset() { iota(a.begin(), a.end(), 0); }
};

class K_means {
private:
	int n;
	int m;
	int k;
	mt19937 gen{ random_device()() };
	uniform_real_distribution<float> dis{ -10.0f, 10.0f };
	mat kdot;

	float cul(const point& a, const point& b) const {
		float sum = 0;
		for (int i = 0; i < m; i++) sum += pow(a[i] - b[i], 2);
		return sqrt(sum);
	}
public:
	K_means(int nn, int mm, int kk) :n(nn), m(mm), k(kk), kdot(kk, point(mm)) {
		for (auto& x : kdot)
			generate(x.begin(), x.end(), [&]() {return dis(gen); });
	}
	vector<vector<int>> train(const mat& x) {
		vector<vector<int>> re(k), lare(k);
		do {
			lare = re;
			for (auto& a : re) a.clear();
			for (int i = 0; i < n; i++) {
				int idx = -1;
				float mi = numeric_limits<float>::max();
				for (int j = 0; j < k; j++) {
					auto dis = cul(x[i], kdot[j]);
					if (dis < mi) {
						mi = dis;
						idx = j;
					}
				}
				re[idx].push_back(i);
			}
			vector<point> mean(k, point(m, 0));
			for (int i = 0; i < k; i++) {
				int sz = re[i].size();
				for (auto xx : re[i]) {
					for (int j = 0; j < m; j++) mean[i][j] += x[xx][j];
				}
				for (auto& xx : mean[i]) xx /= sz;
			}
			kdot = move(mean);
		} while (re != lare);
		return re;
	}
};