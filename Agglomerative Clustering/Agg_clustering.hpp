#pragma once
#include <vector>
#include <algorithm>
#include <string>
#include <cmath>
#include <Eigen/Dense>
#include <numeric>
#include <limits>

using namespace std;
using namespace Eigen;
using point = vector<float>;

class UnionFind {
private:
	std::vector<int> parent;
	std::vector<int> rank;
public:
	UnionFind(int n) : parent(n), rank(n, 0) {
		for (int i = 0; i < n; ++i) {
			parent[i] = i;
		}
	}

	int find(int x) {
		if (parent[x] != x) {
			parent[x] = find(parent[x]);
		}
		return parent[x];
	}

	bool unionElements(int x, int y) {
		int rootX = find(x);
		int rootY = find(y);
		if (rootX == rootY) {
			return false;
		}
		if (rank[rootX] < rank[rootY]) {
			parent[rootX] = rootY;
		}
		else if (rank[rootX] > rank[rootY]) {
			parent[rootY] = rootX;
		}
		else {
			parent[rootY] = rootX;
			rank[rootX]++;
		}
		return true;
	}

	bool isConnected(int x, int y) {
		return find(x) == find(y);
	}

	int getCount() const {
		int count = 0;
		for (int i = 0; i < (int)parent.size(); ++i) {
			if (parent[i] == i) {
				count++;
			}
		}
		return count;
	}

	int size() const {
		return parent.size();
	}

	void reset() {
		for (int i = 0; i < (int)parent.size(); ++i) {
			parent[i] = i;
			rank[i] = 0;
		}
	}
};

class agglomerative {
private:
	vector<point> x;
	int dim;
	int n;
	string dis_type;
	string class_type;
	UnionFind ff;
	int p;
	vector<point> dismap;
	int num;

	float Minkowski(int i, int j, int p) const {
		float sum = 0;
		for (int k = 0; k < dim; k++) sum += pow(abs(x[i][k] - x[j][k]), p);
		return pow(sum, 1.0 / p);
	}

	float Mahalanobis(int i, int j) const {
		point mean(dim, 0.0);
		for (int k = 0; k < n; k++)
			for (int d = 0; d < dim; d++)
				mean[d] += x[k][d];
		for (auto& y : mean) y /= n;
		vector<point> cov(dim, point(dim, 0.0));
		for (int k = 0; k < n; k++)
			for (int d1 = 0; d1 < dim; d1++)
				for (int d2 = 0; d2 < dim; d2++)
					cov[d1][d2] += (x[k][d1] - mean[d1]) * (x[k][d2] - mean[d2]);
		for (auto& xx : cov)
			for (auto& y : xx)
				y /= (n - 1);
		point diff(dim);
		for (int d = 0; d < dim; d++)
			diff[d] = x[i][d] - x[j][d];
		Map<MatrixXf> covEigen(cov[0].data(), dim, dim);
		Map<VectorXf> diffEigen(diff.data(), dim);
		MatrixXf covInv = covEigen.inverse();
		float distSq = diffEigen.transpose() * covInv * diffEigen;
		return sqrt(max(0.0f, distSq));
	}

	float correlation(int i, int j) const {
		float xi = accumulate(x[i].begin(), x[i].end(), 0.0) / dim;
		float xj = accumulate(x[j].begin(), x[j].end(), 0.0) / dim;
		float zi = 0;
		for (int d = 0; d < dim; d++)
			zi += (x[i][d] - xi) * (x[j][d] - xj);
		float xi2 = 0, xj2 = 0;
		for (int d = 0; d < dim; d++) {
			xi2 += pow((x[i][d] - xi), 2);
			xj2 += pow((x[j][d] - xj), 2);
		}
		float mu = sqrt(xi2 * xj2);
		return zi / mu;
	}

	float cosine(int i, int j) const {
		float zi = 0;
		float xi2 = 0, xj2 = 0;
		for (int d = 0; d < dim; d++) {
			zi += x[i][d] * x[j][d];
			xi2 += x[i][d] * x[i][d];
			xj2 += x[j][d] * x[j][d];
		}
		float mu = sqrt(xi2 * xj2);
		return zi / mu;
	}

	float distance(int i, int j) const {
		if (dis_type == "Minkowski") return Minkowski(i, j, p);
		else if (dis_type == "Mahalanobis") return Mahalanobis(i, j);
		else if (dis_type == "correlation") return 1 - correlation(i, j);
		else if (dis_type == "cosine") return 1 - cosine(i, j);
		return 0.0f;
	}

	bool is(const vector<int>& list, float t) const {
		for (int i = 0; i < (int)list.size() - 1; i++) {
			for (int j = i + 1; j < (int)list.size(); j++) {
				if (distance(list[i], list[j]) > t) return false;
			}
		}
		return true;
	}

	float class_dis(int r1, int r2) {
		vector<int> list1, list2;
		for (int i = 0; i < n; ++i) {
			if (ff.find(i) == r1) list1.push_back(i);
			else if (ff.find(i) == r2) list2.push_back(i);
		}
		if (list1.empty() || list2.empty()) return numeric_limits<float>::max();

		if (class_type == "min") {
			float minD = numeric_limits<float>::max();
			for (int a : list1)
				for (int b : list2)
					minD = min(minD, distance(a, b));
			return minD;
		}
		else if (class_type == "max") {
			float maxD = -numeric_limits<float>::max();
			for (int a : list1)
				for (int b : list2)
					maxD = max(maxD, distance(a, b));
			return maxD;
		}
		else if (class_type == "average") {
			float sum = 0;
			int cnt = 0;
			for (int a : list1)
				for (int b : list2) {
					sum += distance(a, b);
					cnt++;
				}
			return sum / cnt;
		}
		else if (class_type == "centroid") {
			point cen1(dim, 0.0), cen2(dim, 0.0);
			for (int idx : list1) for (int d = 0; d < dim; ++d) cen1[d] += x[idx][d];
			for (int idx : list2) for (int d = 0; d < dim; ++d) cen2[d] += x[idx][d];
			for (int d = 0; d < dim; ++d) { cen1[d] /= list1.size(); cen2[d] /= list2.size(); }
			float sum = 0;
			for (int d = 0; d < dim; ++d) sum += (cen1[d] - cen2[d]) * (cen1[d] - cen2[d]);
			return sqrt(sum);
		}
		return 0.0f;
	}

public:
	agglomerative(const vector<point>& xx, int nn, int numm, string d_type = "Minkowski", string c_type = "min", int pp = 2)
		: ff(nn) {
		p = pp;
		dis_type = d_type;
		class_type = c_type;
		x = xx;
		n = nn;
		dim = xx[0].size();
		num = numm;
		dismap.resize(n, point(n, 0.0f));
	}

	vector<vector<int>> train() {
		for (int i = 0; i < n - 1; ++i)
			for (int j = i + 1; j < n; ++j)
				dismap[i][j] = dismap[j][i] = distance(i, j);

		vector<bool> valid(n, true);

		while (ff.getCount() > num) {
			float minDist = numeric_limits<float>::max();
			int minI = -1, minJ = -1;
			for (int i = 0; i < n; ++i) {
				if (!valid[i]) continue;
				for (int j = i + 1; j < n; ++j) {
					if (!valid[j]) continue;
					if (dismap[i][j] < minDist) {
						minDist = dismap[i][j];
						minI = i; minJ = j;
					}
				}
			}
			if (minI == -1) break;

			ff.unionElements(minI, minJ);
			int newRoot = ff.find(minI);
			int oldRoot = (newRoot == minI) ? minJ : minI;
			valid[oldRoot] = false;

			for (int k = 0; k < n; ++k) {
				if (valid[k] && k != newRoot) {
					float d = class_dis(newRoot, k);
					dismap[newRoot][k] = dismap[k][newRoot] = d;
				}
			}

			for (int i = 0; i < n; ++i) {
				dismap[oldRoot][i] = dismap[i][oldRoot] = numeric_limits<float>::max();
			}
		}

		vector<vector<int>> result;
		for (int i = 0; i < n; ++i) {
			if (valid[i]) {
				vector<int> cluster;
				for (int j = 0; j < n; ++j) {
					if (ff.find(j) == i) cluster.push_back(j);
				}
				result.push_back(cluster);
			}
		}
		return result;
	}
};