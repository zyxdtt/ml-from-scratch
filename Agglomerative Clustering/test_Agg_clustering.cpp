#include "Agg_clustering.hpp"
#include <iostream>

int main() {
	vector<point> a{ {1,1},{0,1},{1,0},{0,0},{5,5},{4,4},{4,5},{5,4},{-7,-7},{-6,-7},{-6,-6},{-7,-6} };
	agglomerative aa(a, 12, 3);
	auto re = aa.train();
	for (auto x : re) {
		for (auto y : x) cout << y << ' ';
		cout << endl;
	}
}