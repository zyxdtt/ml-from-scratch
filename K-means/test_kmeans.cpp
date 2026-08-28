#include "K_means.hpp"
#include <iostream>

int main() {
	K_means kmeans(12, 2, 3);
	vector<point> x{ {0,0},{0,1},{1,0},{1,1},{10,11},{11,11},{11,10},{10,10},{-5,-6},{-6,-5},{-5,-5},{-6,-6} };
	auto re=kmeans.train(x);
	for (auto x : re) {
		for (auto y : x) cout << y << ' ';
		cout << endl;
	}
}