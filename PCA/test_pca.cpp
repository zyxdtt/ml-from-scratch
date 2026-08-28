#include "PCA.hpp"

int main() {
	mat a{ {1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15} };
	PCA pp(3, 5, 2);
	auto re = pp.pca(a);
	for (auto x : re) {
		for (auto y : x) cout << y << ' ';
		cout << endl;
	}
}