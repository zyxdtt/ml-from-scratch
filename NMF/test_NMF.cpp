#include <iostream>
#include <iomanip>
#include "NMF.hpp"

using namespace std;

void print_mat(const mat& m, const char* name) {
    cout << name << " (" << m.size() << "x" << m[0].size() << "):\n";
    for (auto& row : m) {
        for (auto v : row) cout << setw(10) << fixed << setprecision(4) << v;
        cout << "\n";
    }
    cout << "\n";
}

int main() {
    mat x = {
        {3, 1, 4, 1},
        {3, 1, 4, 1},
        {3, 1, 4, 1},
        {1, 2, 1, 3},
        {1, 2, 1, 3},
        {1, 2, 1, 3}
    };

    int k = 2;
    NMF nmf(k);
    auto [u, v] = nmf.train(x, 500);

    print_mat(x, "Original X");
    print_mat(u, "Factor U");
    print_mat(v, "Factor V");

    // Reconstruct UV
    mat uv(x.size(), vector<double>(x[0].size(), 0));
    for (int i = 0; i < (int)x.size(); i++)
        for (int j = 0; j < (int)x[0].size(); j++)
            for (int p = 0; p < k; p++)
                uv[i][j] += u[i][p] * v[p][j];

    print_mat(uv, "Reconstructed UV");

    // Frobenius norm of (X - UV)
    double err = 0;
    for (int i = 0; i < (int)x.size(); i++)
        for (int j = 0; j < (int)x[0].size(); j++)
            err += pow(x[i][j] - uv[i][j], 2);
    err = sqrt(err);

    cout << "Frobenius reconstruction error: " << err << "\n";

    if (err < 0.01)
        cout << "PASS: reconstruction error is negligible.\n";
    else
        cout << "FAIL: reconstruction error is too large.\n";

    return 0;
}