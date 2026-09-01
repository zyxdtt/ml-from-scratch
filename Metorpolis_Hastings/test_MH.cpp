#include <fstream>
#include <cmath>
#include "Metropolis_Hastings.hpp"

// 双峰高斯混合分布
// p(x) = 0.5 * N(x; [-3,-3], I) + 0.5 * N(x; [3,3], I)
double bimodal(point x) {
    double d1 = 0.0, d2 = 0.0;
    for (int i = 0; i < x.size(); i++) {
        d1 += (x[i] + 3.0) * (x[i] + 3.0);
        d2 += (x[i] - 3.0) * (x[i] - 3.0);
    }
    return 0.5 * exp(-d1 / 2.0) + 0.5 * exp(-d2 / 2.0);
}

double ring(point x) {
    double r = 0.0;
    for (int i = 0; i < x.size(); i++) {
        r += x[i] * x[i];
    }
    r = sqrt(r);
    return exp(-(r - 5.0) * (r - 5.0) / 0.5);
}

double banana(point x) {
    double x1 = x[0], x2 = x[1];
    double val = (1 - x1) * (1 - x1) + 100.0 * (x2 - x1 * x1) * (x2 - x1 * x1);
    return exp(-val / 20.0);
}

int main() {
    int dim = 2;
    int burnin = 1000;
    int samples = 10000;

    MH mh(dim, burnin, burnin + samples);
    //cout << "kjdioawujh" << endl;
    auto result = mh.sample(banana);

    // 输出到文件，方便画图
    ofstream out("samples.txt");
    for (auto& p : result) {
        for (int i = 0; i < p.size(); i++) {
            out << p[i] << (i + 1 < p.size() ? "\t" : "\n");
        }
    }
    out.close();

    cout << "complete" << result.size() << "samples.txt" << endl;
    return 0;
}