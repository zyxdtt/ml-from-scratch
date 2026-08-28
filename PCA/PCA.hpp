#pragma once
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <vector>
#include <algorithm>
#include <iostream>

using namespace Eigen;
using namespace std;
using point = vector<float>;
using mat = vector<point>;

class PCA {
private:
    int n;
    int m;
    int k;

    MatrixXf toEigenMatrix(const mat& matrix) {
        if (matrix.empty()) return MatrixXf();
        size_t rows = matrix.size();
        size_t cols = matrix[0].size();
        MatrixXf outMat(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            outMat.row(i) = VectorXf::Map(matrix[i].data(), cols);
        }
        return outMat;
    }

    mat eigenToVector(MatrixXf& mm) {
        size_t rows = mm.rows();
        size_t cols = mm.cols();
        mat vec(rows, point(cols));
        for (size_t i = 0; i < rows; ++i) {
            VectorXf::Map(vec[i].data(), cols) = mm.row(i);
        }
        return vec;
    }

public:
    PCA(int nn, int mm, int kk) : n(nn), m(mm), k(kk) {}

    mat pca(const mat& x) {
        auto mm = toEigenMatrix(x);
        auto vec = mm.colwise().mean();
        mm.rowwise() -= vec;
        mm /= sqrt(n - 1);
        JacobiSVD<MatrixXf, ComputeThinU | ComputeThinV> svd;
        svd.compute(mm);
        MatrixXf V_k = svd.matrixV().leftCols(k);
        MatrixXf result = mm * V_k;
        return eigenToVector(result);
    }
};