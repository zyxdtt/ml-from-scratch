#pragma once
//2026 07 31
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <Eigen/Dense>
#include <fstream>

using namespace std;
using namespace Eigen;
using point = vector<double>;
const double M_PI = 3.14159265358979323846;

struct GS_xD {
    int dim;
    Eigen::VectorXd mu;
    Eigen::MatrixXd cov;
    Eigen::MatrixXd cov_inv;
    double log_det;
    double two_pi_log;

    GS_xD() : dim(0), log_det(0.0), two_pi_log(0.0) {}

    GS_xD(int d) : dim(d), mu(Eigen::VectorXd::Zero(d)), cov(Eigen::MatrixXd::Identity(d, d)) {
        updateCache();
    }

    GS_xD(const std::vector<double>& mean, const std::vector<std::vector<double>>& covariance) {
        int d = (int)mean.size();
        dim = d;
        mu = Eigen::VectorXd::Zero(d);
        cov = Eigen::MatrixXd::Zero(d, d);
        for (int i = 0; i < d; ++i) {
            mu(i) = mean[i];
            for (int j = 0; j < d; ++j) cov(i, j) = covariance[i][j];
        }
        updateCache();
    }

    GS_xD(const Eigen::VectorXd& mean, const Eigen::MatrixXd& covariance)
        : dim(mean.size()), mu(mean), cov(covariance) {
        updateCache();
    }

    void setParameters(const std::vector<double>& mean, const std::vector<std::vector<double>>& covariance) {
        int d = (int)mean.size();
        dim = d;
        mu = Eigen::VectorXd::Zero(d);
        cov = Eigen::MatrixXd::Zero(d, d);
        for (int i = 0; i < d; ++i) {
            mu(i) = mean[i];
            for (int j = 0; j < d; ++j) cov(i, j) = covariance[i][j];
        }
        updateCache();
    }

    double logPdf(const std::vector<double>& y) const {
        Eigen::Map<const Eigen::VectorXd> y_map(y.data(), y.size());
        return logPdf(y_map);
    }

    double logPdf(const Eigen::VectorXd& y) const {
        Eigen::VectorXd diff = y - mu;
        double mahalanobis_dist_sq = diff.transpose() * cov_inv * diff;
        return -0.5 * (two_pi_log + log_det + mahalanobis_dist_sq);
    }

    std::pair<std::vector<double>, std::vector<std::vector<double>>> getParameters() const {
        std::vector<double> mean(dim);
        std::vector<std::vector<double>> covariance(dim, std::vector<double>(dim));
        for (int i = 0; i < dim; ++i) {
            mean[i] = mu(i);
            for (int j = 0; j < dim; ++j) {
                covariance[i][j] = cov(i, j);
            }
        }
        return { mean, covariance };
    }
private:
    void updateCache() {
        if (dim == 0) return;
        double epsilon = 1e-6;
        Eigen::MatrixXd reg_cov = cov + epsilon * Eigen::MatrixXd::Identity(dim, dim);
        Eigen::LLT<Eigen::MatrixXd> llt(reg_cov);
        if (llt.info() == Eigen::NumericalIssue) {
            reg_cov.diagonal().array() += 1e-3;
            cov_inv = reg_cov.inverse();
        }
        else {
            cov_inv = llt.solve(Eigen::MatrixXd::Identity(dim, dim));
        }
        log_det = 0.0;
        if (llt.info() != Eigen::NumericalIssue) {
            const Eigen::VectorXd& diag = llt.matrixLLT().diagonal();
            log_det = 2.0 * diag.array().abs().log().sum();
        }
        else {
            double det = reg_cov.determinant();
            log_det = std::log(std::abs(det) + 1e-300);
        }
        two_pi_log = dim * std::log(2.0 * M_PI);
    }
};

class GMM_EM {
private:
    int K, D;
    bool trained;
    vector<GS_xD> model;
    vector<double> ak;
public:
    GMM_EM(int k, int d) {
        K = k; D = d;
        std::mt19937 rng(42);
        std::normal_distribution<double> dist(0.0, 1.0);
        for (int i = 0; i < k; ++i) {
            Eigen::VectorXd mu(d);
            for (int j = 0; j < d; ++j) mu(j) = dist(rng);
            Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(d, d);
            model.emplace_back(mu, cov);
        }
        ak.assign(k, 1.0 / k);
        trained = false;
    }
    ~GMM_EM() = default;
    void train(const vector<point>& X_train, const int max_iter = 5000, const double tol = 1e-6) {
        trained = true;
        double prev_log_likelihood = -1e300;
        int j = X_train.size();
        vector<vector<double>> lamda(j, vector<double>(K));
        vector<double> cache(K);
        vector<double> mean(D, 0), diff(D);
        vector<vector<double>> var(D, vector<double>(D, 0));
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dis(0, j - 1);

        for (int iter = 0; iter < max_iter; ++iter) {
            // ---------- E-step ----------
            double log_likelihood = 0.0;
            for (int c = 0; c < j; ++c) {
                for (int i = 0; i < K; ++i)
                    cache[i] = log(ak[i]) + model[i].logPdf(X_train[c]);
                auto p = max_element(cache.begin(), cache.end());
                double m = 0.0;
                for (int i = 0; i < K; ++i) m += exp(cache[i] - *p);
                m = log(m) + *p;
                log_likelihood += m;
                for (int i = 0; i < K; ++i)
                    lamda[c][i] = exp(cache[i] - m);
            }

            if (iter > 0 && fabs(log_likelihood - prev_log_likelihood) < tol) break;
            prev_log_likelihood = log_likelihood;

            // ---------- M-step ----------
            for (int i = 0; i < K; ++i) {
                // 1. 计算有效样本数 w
                double w = 0.0;
                for (int c = 0; c < j; ++c) w += lamda[c][i];

                // 2. 如果成分死亡，复活它
                if (w < 1e-12) {
                    int idx = dis(gen);
                    vector<vector<double>> init_cov(D, vector<double>(D, 0.0));
                    for (int d = 0; d < D; ++d) init_cov[d][d] = 1.0;
                    model[i].setParameters(X_train[idx], init_cov);
                    ak[i] = 1.0 / K;
                    continue;
                }

                // 3. 更新均值
                fill(mean.begin(), mean.end(), 0.0);
                for (int c = 0; c < j; ++c) {
                    double gamma = lamda[c][i];
                    for (int d = 0; d < D; ++d)
                        mean[d] += gamma * X_train[c][d];
                }
                for (int d = 0; d < D; ++d) mean[d] /= w;

                // 4. 更新协方差
                for (int r = 0; r < D; ++r) fill(var[r].begin(), var[r].end(), 0.0);
                for (int c = 0; c < j; ++c) {
                    double gamma = lamda[c][i];
                    for (int d = 0; d < D; ++d)
                        diff[d] = X_train[c][d] - mean[d];
                    for (int r = 0; r < D; ++r)
                        for (int col = 0; col < D; ++col)
                            var[r][col] += gamma * diff[r] * diff[col];
                }
                for (int r = 0; r < D; ++r)
                    for (int col = 0; col < D; ++col)
                        var[r][col] /= w;

                // 5. 更新模型和权重
                model[i].setParameters(mean, var);
                ak[i] = w / j;
            }
        }
    }
    /*void train(const vector<point>& X_train, const int max_iter = 5000, const double tol = 1e-6) {
        trained = true;
        double prev_log_likelihood = -1e300;
        int j = X_train.size();
        vector<vector<double>> lamda(j, vector<double>(K));
        vector<double> cache(K);
        vector<double> mean(D, 0), diff(D);
        vector<vector<double>> var(D, vector<double>(D, 0));
        for (int iter = 0; iter < max_iter; iter++) {
            double log_likelihood = 0.0;
            for (int c = 0; c < j; c++) {
                for (int i = 0; i < K; i++) cache[i] = log(ak[i]) + model[i].logPdf(X_train[c]);
                auto p = max_element(cache.begin(), cache.end());
                double m = 0.0;
                for (int i = 0; i < K; i++) m += exp(cache[i] - *p);
                m = log(m) + *p;
                log_likelihood += m;
                for (int i = 0; i < K; i++) lamda[c][i] = exp(cache[i] - m);
            }
            if (iter > 0 && fabs(log_likelihood - prev_log_likelihood) < tol) break;
            prev_log_likelihood = log_likelihood;
            for (int i = 0; i < K; i++) {
                fill(mean.begin(), mean.end(), 0.0);
                for (int r = 0; r < D; r++) fill(var[r].begin(), var[r].end(), 0.0);
                double w = 0;
                for (int c = 0; c < j; c++) w += lamda[c][i];
                ak[i] = w / j;
                for (int c = 0; c < j; c++) {
                    for (int d = 0; d < D; d++) mean[d] += lamda[c][i] * X_train[c][d];
                }
                for (int d = 0; d < D; d++) mean[d] /= w;
                for (int c = 0; c < j; c++) {
                    double gamma = lamda[c][i];
                    for (int d = 0; d < D; d++) diff[d] = X_train[c][d] - mean[d];
                    for (int r = 0; r < D; r++) {
                        for (int col = 0; col < D; col++) var[r][col] += gamma * diff[r] * diff[col];
                    }
                }
                for (int r = 0; r < D; r++) {
                    for (int col = 0; col < D; col++) {
                        var[r][col] /= w;
                    }
                }
                model[i].setParameters(mean, var);
            }
        }
    }*/
    bool save_model() const {
        if (!trained) return false;
        ofstream file("weight.txt");
        if (!file.is_open()) return false;
        file << K << ' ' << D << '\n';
        for (int i = 0; i < K; i++) {
            auto [mean, var] = model[i].getParameters();
            file << ak[i] << '\n';
            for (int d = 0; d < D; d++) file << mean[d] << ' ';
            file << '\n';
            for (int r = 0; r < D; r++) {
                for (int col = 0; col < D; col++) file << var[r][col] << ' ';
                file << '\n';
            }
        }
        return true;
    }
    int predict(const point& x) const {
        if (!trained) return -1;
        double best_log = -1e300;
        int best_idx = 0;
        for (int i = 0; i < K; ++i) {
            double log_val = log(ak[i]) + model[i].logPdf(x);
            if (log_val > best_log) {
                best_log = log_val;
                best_idx = i;
            }
        }
        return best_idx;
    }
};