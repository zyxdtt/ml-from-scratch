#pragma once
//2026 08 03
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iterator>
#include <random>

using namespace std;
using point = vector<double>;

class HMM {//only support one and discrete observation,target is to get/
private:   //the core idea of Markov model/
		   //if you write high dim and continuous code,the trick will be over complicated.
	vector<point> A;
	vector<point> B;
	point pi;
	bool trained;
	int num_box;
	int num_ball;
public:
	HMM(int state, int choice) :A(state, point(state, 0)),
		B(state, point(choice, 0)), pi(state, 0) {
		num_box = state;
		num_ball = choice;
		trained = false;
	}
	void print_weight() const {
		if (!trained) cout << "You have not trained!" << endl;
		if (num_box >= 100) {
			cout << "The output too large, watch it in HMM_weight.txt" << endl;
			save_model();
			return;
		}
		cout << "A:" << endl;
		for (const auto& x : A) {
			copy(x.begin(), x.end(), ostream_iterator<double>(cout, " "));
			cout << '\n';
		}
		cout << "B:" << endl;
		for (const auto& x : B) {
			copy(x.begin(), x.end(), ostream_iterator<double>(cout, " "));
			cout << '\n';
		}
		cout << "PI:" << endl;
		copy(pi.begin(), pi.end(), ostream_iterator<double>(cout, " "));
		cout << '\n';
		cout << "Done." << endl;
		return;
	}
	void save_model() const {
		ofstream file("HMM_weight.txt");
		streambuf* original_cout_buf = cout.rdbuf(file.rdbuf());
		cout << "A:" << endl;
		for (const auto& x : A) {
			copy(x.begin(), x.end(), ostream_iterator<double>(cout, " "));
			cout << '\n';
		}
		cout << "B:" << endl;
		for (const auto& x : B) {
			copy(x.begin(), x.end(), ostream_iterator<double>(cout, " "));
			cout << '\n';
		}
		cout << "PI:" << endl;
		copy(pi.begin(), pi.end(), ostream_iterator<double>(cout, " "));
		cout << '\n';
		cout << "Done." << endl;
		cout.rdbuf(original_cout_buf);
		file.close();
		return;
	}
	void train(const vector<vector<int>>& obs,const vector<vector<int>>& state) {
		trained = true;
		int s = obs.size();
		for (int i = 0; i < num_box; i++) {
			int start = state[0][i];
			pi[start]++;
		}
		for (auto& x : pi) x /= (double)num_box;
		for (int i = 0; i < s; i++) {
			for (int j = 0; j < obs[i].size() - 1; j++) {
				int ii = state[i][j], jj = state[i][j + 1];
				A[ii][jj]++;
			}
		}
		for (auto& x : A)
			for (auto& y : x) y /= (double)num_box;
		for (int i = 0; i < s; i++) 
			for (int j = 0; j < obs[i].size(); j++) B[state[i][j]][obs[i][j]]++;
		for (auto& x : B)
			for (auto& y : x) y /= (double)num_ball;
	}
    void train(const vector<vector<int>>& obs, int max_iter = 0) {
        int T = 0;
        for (int s = 0; s < obs.size(); s++) T = max(T, (int)obs[s].size());
        trained = true;
        static random_device rd;
        static mt19937 gen(rd());
        uniform_real_distribution<double> dis(0.0, 1.0);
        double sum = 0.0;
        for (int i = 0; i < num_box; ++i) { pi[i] = dis(gen); sum += pi[i]; }
        for (int i = 0; i < num_box; ++i) pi[i] /= sum;
        for (int i = 0; i < num_box; ++i) {
            sum = 0.0;
            for (int j = 0; j < num_box; ++j) { A[i][j] = dis(gen); sum += A[i][j]; }
            for (int j = 0; j < num_box; ++j) A[i][j] /= sum;
        }
        for (int i = 0; i < num_box; ++i) {
            sum = 0.0;
            for (int j = 0; j < num_ball; ++j) { B[i][j] = dis(gen); sum += B[i][j]; }
            for (int j = 0; j < num_ball; ++j) B[i][j] /= sum;
        }
        vector<point> a(T, point(num_box, 0)), b(T, point(num_box, 0));
        vector<double> c(T, 0);
        bool yes = true, first = true;
        int cur = 0; char ch;
        vector<point> Ac(num_box, point(num_box, 0)), Bc(num_box, point(num_ball, 0));
        point pic(num_box, 0);
        while (yes) {
            if (first) {
                if (!max_iter) max_iter += 50;
                first = false;
            }
            else max_iter += 50;
            for (; cur < max_iter; cur++) {
                vector<point> num_A(num_box, point(num_box, 0));
                point den_A(num_box, 0);
                vector<point> num_B(num_box, point(num_ball, 0));
                point den_B(num_box, 0);
                point num_pi(num_box, 0);
                for (int s = 0; s < obs.size(); s++) {
                    int t = obs[s].size();
                    for (int j = 0; j < num_box; j++) a[0][j] = pi[j] * B[j][obs[s][0]];
                    c[0] = 0;
                    for (int j = 0; j < num_box; j++) c[0] += a[0][j];
                    if (c[0] < 1e-300) c[0] = 1.0;
                    for (int j = 0; j < num_box; j++) a[0][j] /= c[0];
                    for (int i = 1; i < t; i++) {
                        for (int j = 0; j < num_box; j++) {
                            double sum = 0;
                            for (int n = 0; n < num_box; n++) sum += a[i - 1][n] * A[n][j];
                            a[i][j] = sum * B[j][obs[s][i]];
                        }
                        c[i] = 0;
                        for (int j = 0; j < num_box; j++) c[i] += a[i][j];
                        if (c[i] < 1e-300) c[i] = 1.0;
                        for (int j = 0; j < num_box; j++) a[i][j] /= c[i];
                    }
                    for (int j = 0; j < num_box; j++) b[t - 1][j] = 1.0;
                    for (int i = t - 2; i >= 0; i--) {
                        for (int j = 0; j < num_box; j++) {
                            double sum = 0;
                            for (int n = 0; n < num_box; n++) {
                                sum += A[j][n] * B[n][obs[s][i + 1]] * b[i + 1][n];
                            }
                            b[i][j] = sum / c[i + 1];
                        }
                    }
                    for (int i = 0; i < num_box; i++) {
                        num_pi[i] += a[0][i] * b[0][i];
                    }
                    for (int i = 0; i < num_box; i++) {
                        for (int tt = 0; tt < t; tt++) {
                            double gamma = a[tt][i] * b[tt][i];
                            den_B[i] += gamma;
                            num_B[i][obs[s][tt]] += gamma;
                            if (tt < t - 1) {
                                den_A[i] += gamma;
                            }
                        }
                        if (t > 1) {
                            for (int j = 0; j < num_box; j++) {
                                for (int tt = 0; tt < t - 1; tt++) {
                                    num_A[i][j] += a[tt][i] * A[i][j] * B[j][obs[s][tt + 1]] * b[tt + 1][j] / c[tt + 1];
                                }
                            }
                        }
                    }
                }
                for (int i = 0; i < num_box; i++) {
                    if (num_pi[i] > 1e-300 || true) {
                    }
                    pic[i] = num_pi[i];
                    for (int j = 0; j < num_box; j++) {
                        if (den_A[i] > 1e-300)
                            A[i][j] = num_A[i][j] / den_A[i];
                        else
                            A[i][j] = 1.0 / num_box;
                    }

                    for (int j = 0; j < num_ball; j++) {
                        if (den_B[i] > 1e-300)
                            B[i][j] = num_B[i][j] / den_B[i];
                        else
                            B[i][j] = 1.0 / num_ball;
                    }
                }
                double pi_sum = 0;
                for (int i = 0; i < num_box; i++) pi_sum += pic[i];
                if (pi_sum > 1e-300) {
                    for (int i = 0; i < num_box; i++) pi[i] = pic[i] / pi_sum;
                }
            }
            cout << "current iters: " << cur << endl;
            print_weight();
            cout << "Do you want to train another 50 epoches?[y/n]";
            cin >> ch;
            if (ch == 'n') yes = false;
        }
        cout << "You have completed the training process.Do you want to save model?[y/n]";
        cin >> ch;
        if (ch == 'y') save_model();
    }

};