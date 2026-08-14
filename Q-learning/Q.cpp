#include <vector>
#include <algorithm>
#include <string>
#include <utility>
#include <random>
#include <unordered_map>
#include <iostream>

using namespace std;
using pp = pair<int, int>;
mt19937 gen{ random_device{}() };

template<class Ty1, class Ty2>
std::pair<Ty1, Ty2>& operator+=(std::pair<Ty1, Ty2>& p1, const std::pair<Ty1, Ty2>& p2) {
	p1.first += p2.first;
	p1.second += p2.second;
	return p1;
}

int main() {
	vector<string> mi{ "S****",
	"*****",
	"*****",
	"*****",
	"****#" };
	vector<vector<double>> Q(49, vector<double>(4, 0));
	int K , T = 15;
	cin >> K;
	double eps = 0.1;
	double lr = 0.1, gamma = 0.9;
	uniform_real_distribution<double> dis(0.0, 1.0);
	uniform_int_distribution<int> in(0, 3);
	unordered_map<int, pp> mp{ {0,{0,1}},{1,{1,0}},{2,{-1,0}},{3,{0,-1}} };
	for (int k = 0; k < K; k++) {
		pp state = { 1,1 }; int reward = 0;
		for (int t = 0; t < T; t++) {
			int pa = state.first * 7 + state.second;
			double g = dis(gen);
			int ac;
			if (g < eps) ac = in(gen);
			else ac = max_element(Q[pa].begin(), Q[pa].end()) - Q[pa].begin();
			state += mp[ac];
			int re, paa = state.first * 7 + state.second;
			double ma = *max_element(Q[paa].begin(), Q[paa].end());
			if (state == make_pair(4,4)) {
				re = 100;
				reward += re;
				Q[pa][ac] += lr * (re + gamma * ma - Q[pa][ac]);
				break;
			}
			else if (state.first < 1 || state.second < 1 || state.first>5 || state.second>5) {
				re = -100;
				reward += re;
				Q[pa][ac] += lr * (re + gamma * ma - Q[pa][ac]);
				break;
			}
			else re = -2;
			reward += re;
			Q[pa][ac] += lr * (re + gamma * ma - Q[pa][ac]);
		}
		cout << "epoch " << k + 1 << ": " << "reward: " << reward << endl;
	}
}