#pragma once
//2026 08 09
#include <vector>
#include <algorithm>
#include <utility>
#include <random>
#include <initializer_list>
#include <string>
#include <map>
#include <iostream>
#include <cmath>

using namespace std;

class agent {
public:
	virtual ~agent() = default;
	virtual int exploration() = 0;
	virtual void learn(int choose, int reward) = 0;
	virtual string name() const = 0;
	virtual int get_T() const = 0;
	virtual int get_num_arms() const = 0;
	virtual void reset() = 0;
};

class bandits {
private:
	int num_arms;
	vector<double> pro;
public:
	mt19937 gen{ random_device{}() };
	bandits(int num, const vector<double>& p) :
		num_arms(num), pro(p) {}

	bandits() = default;

	~bandits() = default;

	int produce(int arm) {
		bernoulli_distribution d(pro[arm]);
		return d(gen);
	}

	void reset(int num, const vector<double>& p) {
		num_arms = num;
		pro = p;
	}
};

class test {
private:
	vector<agent*> agents;
public:
	test(initializer_list<agent*> agent_list) :agents(agent_list) {}
	test(const vector<agent*>& agent_list) : agents(agent_list) {}
	test(vector<agent*>&& agent_list) : agents(std::move(agent_list)) {}

	~test() = default;

	map<double, string> regret(int cnt = 100, int T = -1, int num = -1) const {
		map<double, string> reg;
		bandits ban;
		uniform_real_distribution<double> dis(0.0, 1.0);
		bool together = true;
		if (T == -1) together = false;
		vector<double> point(agents.size(), 0);
		vector<double> re(agents.size(), 0);
		for (int c = 0; c < cnt; c++) {
			double r = 0;
			if (together) {
				vector<double> pro(num);
				generate(pro.begin(), pro.end(), [&]() {return dis(ban.gen); });
				ban.reset(num, pro);
				r = T * (*max_element(pro.begin(), pro.end()));
			}
			for (int i = 0; i < agents.size(); i++) {
				
				int pp = 0;
				auto agent = agents[i];
				agent->reset();
				if (!together) {
					vector<double> pro(agent->get_num_arms());
					generate(pro.begin(), pro.end(), [&]() {return dis(ban.gen); });
					ban.reset(num, pro);
					r = agent->get_T() * (*max_element(pro.begin(), pro.end()));
				}

				for (int t = 0; t < agent->get_T(); t++) {
					int choose = agent->exploration();
					int reward = ban.produce(choose);
					pp += reward;
					agent->learn(choose, reward);
				}
				re[i] += (r - pp) / r;
				point[i] += pp;
			}
		}
		for (int i = 0; i < agents.size(); i++) {
			cout << agents[i]->name() << ": " << point[i] / cnt << " points" << endl;
			reg[re[i] / cnt] = agents[i]->name();
		}
		return reg;
	}
};

class exploration_first_agent :public agent {
private:
	vector<pair<int, int>> memory;
	int limit;
	int t;
	int winner;
	int choose;
	int T;
	int num_arms;
public:
	exploration_first_agent(int TT, int num) :T(TT),
		num_arms(num), t(-1), choose(-1) {
		limit = T / 2;
		winner = -1;
	}

	~exploration_first_agent() = default;

	int exploration() override {
		t++;
		if (t > limit && winner != -1) return winner;
		else if (t > limit) {
			vector<vector<int>> temp(num_arms, { 0,0 });
			for (auto [arm, reward] : memory) temp[arm][reward]++;
			vector<double> tt(num_arms);
			for (int i = 0; i < num_arms; i++)
				tt[i] = temp[i][1] * 1.0 / (temp[i][1] + temp[i][0]);
			winner = max_element(tt.begin(), tt.end()) - tt.begin();
			return winner;
		}
		else {
			choose++;
			if (choose >= num_arms) choose = 0;
			return choose;
		}
	}

	void learn(int choose, int reward) override {
		memory.emplace_back(choose, reward);
	}

	void reset() override {
		memory.clear();
		t = -1; choose = -1; winner = -1;
	}

	string name() const override { return "exploration_first_agent"; }

	int get_T() const override { return T; }
	int get_num_arms() const override { return num_arms; }
};

class greedy_agent :public agent {
private:
	int T;
	int num_arms;
	string type;
	int t;
	double eps;
	vector<int> counts;
	vector<double> values;
	mt19937 gen{ random_device{}() };
	uniform_real_distribution<double> urd{0.0, 1.0};

public:
	greedy_agent(int T_, int num_arms_, double eps = 0.1, string type_ = "normal") :
		T(T_), num_arms(num_arms_), type(type_), t(0), counts(num_arms_, 0),
		values(num_arms_, 0.0), eps(eps) {
		if (!(type == "normal" || type == "cosine" || type == "linear"))
			type = "normal";
	}

	~greedy_agent() override = default;

	int exploration() override {
		double frac = t * 1.0 / T;
		double epss = eps;
		if (type == "linear") epss = eps * (1 - frac);
		else if (type == "cosine") epss = eps * (cos(acos(-1.0) * frac) + 1e-3);
		double r = urd(gen);
		if (r < epss) {
			uniform_int_distribution<int> idist(0, num_arms - 1);
			return idist(gen);
		}
		return static_cast<int>(max_element(values.begin(), values.end()) - values.begin());
	}

	void learn(int choose, int reward) override {
		t++;
		counts[choose]++;
		double n = static_cast<double>(counts[choose]);
		values[choose] += (static_cast<double>(reward) - values[choose]) / n;
	}

	string name() const override { 
		string temp = "greedy_agent_" + type + "_" + to_string(eps);
		return temp;
	}

	void reset() override {
		fill(counts.begin(), counts.end(), 0);
		t = 0;
		fill(values.begin(), values.end(), 0.0);
	}

	int get_T() const override { return T; }
	int get_num_arms() const override { return num_arms; }
};

class UCB_agent :public agent {
private:
	int T;
	int num_arms;
	int t;
	vector<double> hope;
	vector<int> cnt;
public:
	UCB_agent(int TT, int num) :
		T(TT), num_arms(num), hope(num, 0), cnt(num, 0) {}

	~UCB_agent() override = default;

	int exploration() override {
		if (t < num_arms) return t;
		double ma = -1e9;
		int idx = 0;
		for (int i = 0; i < num_arms; i++) {
			double up = hope[i] + sqrt(2.0 * log(t+1) / cnt[i]);
			if (ma < up) ma = up, idx = i;
		}
		return idx;
	}

	void learn(int choose, int reward) override {
		t++;
		cnt[choose]++;
		hope[choose] += (reward - hope[choose]) / cnt[choose];
	}

	
	string name() const override { return "UCB_agent"; }

	int get_T() const override { return T; }
	int get_num_arms() const override { return num_arms; }

	void reset() override {
		fill(hope.begin(), hope.end(), 0);
		fill(cnt.begin(), cnt.end(), 0);
		t = 0;
	}
};