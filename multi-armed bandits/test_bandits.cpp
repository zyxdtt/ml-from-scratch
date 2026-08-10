#include "multi_armed_bandits.hpp"

int main() {
	greedy_agent a(100, 10, 0.2, "cosine");
	vector<agent*> agents;
	agents.push_back(&a);
	UCB_agent u(100, 10);
	agents.push_back(&u);

	test tt(agents);
	auto temp = tt.regret(10000,100,10);
	for (auto [regret, agent] : temp) cout << agent << ": " << regret << endl;
}