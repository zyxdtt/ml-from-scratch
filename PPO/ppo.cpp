#include "pch.h"
#include "CartPole.hpp"
#include <stddef.h>

using namespace torch;
using namespace std;

struct MLPImpl : public nn::Module {
    nn::Linear ln1{ nullptr };
    nn::Linear ln2{ nullptr };
    nn::Linear ln3{ nullptr };

    MLPImpl(int in, int out, int hide1, int hide2) {
        ln1 = register_module("ln1", nn::Linear(in, hide1));
        ln2 = register_module("ln2", nn::Linear(hide1, hide2));
        ln3 = register_module("ln3", nn::Linear(hide2, out));
    }

    Tensor forward(Tensor x) {
        x = ln1->forward(x);
        x = relu(x);
        x = ln2->forward(x);
        x = relu(x);
        x = ln3->forward(x);
        return x;
    }
};

TORCH_MODULE(MLP);

using pp = pair<array<float, 4>, bool>;



void eva(MLP& actor, CartPoleEnv& env, int max_st = 10000) {
    actor->eval();
    NoGradGuard no;
    env.reset();
    int total = 0;
    for (int s = 0; s < max_st; s++) {
        auto state = env.get_state();
        if (state.second) break;
        Tensor ss = torch::from_blob(state.first.data(), { 1,4 }, torch::kFloat32).clone();
        Tensor logits = actor->forward(ss);
        int action = argmax(logits, -1).item<int>();
        env.step(action);
        total++;
    }
    std::cout << "Test point: " << total << std::endl;
}

int main() {
    // create separate module instances for old and new actors
    MLP actor_old(4, 2, 32, 16);
    MLP actor_new(4, 2, 32, 16);
    MLP critic(4, 1, 16, 8);
    auto op_actor = make_unique<optim::Adam>(
        actor_new->parameters(),
        optim::AdamOptions(0.001)
    );
    auto op_c = make_unique<optim::Adam>(
        critic->parameters(),
        optim::AdamOptions(0.003)
    );

    CartPoleEnv env;
    int K = 250, N = 8, epoch = 500;
    float gamma = 0.95, eps = 0.2;

    vector<vector<Tensor>> ad(N), tra(N), pro(N);
    vector<vector<int>> acts(N);

    for (int i = 0; i < epoch; ++i) {

        // clear buffers
        for (int ni = 0; ni < N; ++ni) {
            tra[ni].clear();
            ad[ni].clear();
            pro[ni].clear();
            acts[ni].clear();
        }

        // copy parameters from actor_new to actor_old
        {
            auto params_new = actor_new->parameters();
            auto params_old = actor_old->parameters();
            for (size_t pi = 0; pi < params_old.size() && pi < params_new.size(); ++pi) {
                params_old[pi].data().copy_(params_new[pi].data());
            }
        }

        actor_old->eval();
        Tensor loss_critic = torch::zeros({1,1});
        Tensor loss_actor = torch::zeros({1,1});
        int total_steps = 0;

        // collect trajectories
        for (int n = 0; n < N; ++n) {
            env.reset();
            for (int k = 0; k < K; ++k) {
                auto state = env.get_state();
                if (state.second) break;
                Tensor ss = torch::from_blob(state.first.data(), {1,4}, torch::kFloat32).clone();
                tra[n].push_back(ss);

                Tensor logits = actor_old->forward(ss);
                    if (torch::isnan(logits).any().item<bool>() || torch::isinf(logits).any().item<bool>()) return -1;
                Tensor probs = torch::softmax(logits, -1).clamp(1e-6, 1.0);
                auto sum_p = probs.sum().item<double>();
                    if (!(sum_p > 0.0) || torch::isnan(probs).any().item<bool>() || torch::isinf(probs).any().item<bool>()) return -1;
                int action = torch::multinomial(probs, 1).item<int>();
                if (action < 0 || action >= (int)probs.size(1)) return -1;
                acts[n].push_back(action);
                pro[n].push_back(probs[0][action]);

                auto next = env.step(action);
                total_steps++;

                Tensor now_p = critic->forward(ss);
                Tensor G;
                if (next.second) G = torch::tensor(1.0f);
                else {
                    Tensor ne = torch::from_blob(next.first.data(), {1,4}, torch::kFloat32).clone();
                    G = torch::tensor(1.0f) + gamma * critic->forward(ne);
                }
                    ad[n].push_back((G - now_p).detach());
                loss_critic = loss_critic + 0.5 * (G - now_p).pow(2);
                if (next.second) break;
            }
        }

        if (total_steps == 0) total_steps = 1;

        // compute actor loss
        for (int n = 0; n < N; ++n) {
            (void)0;
            for (size_t k = 0; k < tra[n].size(); ++k) {
                Tensor logits = actor_new->forward(tra[n][k]);
                    if (torch::isnan(logits).any().item<bool>() || torch::isinf(logits).any().item<bool>()) return -1;
                Tensor probs = torch::softmax(logits, -1);
                int old_action = acts[n][k];
                Tensor stored_p = pro[n][k];
                if (torch::isnan(stored_p).any().item<bool>() || torch::isinf(stored_p).any().item<bool>()) return -1;
                double sp = stored_p.item<double>();
                if (fabs(sp) < 1e-12) return -1;
                auto ratio = probs[0][old_action] / stored_p;
                    if (torch::isnan(ratio).any().item<bool>() || torch::isinf(ratio).any().item<bool>()) return -1;
                Tensor surr1 = ratio * ad[n][k];
                Tensor surr2 = torch::clamp(ratio, 1.0f - eps, 1.0f + eps) * ad[n][k];
                loss_actor += torch::min(surr1, surr2);
            }
        }

        loss_actor = -loss_actor / total_steps;
        loss_critic = loss_critic / total_steps;

        // debug: check loss values before backward
        if (torch::isnan(loss_actor).item<bool>() || torch::isinf(loss_actor).item<bool>() || torch::isnan(loss_critic).item<bool>() || torch::isinf(loss_critic).item<bool>()) return -1;

        op_actor->zero_grad();
        loss_actor.backward();
        op_actor->step();

        op_c->zero_grad();
        loss_critic.backward();
        op_c->step();

        std::cout << "epoch " << i + 1 << ": steps " << total_steps
            << ", actor loss " << loss_actor.item<float>()
            << ", critic loss " << loss_critic.item<float>() << std::endl;
    }
    std::cout << "Train done.Test begin." << std::endl;
    eva(actor_new, env);
}
