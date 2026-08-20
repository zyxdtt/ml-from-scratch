#include "pch.h"
#include "CartPole.hpp"

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
    cout << "Test point: " << total << endl;
}

int main() {
    MLP actor(4, 2, 32, 16), critic(4, 1, 16, 8);
    auto op_actor = make_unique<optim::Adam>(
        actor->parameters(),
        optim::AdamOptions(0.001) 
    );
    auto op_c = make_unique<optim::Adam>(
        critic->parameters(),
        optim::AdamOptions(0.003)
    );

    CartPoleEnv env;
    int K = 250, N = 8, epoch = 500;
    float gamma = 0.95;

    for (int i = 0; i < epoch; i++) {
        pp state, next;
        Tensor loss_actor = torch::tensor(0.0f);
        Tensor loss_critic = torch::tensor(0.0f);
        int total_steps = 0;
        for (int n = 0; n < N; n++) {
            env.reset();
            for (int k = 0; k < K; k++) {
                state = env.get_state();
                if (state.second == true) break;
                Tensor ss = torch::from_blob(state.first.data(), { 1, 4 }, torch::kFloat32).clone();
                auto logits = actor->forward(ss);
                auto probs = torch::softmax(logits, -1);
                int action = torch::multinomial(probs, 1).item<int>();
                next = env.step(action);
                total_steps++; 
                Tensor now_p = critic->forward(ss);
                Tensor td_error;
                if (next.second == true) {
                    td_error = torch::tensor({ 1.0f }) - now_p;
                }
                else {
                    Tensor ne = torch::from_blob(next.first.data(), { 1, 4 }, torch::kFloat32).clone();
                    Tensor next_p = critic->forward(ne);
                    td_error = torch::tensor({ 1.0f }) + gamma * next_p - now_p;
                }
                loss_actor = loss_actor + (-torch::log(probs[0][action]) * td_error.detach());
                loss_critic = loss_critic + (0.5 * td_error.pow(2));
                if (next.second == true) break; 
            }
        }
        float safe_steps = std::max(total_steps, 1);
        loss_actor = loss_actor / safe_steps;
        loss_critic = loss_critic / safe_steps;

        op_actor->zero_grad();
        op_c->zero_grad();
        loss_actor.backward();
        loss_critic.backward();

        op_actor->step();
        op_c->step();

        cout << "epoch " << i + 1 << ": steps " << total_steps
            << ", actor loss " << loss_actor.item<float>()
            << ", critic loss " << loss_critic.item<float>() << endl;
    }
    cout << "Train done.Test begin." << endl;
    eva(actor, env);
}