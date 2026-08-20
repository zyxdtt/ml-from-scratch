#pragma once
#include <cmath>
#include <array>
#include <random>

using namespace std;

class CartPoleEnv {
private:
	const float PI = 3.1415926535f;
	const float g = 9.8f;
	const float m_cart = 1.0f;
	const float m_pole = 0.1f;
	const float m_total = m_cart + m_pole;
	const float l = 0.5f;
	const float ml = m_pole * l;
	const float tau = 0.02f;
	const float x_the = 2.4f;
	const float forcee = 10.0f;
	const float theta_the = 12.0f * PI / 180.0f;
	array<float, 4> state;
	bool done;
	int step_count;
	mt19937 gen;
	uniform_real_distribution<float> dis;
public:
	CartPoleEnv() :done(false), step_count(0), dis(-0.05f, 0.05f) {
		random_device rd;
		gen.seed(rd());
		reset();
	}
	void reset() {
		for (auto& x : state) x = dis(gen);
		done = false;
		step_count = 0;
	}
	pair<array<float, 4>, bool> step(int action) {
		float force = (action == 1) ? forcee : -forcee;
		auto [x, xs, theta, thetas] = state;
		float cos_theta = cos(theta);
		float sin_theta = sin(theta);
		float temp = (force + ml * thetas * thetas * sin_theta) / m_total;
		float theta_acc = (g * sin_theta - cos_theta * temp) / (l * (4.0f / 3.0f - m_pole * cos_theta * cos_theta / m_total));
		float x_acc = temp - ml * theta_acc * cos_theta / m_total;
		x += tau * xs;
		xs += tau * x_acc;
		theta += tau * thetas;
		thetas += tau * theta_acc;
		state = { x,xs,theta,thetas };
		bool out = (x<-x_the || x>x_the);
		bool po = (theta<-theta_the || theta>theta_the);
		done = out || po;
		step_count++;
		return { state,done };
	}
	// alternative accessor to avoid name collisions in some build environments
	pair<array<float, 4>, bool> get_state() const { return { state, done }; }
	bool is() const { return done; }
};