<div align="center">

# 🎰 Multi-Armed Bandit Playground

### *When Cosine Annealing Meets the Slot Machine*

*A C++ framework for exploring the exploration-exploitation tradeoff — from first principles to hard-won insights.*

[![C++](https://img.shields.io/badge/C%2B%2B-17-blue?logo=c%2B%2B)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen)]()

</div>

---

## 📖 Theoretical Foundation

This project is grounded in **Chapter 36 (Reinforcement Learning)** from Professor Hang Li's classic textbook *"Methods of Machine Learning"*. The MAB framework, regret analysis, and the exploration-exploitation dilemma discussed in that chapter served as the blueprint for everything you see here.

---

## 🧠 Algorithms

### 1. 🏃 Exploration-First Agent

> *"Look before you leap — but only look for a while."*

A dead-simple baseline: spend the first 50% of steps exploring randomly, then go full greedy for the rest. No tuning required. Robust, but rigid — like a sprinter who sprints the first lap and then just... coasts.

---

### 2. 🎯 UCB (Upper Confidence Bound)

> *"Be optimistic in the face of uncertainty."*

The textbook classic. UCB1 selects actions by balancing estimated reward with an uncertainty bonus:

$$a_t = \arg\max_i \left( \bar{x}_i + \sqrt{\frac{2 \ln t}{n_i}} \right)$$

It comes with a beautiful $O(\log T)$ regret guarantee — but only when $T \to \infty$. More on that later... 😏

---

### 3. 🔥 Cosine Annealing Greedy — *Our Star Player*

#### 💡 Motivation: Why Fix What's Broken?

The vanilla $\epsilon$-Greedy algorithm uses a **fixed** exploration rate from start to finish. This is like driving at 60 mph whether you're on a highway or in a school zone:

| Problem | What Happens |
|---------|-------------|
| 🐌 $\epsilon$ too low | You never explore enough → premature convergence to a suboptimal arm |
| 🤡 $\epsilon$ too high | You keep randomly exploring late into the game → wasting precious steps |

**Neither extreme works.** So why not let $\epsilon$ *breathe*?

Our inspiration comes from **Cosine Annealing** — the learning rate schedule that took the deep learning world by storm. The cosine curve has a magical shape:

- **Flat at the top** → high exploration early on, giving every arm a fair shot
- **Steep in the middle** → rapid transition from exploration to exploitation
- **Flat at the bottom** → near-zero exploration late, maximizing reward harvest

It's like a well-paced marathon runner: start steady, push hard in the middle, and cruise to the finish. 🏅

#### ⚙️ Implementation

We support three decay schedules: **constant**, **linear**, and **cosine annealing**. The core logic is beautifully compact:

```cpp
double frac = t * 1.0 / T;
double epss = eps;
if (type == "linear") epss = eps * (1 - frac);
else if (type == "cosine") epss = eps * (cos(acos(-1.0) * frac) + 1e-3);
```

Here, `frac` is the normalized progress ($t/T$), `eps` is the initial exploration rate, and `epss` is the dynamically adjusted rate at step `t`. The `1e-3` floor prevents $\epsilon$ from hitting exact zero, keeping a tiny safety net of exploration.

---

## 🧪 Experiments

All experiments are averaged over **10,000 independent runs** — because one lucky run means nothing. 💪

### Standard Benchmark: T = 500

We threw every algorithm into the arena — Exploration-First, UCB, and all Greedy variants (constant / linear / cosine × multiple $\epsilon$ values) — and let them fight over 500 steps.

| Metric | Winner |
|--------|--------|
| 🏆 Highest Average Reward | **Cosine Annealing Greedy ($\epsilon = 0.2$)** |
| 🏆 Lowest Regret | **Cosine Annealing Greedy ($\epsilon = 0.2$)** |

It swept the floor with every other contender — linear decay, constant $\epsilon$, Exploration-First, and even UCB. 🧹

---

### ⚔️ Head-to-Head: Cosine Annealing vs. UCB

To really understand *why* cosine won, we isolated the two champions and pitted them against each other in two extreme environments:

#### 🏃 Short Sprint: T = 100

> *100 steps. No second chances. Make every step count.*

| | Cosine Annealing | UCB |
|---|---|---|
| Reward | **Higher** ✅ | Lower ❌ |
| Regret | **Lower** ✅ | Higher ❌ |

**Verdict**: Cosine Annealing **dominates**. UCB's $\sqrt{\ln t}$ bonus is far too timid in such a short window — it under-explores and locks onto a suboptimal arm before it even realizes what happened.

#### 🏔️ Marathon: T = 50,000

> *50,000 steps. The long game. Patience is rewarded.*

| | Cosine Annealing | UCB |
|---|---|---|
| Reward | Lower ❌ | **Higher** ✅ |
| Regret | Higher ❌ | **Lower** ✅ |

**Verdict**: UCB **strikes back**. Given enough time, its principled uncertainty-based exploration slowly but surely overtakes any fixed-schedule strategy. The $O(\log T)$ regret bound is not just theory — it's real. 📈

---

## 🎯 Conclusion

<div align="center">

### *There is no universally best algorithm.*
### *There is only the best algorithm **for your scenario**.*

</div>

| Scenario | Recommended Algorithm | Why |
|----------|----------------------|-----|
| 🏢 A/B Testing (days) | **Cosine Annealing Greedy** | Known horizon, limited budget, need fast convergence |
| 📢 Ad Placement (hours) | **Cosine Annealing Greedy** | Every impression counts, can't afford slow exploration |
| 🧊 Cold-Start Recommendation | **Cosine Annealing Greedy** | Users won't wait — you must get it right, fast |
| 📰 Long-term Content Rec | **UCB** | Infinite horizon, low-stakes, let the math do its thing |
| 🔬 Lifelong Learning | **UCB** | Time is cheap, mistakes are cheap, let it converge |

> 💡 **The bottom line**: Most real-world scenarios are **short-horizon, high-stakes** problems. You don't get 50,000 tries — you get 500, maybe 100. In that world, a well-tuned Cosine Annealing schedule beats a theoretically elegant UCB every single time.

---

<div align="center">

*Built with ☕, 🐛, and an unhealthy obsession with regret bounds.*

*Star ⭐ this repo if you found it useful!*

</div>
