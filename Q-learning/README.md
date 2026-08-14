# Q-Learning — A Minimal Implementation

A from-scratch, single-file implementation of tabular Q-learning.
Written to **understand the core idea**, not for production use.

> ~150 lines of C++. No dependencies beyond the C++ standard library.

---

## Reference

This implementation follows **Chapter 37 (Reinforcement Learning)** of
**李航 (Li Hang), 《统计学习方法》 / 《机器学习方法》 (*Statistical Learning Method* / *Machine Learning Methods*)**.

The chapter introduces the Markov Decision Process (MDP) framework, the
Bellman optimality equation, and the Q-learning algorithm. This repo
implements that algorithm directly in C++, on a small grid-world
environment so the behavior of every line of code can be traced.

---

## The Algorithm

### Setup

| Symbol    | Meaning                                       |
|-----------|-----------------------------------------------|
| `S`       | State space (finite set of environment states)|
| `A`       | Action space (finite set of actions)          |
| `r(s,a,s')`| Reward received for transition               |
| `γ`       | Discount factor, 0 ≤ γ ≤ 1                    |
| `α`       | Learning rate, 0 < α ≤ 1                      |
| `ε`       | Exploration probability, 0 ≤ ε ≤ 1           |

### Bellman Optimality

The optimal action-value function `Q*` satisfies:

```
Q*(s, a) = E[ r(s, a, s') + γ · max_a' Q*(s', a') ]
```

Q-learning iteratively approximates `Q*` by sampling transitions and
applying the update below.

### Q-Value Update

After observing transition `(s, a, s', r)`:

```
Q(s, a) ← Q(s, a) + α · [ r + γ · max_a' Q(s', a') − Q(s, a) ]
```

The bracketed term is the **TD error**: the difference between the
predicted value and the bootstrapped target.

### ε-Greedy Action Selection

```
with probability ε:      pick a random action  (exploration)
with probability 1 − ε:  pick argmax_a Q(s, a) (exploitation)
```

### Training Loop

```
Initialize Q(s, a) = 0 for all s, a
for each episode:
    observe initial state s
    while episode not terminated:
        select action a via ε-greedy
        take a, observe (r, s')
        Q(s, a) ← Q(s, a) + α · [ r + γ · max Q(s', ·) − Q(s, a) ]
        s ← s'
    end while
end for
```

The Q-table is the **accumulated knowledge** across all episodes; it
does **not** reset between episodes.

---

## Implementation

### Environment

A 5×5 grid:

```
S****
*****
*****
*****
****#      ← # is the goal
```

| Item             | Value                           |
|------------------|---------------------------------|
| Start            | `(1, 1)`                        |
| Goal             | `(4, 4)`                        |
| Actions          | 4: right, down, up, left         |
| State space size | 25 cells                         |
| Q-table size     | 25 × 4 = 100 entries             |

### Rewards

| Event                | Reward | Episode terminates? |
|----------------------|-------:|:--------------------:|
| Reach goal `(4, 4)`  | `+100` | yes                  |
| Step out of bounds   | `−100` | yes                  |
| Any other step       |  `−2`  | no                   |

### Hyperparameters

| Symbol    | Code   | Value |
|-----------|--------|------:|
| α (lr)    | `lr`   | 0.1   |
| γ (gamma) | `gamma` | 0.9  |
| ε (eps)   | `eps`  | 0.1   |
| T         | max steps per episode | 15 |
| K         | episodes (user input) | — |

---

## How to Build and Run

Requires MSVC `cl.exe` (Visual Studio). Single source file, no
dependencies.

```bat
cl /EHsc /std:c++17 /O2 Q.cpp /Fe:Q.exe
echo 1000 | Q.exe
```

Each line of stdout prints `epoch N: reward: R` so you can watch the
learning curve episode by episode.

### Expected Learning Curve

After ~1000 episodes:

| Window              | Mean reward | Comment                                |
|---------------------|------------:|----------------------------------------|
| First 100 episodes  | ~ +44       | mostly random exploration             |
| Last  100 episodes  | ~ +80       | near-optimal policy has formed        |
| Best single episode |    +90      | shortest path (5 steps × −2 = −10) + 100 |

If your numbers look like these, the algorithm is working.

---

## Limitations

This is a **reference implementation** for understanding the core
Bellman-update / ε-greedy / MDP loop. It deliberately omits the
machinery that makes Q-learning work on real problems. Limitations:

1. **Tabular only.** The Q-table is `O(|S| × |A|)`. It cannot scale to
   continuous or high-dimensional state spaces (Atari pixels, robotic
   joint angles, etc.). For those, use function approximation —
   DQN, DDPG, PPO — all built on the same Bellman target.

2. **No ε decay.** Exploration is held at ε = 0.1 throughout training.
   A real schedule decays ε from ≈ 1.0 down to ≈ 0.05 so the agent
   explores widely at first and exploits reliably later.

3. **No replay buffer.** Each transition is consumed once. Deep RL
   reuses past transitions by sampling from a buffer to break
   autocorrelation between consecutive updates.

4. **No batching / GPU.** Updates are scalar and serial. Production
   training batches many transitions per update step on a GPU.

5. **No target network.** Tabular Q-learning is stable because the
   target `r + γ · max Q(s', ·)` is frozen between updates. With neural
   function approximation, the target must move slowly, hence a
   periodically-updated target network (DQN).

6. **No generalization.** The Q-table treats every state as independent.
   Two adjacent grid cells have no relationship in the table, even
   though they should. Neural function approximation learns a
   representation that generalizes across similar states.

7. **Single-threaded.** One environment at a time. Production training
   uses many parallel environment copies (vectorized environments).

In short: **this code teaches you the core update rule**. Everything
listed above is what the deep RL literature builds on top of that
one line `Q(s, a) ← Q(s, a) + α · [r + γ · max Q(s', ·) − Q(s, a)]`.

---

## What I Did Differently from the Book

The book assumes a 0-indexed `(0..N-1)²` grid. This implementation
uses a 1-indexed `(1..N)²` grid with bounds checks rejecting
`(0, ·)` and `(N+1, ·)`. The two are mathematically equivalent; the
1-indexed variant makes the bounds check easier to read (`< 1 || > 5`
versus `< 0 || > N-1`).

The Q-table is laid out row-stride = `N+2` (here `7`) rather than the
dense `N` layout. This wastes 24 indices but makes the print/debug
mapping `pa = r·7 + c` easy to read off by eye.

---

## File Layout

```
Statistic_Learning/
├── Q.cpp          ← the entire implementation (~150 lines)
└── README.md      ← this file
```

---

## References

- 李航 (Li Hang), 《统计学习方法》 / 《机器学习方法》 (*Statistical
  Learning Method / Machine Learning Methods*), **Chapter 37**:
  强化学习 (Reinforcement Learning).

- Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd edition.
  Chapters 3 (MDPs), 6 (TD learning), and the canonical Q-learning
  treatment. Freely available at incompleteideas.net/book/the-book.html.

- Mnih et al., *Playing Atari with Deep Reinforcement Learning* (2013) —
  the paper that scaled Q-learning to neural networks and Atari.

---

## License

MIT — read, modify, and learn freely.