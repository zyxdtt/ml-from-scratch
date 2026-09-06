# PLSA (Probabilistic Latent Semantic Analysis)

This repository contains a C++ implementation of the Probabilistic Latent Semantic Analysis (PLSA) algorithm, as described in Chapter 21 of Li Hang's *"Statistical Learning Methods"* (《统计学习方法》). PLSA is an unsupervised learning method for discovering latent topics in document-word co-occurrence data.

## Reference

This implementation follows the Expectation-Maximization (EM) algorithm for PLSA as introduced in:

> Li Hang, *Statistical Learning Methods*, Chapter 21: Probabilistic Latent Semantic Analysis.

The model assumes a generative process where each document is a mixture of latent topics, and each topic is a distribution over words.

---

## Algorithm Overview

### Model Definition

Given:
- `ci` = number of words (vocabulary size)
- `ben` = number of documents
- `ti` = number of latent topics

PLSA models the joint probability of a word `w` and a document `d` as:

$$
P(w, d) = \sum_{z=1}^{K} P(z) \, P(w \mid z) \, P(d \mid z)
$$

Equivalently, the conditional probability of a word given a document is:

$$
P(w \mid d) = \sum_{z=1}^{K} P(w \mid z) \, P(z \mid d)
$$

where:
- $P(w \mid z)$: word distribution for topic `z` (citi in code)
- $P(z \mid d)$: topic distribution for document `d`

### Parameters

| Variable | Description | Code Name |
|----------|-------------|-----------|
| $P(w \mid z)$ | probability of word `w` given topic `z` | `citi[z][w]` |
| $P(d \mid z)$ | probability of document `d` given topic `z` | `tiben[z][d]` |
| $P(z \mid d, w)$ | posterior topic probability | `prob[z][d][w]` |

---

## Expectation-Maximization (EM) Algorithm

### E-Step

Compute the posterior probability of topic `z` given document `d` and word `w`:

$$
P(z \mid d, w) = \frac{P(w \mid z) \, P(d \mid z)}{\sum_{z'=1}^{K} P(w \mid z') \, P(d \mid z')}
$$

In code:
```cpp
for (int l = 0; l < ti; l++) 
    temp[l] = citi[j][l] * tiben[l][k];
double sum = accumulate(temp.begin(), temp.end(), 0.0);
if (sum < 1e-9) sum = 1e-9;
for (int l = 0; l < ti; l++) 
    prob[l][j][k] = temp[l] / sum;
```

### M-Step

Update parameters:

**Update $P(w \mid z)$:**

$$
P(w \mid z) = \frac{\sum_{d} n(d, w) \, P(z \mid d, w)}{\sum_{w'} \sum_{d} n(d, w') \, P(z \mid d, w')}
$$

**Update $P(d \mid z)$:**

$$
P(d \mid z) = \frac{\sum_{w} n(d, w) \, P(z \mid d, w)}{\sum_{d'} \sum_{w} n(d', w) \, P(z \mid d', w)}
$$

where $n(d, w)$ is the word frequency in document `d`.

---

## Implementation Details

- **Language**: C++17
- **Dependencies**: Standard Template Library (STL) only
- **Random Initialization**: Parameters are initialized uniformly at random using `mt19937`
- **Numerical Stability**: Small epsilon ($10^{-9}$) is added to denominators to avoid division by zero

### Class Interface

```cpp
class PLSA {
public:
    PLSA(int ci, int ben, int ti);
    pair<mat, mat> train(const mat& x, int iter = 100);
};
```

- `ci`: number of words
- `ben`: number of documents
- `ti`: number of topics
- `x`: `ci × ben` word-document frequency matrix
- Returns: `(citi, tiben)` where:
  - `citi`: `ci × ti` — $P(z \mid w)$
  - `tiben`: `ti × ben` — $P(d \mid z)$

---

## Experiment

### Dataset

A synthetic document-word frequency matrix with:
- `ci = 5` words
- `ben = 4` documents
- `ti = 2` latent topics

| Word \ Doc | Doc0 | Doc1 | Doc2 | Doc3 |
|------------|------|------|------|------|
| word0      | 8    | 7    | 1    | 1    |
| word1      | 6    | 5    | 1    | 1    |
| word2      | 1    | 2    | 8    | 9    |
| word3      | 1    | 1    | 7    | 6    |
| word4      | 0    | 0    | 0    | 0    |

**Design**: Documents 0–1 are biased toward words 0–1 (sports), while documents 2–3 are biased toward words 2–3 (finance). Word4 is a stopword with zero frequency.

### Training

```cpp
PLSA model(ci, ben, ti);
auto [citi, tiben] = model.train(x, 200);
```

---

## Results

### $P(z \mid w)$ — Topic Distribution per Word

| Word   | Topic 0 | Topic 1 |
|--------|---------|---------|
| word0  | 0.509   | 0.044   |
| word1  | 0.372   | 0.048   |
| word2  | 0.069   | 0.518   |
| word3  | 0.050   | 0.390   |
| word4  | 0.000   | 0.000   |

**Interpretation**: Words 0 and 1 have high probability under Topic 0 (sports); words 2 and 3 have high probability under Topic 1 (finance). Word4 is uninformative.

### $P(d \mid z)$ — Document Distribution per Topic

| Topic   | Doc0  | Doc1  | Doc2  | Doc3  |
|---------|-------|-------|-------|-------|
| Topic 0 | 0.992 | 0.898 | 0.033 | 0.033 |
| Topic 1 | 0.008 | 0.102 | 0.967 | 0.967 |

**Interpretation**: 
- Topic 0 (sports) dominates Documents 0 and 1.
- Topic 1 (finance) dominates Documents 2 and 3.

The model successfully discovers the latent thematic structure with clear separation between topics.

---

## How to Compile and Run

```bash
g++ -std=c++17 -O2 main.cpp -o plsa
./plsa
```

## License

MIT
