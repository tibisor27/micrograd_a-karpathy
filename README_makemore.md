# Makemore — Character-Level Language Model (From Scratch)

> Following **Andrej Karpathy's** tutorial on building a character-level language model from scratch.
> Makemore **"makes more"** of things — given a dataset of names, it generates new, plausible names.

---

## Overview

### What is a Bigram Character-Level Language Model?

A model that predicts the **next character** based on the **current character** (one character of context).
For example, given the name `"emma"`:

```
Input (xs)    →   Target (ys)
    .         →      e          "after . comes e"
    e         →      m          "after e comes m"
    m         →      m          "after m comes m"
    m         →      a          "after m comes a"
    a         →      .          "after a comes ."
```

Where `.` is the special start/end token.

---

## Two Approaches — Same Result

### Approach 1: Counting + Normalization

1. **Count** how many times each bigram (character pair) appears in the dataset → build a 27×27 count matrix `N`
2. **Normalize** each row to get probabilities: `P = (N + 1) / (N + 1).sum(1, keepdim=True)`
   - The `+1` is **Laplace smoothing** — ensures no probability is exactly zero
3. **Sample** from the distribution using `torch.multinomial`

### Approach 2: Gradient-Based Neural Network

1. Encode inputs as **one-hot vectors** (27-dimensional)
2. Multiply by a **weight matrix** `W` (27×27) → get **logits** (raw scores)
3. Apply **softmax** (exp + normalize) → get probabilities
4. Compute **negative log likelihood loss**
5. **Backpropagate** to get gradients
6. **Update weights** via gradient descent
7. Repeat until loss is minimized

> Both approaches converge to the **same result** — the gradient-based method finds the same bigram statistics that counting finds directly!

---

## Key Concepts & PyTorch Operations

### 1. Random Number Generator & Reproducibility

```python
g = torch.Generator().manual_seed(2147483647)
```

- `torch.Generator()` — object that controls **pseudo-random number generation** (PRNG)
- `.manual_seed(2147483647)` — sets the starting point (seed) of the random sequence
- `2147483647` = 2³¹ - 1 (max signed 32-bit int, just a convention)
- **Purpose**: For the same seed, we always get the **same sequence** of numbers → **reproducibility**
- The generator maintains an **internal state** that advances with each use. Resetting the seed resets the state

---

### 2. Sampling from a Distribution — `torch.multinomial`

```python
torch.multinomial(p, num_samples=20, replacement=True, generator=g)
```

- **Returns INDICES** (not values!) drawn according to the probability distribution `p`
- `num_samples=20` — draw 20 samples
- `replacement=True` — same index **can appear multiple times** (element is "put back")
- `replacement=False` — each index appears **at most once** (and `num_samples ≤ len(p)`)
- `generator=g` — uses our seeded generator for reproducible results

---

### 3. Creating a Probability Distribution — Normalization

```python
p = p / p.sum()
```

- **Operation**: Normalization — divides each element by the total sum
- **Result**: A **probability distribution** (specifically a **categorical distribution**)
- **Two conditions** for a valid distribution: all values ≥ 0 AND sum = 1
- `p` values do NOT need to sum to 1 before feeding to `multinomial` (it auto-normalizes), but we do it explicitly for clarity

---

### 4. `keepdim=True` — Preserving Dimensions for Broadcasting

```python
P.sum(0, keepdim=True)   # shape (1, 27) — dimension preserved as 1
P.sum(0)                 # shape (27,)   — dimension REMOVED
```

# CORRECT — shapes align for broadcasting
row_sums = P.sum(1, keepdim=True)   # (N, 1)
P_norm = P / row_sums               # (N, 27) / (N, 1) → broadcasts correctly!

# WRONG — shapes mismatch
row_sums = P.sum(1)                 # (N,)
P_norm = P / row_sums               # ambiguous broadcasting!
```

---

### 5. One-Hot Encoding

```python
import torch.nn.functional as F
xenc = F.one_hot(xs, num_classes=27).float()
```

- **Why**: Characters are **categories**, not numbers. Index 13 is NOT "greater than" index 5
- **What**: Transforms each index into a **binary vector** with exactly one `1` at the character's position
- `num_classes=27` — vector length = vocabulary size (26 letters + `.` separator)
- `.float()` — converts from `int64` to `float32` (required for neural network operations)
- **Shape**: input `(N,)` → output `(N, 27)` — each row is one character's representation
- **Key insight**: One-hot × Weight matrix = **selecting a row from W** (acts as a differentiable lookup table!)

---

### 6. The Neural Network — Weight Matrix & Forward Pass

```python
W = torch.randn((27, 27), generator=g, requires_grad=True)
logits = xenc @ W  # matrix multiplication → raw scores (logits)
```

**Architecture (single layer, no hidden layers):**

```
INPUT (27 neurons)        W (27×27)         OUTPUT (27 neurons)
  one-hot vector    ×    weights     =     27 logits (one per character)
```

- **`W.shape = (27, 27)`**: 27 input features → 27 output scores (one per possible next character)
- Each **row** of W represents: "scores for what comes after this character"
- Each **column** of W represents: "how likely is this character as a next character"
- Since input is one-hot, `xenc @ W` simply **selects the corresponding row** from W

**From logits to probabilities (Softmax):**

```python
counts = logits.exp()                          # exponentiate → all positive
probs = counts / counts.sum(1, keepdim=True)   # normalize → sum to 1
# Equivalent to: probs = torch.softmax(logits, dim=1)
```

| Stage | Values | Range |
|---|---|---|
| **Logits** (`xenc @ W`) | Raw scores | `(-∞, +∞)` |
| **Counts** (`logits.exp()`) | Exponentiated | `(0, +∞)` |
| **Probs** (`counts / sum`) | Probabilities | `[0, 1]`, sum = 1 |

---

### 7. Fancy Indexing — Extracting Target Probabilities

```python
probs[torch.arange(5), ys]
```

- Uses **advanced (fancy) indexing** to create `(row, column)` pairs element-wise:
  - `(0, ys[0])`, `(1, ys[1])`, `(2, ys[2])`, ...
- Extracts the **probability assigned to the CORRECT next character** for each example
- If model is good → these probabilities are **high** (close to 1)
- If model is bad → these probabilities are **low** (close to 0)
- This is the basis for computing the **loss function**

---

### 8. Loss, Backpropagation & Weight Updates

```python
# FORWARD PASS — compute prediction and loss
logits = xenc @ W
counts = logits.exp()
probs = counts / counts.sum(1, keepdim=True)
loss = -probs[torch.arange(num), ys].log().mean()   # negative log likelihood

# BACKWARD PASS — compute gradients (does NOT update weights!)
W.grad = None       # reset old gradients (they accumulate by default!)
loss.backward()     # computes ∂loss/∂W → stored in W.grad

# UPDATE — adjust weights (gradient descent)
W.data += -learning_rate * W.grad   # step in the direction that reduces loss
```

**Critical points:**
- `requires_grad=True` on W tells PyTorch to **track operations** and build a computational graph
- `loss.backward()` **ONLY calculates** gradients — it does **NOT update** weights
- `W.grad` must be **reset to None** before each `backward()` call (gradients accumulate by default!)

---

### 9. Smooth Distributions & Regularization

**Why we need smooth distributions:**

1. **Avoid `log(0) = -∞`** — if the model assigns probability 0 to a character that actually appears, loss becomes infinite
2. **Prevent overfitting** — peaky distributions mean the model memorized training data and fails on new data

**How to achieve smoothing:**

| Method | Where used | How |
|---|---|---|
| **Laplace smoothing** | Counting approach | Add +1 to all counts before normalizing |
| **Regularization** | Neural network | Add penalty term: `+ λ * (W²).mean()` to loss |

**Regularization mechanics:**
- Large weights → large logits → softmax produces **peaky** distributions
- Small weights → small logits → softmax produces **smooth/uniform** distributions
- Regularization **pulls weights toward zero** → encourages smoother distributions
- **W = all zeros** → all logits = 0 → exp(0) = 1 → probs = 1/27 ≈ 0.037 for all characters → **perfectly uniform** (maximum smooth)

**The ideal balance:** Learned enough to predict well, but smooth enough to handle unseen data.

---

## Training Quality: Negative Log Likelihood Loss

```python
loss = -probs[torch.arange(num), ys].log().mean()
```

- **Lower loss = better model**
- Perfect model (always predicts correct character with probability 1): `loss = -log(1) = 0`
- Random model (uniform distribution, 1/27 chance): `loss = -log(1/27) ≈ 3.29`
- Terrible model (assigns probability 0 to correct character): `loss = -log(0) = ∞`

> If your loss is significantly above 3.29, your model is **worse than random guessing**!
