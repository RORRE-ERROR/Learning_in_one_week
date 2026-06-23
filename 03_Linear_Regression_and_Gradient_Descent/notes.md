# Chapter 3 — Linear Regression & Gradient Descent

> **Day 2 of 7** · Prerequisites: Ch 1–2 (what ML is, supervised vs. unsupervised, basic data prep) · Est. study time: ~75–90 min

**In one sentence:** Linear regression draws the best straight line through your data, and gradient descent is the step-by-step procedure that finds that line by repeatedly nudging the line's parameters downhill on an error surface.

## Why this matters

Linear regression is the "hello world" of supervised learning, and almost every idea you meet later — cost functions, gradient descent, learning rates, overfitting, regularization — shows up here first in its simplest, most visible form. The exam leans on this chapter heavily: expect MCQs on the learning rate, the LMS rule, batch vs. stochastic GD, the normal equation, feature scaling, and regularization, **plus** numeric problems where you compute a cost or normalize a feature by hand.

## Learning goals

- [ ] Write the hypothesis $h_\theta(x)$ and say what each $\theta$ means
- [ ] Compute the MSE cost $J(\theta)$ for given parameters
- [ ] State the gradient descent update rule and explain the role of the learning rate $\alpha$
- [ ] Explain the LMS rule (update size is proportional to the error)
- [ ] Distinguish batch vs. stochastic gradient descent
- [ ] Use the normal equation and know when to prefer GD instead
- [ ] Apply feature scaling / mean normalization and say why it helps
- [ ] Explain regularization, residual analysis, and MSE vs. MAE
- [ ] Tell apart overfitting / underfitting and local vs. global minima

---

## 1. The hypothesis: the line you're fitting

**Intuition.** You have data points (e.g. study hours → exam score) and you believe a straight line roughly captures the trend. The hypothesis is just the equation of that line — your model's guess for $y$ given $x$.

**Formula (one feature):**

$$h_\theta(x)=\theta_0+\theta_1 x$$

- $x$ = the input feature (e.g. hours studied)
- $h_\theta(x)$ = the predicted output (e.g. predicted score)
- $\theta_0$ = the **intercept** (where the line crosses the y-axis; the prediction when $x=0$)
- $\theta_1$ = the **slope** / coefficient (how much $y$ changes when $x$ goes up by 1)

**Many features (multivariate / "polynomial" in the tutorial):**

$$h_\theta(x)=\theta_0+\theta_1 x_1+\theta_2 x_2+\dots+\theta_n x_n$$

Each $\theta_j$ is the weight on feature $x_j$.

**Notation you must lock in:**
- $m$ = number of training examples (rows of data)
- $n$ = number of features (not counting the intercept)
- $x^{(i)}$, $y^{(i)}$ = the $i$-th training example's input and true output
- $x_j^{(i)}$ = the $j$-th feature of the $i$-th example

**Worked mini-example (tutorial Q3).** If $\theta_0=-1$, $\theta_1=0.5$, then

$$h_\theta(3)=-1+0.5\times 3=-1+1.5=0.5.$$

### What the coefficient *means* (midterm Q35)

The coefficient $\theta_1$ carries **two** pieces of information at once:
1. **Direction** of the relationship — a positive $\theta_1$ means $y$ rises as $x$ rises; negative means it falls.
2. **Magnitude** — it is the change in the dependent variable $y$ for a **one-unit** change in the independent variable $x$.

> 💡 **Exam tip:** The "role of the coefficient" answer is the option that mentions **both** direction **and** the change-in-$y$-per-one-unit-change-in-$x$ — not just one of them. (Strength of the relationship is correlation, a different idea.)

---

## 2. The cost function: how wrong is the line?

**Intuition.** Pick any line and measure how far off its predictions are from the true values. We want a single number that is small for a good line and large for a bad one. That number is the cost.

**Formula (Mean Squared Error, MSE):**

$$J(\theta_0,\theta_1)=\frac{1}{2m}\sum_{i=1}^{m}\bigl(h_\theta(x^{(i)})-y^{(i)}\bigr)^2$$

The term $\bigl(h_\theta(x^{(i)})-y^{(i)}\bigr)$ is the **error** (residual) on example $i$: prediction minus truth.

**Why squared?** Two reasons: (1) squaring makes every error positive so they don't cancel out, and (2) it penalizes **large** errors much more heavily than small ones (an error of 4 costs 16, an error of 2 costs only 4). The line is therefore strongly discouraged from being badly wrong on any single point.

**Why the $\frac{1}{2m}$?** The $\frac{1}{m}$ averages over all examples so cost doesn't just grow with dataset size. The extra $\frac{1}{2}$ is a convenience: when you differentiate the square, the exponent 2 comes down and cancels the $\frac{1}{2}$, giving a cleaner gradient. It does **not** change where the minimum is.

### Worked example (tutorial Q2) — reproduce every step

Training set:

| $x$ | $y$ |
|---|---|
| 3 | 2 |
| 2 | 3 |
| 4 | 5 |
| 1 | 1 |
| 5 | 4 |

Here $m=5$. Use $\theta_0=0,\ \theta_1=1$, so $h_\theta(x)=x$. Compute $J(0,1)$.

**Step 1 — predictions and errors** (error $=h_\theta(x)-y$):

| $x$ | $h_\theta(x)=x$ | $y$ | error | error² |
|---|---|---|---|---|
| 3 | 3 | 2 | $3-2=1$ | 1 |
| 2 | 2 | 3 | $2-3=-1$ | 1 |
| 4 | 4 | 5 | $4-5=-1$ | 1 |
| 1 | 1 | 1 | $1-1=0$ | 0 |
| 5 | 5 | 4 | $5-4=1$ | 1 |

**Step 2 — sum of squared errors:** $1+1+1+0+1=4$.

**Step 3 — divide by $2m$:** $2m=2\times 5=10$, so

$$J(0,1)=\frac{4}{10}=\boxed{0.4}.$$

> 💡 **Exam tip:** When given $\theta_0=0$ and $\theta_1=1$, the hypothesis collapses to $h_\theta(x)=x$. Don't overthink it — just plug $x$ in directly.

> ⚠️ **Common trap:** Forgetting the **2** in the denominator. The sum of squared errors here is 4; the cost is $4/(2\cdot5)=0.4$, **not** $4/5=0.8$.

---

## 3. Gradient descent: rolling downhill to the best line

**Intuition.** Think of $J(\theta)$ as a valley (a bowl shape for linear regression). You're standing somewhere on the slope blindfolded. To reach the bottom, you feel which way is downhill (the gradient) and take a small step that way. Repeat until you stop moving. That's gradient descent.

**The update rule** (do this for every parameter $\theta_j$, simultaneously):

$$\theta_j := \theta_j - \alpha\,\frac{\partial}{\partial\theta_j}J(\theta)$$

- $\alpha$ = the **learning rate** (step size)
- $\frac{\partial}{\partial\theta_j}J(\theta)$ = the slope of the cost in the direction of $\theta_j$ (which way is uphill); the minus sign turns it into a downhill step.

**Expanded for linear regression** (after differentiating the MSE — the $\frac12$ cancels the 2):

$$\theta_0 := \theta_0 - \alpha\,\frac{1}{m}\sum_{i=1}^{m}\bigl(h_\theta(x^{(i)})-y^{(i)}\bigr)$$

$$\theta_j := \theta_j - \alpha\,\frac{1}{m}\sum_{i=1}^{m}\bigl(h_\theta(x^{(i)})-y^{(i)}\bigr)\,x_j^{(i)}\qquad(j\ge 1)$$

> ⚠️ **Common trap:** Update **all** $\theta_j$ using the **old** values, then assign. If you update $\theta_0$ first and use the new $\theta_0$ inside the $\theta_1$ formula, you've done it wrong.

### The learning rate $\alpha$ (midterm Q34, tutorial Q7)

$\alpha$ controls step size, and it's a Goldilocks parameter:

- **Too large** → you overshoot the bottom of the valley and bounce to the other side, possibly higher than before. The cost **oscillates or diverges** (blows up). If you see $J(\theta)$ **increasing** during training, the learning rate is almost certainly too large.
- **Too small** → every step is tiny, so convergence is correct but painfully **slow**.
- **Just right** → $J(\theta)$ decreases smoothly every iteration.

> 💡 **Exam tip:** "$J(\theta)$ increases during training" ⇒ learning rate too large (tutorial Q7). "Learning rate too high" ⇒ divergence/oscillation (midterm Q34). These are the same idea phrased two ways.

### The LMS rule (midterm Q4)

Look at the update term: it contains the factor $\bigl(h_\theta(x^{(i)})-y^{(i)}\bigr)$, the **error**. This is the **Least Mean Squares (LMS)** rule, and the key insight is:

> The **magnitude of the update is proportional to the error**.

- Big error on an example → big correction to the parameters.
- Prediction already accurate (error ≈ 0) → the update term ≈ 0, so the parameters barely move.

It's a self-correcting, intuitive behaviour: the model adjusts hardest exactly where it's most wrong.

> ⚠️ **Common trap:** The relationship is **proportional**, not inversely proportional. An answer claiming "inversely proportional to the error" (or "updates big when the prediction is already correct") is wrong.

### Local vs. global minima (tutorial Q5)

- **Global minimum** = the lowest point of the **entire** cost surface (the true best parameters).
- **Local minimum** = the lowest point within some **neighbourhood**, but not necessarily the lowest overall — a dip you can get stuck in.

Good news for linear regression: its MSE cost is a convex bowl with a **single** minimum, so gradient descent can't get trapped. (Local minima become a real worry later, with neural networks.)

---

## 4. Batch vs. stochastic (incremental) gradient descent (midterm Q20)

The two differ in **how much data** they use before each parameter update:

| | Batch GD | Stochastic / Incremental GD |
|---|---|---|
| Data per update | The **entire** training set | A **single** training example |
| Update rule | Sum over all $m$ examples, then step | Step after each example $i$ |
| Per-step cost | Expensive (touches all data) | Cheap |
| Path to minimum | Smooth, direct | Noisy, zig-zaggy but fast to get moving |

- **Batch**: parameters are updated **once after processing the whole training set** each iteration.
- **Stochastic (incremental)**: parameters are updated **after each single training example**, using the gradient of the error for that one example.

(There's also **mini-batch** GD in between — updates on small batches — which is the most common choice in practice. Mentioned for completeness.)

> 💡 **Exam tip:** The distinguishing answer is purely "entire training set per iteration" (batch) vs. "single example per update" (stochastic). Ignore distractor options about local minima or one being universally cheaper.

---

## 5. The normal equation: solving it in one shot

**Intuition.** Instead of iterating downhill, you can solve for the best $\theta$ directly with a formula from linear algebra — no steps, no learning rate.

**Formula:**

$$\theta=(X^TX)^{-1}X^Ty$$

- $X$ = the **design matrix**: one row per training example, one column per feature, **plus a leading column of all 1's** for the intercept $\theta_0$.
- $y$ = column vector of true outputs.
- $\theta$ = column vector of parameters you solve for.

**Pros vs. gradient descent:**
- No learning rate $\alpha$ to tune.
- No iterations; no risk of getting the step size wrong.
- No local-minima concerns (it's a direct closed-form solution — tutorial Q10c: the normal equation does **not** suffer from local minima).

**Con:** computing the inverse $(X^TX)^{-1}$ costs roughly $O(n^3)$ where $n$ is the number of features. When $n$ is huge, this is far too slow.

### Worked dimensions example (tutorial Q8)

$m=23$ examples, $n=5$ features. Add the all-ones intercept column, so each example now has $5+1=6$ entries.

- $X$ is $m\times(n+1)=23\times 6$
- $y$ is $m\times 1=23\times 1$
- $\theta$ is $(n+1)\times 1=6\times 1$

Quick sanity check via dimensions: $X^T$ is $6\times23$, so $X^TX$ is $6\times6$ (invertible square matrix ✓), $X^Ty$ is $6\times1$, and $(6\times6)(6\times1)=6\times1=\theta$. ✓

### When to prefer GD over the normal equation (tutorial Q9)

For $m=1{,}000{,}000$, $n=200{,}000$: choose **gradient descent**. Inverting a $200{,}001\times200{,}001$ matrix at $O(n^3)$ is computationally hopeless, whereas GD scales fine to large $n$.

> 💡 **Exam tip:** Rough rule of thumb — small number of features ⇒ normal equation is convenient; **huge** number of features ⇒ gradient descent (the $O(n^3)$ inversion kills the normal equation).

---

## 6. Feature scaling & mean normalization (midterm Q39)

**Intuition.** If one feature ranges 0–2000 (sqft) and another ranges 1–5 (bedrooms), the cost surface becomes a long, skewed ravine. Gradient descent then zig-zags slowly. Rescaling features to a comparable range turns the ravine into a round bowl, so GD heads straight for the minimum — **it speeds up convergence** and lets all features contribute comparably.

**Mean normalization** rescales each feature to be roughly centred at 0:

$$x_j^{(i)} \leftarrow \frac{x_j^{(i)}-\mu_j}{\text{range}_j}$$

where $\mu_j$ is the feature's mean and $\text{range}_j = \max_j - \min_j$ (the tutorial uses the range; standard deviation is also common).

### Worked example (tutorial Q6) — reproduce the arithmetic

Houses, with feature $x_2$ = square footage:

| No | Bedrooms ($x_1$) | Sqft ($x_2$) | Price ($y$) |
|---|---|---|---|
| 1 | 1 | 880 | 490000 |
| 2 | 3 | 1930 | 630000 |
| 3 | 4 | 1940 | 640000 |
| 4 | 3 | 1350 | 570000 |

Compute the normalized 4th sqft value, $x_2^{(4)}$.

**Step 1 — mean of sqft:**
$$\mu_2=\frac{880+1930+1940+1350}{4}=\frac{6100}{4}=1525.$$

**Step 2 — range of sqft:**
$$\text{range}_2=\max-\min=1940-880=1060.$$

**Step 3 — normalize the 4th value** ($x_2^{(4)}=1350$):
$$x_2^{(4)}=\frac{1350-1525}{1060}=\frac{-175}{1060}\approx \boxed{-0.16}.$$

The negative sign tells you 1350 sqft is **below** the average house size in this set.

> ⚠️ **Common trap:** Use the **range** (max − min = 1060) as the denominator here, not the standard deviation. And subtract the **mean** in the numerator — forgetting to subtract gives a positive number near 1.27, which is wrong.

---

## 7. Regularization (midterm Q31)

**Intuition.** An overfit model often has wildly large coefficients that let the line wiggle through every training point. Regularization adds a penalty for large coefficients to the cost, so the optimizer is pushed toward simpler, smaller-weight solutions that generalize better.

**Regularized cost (L2 / ridge):**

$$J(\theta)=\frac{1}{2m}\sum_{i=1}^{m}\bigl(h_\theta(x^{(i)})-y^{(i)}\bigr)^2+\frac{\lambda}{2m}\sum_{j=1}^{n}\theta_j^2$$

- $\lambda$ = regularization strength. Bigger $\lambda$ ⇒ heavier penalty ⇒ smaller weights ⇒ simpler model. (Too big and you underfit.)
- The penalty sum starts at $j=1$: we conventionally don't penalize the intercept $\theta_0$.

**Two jobs regularization does:**
1. **Reduces overfitting** by penalizing large coefficients (midterm Q31).
2. **Makes the matrix invertible** in the normal equation (tutorial Q10a): the regularized form is $\theta=(X^TX+\lambda I)^{-1}X^Ty$, and adding $\lambda I$ guarantees the matrix is non-singular even when $X^TX$ alone isn't.

> 💡 **Exam tip:** "Purpose of regularization" = reduce overfitting by penalizing large coefficients. It does **not** normalize feature scales (that's feature scaling) and does **not** increase model complexity — it does the opposite.

---

## 8. Residual analysis (midterm Q7)

A **residual** is the leftover error for an example: $y^{(i)} - h_\theta(x^{(i)})$. **Residual analysis** plots and inspects these residuals to **check the assumptions of regression** — e.g. that residuals are randomly scattered around zero with roughly constant spread and no leftover pattern. A clear pattern (a curve, a funnel shape) signals a violated assumption (non-linearity, non-constant variance).

> 💡 **Exam tip:** Purpose of residual analysis = **check the assumptions of regression**. It is not dimensionality reduction and not a primary "performance metric" — those are tempting distractors.

---

## 9. Error metrics: MSE vs. MAE (tutorial Q10)

| Metric | Formula | Behaviour |
|---|---|---|
| **MSE** (Mean Squared Error) | $\frac{1}{m}\sum (h_\theta(x^{(i)})-y^{(i)})^2$ | Squares errors ⇒ **penalizes large errors more heavily**; very sensitive to outliers |
| **MAE** (Mean Absolute Error) | $\frac{1}{m}\sum \lvert h_\theta(x^{(i)})-y^{(i)}\rvert$ | Absolute value ⇒ **grows linearly** with the error; more robust to outliers |

So if a few big mistakes should be punished hard, MSE; if you don't want outliers to dominate, MAE.

> 💡 **Exam tip:** MSE penalizes large errors **more** than MAE (because of the square); MAE is **linear** in the error. Both are true statements the tutorial flags as correct.

---

## 10. Overfitting, underfitting, best fit (tutorial Q4)

Three models on the same data:

- **Underfit** (e.g. a too-simple straight line through curved data): high bias. **High robustness** (stable, won't swing with new data) but **poor fit** — it misses the real pattern.
- **Best fit** (a smooth curve that captures the trend): the sweet spot — good fit and good generalization.
- **Overfit** (a wiggly curve hitting every point): **high fit** on training data but **poor robustness** — it memorized the noise and fails on new data.

> 💡 **Exam tip:** Map the picture: leftmost/straight = underfit (high robustness, poor fit); rightmost/wiggly = overfit (high fit, poor robustness). Regularization and more data are the standard cures for overfitting.

---

## Key formulas / facts at a glance

| Thing | Formula / Rule | Plain meaning |
|---|---|---|
| Hypothesis (1 feature) | $h_\theta(x)=\theta_0+\theta_1 x$ | Equation of the fitted line |
| Hypothesis (n features) | $h_\theta(x)=\theta_0+\theta_1x_1+\dots+\theta_nx_n$ | Weighted sum of features |
| Coefficient meaning | direction + Δy per one-unit Δx | What $\theta_1$ tells you |
| Cost (MSE) | $J=\frac{1}{2m}\sum(h_\theta(x^{(i)})-y^{(i)})^2$ | Average squared error; 0.4 in tutorial Q2 |
| GD update | $\theta_j:=\theta_j-\alpha\frac{\partial}{\partial\theta_j}J$ | Step downhill on the cost |
| GD for linreg ($j\ge1$) | $\theta_j:=\theta_j-\alpha\frac1m\sum(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}$ | Expanded update |
| LMS rule | update magnitude ∝ error | Big error → big update |
| Learning rate | too large→diverge/oscillate; too small→slow | $\alpha$ is step size |
| Batch vs. stochastic | whole set / single example per update | How much data per step |
| Normal equation | $\theta=(X^TX)^{-1}X^Ty$ | Closed form; $O(n^3)$ inversion |
| Reg. normal equation | $\theta=(X^TX+\lambda I)^{-1}X^Ty$ | Always invertible |
| Mean normalization | $\frac{x-\mu}{\max-\min}$ | Centre & rescale; −0.16 in tutorial Q6 |
| Regularization (L2) | $+\frac{\lambda}{2m}\sum_{j\ge1}\theta_j^2$ | Penalize big weights → less overfit |
| MSE vs MAE | square vs. absolute | MSE punishes big errors more; MAE linear |

## Exam tips — quick recap

- Coefficient role = **both** direction **and** Δy per one-unit Δx (midterm Q35).
- $J(\theta)$ rising during training ⇒ **learning rate too large** (tutorial Q7 / midterm Q34).
- LMS: update magnitude **proportional** to error — never *inversely* (midterm Q4).
- Batch = whole set per step; stochastic = single example per step (midterm Q20).
- Huge $n$ ⇒ use gradient descent, not the normal equation (tutorial Q9).
- With intercept column: $X$ is $m\times(n{+}1)$, $\theta$ is $(n{+}1)\times1$ (tutorial Q8).
- Feature scaling **speeds up GD** / equalizes contribution (midterm Q39); normalize with range = max − min (tutorial Q6).
- Regularization = reduce overfitting by penalizing big coefficients + makes $X^TX+\lambda I$ invertible (midterm Q31, tutorial Q10a).
- Residual analysis = check regression **assumptions** (midterm Q7).
- MSE penalizes large errors more; MAE is linear (tutorial Q10).
- Don't forget the **2** in the $\frac{1}{2m}$ when computing cost.

## Self-check

1. With $\theta_0=0,\theta_1=1$ on the tutorial Q2 data, why is the cost 0.4 and not 0.8?
2. Your $J(\theta)$ goes UP each iteration. What's the single most likely cause, and what do you change?
3. In one sentence each: how does batch GD differ from stochastic GD?
4. For $m=23$, $n=5$, what are the dimensions of $X$, $y$, and $\theta$ in the normal equation, and why is there a "+1"?
5. Normalize the value 1350 in a feature with mean 1525 and range 1060 — show the arithmetic.

**→ Now test yourself with [`quiz.md`](./quiz.md).**
