# Chapter 4 — Logistic Regression

> **Day 3 of 7** · Prerequisites: Ch 3 (Linear Regression + Gradient Descent) · Est. study time: ~60–75 min

**In one sentence:** Logistic regression takes the linear-regression machinery you already know, squashes its output through an S-curve to get a probability between 0 and 1, and uses that to make **yes/no** predictions.

## Why this matters

In the GhostBusters project, the target is `is_ghosted` — either 1 (Ghosted) or 0 (Not Ghosted). That is a **binary classification** problem, and logistic regression is the project's *primary interpretable baseline*: it is simple, fast, and you can read off which features push a prediction toward "ghost". The exam leans hard on logistic regression because it sits exactly at the boundary between regression (Ch 3) and classification, so it tests whether you really understood both.

## Learning goals
- [ ] Explain why linear regression is the wrong tool for predicting a 0/1 label
- [ ] Write down the sigmoid function and describe its shape and range
- [ ] State the decision boundary rule and connect it to $\theta^T x$
- [ ] Define **odds**, **log-odds**, and the **logit**, and say precisely what each direction of the map does
- [ ] Explain why we use log-loss (cross-entropy) instead of MSE
- [ ] Describe the stochastic gradient descent update for logistic regression
- [ ] List the main limitations of logistic regression

---

## 1. The problem: binary classification

Linear regression (Ch 3) predicts a **continuous** number: a price, a temperature, a usage time. The output can be any real value — negative, huge, fractional.

But many real questions are **yes/no**:
- Will this user ghost their match? (1 = yes, 0 = no)
- Is this email spam?
- Did the tumour turn out malignant?

These have only two possible answers, encoded as **1** (the positive class) and **0** (the negative class). This is **binary classification**.

**Why not just use linear regression and round?** Suppose you fit a straight line to predict `is_ghosted`. The line can output 1.7 or −0.4 — values that make no sense as a class label, and worse, *cannot* be read as a probability (probabilities live in [0, 1]). A single far-away point can also tilt the whole line and move your cut-off. You need an output that is always between 0 and 1, that you can interpret as **"probability of class 1"**. That is what logistic regression gives you.

> 💡 **Exam tip:** The cleanest one-liner the exam wants (midterm Q1): *linear regression predicts continuous outcomes; logistic regression predicts binary outcomes* — **and** *linear regression assumes a linear relationship while logistic regression does not assume linearity in the raw output*. Both halves together are the "most complete" answer. The student's answer (just the continuous-vs-binary half) is correct but partial; the fuller option also mentions the linearity assumption.

> ⚠️ **Common trap:** "Logistic regression is a regression algorithm because of the name." It has *regression* in its name and reuses linear-regression math, but its **job is classification**.

## 2. The sigmoid (logistic) function

We still build a linear score from the features, exactly like Ch 3:

$$z = \theta^T x = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n$$

Here $x$ is the feature vector for one example (e.g. one user's `message_sent_count`, `swipe_right_ratio`, …), and $\theta$ is the vector of weights we learn. The score $z$ can be any real number, from $-\infty$ to $+\infty$.

To turn $z$ into a probability we pass it through the **sigmoid** (a.k.a. **logistic**) function:

$$g(z) = \frac{1}{1 + e^{-z}}$$

So the full model — the **hypothesis** — is:

$$h_\theta(x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$

**What the sigmoid does, intuitively:** it is a smooth **S-curve** that squashes any real number into the open interval $(0, 1)$.
- Very large positive $z$ → $e^{-z}$ ≈ 0 → $g(z)$ ≈ **1**
- Very large negative $z$ → $e^{-z}$ huge → $g(z)$ ≈ **0**
- $z = 0$ → $e^{0} = 1$ → $g(0) = \frac{1}{1+1} = \mathbf{0.5}$

```
g(z)
 1.0 |                         _____------
     |                    __--
 0.5 |- - - - - - - - - +   <- crosses 0.5 exactly at z = 0
     |             __--
 0.0 |______------_________________________ z
          -          0          +
```

We read $h_\theta(x)$ as **"the model's estimated probability that y = 1"**. For a user, $h_\theta(x) = 0.8$ means "80% chance this user ghosts".

> 💡 **Exam tip:** Memorise three facts: the range of sigmoid is **(0, 1)** (open — it never *exactly* reaches 0 or 1), it is **monotonically increasing**, and $g(0) = 0.5$.

> ⚠️ **Common trap:** The range is **not** [−1, 1] (that's `tanh`) and **not** all reals. Sigmoid output is a probability, so it lives strictly between 0 and 1.

## 3. The decision boundary

A probability is nice, but eventually you must commit to a label. The standard rule uses a **threshold of 0.5**:

$$\text{predict } y = 1 \quad \text{if} \quad h_\theta(x) \ge 0.5, \qquad \text{else predict } y = 0$$

Now connect this back to $z$. Because the sigmoid crosses 0.5 *exactly* at $z = 0$, and is increasing:

$$h_\theta(x) \ge 0.5 \iff g(\theta^T x) \ge 0.5 \iff \theta^T x \ge 0$$

So the rule simplifies beautifully:

$$\text{predict } 1 \iff \theta^T x \ge 0$$

The set of points where $\theta^T x = 0$ is the **decision boundary** — the dividing surface between the "predict 1" region and the "predict 0" region. Because $\theta^T x$ is *linear* in $x$, this boundary is a straight line (or a flat plane in higher dimensions). That is why plain logistic regression is called a **linear classifier**.

> 💡 **Exam tip:** Default classification threshold = **0.5**, which corresponds to $\theta^T x = 0$. You *can* move the threshold (e.g. raise it to catch fewer false positives), but 0.5 is the default they expect.

> ⚠️ **Common trap:** The decision boundary is a property of **$\theta$**, not of the data points themselves. You don't compute it from one example — it's the line where the model is exactly undecided (p = 0.5).

## 4. Odds and the logit (read this slowly — the exam tests it)

This section is the one students get wrong on the exam, so let's be precise.

**Odds.** If $p$ is the probability of class 1, the **odds** are the ratio of "happening" to "not happening":

$$\text{odds} = \frac{p}{1 - p}$$

If $p = 0.8$, odds $= 0.8 / 0.2 = 4$, i.e. "4 to 1 on". Odds range over $(0, \infty)$.

**Logit = log-odds.** The **logit** function takes the natural log of the odds:

$$\text{logit}(p) = \ln\!\left(\frac{p}{1 - p}\right)$$

And here is the heart of logistic regression — the logit of the predicted probability equals the linear score:

$$\ln\!\left(\frac{p}{1 - p}\right) = \theta^T x$$

So **the thing that is linear in $x$ is the log-odds**, not the probability itself. The sigmoid is just the *inverse* of the logit, rearranged so you can go the other way:

$$\underbrace{\text{logit}}_{p \,\rightarrow\, \text{log-odds}} \qquad\text{vs.}\qquad \underbrace{\text{sigmoid}}_{\text{log-odds} \,\rightarrow\, p}$$

- **logit (log-odds): probability → real number.** It takes a probability $p \in (0,1)$ and maps it to a log-odds value in $(-\infty, \infty)$.
- **sigmoid: real number → probability.** It takes the log-odds (your score $z = \theta^T x$) back to a probability.

> ⚠️ **Common trap (midterm Q12):** The exam asked *"what does the logit function do?"* and the student picked **(a) "It converts probabilities into odds."** That is **not quite right**, and it's worth knowing why:
> - The logit converts a probability into **log-odds**, i.e. $\ln\big(\tfrac{p}{1-p}\big)$ — it includes the **log**. "Converts probabilities into odds" describes only $\tfrac{p}{1-p}$, missing the logarithm.
> - **(c) "converts odds into probabilities"** describes the *inverse* direction (that's what sigmoid does), so it's wrong too.
> Of the options offered, (a) is the closest to correct and is the intended answer, but say it precisely on paper: **the logit maps a probability to its log-odds**. If a cleaner option ("converts probability to log-odds") is ever available, pick that.

> 💡 **Exam tip:** One sentence to lock in: *"In logistic regression the **log-odds** are a linear function of the features: $\ln\frac{p}{1-p} = \theta^T x$."* Everything else (sigmoid, decision boundary) follows from this.

## 5. The cost function: why not MSE, and what we use instead

In linear regression you minimised **mean squared error (MSE)**. The natural instinct is to reuse it here. Don't — and here's the intuition.

**Why MSE fails for logistic regression.** When you plug the sigmoid $h_\theta(x)$ into the squared-error formula, the resulting cost surface $J(\theta)$ becomes **non-convex** — it's wavy, with many local minima. Gradient descent can get stuck in a bad dip and never reach the true best $\theta$. We want a **convex** (single-bowl) cost so gradient descent reliably slides to the global minimum.

**The fix: log-loss / cross-entropy.** We use a cost that is large when the model is *confidently wrong* and near zero when it is *confidently right*. For a single example with true label $y$:

$$\text{cost} = \begin{cases} -\log\big(h_\theta(x)\big) & \text{if } y = 1 \\ -\log\big(1 - h_\theta(x)\big) & \text{if } y = 0 \end{cases}$$

Intuition: if $y = 1$ and the model says $h_\theta(x) \approx 1$, then $-\log(1) = 0$ → no penalty. But if it confidently says $h_\theta(x) \approx 0$, then $-\log(0) \to \infty$ → huge penalty. The model is punished hard for being confidently wrong.

The two cases combine into one tidy formula (because exactly one of $y$, $1-y$ is 1):

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \Big[\, y^{(i)} \log h_\theta(x^{(i)}) + (1 - y^{(i)}) \log\big(1 - h_\theta(x^{(i)})\big) \,\Big]$$

where $m$ is the number of training examples. This is the **cross-entropy** / **log-loss** cost, and it is convex.

> 💡 **Exam tip:** "Why not MSE for logistic regression?" → *Because MSE with the sigmoid is **non-convex**, so gradient descent can get stuck in local minima; log-loss (cross-entropy) is convex.* That single word **non-convex** is usually what they're fishing for.

## 6. Training: stochastic gradient descent (SGD)

We minimise $J(\theta)$ with gradient descent, just like Ch 3. Two flavours:
- **Batch gradient descent:** compute the gradient over the **entire** training set, then take one step. (Midterm Q20 contrasts this with stochastic.)
- **Stochastic gradient descent (SGD):** update the parameters after **each single training example**, using the gradient of the error for *that one example*.

The SGD update rule for logistic regression, for each example $(x^{(i)}, y^{(i)})$:

$$\theta_j \;:=\; \theta_j \;-\; \alpha \,\big(h_\theta(x^{(i)}) - y^{(i)}\big)\, x_j^{(i)}$$

where $\alpha$ is the learning rate. The term $\big(h_\theta(x^{(i)}) - y^{(i)}\big)$ is the **error** (prediction minus truth) for that example.

**The neat surprise:** this update has the **exact same algebraic form** as the linear-regression update from Ch 3 — `(prediction − actual) × feature`. The *only* difference is that $h_\theta(x)$ is now the sigmoid output instead of a raw linear value. So if you learned the LMS / gradient-descent update in Ch 3, you already know this one.

> 💡 **Exam tip (midterm Q3):** The correct description of the SGD update is: *parameters are updated **after each single training example**, using the gradient of the error with respect to that one example.* Reject the distractors: it does **not** use the whole set per update (that's batch), it does **not** average over a mini-batch (that's mini-batch GD), and the learning rate does **not** automatically increase to "accelerate convergence".

> ⚠️ **Common trap:** Don't confuse "updates after each example" (stochastic) with "updates once after the whole set" (batch). Midterm Q20 is the same idea framed as a comparison.

## 7. Regularization (carried over from Ch 3)

Just like in linear regression, you can add a **regularization** penalty to the cost (an extra term that penalises large weights $\theta_j$). Its purpose is unchanged: **reduce overfitting** so the model generalises to unseen users instead of memorising the training set. In the GhostBusters notebook this is implicit in scikit-learn's `LogisticRegression`, which is L2-regularized by default.

> 💡 **Exam tip (midterm Q31):** Regularization = *reduce overfitting by penalising large coefficients.* It does **not** normalise feature scales or handle missing values.

## 8. Limitations of logistic regression

Logistic regression is a great, interpretable baseline — but it has real weaknesses (midterm Q27):

- **It assumes linearity in the log-odds.** The boundary $\theta^T x = 0$ is a straight line. If the true relationship between features and class is curved, plain logistic regression can't capture it well.
- **It is sensitive to outliers.** Extreme feature values can drag the boundary around.
- **It struggles with inherently non-linear relationships** unless you do **feature engineering** (e.g. add polynomial or interaction terms) to bend the boundary.

This is exactly why the GhostBusters project also trains Decision Trees, Random Forests, and an MLP — non-linear models that can capture the messy, non-linear nature of human dating behaviour that logistic regression misses.

> 💡 **Exam tip (midterm Q27, "select all"):** The limitations are **(c) it assumes linearity in the log-odds** and **(d) it is sensitive to outliers**. Option (b) "it can handle non-linear relationships" is a *strength claim* and is **false for plain logistic regression**, so it is NOT a limitation — don't select it.

---

## Key formulas / facts at a glance

| Thing | Formula / Rule | Plain meaning |
|---|---|---|
| Linear score | $z = \theta^T x$ | Same weighted sum as linear regression |
| Sigmoid | $g(z) = \dfrac{1}{1 + e^{-z}}$ | Squashes any real number into $(0,1)$ |
| Hypothesis | $h_\theta(x) = g(\theta^T x)$ | Estimated probability that $y = 1$ |
| Key value | $g(0) = 0.5$ | Sigmoid crosses 0.5 at $z = 0$ |
| Range | $(0, 1)$ | A probability — never exactly 0 or 1 |
| Decision rule | predict 1 if $h_\theta(x) \ge 0.5 \iff \theta^T x \ge 0$ | Threshold at 0.5 |
| Odds | $\dfrac{p}{1-p}$ | "Happening vs not", range $(0,\infty)$ |
| Logit (log-odds) | $\ln\!\big(\tfrac{p}{1-p}\big) = \theta^T x$ | The **log-odds** are linear in $x$ |
| Cost (log-loss) | $-\tfrac{1}{m}\sum[y\log h_\theta(x) + (1-y)\log(1-h_\theta(x))]$ | Convex; punishes confident mistakes |
| Why not MSE | MSE + sigmoid is **non-convex** | Gradient descent can get stuck |
| SGD update | $\theta_j := \theta_j - \alpha(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}$ | Update per example; same form as linear reg |
| Limitations | linear log-odds, outlier-sensitive, needs feature engineering for non-linearity | Why we also use trees / NNs |

## Exam tips — quick recap
- **Q1:** Linear → continuous + assumes linearity; logistic → binary + does not assume linearity in the raw output.
- **Sigmoid:** range $(0,1)$, increasing, $g(0)=0.5$. Not $[-1,1]$.
- **Decision boundary:** threshold 0.5 ⟺ $\theta^T x \ge 0$.
- **Q12 (logit):** logit maps **probability → log-odds** = $\ln\frac{p}{1-p}=\theta^T x$. Sigmoid is the inverse (log-odds → probability).
- **Cost:** use **log-loss/cross-entropy** because MSE here is **non-convex**.
- **Q3 (SGD):** update after **each single example** using that example's gradient; same form as linear regression.
- **Q31:** regularization reduces overfitting by penalising large coefficients.
- **Q27:** limitations = assumes linearity in log-odds **and** sensitive to outliers (not "can handle non-linear relationships").

## Self-check
1. Why can't you use ordinary linear regression to predict a 0/1 label directly?
2. What is $g(0)$, and what is the range of the sigmoid?
3. Translate "predict 1 if $h_\theta(x) \ge 0.5$" into a condition on $\theta^T x$. Why are they equivalent?
4. State exactly what the logit (log-odds) function does, and what the sigmoid does — in which direction does each map?
5. Why do we minimise log-loss instead of MSE for logistic regression?
6. Describe the stochastic gradient descent update in one sentence, and say how it differs from batch gradient descent.

**→ Now test yourself with [`quiz.md`](./quiz.md).**
