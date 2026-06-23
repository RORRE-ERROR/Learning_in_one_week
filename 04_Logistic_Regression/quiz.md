# Chapter 4 Quiz — Logistic Regression

> Format mirrors the WIA1006 quiz/midterm: multiple-choice + true/false + a few worked problems.
> Try it closed-book first. Answers + explanations are at the very bottom.

## Section A — Multiple Choice

**A1.** What is the difference between linear and logistic regression?
- a. Linear regression assumes linearity, while logistic regression does not.
- b. Linear regression is used for classification, while logistic regression is used for regression.
- c. Linear regression predicts continuous outcomes while logistic regression predicts binary outcomes, **and** linear regression assumes linearity while logistic regression does not.
- d. Linear regression predicts continuous outcomes, while logistic regression predicts binary outcomes.

**A2.** What is the range of the sigmoid (logistic) function $g(z) = \frac{1}{1+e^{-z}}$?
- a. $[-1, 1]$
- b. $(0, 1)$
- c. $[0, \infty)$
- d. All real numbers

**A3.** Using the standard threshold, a logistic regression model predicts class 1 when:
- a. $\theta^T x \ge 0$
- b. $\theta^T x \le 0$
- c. $\theta^T x = 1$
- d. $h_\theta(x) < 0.5$

**A4.** In logistic regression, what does the logit function do?
- a. It converts probabilities into odds.
- b. It normalizes feature scales.
- c. It converts odds into probabilities.
- d. It handles missing values.

**A5.** Which statement accurately describes the update process in the stochastic gradient descent rule for logistic regression?
- a. Parameters updated using a learning rate that increases over time to accelerate convergence.
- b. Parameters updated once after processing the entire training set, ensuring convergence to the global optimum.
- c. Parameters updated based on the average gradient across a small batch of training examples to reduce noise.
- d. Parameters updated after each training example, using the gradient of the error with respect to that single example.

**A6.** Why is mean squared error (MSE) a poor choice of cost function for logistic regression?
- a. It cannot be differentiated.
- b. Combined with the sigmoid it produces a non-convex cost surface, so gradient descent can get stuck in local minima.
- c. It always outputs values greater than 1.
- d. It requires the labels to be continuous.

**A7.** What is the value of the sigmoid function at $z = 0$?
- a. 0
- b. 1
- c. 0.5
- d. Undefined

**A8.** In logistic regression, which quantity is modelled as a *linear* function of the features $x$?
- a. The probability $p$ directly.
- b. The odds $\frac{p}{1-p}$.
- c. The log-odds $\ln\frac{p}{1-p}$.
- d. The squared error.

**A9.** Which is a limitation of logistic regression? (Select one or more)
- a. None of them
- b. It can handle non-linear relationships.
- c. It assumes linearity in the log-odds.
- d. It is sensitive to outliers.

**A10.** What is the purpose of regularization in logistic (and linear) regression?
- a. To normalize feature scales.
- b. To handle missing values.
- c. To reduce overfitting by penalizing large coefficients.
- d. To increase the model's complexity.

**A11.** A logistic regression model outputs $h_\theta(x) = 0.73$ for a user. What does this number mean?
- a. The user definitely belongs to class 1.
- b. The estimated probability that the user belongs to class 1 is 0.73.
- c. The raw linear score $\theta^T x$ is 0.73.
- d. The model is 73% accurate.

**A12.** Which best differentiates Batch Gradient Descent from Stochastic (Incremental) Gradient Descent?
- a. Batch GD calculates the gradient using the entire training set each iteration, while Stochastic GD updates parameters based on the gradient of a single training example.
- b. Stochastic GD is guaranteed to find the global optimum, while Batch GD is not.
- c. Batch GD updates after each training example, while Stochastic GD updates after the whole set.
- d. Batch GD is always faster per pass over the data.

## Section B — True / False

**B1.** Logistic regression is used to predict continuous numerical outcomes. — (True / False)

**B2.** The output of the sigmoid function can be interpreted as a probability and always lies strictly between 0 and 1. — (True / False)

**B3.** The decision boundary of a plain logistic regression model is linear in the input features. — (True / False)

**B4.** The sigmoid function maps log-odds (a real number) to a probability, which is the inverse of what the logit function does. — (True / False)

**B5.** We use log-loss (cross-entropy) instead of MSE for logistic regression mainly because the MSE-based cost is non-convex here. — (True / False)

**B6.** In stochastic gradient descent for logistic regression, the parameter update has a different algebraic form from the linear regression update. — (True / False)

**B7.** Logistic regression can natively capture highly non-linear relationships without any feature engineering. — (True / False)

## Section C — Worked Problems

**C1.** Compute $g(0)$ for the sigmoid $g(z) = \frac{1}{1+e^{-z}}$, and state the predicted class using the 0.5 threshold.

**C2.** A model has parameters giving a linear score $z = \theta^T x = 2$ for a particular user. Using $e^{-2} \approx 0.135$, compute $h_\theta(x) = g(2)$ and decide the predicted class (threshold 0.5).

**C3.** For another user the score is $z = \theta^T x = -1$. Using $e^{1} \approx 2.718$, compute $g(-1)$ and decide the predicted class. Then, without recomputing the sigmoid, state the predicted class for *any* user whose $\theta^T x$ is negative, and explain why in one sentence.

---

## ✅ Answer Key & Explanations

**A1 — (c)** This is the most complete answer. Logistic regression predicts **binary** outcomes (vs continuous for linear regression) **and** does not assume a linear relationship in the raw output. Option (d) is true but partial — it omits the linearity point. Option (a) alone is incomplete, and (b) reverses the two algorithms.

**A2 — (b)** The sigmoid squashes any real input into the **open interval $(0, 1)$**. It approaches 0 and 1 but never reaches them. $[-1,1]$ is the range of `tanh`, not sigmoid.

**A3 — (a)** Predict 1 when $h_\theta(x) \ge 0.5$. Since sigmoid crosses 0.5 exactly at $z = 0$ and is increasing, $h_\theta(x) \ge 0.5 \iff \theta^T x \ge 0$.

**A4 — (a)** Of the options given, (a) is the intended answer. Be precise on paper, though: the logit actually maps a probability to its **log-odds**, $\ln\frac{p}{1-p}$ — option (a) leaves out the logarithm, and option (c) describes the *inverse* (that's what the sigmoid does). So (a) is closest, but the truthful statement is "logit = probability → log-odds."

**A5 — (d)** Stochastic GD updates the parameters **after each single training example**, using that example's gradient. (b) is batch GD; (c) is mini-batch GD; (a) misdescribes the learning rate.

**A6 — (b)** Plugging the sigmoid into squared error makes the cost surface **non-convex** (many local minima), so gradient descent may not reach the global optimum. Log-loss is convex. MSE is perfectly differentiable, so (a) is wrong.

**A7 — (c)** $g(0) = \frac{1}{1+e^{0}} = \frac{1}{1+1} = 0.5$. This is exactly why the 0.5 threshold corresponds to $\theta^T x = 0$.

**A8 — (c)** The defining equation of logistic regression is $\ln\frac{p}{1-p} = \theta^T x$ — the **log-odds** are linear in $x$. The probability itself is a non-linear (S-shaped) function of $x$.

**A9 — (c) and (d)** Logistic regression **assumes linearity in the log-odds** and is **sensitive to outliers**. Option (b) is a false claim (plain logistic regression cannot natively handle non-linear relationships), so it is not a limitation; (a) is wrong because limitations do exist.

**A10 — (c)** Regularization adds a penalty on large coefficients to **reduce overfitting** and improve generalization. It does not scale features or impute missing values.

**A11 — (b)** $h_\theta(x)$ is the model's **estimated probability of class 1**, here 0.73. It is not a certainty (a), not the raw score (c) — the raw score $\theta^T x$ would be passed through the sigmoid first — and not an accuracy figure (d).

**A12 — (a)** Batch GD uses the **entire** training set per update; Stochastic GD updates from a **single** example at a time. (c) reverses them; (b) and (d) are false in general.

**B1 — False.** That describes linear regression. Logistic regression predicts **binary** (0/1) outcomes via a probability.

**B2 — True.** The sigmoid output lies in $(0,1)$ and is read as the probability of class 1.

**B3 — True.** The boundary is $\theta^T x = 0$, which is linear in the features — hence "linear classifier".

**B4 — True.** Logit: probability → log-odds. Sigmoid: log-odds → probability. They are inverses of each other.

**B5 — True.** MSE with the sigmoid gives a non-convex cost; cross-entropy is convex, so gradient descent reliably reaches the global minimum.

**B6 — False.** The update has the **same form** as linear regression, $\theta_j := \theta_j - \alpha(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}$; the only change is that $h_\theta$ is now the sigmoid output.

**B7 — False.** Plain logistic regression has a linear boundary; capturing non-linear relationships requires feature engineering (e.g. polynomial/interaction terms) or a non-linear model.

**C1 —** $g(0) = \dfrac{1}{1+e^{0}} = \dfrac{1}{1+1} = \mathbf{0.5}$. With the 0.5 threshold ($\ge 0.5 \Rightarrow$ class 1), the prediction is **class 1** (this is the exact tipping point).

**C2 —** $g(2) = \dfrac{1}{1+e^{-2}} = \dfrac{1}{1+0.135} = \dfrac{1}{1.135} \approx \mathbf{0.881}$. Since $0.881 \ge 0.5$ (equivalently $z = 2 \ge 0$), predict **class 1**.

**C3 —** $g(-1) = \dfrac{1}{1+e^{1}} = \dfrac{1}{1+2.718} = \dfrac{1}{3.718} \approx \mathbf{0.269}$. Since $0.269 < 0.5$, predict **class 0**. For *any* user with $\theta^T x < 0$ the prediction is **class 0**, because the sigmoid is increasing and equals 0.5 only at $z=0$, so $z<0 \Rightarrow g(z)<0.5$.
