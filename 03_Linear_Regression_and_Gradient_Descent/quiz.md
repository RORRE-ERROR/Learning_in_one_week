# Chapter 3 Quiz — Linear Regression & Gradient Descent

> Format mirrors the WIA1006 quiz/midterm: multiple-choice + true/false + worked problems.
> Try it closed-book first. Answers + full worked solutions are at the very bottom.

## Section A — Multiple Choice

**A1.** What is the role of the coefficient in linear regression?
- a. It indicates the direction of the relationship between variables.
- b. It indicates the direction of the relationship between variables **and** represents the change in the dependent variable for a one-unit change in the independent variable.
- c. It measures the strength of the relationship between variables.
- d. It represents the change in the dependent variable for a one-unit change in the independent variable.

**A2.** Which is MOST accurate regarding the Least Mean Squares (LMS) algorithm?
- a. The LMS update's magnitude is inversely proportional to the error term, ensuring rapid convergence even with noisy data.
- b. The magnitude of the update is proportional to the error term; larger errors lead to more substantial parameter adjustments, while minimal changes occur with accurate predictions.
- c. When the prediction closely matches the actual value, the parameters are updated significantly to reinforce the correct prediction.
- d. LMS is primarily used for classification and adapted to regression through complex transformations.

**A3.** What happens when the learning rate is too high?
- a. Better generalization
- b. Reduced model capacity
- c. Divergence or oscillation during training
- d. Slow convergence

**A4.** During training, the cost $J(\theta)$ **increases** with each iteration. The most likely issue is:
- a. The learning rate is too small
- b. There are too few features
- c. The learning rate is too large
- d. The dataset is too small

**A5.** What best differentiates Batch Gradient Descent from Incremental (Stochastic) Gradient Descent?
- a. Batch GD calculates the gradient using the entire training set each iteration, while Incremental GD updates parameters based on the gradient of a single training example.
- b. Incremental GD is less sensitive to local minima than Batch GD in all cases.
- c. Batch GD updates after each training example, while Incremental GD updates after processing the entire training set.
- d. Batch GD is computationally cheaper per iteration but requires more iterations.

**A6.** What is the purpose of regularization in regression models?
- a. To normalize feature scales
- b. To handle missing values
- c. To reduce overfitting by penalizing large coefficients
- d. To increase the model's complexity

**A7.** What is the purpose of feature scaling in regression models?
- a. To ensure all features contribute equally (and to speed up gradient descent)
- b. To reduce dimensionality
- c. To increase model complexity
- d. To handle missing values

**A8.** What is the purpose of residual analysis in regression?
- a. To reduce dimensionality
- b. To check the assumptions of regression
- c. To evaluate the model's accuracy as the primary metric
- d. To handle missing values

**A9.** You have a dataset with $m=1{,}000{,}000$ examples and $n=200{,}000$ features. Which method should you use to fit a linear regression?
- a. Gradient descent
- b. The normal equation, because it has no learning rate
- c. The normal equation, because it has no local minima
- d. Neither can handle this data

**A10.** Why does the MSE cost function use a factor of $\frac{1}{2}$ (i.e. $\frac{1}{2m}$)?
- a. It halves the dataset to save computation.
- b. It is required for the cost to be positive.
- c. It cancels the 2 produced when differentiating the square, giving a cleaner gradient.
- d. It moves the location of the minimum closer to zero.

**A11.** Which statement about the normal equation $\theta=(X^TX)^{-1}X^Ty$ is TRUE?
- a. It requires you to tune a learning rate.
- b. It can get stuck in local minima.
- c. It is a closed-form solution but its $(X^TX)^{-1}$ inversion costs about $O(n^3)$.
- d. It always converges faster than gradient descent regardless of the number of features.

**A12.** Comparing error metrics, which pair of statements is correct?
- a. MSE grows linearly with error; MAE penalizes large errors more.
- b. MSE penalizes large errors more heavily; MAE grows linearly with error.
- c. Both grow linearly with error.
- d. Both penalize large errors equally.

## Section B — True / False

**B1.** In the cost $J(\theta)=\frac{1}{2m}\sum(h_\theta(x^{(i)})-y^{(i)})^2$, the symbol $m$ is the number of training examples. — (True / False)

**B2.** A learning rate that is too small will cause the cost to diverge and blow up. — (True / False)

**B3.** The normal equation suffers from local minima, just like gradient descent. — (True / False)

**B4.** Adding the regularization term $\lambda I$ makes $X^TX+\lambda I$ invertible, helping avoid a singular (non-invertible) matrix. — (True / False)

**B5.** For linear regression, the MSE cost surface is a convex bowl with a single global minimum, so gradient descent cannot get trapped in a local minimum. — (True / False)

**B6.** In stochastic (incremental) gradient descent, parameters are updated only once after the entire training set has been processed. — (True / False)

**B7.** Mean normalization centres a feature by subtracting its mean before dividing by its range. — (True / False)

## Section C — Worked Problems

**C1.** A training set for predicting sunny days:

| $x$ | $y$ |
|---|---|
| 3 | 2 |
| 2 | 3 |
| 4 | 5 |
| 1 | 1 |
| 5 | 4 |

What is $m$, the number of training examples?

**C2.** Using the data in C1 with $\theta_0=0$ and $\theta_1=1$, compute the cost $J(0,1)=\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2$. Show every step.

**C3.** Given $\theta_0=-1$ and $\theta_1=0.5$, compute $h_\theta(3)$.

**C4.** For the house data below, the model is $h_\theta(x)=\theta_0+\theta_1 x_1+\theta_2 x_2$ with $x_1$ = bedrooms and $x_2$ = sqft. Apply mean normalization with range $=\max-\min$ and find the normalized value $x_2^{(4)}$ (the 4th house's sqft). Show the mean, the range, and the final value.

| No | Bedrooms ($x_1$) | Sqft ($x_2$) | Price ($y$) |
|---|---|---|---|
| 1 | 1 | 880 | 490000 |
| 2 | 3 | 1930 | 630000 |
| 3 | 4 | 1940 | 640000 |
| 4 | 3 | 1350 | 570000 |

**C5.** A dataset has $m=23$ training examples and $n=5$ features. You add an intercept (all-ones) column and use the normal equation $\theta=(X^TX)^{-1}X^Ty$. State the dimensions of $X$, $y$, and $\theta$, and explain the "+1".

**C6.** For the dataset in C5, you instead have $m=1{,}000{,}000$ and $n=200{,}000$. Should you use gradient descent or the normal equation? Justify with the cost of the matrix inversion.

---

## ✅ Answer Key & Explanations

**A1 — (b)** The coefficient carries two things at once: the **direction** of the relationship (sign) **and** the change in $y$ per one-unit change in $x$ (magnitude). Option (d) gives only the magnitude and (a) only the direction; (c) describes correlation/strength, a different concept.

**A2 — (b)** This is the LMS rule: the update term contains the error factor $(h_\theta(x^{(i)})-y^{(i)})$, so the update is **proportional** to the error — big error, big correction; accurate prediction, near-zero update. (a) says *inversely* proportional (wrong direction) and (c) claims big updates when already correct (backwards).

**A3 — (c)** Too large a step overshoots the minimum and bounces around, so the cost oscillates or diverges. Slow convergence (d) is the symptom of a learning rate that's too *small*.

**A4 — (c)** A rising cost is the signature of an oversized learning rate: the steps overshoot and climb the far wall of the valley. A too-small rate (a) still decreases the cost, just slowly.

**A5 — (a)** The defining difference is how much data per update: batch uses the **entire** training set each iteration; stochastic updates from a **single** example. (c) reverses the two; (b) and (d) are overstated distractors.

**A6 — (c)** Regularization adds a penalty on large coefficients to discourage the overly-wiggly fits that cause overfitting. (a) is feature scaling; (d) is the opposite of what regularization does (it reduces effective complexity).

**A7 — (a)** Scaling puts features on comparable ranges so none dominates and so the cost surface is round rather than a skewed ravine — which **speeds up gradient descent**. It is unrelated to dimensionality, complexity, or missing values.

**A8 — (b)** Residual analysis inspects the leftover errors to verify regression assumptions (random scatter, constant spread, no leftover pattern). It is not dimensionality reduction, and "primary performance metric" is a distractor — its job is assumption-checking.

**A9 — (a)** With $n$ in the hundreds of thousands, the normal equation's $(X^TX)^{-1}$ inversion at roughly $O(n^3)$ is computationally hopeless. Gradient descent scales to large $n$, so it's the right choice. (b)/(c) state true facts about the normal equation but ignore the fatal cost issue here.

**A10 — (c)** The $\frac12$ is a convenience: differentiating $(\cdot)^2$ brings down a 2 that cancels the $\frac12$, giving a cleaner gradient. It does **not** change where the minimum lies, and it doesn't halve the data or affect positivity.

**A11 — (c)** The normal equation is a closed-form solution (no learning rate, no local minima — so (a) and (b) are false), but it requires inverting $X^TX$ at about $O(n^3)$, which is why it's bad for huge $n$. (d) is false precisely because of that inversion cost.

**A12 — (b)** MSE squares the errors, so large errors are penalized more heavily; MAE uses absolute value, so it grows linearly with the error. The other options swap or equate the two.

**B1 — True.** $m$ is the number of training examples (rows). $n$ would be the number of features.

**B2 — False.** A too-small learning rate makes convergence **slow**, not divergent. Divergence/oscillation comes from a learning rate that is too **large**.

**B3 — False.** The normal equation is a closed-form solution; it computes the optimum directly and has no local-minima problem (the issue gradient descent can face on non-convex surfaces).

**B4 — True.** $X^TX$ can be singular (non-invertible). Adding $\lambda I$ (regularization) gives $X^TX+\lambda I$, which is guaranteed invertible — this is exactly the benefit noted in the tutorial.

**B5 — True.** Linear regression's MSE cost is convex (a single bowl), so there's one global minimum and no local traps. Local minima become a concern with more complex models like neural networks.

**B6 — False.** That describes **batch** GD. Stochastic/incremental GD updates after **each single** example.

**B7 — True.** Mean normalization is $\dfrac{x-\mu}{\max-\min}$: subtract the mean (centre), then divide by the range (rescale).

---

### Section C — full worked solutions

**C1.** Count the rows: there are 5 examples, so $\boxed{m=5}$.

**C2.** With $\theta_0=0,\ \theta_1=1$, the hypothesis is $h_\theta(x)=0+1\cdot x=x$.

Compute the error $(h_\theta(x)-y)$ and its square for each example:

| $x$ | $h_\theta(x)=x$ | $y$ | error | error² |
|---|---|---|---|---|
| 3 | 3 | 2 | $3-2=1$ | 1 |
| 2 | 2 | 3 | $2-3=-1$ | 1 |
| 4 | 4 | 5 | $4-5=-1$ | 1 |
| 1 | 1 | 1 | $1-1=0$ | 0 |
| 5 | 5 | 4 | $5-4=1$ | 1 |

Sum of squared errors $=1+1+1+0+1=4$.
Denominator $2m=2\times5=10$.
$$J(0,1)=\frac{4}{10}=\boxed{0.4}.$$
(Trap check: it's $4/10$, not $4/5=0.8$ — don't drop the 2.)

**C3.** $h_\theta(3)=\theta_0+\theta_1\cdot 3=-1+0.5\times 3=-1+1.5=\boxed{0.5}.$

**C4.** Work on feature $x_2$ = sqft, values $\{880, 1930, 1940, 1350\}$.

Mean: $\mu_2=\dfrac{880+1930+1940+1350}{4}=\dfrac{6100}{4}=1525.$

Range: $\max-\min=1940-880=1060.$

Normalize the 4th value ($x_2^{(4)}=1350$):
$$x_2^{(4)}=\frac{1350-1525}{1060}=\frac{-175}{1060}\approx \boxed{-0.16}.$$
The negative sign means house 4 is below the average size in this set.

**C5.** Add the all-ones intercept column, so each example has $n+1=5+1=6$ entries.
- $X$: $m\times(n+1)=\boxed{23\times 6}$
- $y$: $m\times 1=\boxed{23\times 1}$
- $\theta$: $(n+1)\times 1=\boxed{6\times 1}$

The "+1" is the intercept term $\theta_0$: the leading column of 1's lets the model fit a non-zero $y$-intercept. Dimension check: $X^TX$ is $6\times6$ (square, invertible), $X^Ty$ is $6\times1$, and $(6\times6)(6\times1)=6\times1=\theta$. ✓

**C6.** Use **gradient descent**. The normal equation requires inverting $X^TX$, which is a $(n+1)\times(n+1)=200{,}001\times200{,}001$ matrix; matrix inversion costs roughly $O(n^3)$, which for $n\approx200{,}000$ is astronomically expensive. Gradient descent scales well to large numbers of features, so it is the practical choice here.
