# Chapter 7 Quiz — Support Vector Machines (SVM)

> Format mirrors the WIA1006 quiz/midterm: multiple-choice + true/false + a short worked
> reasoning problem. Try it closed-book first. Answers + explanations are at the very bottom.

## Section A — Multiple Choice

**A1.** You train a Linear SVM on a 2-class problem and circle the support vectors. If you **remove one of the support vectors**, what happens to the decision boundary?
- a. It stays exactly the same.
- b. It changes.
- c. It only changes if you also remove a non-support-vector.
- d. The model can no longer make predictions.

**A2.** You remove a point that sits **well outside the margin** (a non-support-vector). The decision boundary will:
- a. Change, because every training point matters equally.
- b. Change, but only its position, not its orientation.
- c. Not change, because that point had no influence on the geometry.
- d. Flip to the opposite class.

**A3.** You trained a Gaussian-kernel SVM and suspect it is **underfitting** the training data. Which adjustment is reasonable?
- a. Decrease `C`, increase `σ²`.
- b. Decrease `C`, decrease `σ²`.
- c. **Increase `C`, decrease `σ²`.**
- d. Increase `C`, increase `σ²`.

**A4.** A Gaussian-kernel SVM does **poorly on the cross-validation set** (overfitting). Which adjustment is reasonable?
- a. Decrease `C`, increase `σ²`.
- b. Decrease `C`, decrease `σ²`.
- c. Increase `C`, decrease `σ²`.
- d. Increase `C`, increase `σ²`.

**A5.** Your logistic-regression classifier (`n = 10` features, `m = 5000` examples) **underfits** on both the training and cross-validation sets. Which steps are promising? (Choose all that apply.)
- a. Use an SVM with a Gaussian kernel.
- b. Create / add new polynomial features.
- c. Use an SVM with a *linear* kernel and no new features.
- d. Use a neural network with many hidden units.
- e. Reduce the number of training examples.

**A6.** Why is feature normalization important **before** using a Gaussian kernel?
- a. It speeds up matrix inversion.
- b. The kernel uses squared Euclidean distance, so large-range features dominate and small-scale features get ignored.
- c. It converts categorical features into numbers.
- d. It is required to compute probabilities.

**A7.** When the `C` parameter is set to **infinity**, which statement holds?
- a. The optimal hyperplane, if it exists, will be the one that completely separates the data.
- b. The soft-margin classifier will separate the data.
- c. The margin becomes as wide as possible regardless of errors.
- d. None of the above.

**A8.** The effectiveness of an SVM depends upon:
- a. Selection of the kernel trick.
- b. Kernel parameters.
- c. The soft-margin parameter `C`.
- d. All of the above.

**A9.** Which statement about the `C` parameter is **correct**?
- a. A large `C` widens the margin and tolerates more misclassifications.
- b. A large `C` narrows the margin and tolerates fewer misclassifications.
- c. `C` has no effect on the margin width.
- d. A small `C` forces a perfect (hard) margin.

**A10.** What does the **kernel trick** accomplish?
- a. It deletes outliers before training.
- b. It lets a linear separator work by implicitly mapping data into a higher-dimensional space, without computing the new coordinates explicitly.
- c. It normalizes the features automatically.
- d. It guarantees zero training error on any dataset.

**A11.** Which is the defining property of the hyperplane an SVM chooses?
- a. It passes through the mean of each class.
- b. It minimizes the number of support vectors.
- c. It is the separating boundary with the **largest margin**.
- d. It is the boundary with the steepest slope.

## Section B — True / False

**B1.** Removing a non-support-vector from the training set changes the decision boundary. — (True / False)

**B2.** A larger value of `C` makes the SVM *more* tolerant of misclassified points. — (True / False)

**B3.** For a Gaussian kernel, a **larger `σ²`** produces a smoother, simpler decision boundary. — (True / False)

**B4.** A linear-kernel SVM with no added features is a good way to fix an underfitting (high-bias) model. — (True / False)

**B5.** `SGDClassifier(loss='hinge')` — as used in the GhostBusters project — implements a **linear** SVM. — (True / False)

**B6.** The number of support vectors is typically much smaller than the total number of training examples. — (True / False)

**B7.** Setting `C → ∞` corresponds to a **soft**-margin SVM that tolerates some violations. — (True / False)

## Section C — Worked Problem (bias–variance reasoning)

**C1.** You train two Gaussian-kernel SVMs on the same dataset.

- **Model X** scores 99% on the training set but only 71% on the cross-validation set.
- **Model Y** scores 68% on the training set and 67% on the cross-validation set.

For each model, (i) diagnose under- vs overfitting and name whether it's a bias or variance problem, and (ii) state which direction to move `C` and `σ²` to improve it.

---

## ✅ Answer Key & Explanations

**A1 — (b) It changes.** Support vectors are the points that pin the margin in place. Remove one and the optimal maximum-margin boundary must be recomputed, so it moves. (a) is the rule for *non*-support-vectors, not support vectors.

**A2 — (c) Not change.** A point sitting outside the margin contributes nothing to the boundary's geometry — only support vectors do. Deleting it leaves the solution identical. (a) wrongly assumes all points matter equally; in SVM they don't.

**A3 — (c) Increase `C`, decrease `σ²`.** Underfitting = too simple = high bias, so add complexity. Increasing `C` penalises errors harder (more intricate boundary); decreasing `σ²` narrows each Gaussian so the model reacts to local variation. The other options either soften the model (↓C) or smooth it (↑σ²), which make underfitting worse.

**A4 — (a) Decrease `C`, increase `σ²`.** Poor cross-validation with overfitting = too complex = high variance, so simplify. Decreasing `C` softens the margin (allows some errors, generalises better); increasing `σ²` broadens the kernel for a smoother boundary. This is the exact mirror image of A3.

**A5 — (a), (b), (d).** Underfitting is high bias, fixed by *adding non-linearity/complexity*: a Gaussian-kernel SVM (a), polynomial features (b), and a large neural network (d) all add capacity. (c) stays linear so it still underfits, and (e) reducing data never cures high bias — it usually hurts.

**A6 — (b).** The Gaussian kernel computes $\exp(-\lVert x-x'\rVert^2 / 2\sigma^2)$, which is driven by squared Euclidean distance. Without scaling, a feature with a large numeric range swamps the distance and the model effectively ignores small-scale features. (a), (c), (d) are unrelated to the kernel's distance computation.

**A7 — (a).** An infinite `C` makes any margin violation infinitely costly, removing the soft margin entirely — this is a hard margin, so the chosen hyperplane (if one exists) completely separates the data. (b) describes a *soft*-margin classifier, which is what `C → ∞` eliminates.

**A8 — (d) All of the above.** SVM performance hinges on the kernel choice, the kernel's parameters (e.g. `σ²`), *and* the soft-margin `C`. All three must be tuned together.

**A9 — (b).** Large `C` = high penalty for errors = the model squeezes the margin to classify everything correctly (overfit risk). (a) and (d) invert the relationship; (c) is false since `C` directly trades off margin width against errors.

**A10 — (b).** The kernel trick implicitly maps data into a higher-dimensional space where a linear separator exists, computing only inner products (kernel values) rather than the explicit new coordinates. It does not remove outliers (a), normalize (c), or guarantee zero error (d).

**A11 — (c).** SVM picks the separating hyperplane with the **largest margin** (widest gap to the nearest points of each class). The other options are not what the SVM optimises.

**B1 — False.** Only support vectors define the boundary; removing a non-support-vector leaves it unchanged.

**B2 — False.** It's the opposite. Larger `C` = *less* tolerance for misclassification (heavier penalty, narrower margin). Smaller `C` is the tolerant one.

**B3 — True.** Larger `σ²` widens each Gaussian's reach, blending influence across points into a smoother, lower-complexity (higher-bias) boundary.

**B4 — False.** A linear kernel with no new features stays linear and cannot capture the structure the model is missing — it still underfits. To fix high bias you need to *add* complexity (RBF kernel, polynomial features, bigger network).

**B5 — True.** Hinge loss with linear weights is exactly a linear SVM; `SGDClassifier(loss='hinge')` optimises that objective with stochastic gradient descent, which is why the project used it for speed on 50,000 rows.

**B6 — True.** Typically only a small subset of points lie on the margin edge and act as support vectors, which is why SVMs are memory-efficient at prediction time.

**B7 — False.** `C → ∞` is a **hard**-margin SVM with zero tolerance for violations — the opposite of a soft margin.

**C1 —**
- **Model X (99% train, 71% CV):** the large gap between training and cross-validation accuracy means it memorised the training data — **overfitting / high variance**. Fix by simplifying: **decrease `C`** (softer margin) and **increase `σ²`** (smoother boundary).
- **Model Y (68% train, 67% CV):** both scores are low and close together, so it can't even fit the training data — **underfitting / high bias**. Fix by adding complexity: **increase `C`** (penalise errors harder) and **decrease `σ²`** (narrower, more flexible kernel).

Memory hook: a big *train-vs-CV gap* ⇒ variance ⇒ simplify (↓C, ↑σ²); *both low* ⇒ bias ⇒ complexify (↑C, ↓σ²).
