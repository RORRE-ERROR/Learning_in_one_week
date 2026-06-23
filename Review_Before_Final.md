# 🎯 Review Before the Final — WIA1006 Machine Learning

**How to use this:** This is a *consolidation* sheet, not a teaching one. Work through the 10
chapters first. The night before the exam, read this top to bottom — it's the highest-yield
facts, formulas, and traps in one place. If a line here looks unfamiliar, go back to that
chapter's `notes.md`.

Legend: 💡 = frequently tested · ⚠️ = common trap.

---

## 1. Foundations

- **ML = computer learns the rules from data**, instead of a human hand-coding them.
  Traditional: `Rules + Data → Answers`. ML: `Data + Answers → Rules (model)`.
- **Three paradigms:**
  - **Supervised** — labelled data. → **Regression** (predict a *continuous number*) vs
    **Classification** (predict a *category*). The project = supervised *binary classification*.
  - **Unsupervised** — no labels; find structure. → clustering, dimensionality reduction (PCA).
  - **Reinforcement** — an **agent acts on an environment to obtain rewards**. 💡 (midterm: RL → *rewards*).
- **Workflow:** data → preprocessing → train model → evaluate → deploy/explain.
- **Overfitting** = memorises training data, fails on new data (low train error, high test error).
  **Underfitting** = too simple, fails on both. Goal = **generalization**.
- **Metrics:** Accuracy (can mislead under imbalance), **Precision** (of predicted-positives, how many right),
  **Recall** (of actual-positives, how many caught), **F1** = harmonic mean of P & R, **AUC-ROC** (separability across thresholds).

---

## 2. Data Preprocessing  *(most-tested topic)*

- **Why:** raw data has missing values, outliers, noise, meaningless combos. Garbage in → garbage out.
  Improper handling → *inaccurate/faulty conclusions*. 💡
- **Step order:** **Cleaning → Integration → Transformation → Reduction.** First step = **data cleaning**. 💡
- **Missing values:** drop, or impute (mean / median / mode). Project uses `SimpleImputer`.
- **Noisy data:** **binning, clustering, regression.** ⚠️ **SMOTE is NOT for noise** — it's for *class imbalance*.
- **Normalization / feature scaling:** rescale features to a range (0–1 or −1 to 1) or z-score
  (mean 0, std 1). **Why:** features contribute *equally* to distance calcs + gradient descent converges faster. 💡
- **Encoding:** categorical → numbers via **one-hot encoding**.
- **Sampling:** **stratified > random** when you need *equal class representation* (imbalanced data). 💡
- **Train/test split:** evaluate on *unseen* data, prevent overfitting. ⚠️ **Fit the scaler on train only**, then
  transform test — otherwise *data leakage*.
- **Feature selection** helps avoid overfitting. **Dimensionality reduction** = PCA *and* feature selection.

---

## 3. Linear Regression & Gradient Descent  *(know the math)*

- **Hypothesis:** $h_\theta(x)=\theta_0+\theta_1 x$ (multivariate: $\theta_0+\theta_1x_1+\dots+\theta_nx_n$).
  **Coefficient** = direction of relationship **and** change in $y$ per one-unit change in $x$. 💡
- **Cost (MSE):** $J(\theta)=\dfrac{1}{2m}\sum_{i=1}^m\big(h_\theta(x^{(i)})-y^{(i)}\big)^2$. ($m$ = #training examples.)
- **Gradient descent:** $\theta_j := \theta_j-\alpha\dfrac{\partial}{\partial\theta_j}J(\theta)$.
  - $\alpha$ too **large** → diverges / oscillates (J *increases*). 💡  $\alpha$ too small → slow.
- **LMS rule:** update is **proportional to the error** — big error → big update; correct prediction → ~no update. 💡
- **Batch vs Stochastic GD:** Batch uses the **whole** training set per step; **Stochastic** updates after **each single example**. 💡
- **Normal equation:** $\theta=(X^TX)^{-1}X^Ty$ — closed form, no $\alpha$, no local minima. ⚠️ inversion is ~$O(n^3)$ →
  **prefer gradient descent when $n$ is huge.** With $m$ examples, $n$ features (+intercept): $X$ is $m\times(n{+}1)$, $y$ is $m\times1$, $\theta$ is $(n{+}1)\times1$.
- **Regularization:** penalize large coefficients → reduces overfitting; also makes $X^TX+\lambda I$ invertible. 💡
- **Residual analysis** checks regression *assumptions*. **MSE** penalizes large errors more; **MAE** grows linearly.
- **Worked:** $J(0,1)=0.4$ on the sunny-days set; scaled $x_2^{(4)}=\frac{1350-1525}{1060}\approx-0.16$. (Re-derive these!)

---

## 4. Logistic Regression

- **For binary classification** (output 0/1). ⚠️ Linear reg → continuous; logistic → binary. 💡
- **Sigmoid:** $g(z)=\dfrac{1}{1+e^{-z}}$, $z=\theta^Tx$. Squashes to $(0,1)$ → a probability. $g(0)=0.5$.
- **Decision boundary:** predict 1 if $h_\theta(x)\ge0.5 \iff \theta^Tx\ge0$.
- **Odds & logit:** odds $=\frac{p}{1-p}$; **logit = log-odds $=\ln\frac{p}{1-p}=\theta^Tx$** (linear in $x$).
  Sigmoid is the *inverse* (log-odds → probability). 💡 (Know this both directions.)
- **Cost = log-loss / cross-entropy** (MSE would be non-convex here).
- **SGD update:** parameters updated **after each single training example**, using that example's gradient. 💡
- **Limitations:** assumes **linearity in the log-odds**; **sensitive to outliers**; needs feature engineering for non-linear patterns.

---

## 5. Decision Trees  *(know entropy + information gain by hand)*

- **Nodes:** **root** (first/top split), **decision/internal**, **leaf** (final, most homogeneous). 💡
- **Objective:** split data into **homogeneous** subsets. ⚠️ Trees use **entropy/info-gain, NOT Euclidean distance**. 💡
- **Entropy:** $H(S)=-\sum_i p_i\log_2 p_i$ — measures **impurity/randomness**. $=0$ pure, $=1$ at 50/50.
- **Information Gain** = entropy(before) − weighted entropy(after). **ID3 picks the highest-IG attribute.**
  Good split = high IG, low impurity, balanced.
- **ID3 loop:** entropy → IG per attribute → pick max IG → split → repeat.
  - **Worked:** $H(\text{Passed})\approx0.92$; $IG_{GPA}=0.26$, $IG_{Studied}=0.46$ → **split on Studied first** (uses $\log_2 3\approx1.6$).
- **Continuous features:** split by a **threshold** (e.g. temp ≤ 30 vs > 30).
- **Pruning** fights overfitting: **pre-pruning** (stop early) vs **post-pruning** (grow full, then trim). 💡
  Subtree selection: cross-validation (tuning set) · statistical test (random-chance) · minimum description length (complexity).
- **ID3 improvements:** handling missing attribute values + post-pruning.
- ⚠️ Midterm keyed "ID3 produces trees as large as possible" as **False** — the *intended* point is that
  good practice uses stopping rules + pruning (greedy ID3 alone would over-grow). Know both readings.

---

## 6. Neural Networks (ANN / MLP)

- **Biological map:** dendrite→**input**, cell body→**summation + activation**, axon→**output**, brain→**ANN**. 💡
- **Neuron:** $v=\sum_i w_i x_i + b$, then $y=f(v)$.
  - **Worked perceptron** (w1=w2=1, b=−1, step activation): fires only when *both* inputs = 1 → **AND gate**.
- **Activation functions:** step, sigmoid, linear (also ReLU, tanh). 💡
  **Why needed:** without a non-linear activation the network collapses to a linear model → can't learn non-linear/XOR patterns.
- **MLP** uses **backpropagation** as its learning algorithm. 💡
- **Backprop:** uses the **chain rule**, propagates error **backward**, updates weights **iteratively** (gradient descent). 💡
- **Learning rate** too high → divergence/oscillation.
- **Dropout** = randomly disable neurons during training → **reduce overfitting**. 💡
- **Data augmentation** = more varied training data → **improves generalization**.
- **Well-trained ANN:** low training error + low validation error + good generalization. ANN tolerates noisy data.

---

## 7. Support Vector Machines

- **Idea:** find the **maximum-margin hyperplane** between two classes.
- **Support vectors** = the boundary-defining points. ⚠️ Removing a **support vector changes** the boundary;
  removing a non-support point (outside the margin) does **not**. 💡
- **C parameter (soft margin):** $C\to\infty$ → zero tolerance for misclassification (hard margin, fully separates if possible).
  Large C → less tolerance (overfit risk); small C → softer margin (generalizes).
- **Kernels:** map data to higher dimensions so a linear separator works. Linear vs **Gaussian (RBF)**.
  Effectiveness depends on **kernel choice + kernel params + C** (all of the above). 💡
- **Gaussian σ² ↔ bias/variance:**
  - **Underfit (high bias)** → **increase C** and/or **decrease σ²**.
  - **Overfit / poor CV (high variance)** → **decrease C** and/or **increase σ²**. 💡
- ⚠️ **Always normalize features before a Gaussian kernel** (it uses squared Euclidean distance → big-range features dominate).

---

## 8. Clustering, GMM & PCA  *(unsupervised)*

- **Clustering = unsupervised** (no labels); needs a dissimilarity function + loss + optimizer.
  - **K-means:** assign each point to **nearest centroid (Euclidean)**, then recompute centroids; repeat.
    Worked: point [−1,2] → distances 2, √8≈2.83, 5 → **cluster 1**.
  - **Hierarchical linkages:** single, complete, average (all valid).
- **GMM:** soft clustering with K Gaussians; estimate **means μ, covariances Σ, mixing coefficients π** per cluster (via EM).
  For N points, D dims: each **μ is D×1**, each **Σ is D×D**. ⚠️ No clustering algorithm always wins (GMM ≠ always best).
- **Spectral clustering:** adjacency $A$ → degree matrix $D$ (diagonal = row-sum of weights) → **Laplacian $L=D-A$**;
  normalized $L_{sym}=I-D^{-1/2}AD^{-1/2}$.
- **PCA = unsupervised dimensionality reduction**, finds directions of **maximum variance**.
  - **u₁ = eigenvector of the covariance matrix Σ with the largest eigenvalue.** 💡
  - **Mean-normalize / scale before PCA.** Choose $k\le n$. Compresses $x\in\mathbb{R}^n\to z\in\mathbb{R}^k$.
  - **Good uses:** compression (memory/speed), visualization (2D/3D). ⚠️ **Using PCA to fix overfitting is a *bad* idea — use regularization.** 💡
  - **Eigenfaces:** eigenvectors of the covariance matrix of (normalized) face images; weighted sums reconstruct a face.

---

## 9. Bayes' Theorem & HMM

- **Bayes:** $P(A|B)=\dfrac{P(B|A)\,P(A)}{P(B)}$ — posterior ∝ likelihood × prior, ÷ evidence.
- **Naive Bayes:** assumes features are conditionally independent ("naive") — fast, good for spam/text.
- **Markov property:** next state depends only on the **current** state. Transition matrix rows sum to 1.
- **HMM $M=(A,B,\pi)$:**
  - **A** = transition probs $a_{ij}=P(q_{t+1}=S_j\mid q_t=S_i)$.
  - **B** = emission probs $b_j(k)=P(v_k\mid q_t=S_j)$.
  - **π** = initial state distribution $\pi_i=P(q_1=S_i)$.
- **Forward algorithm** computes $P(O|M)$ efficiently. $\alpha_1(i)=\pi_i b_i(O_1)$;
  $\alpha_{t+1}(j)=\big[\sum_i\alpha_t(i)a_{ij}\big]b_j(O_{t+1})$; $P(O|M)=\sum_i\alpha_T(i)$.
- **Worked (weather):** $P=a_{33}a_{33}a_{31}a_{11}a_{13}a_{32}a_{23}=0.8\cdot0.8\cdot0.1\cdot0.4\cdot0.3\cdot0.1\cdot0.2=\mathbf{0.0001536}$.
- **Staying in state $i$ exactly $d$ days:** $P(d)=(a_{ii})^{d-1}(1-a_{ii})$; **expected duration $=\dfrac{1}{1-a_{ii}}$.**

---

## 10. Python / NumPy / Pandas  *(the project toolkit — "what's the output?" style)*

- **`np.arange(0,10,2)`** → `[0 2 4 6 8]` (stop is exclusive). **`np.ceil(2.1)`** → `3.0`.
- **Broadcasting:** `np.array([1,2,3]) * 2` → `[2 4 6]`. **`axis=0`** = down columns, **`axis=1`** = across rows.
  `np.abs(M).mean(axis=0)` → mean of each *column*.
- **Boolean → int:** `(s == 'Ghosted').astype(int)` → 1/0 Series. **`np.where(cond, a, b)`** picks elementwise.
- **`len("a,b,c".split(","))`** → `3` (the project's `interest_count` trick).
- **pandas:** `df.shape` → `(rows, cols)`; `df['c'].value_counts()` counts; `value_counts(normalize=True)` → proportions;
  `df.groupby('g')['y'].agg(['mean','count'])`; `df.drop(columns=[...])`.
- **f-strings:** `f"{0.1234:.1%}"` → `"12.3%"`; `f"{3.14159:.2f}"` → `"3.14"`.
- **sklearn pipeline:** `train_test_split(..., stratify=y)`; `StandardScaler` → mean 0/std 1;
  ⚠️ **`fit_transform` on train, `transform` (not fit) on test** — fitting on test = leakage.
  `model.fit` → `model.predict` / `predict_proba`; `GridSearchCV` for hyperparameter tuning.

---

## ⏱️ Final 5-minute scan (the absolute must-knows)

1. Three paradigms; RL → **rewards**. Regression = continuous, classification = category.
2. Preprocessing order = **clean → integrate → transform → reduce**; noise → bin/cluster/regress; **SMOTE = imbalance**.
3. Normalization = equal contribution + faster GD. Stratified sampling = equal class representation.
4. $J(\theta)=\frac{1}{2m}\sum(h-y)^2$; GD step; **α too big → diverge**; **LMS update ∝ error**; batch vs stochastic.
5. Normal equation $(X^TX)^{-1}X^Ty$ — no α, $O(n^3)$, use GD for huge $n$.
6. Sigmoid $\frac{1}{1+e^{-z}}$; logit = log-odds = $\theta^Tx$; logistic = binary classification.
7. Entropy $-\sum p\log_2 p$; **highest info gain wins**; pre- vs post-pruning; trees use entropy NOT distance.
8. Neuron = Σ then activation; **need non-linear activation**; MLP → **backprop (chain rule, error backward)**; dropout → less overfit.
9. SVM: support vectors define the boundary; underfit → ↑C/↓σ², overfit → ↓C/↑σ²; normalize before RBF.
10. PCA = max-variance eigenvector (largest eigenvalue), unsupervised, scale first, **not** for overfitting.
11. HMM = $(A,B,\pi)$; forward algorithm for $P(O|M)$; sequence prob = product of transitions.
12. `fit_transform` on train only; `value_counts`, `astype(int)`, `axis=0` vs `1`, f-string `%`/`f`.

**You've got this.** Re-derive the four worked calculations (entropy/IG, cost/GD, sigmoid, HMM sequence) on
blank paper one last time, then rest. 🎓
