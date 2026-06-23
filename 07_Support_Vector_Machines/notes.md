# Chapter 7 — Support Vector Machines (SVM)

> **Day 6 of 7** (morning) · Prerequisites: Ch 4 (Classification) · Est. study time: ~60–75 min

**In one sentence:** An SVM finds the single straight boundary that separates two classes with the *widest possible gap* on either side, and the kernel trick lets it bend that boundary into curves when the data isn't linearly separable.

## Why this matters

SVM is the classic "draw the best line between two groups" classifier, and the exam loves it because it forces you to reason geometrically (support vectors, margins) instead of just plugging into a formula. It's also where the **bias–variance trade-off** shows up most cleanly through two knobs — `C` and the kernel width `σ²`. In your group project, the SVM was a linear classifier (`SGDClassifier(loss='hinge')`) used as one of the five models, so you'll recognise the practical version too.

## Learning goals
- [ ] Explain the maximum-margin idea and why a wider margin generalises better.
- [ ] Define support vectors and predict when removing a point changes the boundary.
- [ ] Distinguish hard vs soft margin and explain what the `C` parameter does (including `C → ∞`).
- [ ] Explain the kernel trick, and contrast a linear kernel with a Gaussian (RBF) kernel.
- [ ] Map `C` and `σ²` onto the bias–variance trade-off and pick the right direction to fix under/overfitting.
- [ ] Explain why you must normalise features before a Gaussian kernel.
- [ ] List concrete ways to fix an underfitting model.

---

## 1. The big idea: the maximum-margin hyperplane

Imagine two clouds of points — red and blue — that you can separate with a straight line. There are infinitely many lines that do the job. Which is *best*?

SVM's answer: the line that sits **as far as possible from the nearest point of either class**. Picture widening a road down the middle of the gap. The **margin** is the width of that road. The best boundary is the one with the **widest margin**.

- The boundary itself is called the **hyperplane** (a line in 2D, a plane in 3D, a "hyperplane" in higher dimensions).
- Wider margin → more breathing room → new points are less likely to land on the wrong side → **better generalisation**.

Formally, for a point $\mathbf{x}$ the model computes a score $\mathbf{w}^\top \mathbf{x} + b$ and predicts class by its sign. The margin is $\frac{2}{\lVert \mathbf{w}\rVert}$, so "maximise the margin" means "minimise $\lVert \mathbf{w}\rVert$" subject to every point being on the correct side. You won't be asked to solve this optimisation by hand — but know that *maximising margin = minimising the weight norm*.

> 💡 **Exam tip:** "Maximum-margin classifier" and "large-margin classifier" both mean SVM. The defining property is the *widest gap*, not just *any* separating line.

## 2. Support vectors — the points that actually matter

Here's the surprising part. Most of your data points are irrelevant to where the boundary ends up. Only the points **right on the edge of the margin** — the ones closest to the boundary — actually pin it in place. Those points are the **support vectors**.

Think of the margin as a tent and the support vectors as the poles holding it up. Move or remove a pole and the tent shifts. Remove a chair sitting far away inside the tent and nothing happens.

This gives two clean rules (straight from tutorial Q1):

- **Remove a support vector → the boundary CHANGES.** It was one of the points defining the margin, so the optimal line has to be recomputed. ✅
- **Remove a non-support-vector (a point sitting outside the margin) → the boundary does NOT change.** That point had zero influence on the geometry. ✅

> 💡 **Exam tip:** The number of support vectors is usually far smaller than the number of training points. That's why SVMs are memory-efficient at prediction time — only the support vectors are needed.

> ⚠️ **Common trap:** Don't assume "removing a point always changes the boundary." It only changes if that point was a support vector. Points comfortably outside the margin are free to delete.

## 3. Hard margin vs soft margin, and the `C` parameter

The clean picture above assumes the data is **perfectly separable** with a straight line. Real data usually isn't — there are outliers and overlap. Two strategies:

- **Hard margin:** demand that *every* point be correctly classified and outside the margin. Zero tolerance for mistakes. Only works if the data is perfectly separable, and is very sensitive to outliers.
- **Soft margin:** allow some points to sit inside the margin or even on the wrong side, paying a penalty for each violation. This is what's used in practice.

The **`C` parameter** controls how harshly margin violations are penalised:

| `C` | Penalty for misclassification | Margin | Behaviour |
|---|---|---|---|
| Large `C` | High (intolerant) | Narrow | Tries hard to classify every point right → can **overfit** |
| Small `C` | Low (tolerant) | Wide | Accepts some errors for a simpler, wider margin → better **generalisation** |

**The extreme: `C → ∞`** (tutorial Q5). An infinite `C` makes any misclassification infinitely costly, so the model refuses to tolerate a single violation. This is exactly a **hard margin**: *the optimal hyperplane, if one exists, will be the one that completely separates the data.* (Answer: **A**.)

> 💡 **Exam tip:** Memorise the mapping. **Large `C` → less tolerance → risk of overfitting (high variance).** **Small `C` → more tolerance → smoother, better generalisation.** `C → ∞` = hard margin = full separation if possible.

> ⚠️ **Common trap:** Larger `C` does *not* mean a wider margin. It's the opposite — large `C` shrinks the margin to avoid mistakes.

## 4. The kernel trick — bending the boundary

What if no straight line can separate the classes (e.g. one class forms a ring around the other)? The **kernel trick** maps the data into a higher-dimensional space where a straight separator *does* exist — without ever explicitly computing those new coordinates. A linear boundary in the high-dimensional space corresponds to a curved boundary back in the original space.

Two kernels you must know:

- **Linear kernel:** no transformation; just the straight-line SVM of Sections 1–3. Use it when the data is (roughly) linearly separable or has very many features. *(This is what your project used: `SGDClassifier(loss='hinge')` is a linear SVM.)*
- **Gaussian / RBF kernel:** measures similarity between two points by how close they are:

$$K(\mathbf{x}, \mathbf{x}') = \exp\!\left(-\frac{\lVert \mathbf{x} - \mathbf{x}' \rVert^2}{2\sigma^2}\right)$$

Here $\lVert \mathbf{x} - \mathbf{x}'\rVert^2$ is the **squared Euclidean distance** between the points. Two identical points give $K = 1$; points far apart give $K \approx 0$. This lets the SVM carve out flexible, curvy regions.

**What determines whether an SVM works well?** (tutorial Q6) The answer is **all of the above**: the **choice of kernel**, the **kernel parameters** (e.g. `σ²`), *and* the **soft-margin parameter `C`**. Tuning all three together is the job.

> 💡 **Exam tip:** "Kernel trick" = separate in higher dimensions without computing the higher-dimensional coordinates. The Gaussian kernel is also called **RBF** (Radial Basis Function).

## 5. The Gaussian kernel's `σ²` and the bias–variance trade-off

The Gaussian kernel has a width parameter `σ²`. It controls how far each point's influence reaches:

- **Small `σ²`** → narrow influence → each support vector only affects its immediate neighbourhood → a **wiggly, complex, high-variance** boundary that hugs the training data. Good for fixing **underfitting**.
- **Large `σ²`** → wide influence → features blend together → a **smooth, simple, high-bias** boundary. Good for fixing **overfitting**.

Now combine with `C` into one master table — this is the most exam-relevant thing in the chapter:

| Symptom | Diagnosis | Fix |
|---|---|---|
| **Underfitting** (does poorly even on training set) | High **bias** — model too simple | **Increase `C`** and/or **decrease `σ²`** (make it more flexible) |
| **Overfitting** (great on training, poor on cross-validation) | High **variance** — model too complex | **Decrease `C`** and/or **increase `σ²`** (make it smoother) |

This is precisely tutorial Q2:

- **Underfitting case:** increase `C` (penalise errors harder → more intricate boundary) and decrease `σ²` (narrow the Gaussian → sensitive to local variation).
- **Poor cross-validation (overfitting) case:** decrease `C` (soften the margin → generalise) and increase `σ²` (broaden the kernel → smoother boundary).

> 💡 **Exam tip:** A memory hook — **C and σ² pull in opposite directions on complexity.** Big `C` *adds* complexity; big `σ²` *removes* it. To fix underfitting you push toward complexity (↑C, ↓σ²); to fix overfitting you push toward simplicity (↓C, ↑σ²).

> ⚠️ **Common trap:** It's easy to flip `σ²`. Remember: **bigger `σ²` = smoother = simpler**. Underfitting already means "too simple," so you *decrease* `σ²` there, not increase it.

## 6. Always normalise before a Gaussian kernel

The Gaussian kernel depends on the **squared Euclidean distance** $\lVert \mathbf{x} - \mathbf{x}'\rVert^2$. If one feature is measured in a huge range (say, income in the tens of thousands) and another in a tiny range (say, a ratio between 0 and 1), the large-range feature dominates the distance completely — the SVM becomes **blind** to the small-scale features (tutorial Q4).

So: **normalise (or standardise) every feature to a comparable scale before applying a Gaussian kernel.** Your project did exactly this with `StandardScaler` in the preprocessing pipeline.

> 💡 **Exam tip:** The reason is specifically that the kernel uses *distance*. Any distance-based method (k-NN, k-means, RBF SVM) needs scaled features.

## 7. Fixing an underfitting model

Tutorial Q3 sets up logistic regression underfitting on a dataset (`n = 10` features, `m = 5000` examples). The cure for **underfitting (high bias)** is to **add non-linearity / complexity**. Promising steps:

- ✅ Use an **SVM with a Gaussian kernel** (adds non-linearity).
- ✅ **Create / add polynomial features** (adds non-linearity).
- ✅ Use a **neural network with many hidden units** (adds capacity).

What does **not** help:

- ❌ SVM with a **linear kernel and no new features** — still linear, still underfits.
- ❌ **Reducing the number of training examples** — fewer data never fixes high bias (and tends to make things worse).

> 💡 **Exam tip:** Underfit = add complexity (more features, non-linear kernel, bigger network). Overfit = remove complexity / regularise / get more data. Memorise the two directions.

## 8. Pros, cons, and when to reach for SVM

**Pros**
- Effective in high-dimensional spaces (even when features > examples).
- Memory-efficient at prediction — only support vectors matter.
- Flexible via kernels (linear, RBF, polynomial).
- Maximising the margin gives strong generalisation.

**Cons**
- Doesn't scale gracefully to very large datasets (the standard kernel SVM is slow — which is exactly why the project used `SGDClassifier` for a *fast linear* SVM on 50,000 rows).
- Sensitive to the choice of `C`, kernel, and kernel parameters (lots of tuning).
- Requires feature scaling.
- No natural probability output (decision scores, not probabilities, by default).

**Good choice when:** you have a clear margin of separation, a moderate-sized dataset, many features, and you're doing binary classification.

---

## Key formulas / facts at a glance

| Thing | Formula / Rule | Plain meaning |
|---|---|---|
| Decision score | $\operatorname{sign}(\mathbf{w}^\top\mathbf{x}+b)$ | Which side of the boundary a point is on |
| Margin width | $\frac{2}{\lVert\mathbf{w}\rVert}$ | Maximise margin ⇔ minimise $\lVert\mathbf{w}\rVert$ |
| Support vector | Point on the margin edge | Defines the boundary; removing it moves the boundary |
| Non-support-vector | Point outside the margin | No effect; removing it leaves the boundary unchanged |
| `C` (soft-margin) | Large = intolerant/narrow margin; small = tolerant/wide | Large `C` ⇒ overfit risk; small `C` ⇒ generalise |
| `C → ∞` | Hard margin | Fully separates the data if a separating hyperplane exists |
| Gaussian (RBF) kernel | $\exp\!\big(-\frac{\lVert\mathbf{x}-\mathbf{x}'\rVert^2}{2\sigma^2}\big)$ | Similarity by distance; needs scaled features |
| `σ²` | Small = wiggly/high-variance; large = smooth/high-bias | Underfit → ↓σ²; overfit → ↑σ² |
| Fix underfitting | ↑`C` and/or ↓`σ²`; add features / non-linearity | Add complexity |
| Fix overfitting | ↓`C` and/or ↑`σ²` | Reduce complexity |
| SVM effectiveness | Kernel choice + kernel params + `C` | All three must be tuned |

## Exam tips — quick recap
- Maximum-margin = widest road between classes; that's *the* definition of SVM.
- Only **support vectors** define the boundary; deleting a non-support-vector changes nothing.
- **Large `C` → less tolerance → overfit; small `C` → more tolerance → generalise; `C → ∞` → hard margin.**
- Larger `C` *narrows* the margin (not widens it).
- **Bigger `σ²` = smoother = simpler.** Underfit → ↓σ² (and ↑C); overfit → ↑σ² (and ↓C).
- Gaussian kernel uses Euclidean distance → **always normalise features first.**
- Fix underfitting by *adding* complexity: RBF SVM, polynomial features, bigger neural net.
- SVM effectiveness depends on kernel + kernel params + `C` (all of the above).

## Self-check
1. You remove a single training point and the decision boundary doesn't move at all. What does that tell you about the point?
2. Your RBF-SVM nails the training set but flops on cross-validation. Which way do you move `C` and `σ²`?
3. Why is `C → ∞` the same thing as a hard-margin SVM?
4. Why must you scale features before using a Gaussian kernel, but it matters less for some other models?
5. A logistic-regression model underfits. Name three changes that could help and one that won't.

**→ Now test yourself with [`quiz.md`](./quiz.md).**
