# Chapter 8 — Unsupervised Learning: Clustering, GMM & PCA

> **Day 6 of 7** (afternoon) · Prerequisites: Ch 1–2 · Est. study time: ~80 min

**In one sentence:** When your data has *no labels*, you can still find structure — group similar points together (clustering), model them as a blend of bell curves (GMM), or squeeze many features into a few that capture most of the variation (PCA).

## Why this matters
Everything before this chapter was **supervised**: you had inputs *and* the correct answers. Unsupervised learning drops the answers — you only have the inputs and must discover the structure yourself. The exam loves this contrast, and it loves a handful of mechanical calculations: the k-means assignment step, GMM parameter dimensions, and the spectral-clustering Laplacian. We'll nail all three.

## Learning goals
- [ ] Explain why clustering is unsupervised and what a dissimilarity function / loss / algorithm each do
- [ ] Run one step of k-means by hand (assign a point to the nearest centroid)
- [ ] Name the three hierarchical-clustering linkages
- [ ] State a GMM's parameters (μ, Σ, π) and give their dimensions for given N and D
- [ ] Build an adjacency matrix, degree matrix, and Laplacian $L = D - A$
- [ ] Explain what PCA does, when to use it (and when *not* to), and what u₁ is
- [ ] Describe eigenfaces at a high level

---

## 1. What "clustering" actually means

**Intuition.** You're handed a pile of unlabelled points and asked: *which ones belong together?* No teacher tells you the right groups — you decide based on how *similar* points are. That's the whole game.

Three ingredients make a clustering method work:

1. **A dissimilarity (or similarity) function** — a way to say how far apart two samples are (e.g. Euclidean distance).
2. **A loss function** — a score that evaluates how good a particular grouping is.
3. **An algorithm** — a procedure that adjusts the groupings to optimize that loss.

> 💡 **Exam tip:** Clustering is **unsupervised** — *no labels, you do not label the raw data*. This is the single most-tested fact in this chapter.

> ⚠️ **Common trap:** "The most likely cluster is calculated according to *hidden states*." **False.** Hidden states belong to sequence models (Hidden Markov Models), *not* to standard clustering. Standard clustering works on distances/variance. (This is the false statement in tutorial Clustering Q1.)

---

## 2. K-means: the workhorse

**Intuition.** Pick *K* "centroids" (cluster centers). Repeat two steps until nothing moves:

- **Assignment step** — assign each point to the **nearest centroid** (by Euclidean distance).
- **Update step** — move each centroid to the **mean** of the points assigned to it.

That's it — assign, average, assign, average. Each round can only lower the loss, so it converges.

**Euclidean distance** between a point $x = [x_1, x_2]$ and a centroid $\mu = [\mu_1, \mu_2]$:

$$d(x, \mu) = \sqrt{(x_1 - \mu_1)^2 + (x_2 - \mu_2)^2}$$

### Worked example (tutorial Clustering Q4)

Three centroids: $\mu_1 = [1, 2]$, $\mu_2 = [-3, 0]$, $\mu_3 = [4, 2]$. A point $x^{(i)} = [-1, 2]$. After the assignment step, what cluster $c^{(i)}$ does it land in?

Compute the distance to each centroid:

$$d(x, \mu_1) = \sqrt{(-1-1)^2 + (2-2)^2} = \sqrt{4 + 0} = \sqrt{4} = 2$$

$$d(x, \mu_2) = \sqrt{(-1-(-3))^2 + (2-0)^2} = \sqrt{2^2 + 2^2} = \sqrt{4+4} = \sqrt{8} \approx 2.83$$

$$d(x, \mu_3) = \sqrt{(-1-4)^2 + (2-2)^2} = \sqrt{25 + 0} = 5$$

Smallest distance is **2**, to $\mu_1$. So $c^{(i)} = 1$ — the point joins **Cluster 1**.

> 💡 **Exam tip:** You don't even need the square root to *compare* distances — whichever has the smallest squared distance wins. Here $4 < 8 < 25$, same answer. But show the full arithmetic if the question asks for the distance value.

> ⚠️ **Common trap:** Don't forget to subtract in the right order and *square* — sign errors vanish under the square, but a forgotten square inflates the wrong term.

---

## 3. Hierarchical clustering & linkages

**Intuition.** Instead of fixing *K* upfront, build a tree. Start with every point as its own cluster, then repeatedly merge the two *closest* clusters until one big cluster remains. The result is a **dendrogram** you can cut at any level.

The catch: "closest *cluster*" needs a definition, because clusters contain many points. That's a **linkage** — how you measure dissimilarity *between two clusters*:

| Linkage | How it measures cluster distance |
|---|---|
| **Single** | distance between the **two closest** points (one from each cluster) |
| **Complete** | distance between the **two farthest** points |
| **Average** | the **average** distance over all cross-cluster point pairs |

> 💡 **Exam tip:** Tutorial Clustering Q3 asks which metrics are available for hierarchical clustering — the answer is **All of the above** (single, complete, and average linkage are all valid).

---

## 4. Gaussian Mixture Models (GMM): soft clustering

**Intuition.** K-means makes a *hard* call: each point belongs to exactly one cluster. A GMM is gentler — it says each point belongs to cluster 1 *with probability 0.7*, cluster 2 *with 0.3*, and so on. It models the data as a **mixture of K Gaussian (bell-curve) distributions**.

Each Gaussian is described by:

- **Mean vector $\mu$** — where the bell is centered.
- **Covariance matrix $\Sigma$** — its shape/spread/orientation.
- **Mixing coefficient $\pi$** — how much weight that Gaussian carries in the mixture (the $\pi$'s sum to 1).

So the parameters to estimate are **μ, Σ, and π, one set per cluster.**

**How it's trained (brief):** the **EM algorithm** (Expectation–Maximization). *E-step:* given current parameters, compute the probability each point came from each Gaussian (soft assignment). *M-step:* given those soft assignments, re-estimate μ, Σ, π. Repeat. *(The detailed EM math is beyond the exam — know the name and the two-step idea.)*

### Parameter dimensions (tutorial GMM Q2.III)

If you have $N = 1000$ points, each $D = 3$ dimensional (so $X$ is $1000 \times 3$), then **per cluster**:

- mean vector $\mu$ is $3 \times 1$ (one entry per dimension),
- covariance matrix $\Sigma$ is $3 \times 3$ (it relates every pair of dimensions),
- mixing coefficient $\pi$ is a single scalar.

> 💡 **Exam tip:** The rule is general: $\mu$ is $D \times 1$ and $\Sigma$ is $D \times D$, **regardless of N**. N is the number of points, not a parameter dimension.

> ⚠️ **Common trap (tutorial GMM Q1 = False):** "GMM *always* outperforms k-means and hierarchical clustering." No algorithm always wins. GMMs fail when the data isn't well-described by a few Gaussian blobs. This is a flavour of the **No Free Lunch** idea.

---

## 5. Spectral clustering: clustering on a graph

**Intuition.** Sometimes clusters aren't round blobs (think two interlocking crescents). Spectral clustering reframes the data as a **graph**: points are vertices, and edge *weights* say how strongly two points are connected. It then uses linear algebra (eigenvectors of a special matrix) to cut the graph into clusters. *(The eigenvector/cut step itself is beyond the exam — focus on building the matrices below.)*

You build three matrices from the graph.

**Adjacency matrix $A$** — entry $A_{ij}$ = weight of the edge between vertex $i$ and $j$ (0 if no edge). It's symmetric, with a 0 diagonal.

**Degree matrix $D$** — a **diagonal** matrix where each diagonal entry is the **sum of that row's weights** in $A$ (the vertex's total connection strength). Off-diagonal entries are 0.

**Unnormalized Laplacian:**
$$L = D - A$$

**Normalized (symmetric) Laplacian:**
$$L_{sym} = I - D^{-1/2} A D^{-1/2}$$

### Worked example (tutorial Spectral Clustering, 6-vertex graph)

The adjacency matrix for vertices A–F:

|   | A | B | C | D | E | F |
|---|---|---|---|---|---|---|
| **A** | 0 | 8 | 6 | 1 | 0 | 0 |
| **B** | 8 | 0 | 8 | 0 | 0 | 0 |
| **C** | 6 | 8 | 0 | 0 | 2 | 0 |
| **D** | 1 | 0 | 0 | 0 | 8 | 8 |
| **E** | 0 | 0 | 2 | 8 | 0 | 7 |
| **F** | 0 | 0 | 0 | 8 | 7 | 0 |

**Degree of A** = sum of row A = $8 + 6 + 1 + 0 + 0 = 15$, so $D_{AA} = 15$.

The full diagonal: $D_{AA}=15,\ D_{BB}=16,\ D_{CC}=16,\ D_{DD}=17,\ D_{EE}=17,\ D_{FF}=15$.

Then the Laplacian's diagonal is $L_{ii} = D_{ii} - A_{ii} = D_{ii} - 0 = D_{ii}$, and off-diagonals are $L_{ij} = -A_{ij}$. For example, $L_{AA} = 15$ and $L_{AB} = -8$.

For the normalized Laplacian, the $D^{-1/2}$ entry for vertex A is $\tfrac{1}{\sqrt{15}}$ (because $D_{AA}=15$).

> 💡 **Exam tip:** The degree of a vertex is just the **sum of its edge weights**. Get the degree right and the rest ($L = D - A$, then $D^{-1/2}$) falls out.

> ⚠️ **Common trap:** The original tutorial sheet mis-stated A's degree as 13. The edges 8, 6, 1 sum to **15** — always recompute, don't trust a printed value.

---

## 6. PCA: dimensionality reduction by maximum variance

**Intuition.** You have data with many features (dimensions), and they're often redundant or noisy. PCA finds a few new **directions** along which the data **varies the most**, and projects the data onto them. You keep the "interesting" directions and drop the flat, uninformative ones — fewer features, most of the information retained.

PCA is **unsupervised** (no labels) and is a form of **dimensionality reduction** — it creates **new features** as linear combinations of the originals, and drops the less-informative projections.

**What u₁ is.** The **first principal direction** $u_1$ is the **eigenvector of the covariance matrix $\Sigma$ that corresponds to the largest eigenvalue**. Large eigenvalue = much variance along that direction = the most informative axis. $u_2$ is the eigenvector with the next-largest eigenvalue, and so on.

**The recipe (high level):**
1. **Mean-normalize** the features (subtract each feature's mean so it has zero mean), and **feature-scale** if features are on different scales.
2. Compute the covariance matrix $\Sigma$.
3. Take its eigenvectors; keep the top $k$ (largest eigenvalues).
4. Project each $x \in \mathbb{R}^n$ onto those $k$ directions to get $z \in \mathbb{R}^k$ (with $k \le n$).

> 💡 **Exam tip:** **Always mean-normalize before PCA** — *even if all features are already on similar scales* (tutorial PCA Q8). If scales differ, feature-scale too. And you only ever pick $k \le n$.

### When to use PCA (and when not to)

**Recommended uses** (tutorial PCA Q7):
- **Data compression** — fewer dimensions → less memory/disk, and supervised algorithms run faster.
- **Visualization** — reduce to 2D or 3D so you can actually plot it.

**Bad idea** (tutorial PCA Q11 / Q2):
- Using PCA **to prevent overfitting**. It might accidentally help by removing features, but it throws away information *blindly* (it never looks at the labels). The right tool for overfitting is **regularization**.

> ⚠️ **Common trap:** "Run PCA every time before any ML algorithm." No — use it when you genuinely want to compress / find latent features / visualize, not reflexively.

> 💡 **Exam tip (midterm Q29):** "Which is an example of dimensionality reduction?" → **PCA *and* feature selection** are both dimensionality reduction. Sampling is not.

---

## 7. Eigenfaces (PCA applied to faces) — short

A classic PCA application for face recognition:

- Face images are first **normalized** to line up eyes, mouths, and other features.
- **Eigenfaces** are the **eigenvectors of the covariance matrix** of the set of face images — "standardized face ingredients" derived from the **statistical analysis** of many faces.
- Any face can be approximated as a **weighted** sum of eigenfaces (a grayscale rendering).
- This lets you store/compare faces compactly → used for **compression** and **identification**.

> 💡 **Exam tip:** Eigenfaces = eigenvectors of the **covariance matrix** of face images, combined by **weighting**. These two blanks are exactly what the tutorial tests (Q12–16).

---

## Key formulas / facts at a glance

| Thing | Formula / Rule | Plain meaning |
|---|---|---|
| Clustering type | Unsupervised | No labels; you discover groups |
| Euclidean distance | $\sqrt{\sum_d (x_d - \mu_d)^2}$ | Straight-line distance to a centroid |
| K-means | assign to nearest centroid, then update centroid = mean | Repeat until stable |
| Hierarchical linkages | single / complete / average | closest / farthest / mean cross-cluster distance |
| GMM parameters | μ ($D{\times}1$), Σ ($D{\times}D$), π (scalar) per cluster | Soft clustering via mixture of Gaussians |
| GMM training | EM (E-step soft-assign, M-step re-estimate) | Iterative fit |
| Degree | $D_{ii} = \sum_j A_{ij}$ | Sum of a vertex's edge weights |
| Laplacian | $L = D - A$ | Unnormalized graph Laplacian |
| Normalized Laplacian | $L_{sym} = I - D^{-1/2} A D^{-1/2}$ | Scale-corrected Laplacian |
| PCA u₁ | eigenvector of Σ with largest eigenvalue | Direction of maximum variance |
| Before PCA | mean-normalize (always); feature-scale if needed | Center & scale first |
| PCA output | $x \in \mathbb{R}^n \to z \in \mathbb{R}^k$, $k \le n$ | Compress to k dimensions |

## Exam tips — quick recap
- Clustering & PCA are both **unsupervised** — no labels.
- "Hidden states" → HMMs, **not** clustering.
- K-means: compare *distances*, smallest wins (squared distance is enough to compare).
- Hierarchical linkages: single, complete, average → "all of the above".
- GMM params per cluster: μ is $D{\times}1$, Σ is $D{\times}D$; N is irrelevant to those dimensions.
- No clustering algorithm always wins (GMM Q1 = False).
- Degree = sum of edge weights; $L = D - A$.
- PCA u₁ = eigenvector of Σ with the largest eigenvalue (= max variance).
- Always mean-normalize before PCA; pick $k \le n$.
- PCA for overfitting = **bad idea**; use regularization.
- PCA & feature selection are both dimensionality reduction (midterm Q29).

## Self-check
1. Why is clustering "unsupervised," and which statement about clustering involves a concept that actually belongs to HMMs?
2. Given centroids $\mu_1=[0,0]$, $\mu_2=[2,2]$ and point $x=[2,1]$, which cluster does $x$ join?
3. For a GMM on $N=500$ points in $D=4$ dimensions, what are the dimensions of one cluster's μ and Σ?
4. A vertex has edges of weight 3, 5, and 4. What is its degree, and what is its diagonal entry in $D^{-1/2}$?
5. What is $u_1$ in PCA, and why is using PCA to fight overfitting a bad idea?

**→ Now test yourself with [`quiz.md`](./quiz.md).**
