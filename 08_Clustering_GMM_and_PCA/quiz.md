# Chapter 8 Quiz — Unsupervised Learning: Clustering, GMM & PCA

> Format mirrors the WIA1006 quiz/midterm: multiple-choice + true/false + worked problems.
> Try it closed-book first. Answers + explanations are at the very bottom.

## Section A — Multiple Choice

**A1.** Which statement is **false** regarding clustering?
- a. A dissimilarity (similarity) function between samples is provided.
- b. Groupings of samples into clusters are evaluated by a loss function.
- c. The most likely cluster is calculated according to hidden states.
- d. There is an algorithm that optimizes this loss function.

**A2.** Which statement is **true** regarding clustering?
- a. Clustering is unsupervised learning.
- b. We must label the raw data.
- c. Clustering requires a separate test set of labels to validate clusters.
- d. Clustering predicts a continuous target value.

**A3.** Which metrics are available for measuring dissimilarity between two clusters in hierarchical clustering?
- a. Single linkage
- b. Complete linkage
- c. Average linkage
- d. All of the above

**A4.** Principal Component Analysis (PCA) is an example of:
- a. Supervised learning
- b. Unsupervised learning
- c. Semi-supervised learning
- d. Reinforcement learning

**A5.** What does PCA do, after all?
- a. Creates clusters so you know the classes.
- b. Predicts your target with high efficiency.
- c. Reduces the dimensionality of the data and creates new features from the given feature set.
- d. Gives you the highest possible number of features.

**A6.** The first principal direction $u_1$ must be the ___ of $\Sigma$ that corresponds to the largest ___.
- a. eigenvalue / eigenvector
- b. eigenvector / eigenvalue
- c. determinant / trace
- d. row / column

**A7.** PCA linearly combines the variables and lets you drop projections that are:
- a. More informative
- b. Less informative
- c. Perfectly correlated with the target
- d. Already mean-normalized

**A8.** Which are **recommended** applications of PCA? (Select all that apply.)
- a. Data compression to save memory/disk
- b. Reducing data to 2D/3D for visualization
- c. Preventing overfitting
- d. Speeding up a supervised learning algorithm by reducing input dimension

**A9.** Which is an example of dimensionality reduction? *(midterm Q29)*
- a. Feature selection
- b. PCA
- c. PCA & feature selection
- d. Sampling

**A10.** When should you use PCA?
- a. Every time, before any machine learning algorithm.
- b. When you want to find latent features and reduce dimensionality.
- c. Only when you have an overfitting problem.
- d. Only when the data is small with few features.

**A11.** In spectral clustering, the diagonal entry $D_{ii}$ of the degree matrix equals:
- a. The number of vertices in the graph
- b. The sum of the edge weights connected to vertex $i$
- c. Always 1
- d. The largest edge weight at vertex $i$

**A12.** Eigenfaces are the eigenvectors of the ___ of the vector space of human faces.
- a. mean vector
- b. adjacency matrix
- c. covariance matrix
- d. Laplacian matrix

## Section B — True / False

**B1.** Clustering is a supervised learning technique that requires labelled data. — (True / False)

**B2.** Of the clustering algorithms covered in class, GMM used for clustering *always* outperforms k-means and hierarchical clustering. — (True / False)

**B3.** Even if all input features are on very similar scales, you should still perform mean normalization before running PCA. — (True / False)

**B4.** Given input $x \in \mathbb{R}^n$, it only makes sense to run PCA with values of $k$ such that $k \le n$. — (True / False)

**B5.** Using PCA to prevent overfitting is a good idea and should be preferred over regularization. — (True / False)

**B6.** When properly weighted, eigenfaces can be summed together to create an approximate grayscale rendering of a human face. — (True / False)

**B7.** In a GMM, the covariance matrix $\Sigma$ for a $D$-dimensional dataset is $D \times D$ regardless of the number of data points $N$. — (True / False)

## Section C — Worked Problems

**C1. (K-means assignment step.)** You have three cluster centroids $\mu_1 = [1, 2]$, $\mu_2 = [-3, 0]$, $\mu_3 = [4, 2]$ and a training example $x^{(i)} = [-1, 2]$. After one assignment step, what is $c^{(i)}$? Show the Euclidean distance to each centroid.

**C2. (GMM parameter dimensions.)** A GMM is fit to a dataset of $N = 1000$ three-dimensional points ($D = 3$), so $X$ is a $1000 \times 3$ matrix. List the parameters to be estimated and give the dimensions of the mean vector $\mu$ and the covariance matrix $\Sigma$ for **one** cluster.

**C3. (Laplacian diagonal entry.)** A 6-vertex weighted graph has the adjacency matrix below. Compute the degree-matrix diagonal entry $D_{AA}$, the Laplacian diagonal entry $L_{AA}$, and the $D^{-1/2}$ entry for vertex A.

|   | A | B | C | D | E | F |
|---|---|---|---|---|---|---|
| **A** | 0 | 8 | 6 | 1 | 0 | 0 |
| **B** | 8 | 0 | 8 | 0 | 0 | 0 |
| **C** | 6 | 8 | 0 | 0 | 2 | 0 |
| **D** | 1 | 0 | 0 | 0 | 8 | 8 |
| **E** | 0 | 0 | 2 | 8 | 0 | 7 |
| **F** | 0 | 0 | 0 | 8 | 7 | 0 |

---

## ✅ Answer Key & Explanations

**A1 — (c)** "Hidden states" is a concept from sequence models (Hidden Markov Models), not standard clustering. Clustering works on distances and variance. The other three (a, b, d) correctly describe the dissimilarity function, loss, and optimizing algorithm.

**A2 — (a)** Clustering is unsupervised: you do **not** label the raw data (rules out b), there's no target to predict (rules out d), and clusters are validated by the loss/structure, not a label test set (rules out c).

**A3 — (d)** Single, complete, and average linkage are all valid ways to measure between-cluster dissimilarity (closest pair, farthest pair, and average pair respectively).

**A4 — (b)** PCA uses no labels — it works purely from the structure (variance) of the inputs, making it unsupervised.

**A5 — (c)** PCA reduces dimensionality and produces new features that are linear combinations of the original ones. It does not classify, predict, or maximize feature count.

**A6 — (b)** $u_1$ is the **eigenvector** of $\Sigma$ corresponding to the **largest eigenvalue** — the direction of maximum variance. (a) reverses the two terms.

**A7 — (b)** PCA keeps the high-variance (informative) directions and drops the **less informative** projections.

**A8 — (a, b, d)** Compression, visualization, and speeding up supervised learning are all recommended. **(c) is not** — using PCA to prevent overfitting is discouraged; use regularization instead.

**A9 — (c)** Both PCA and feature selection reduce the number of features, so both are dimensionality reduction. Sampling reduces the number of *rows*, not dimensions.

**A10 — (b)** Use PCA when you genuinely want latent features / lower dimensionality. It is not a reflexive pre-step for every algorithm (a), not the proper overfitting fix (c), and pointless on already-tiny data (d).

**A11 — (b)** A vertex's degree is the sum of the weights of edges connected to it. The degree matrix is diagonal with these sums on the diagonal.

**A12 — (c)** Eigenfaces are the eigenvectors of the **covariance matrix** of the face images.

**B1 — False.** Clustering is **unsupervised** — it uses no labels.

**B2 — False.** No clustering algorithm always wins (a "no free lunch" result). GMMs fail when the data can't be modelled by a few Gaussians.

**B3 — True.** Mean normalization (zero-mean each feature) is recommended before PCA regardless of scale; feature scaling is additionally needed when scales differ.

**B4 — True.** PCA compresses $x \in \mathbb{R}^n$ to $z \in \mathbb{R}^k$; you can't keep more directions than you started with, so $k \le n$.

**B5 — False.** Using PCA to fight overfitting is a **bad idea** — it discards information without looking at labels. Use regularization instead.

**B6 — True.** A face is approximated as a *weighted* sum of eigenfaces, producing a grayscale rendering.

**B7 — True.** $\Sigma$ relates each pair of the $D$ dimensions, so it is $D \times D$. $N$ (the number of points) does not change this.

---

### C1 — Solution: $c^{(i)} = 1$ (Cluster 1)

Use Euclidean distance $d(x,\mu) = \sqrt{(x_1-\mu_1)^2 + (x_2-\mu_2)^2}$ with $x = [-1, 2]$:

$$d(x, \mu_1) = \sqrt{(-1-1)^2 + (2-2)^2} = \sqrt{4 + 0} = 2$$
$$d(x, \mu_2) = \sqrt{(-1+3)^2 + (2-0)^2} = \sqrt{4 + 4} = \sqrt{8} \approx 2.83$$
$$d(x, \mu_3) = \sqrt{(-1-4)^2 + (2-2)^2} = \sqrt{25 + 0} = 5$$

The smallest distance is $2$, to $\mu_1$, so the point is assigned to **Cluster 1**: $c^{(i)} = 1$.

### C2 — Solution

The parameters to be estimated, **for each cluster**, are:
- the **mean vector** $\mu$,
- the **covariance matrix** $\Sigma$,
- the **mixing coefficient** $\pi$ (the $\pi$'s across clusters sum to 1).

For $D = 3$ dimensions, **per cluster**: $\mu$ is $3 \times 1$ and $\Sigma$ is $3 \times 3$. ($\pi$ is a single scalar.) Note that $N = 1000$ does **not** affect these dimensions — it's the number of data points, not a parameter size.

### C3 — Solution

**Degree of A:** sum of row A in the adjacency matrix:
$$D_{AA} = 8 + 6 + 1 + 0 + 0 = 15$$

**Laplacian diagonal entry:** $L = D - A$, and $A_{AA} = 0$, so
$$L_{AA} = D_{AA} - A_{AA} = 15 - 0 = 15$$

**$D^{-1/2}$ entry for A:**
$$(D^{-1/2})_{AA} = \frac{1}{\sqrt{D_{AA}}} = \frac{1}{\sqrt{15}} \approx 0.258$$

(For reference, the full degree diagonal is $D_{AA}=15,\ D_{BB}=16,\ D_{CC}=16,\ D_{DD}=17,\ D_{EE}=17,\ D_{FF}=15$.)
