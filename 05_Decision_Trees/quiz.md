# Chapter 5 Quiz — Decision Trees

> Format mirrors the WIA1006 quiz/midterm: multiple-choice + true/false + a worked problem.
> Try it closed-book first. Answers + explanations are at the very bottom. Note for Section C: $\log_2 3 \approx 1.6$.

## Section A — Multiple Choice

**A1.** What is the main objective of a decision tree algorithm?
- a. Normalize features
- b. Reduce dimensionality
- c. Minimize training time
- d. Split data into homogeneous subsets

**A2.** What does entropy measure in a dataset?
- a. Model accuracy
- b. Distance between points
- c. Feature importance
- d. Randomness or impurity

**A3.** Which metric is commonly used for classification trees?
- a. Mean Squared Error
- b. Cosine Similarity
- c. Euclidean Distance
- d. Entropy

**A4.** In the ID3 algorithm, which attribute is chosen for the split at a node?
- a. The one with the lowest entropy before splitting
- b. The one with the highest information gain
- c. The one with the most distinct values
- d. A randomly selected attribute

**A5.** Select the improvements made over the years in the ID3 algorithm for decision trees.
- a. handling of missing attribute values
- b. post-pruning trees and handling of missing attribute values
- c. none
- d. post-pruning trees

**A6.** A node has 4 examples, all labelled "Yes". What is its entropy?
- a. 0
- b. 0.5
- c. 1
- d. 2

**A7.** How does a decision tree handle a continuous feature such as temperature?
- a. It normalizes the feature to the range 0–1 first
- b. It creates one branch per unique numeric value
- c. It splits the feature into two ranges using a threshold
- d. It cannot use continuous features at all

**A8.** Information gain for a split is best described as:
- a. the entropy of the resulting branches added together
- b. the entropy before the split minus the weighted entropy after the split
- c. the Euclidean distance between the two child nodes
- d. the number of examples that change class after the split

**A9.** In the badminton example, *Weather* is asked first. What type of node is it?
- a. Leaf node
- b. Decision node only
- c. Root node
- d. Pruned node

**A10.** Match the subtree-selection method used in pruning to its description. Which pairing is correct?
- a. Cross-validation → checks hypothesis complexity
- b. Statistical test → evaluates subtree utility using a tuning set
- c. Minimum description length → checks hypothesis complexity of pruning
- d. Cross-validation → finds regularities that could be random chance

**A11.** Which is the best leaf node in a decision tree?
- a. The one with the highest entropy
- b. The one with the most examples
- c. The most homogeneous (purest) group
- d. The one closest to the root

## Section B — True / False

**B1.** To overcome overfitting, a decision tree could use pre-pruning or post-pruning approaches. (True / False)

**B2.** A decision tree uses Euclidean Distance as its splitting criterion. (True / False)

**B3.** Characteristics of a good split include high information gain, low impurity, and a balanced partition. (True / False)

**B4.** The root node is the first split in a decision tree. (True / False)

**B5.** ID3 algorithms in decision trees tend to produce trees as large as possible. (True / False)

**B6.** Entropy is 0 when a group is completely pure and 1 when a group is split evenly 50/50 between two classes. (True / False)

**B7.** Decision trees can only produce Boolean (Yes/No) outputs; they cannot handle more than two classes. (True / False)

## Section C — Worked Problem

**C1.** We want a decision tree that predicts whether a student **Passed** machine learning (Yes/No) from their **GPA** (High/Medium/Low) and whether they **Studied** (Yes/No). The data for 6 students:

| # | GPA | Studied | Passed |
|---|---|---|---|
| 1 | High | No | No |
| 2 | High | Yes | Yes |
| 3 | Medium | No | Yes |
| 4 | Medium | Yes | Yes |
| 5 | Low | No | No |
| 6 | Low | Yes | Yes |

(a) Compute the entropy of the target, $H(\text{Passed})$.
(b) Compute the conditional entropy $H(\text{Passed}\mid\text{GPA})$ and the information gain $IG(\text{Passed}, \text{GPA})$.
(c) Compute the conditional entropy $H(\text{Passed}\mid\text{Studied})$ and the information gain $IG(\text{Passed}, \text{Studied})$.
(d) Which attribute should ID3 split on first, and why?

Show all arithmetic. Use $\log_2 3 \approx 1.6$.

---

## ✅ Answer Key & Explanations

**A1 — (d) Split data into homogeneous subsets.** That is the entire purpose of a tree. (a)/(b) are preprocessing goals and (c) is irrelevant to how a tree is built.

**A2 — (d) Randomness or impurity.** Entropy quantifies how "mixed" the labels are. "Distance between points" (b) describes metrics like Euclidean distance, not entropy.

**A3 — (d) Entropy.** Classification trees split using entropy / information gain. MSE is for regression, and distance/similarity metrics belong to k-NN and clustering — classic distractors here.

**A4 — (b) The one with the highest information gain.** ID3 is greedy: at each node it picks the attribute whose split reduces entropy the most. Note (a) is a trap — you don't compare entropy *before* the split across attributes; you compare the *gain*.

**A5 — (b) post-pruning trees and handling of missing attribute values.** Both improvements were added to ID3 over time. The single-item options (a)/(d) are incomplete, so the combined option wins.

**A6 — (a) 0.** A pure group has no uncertainty: $H = -1\cdot\log_2 1 = 0$. Entropy of 1 would require a 50/50 split, not a uniform one.

**A7 — (c) It splits the feature into two ranges using a threshold.** For example temperature ≤ 30 vs > 30. The tree searches for the threshold giving the best information gain. (b) would explode into too many tiny branches.

**A8 — (b) entropy before the split minus the weighted entropy after the split.** That difference is exactly how much uncertainty the split removed. The "after" term is weighted by the fraction of examples in each branch.

**A9 — (c) Root node.** The first / topmost split is the root. It is also technically a decision node, but the *specific* name for the first split is "root."

**A10 — (c) Minimum description length → checks hypothesis complexity of pruning.** The correct full matching is: cross-validation → tuning set; statistical test → regularities that could be random chance; MDL → hypothesis complexity. Only option (c) states a correct pair.

**A11 — (c) The most homogeneous (purest) group.** Leaves are where impurity has been driven down toward zero, giving a confident final answer. High entropy (a) is the opposite of what a good leaf looks like.

**B1 — True.** Both pre-pruning (stop growing early) and post-pruning (grow then trim) are standard ways to reduce overfitting.

**B2 — False.** Trees split on entropy / information gain, **not** Euclidean distance. Distance-based splitting is associated with k-NN / clustering. This is a favourite "gotcha."

**B3 — True.** A good split maximizes information gain, leaves the child groups pure (low impurity), and partitions the data in a balanced way.

**B4 — True.** "Root node = first split" is a direct definition.

**B5 — False.** Although raw greedy ID3 keeps splitting until leaves are pure, the statement is treated as false: proper practice (stopping criteria + pruning) deliberately avoids growing the largest possible tree, since an oversized tree overfits. The exam wants you to reject "as large as possible" as a good description of how ID3 *should* be used.

**B6 — True.** $H = 0$ for a pure group; $H = -(\tfrac12\log_2\tfrac12 + \tfrac12\log_2\tfrac12) = 1$ for a perfect 50/50 split.

**B7 — False.** Decision trees extend naturally to multi-class outputs and even real-valued (regression) outputs, though the latter is less common.

---

### C1 — Full worked solution

There are 6 students: **4 Passed (Yes), 2 Failed (No).**

**(a) Entropy of the target.**

$p(\text{Yes}) = \tfrac46 = \tfrac23,\quad p(\text{No}) = \tfrac26 = \tfrac13.$

$$H(\text{Passed}) = -\left(\tfrac13\log_2\tfrac13 + \tfrac23\log_2\tfrac23\right)$$

Using $\log_2\tfrac13 = -\log_2 3$ and $\log_2\tfrac23 = 1-\log_2 3$:

$$H(\text{Passed}) = \tfrac13\log_2 3 + \tfrac23(\log_2 3 - 1) = \log_2 3 - \tfrac23 \approx 1.6 - 0.667 = \boxed{0.92}$$

**(b) Split on GPA.** Three values, two students each:

- High → {No, Yes}: 1 Yes, 1 No → $H = 1$
- Medium → {Yes, Yes}: pure → $H = 0$
- Low → {No, Yes}: 1 Yes, 1 No → $H = 1$

Weight each branch by $\tfrac{2}{6} = \tfrac13$:

$$H(\text{Passed}\mid\text{GPA}) = \tfrac13(1) + \tfrac13(0) + \tfrac13(1) = \tfrac23 \approx 0.67$$

$$IG(\text{Passed}, \text{GPA}) = 0.92 - 0.67 = \boxed{0.26}$$

**(c) Split on Studied.** Two groups of three:

- Studied = No → students 1,3,5 = {No, Yes, No}: 1 Yes, 2 No → $H = -(\tfrac13\log_2\tfrac13 + \tfrac23\log_2\tfrac23) = \log_2 3 - \tfrac23 \approx 0.92$
- Studied = Yes → students 2,4,6 = {Yes, Yes, Yes}: pure → $H = 0$

Weight each branch by $\tfrac{3}{6} = \tfrac12$:

$$H(\text{Passed}\mid\text{Studied}) = \tfrac12(0.92) + \tfrac12(0) = \tfrac12\log_2 3 - \tfrac13 \approx 0.46$$

$$IG(\text{Passed}, \text{Studied}) = 0.92 - 0.46 = \boxed{0.46}$$

**(d) Which split first?**

| Split | Conditional entropy | Information gain |
|---|---|---|
| GPA | ≈ 0.67 | 0.26 |
| **Studied** | **≈ 0.46** | **0.46** |

**Split on Studied first**, because it has the higher information gain (0.46 > 0.26). Intuitively, "Studied = Yes" yields a perfectly pure group (all passed), so it removes the most uncertainty.
