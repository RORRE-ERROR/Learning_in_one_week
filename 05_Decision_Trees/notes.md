# Chapter 5 — Decision Trees

> **Day 4 of 7** · Prerequisites: Ch 1–2 (what supervised learning / classification is, and basic train/test thinking) · Est. study time: ~75–90 min

**In one sentence:** A decision tree is a flowchart of yes/no questions that splits your data into ever-purer groups, and we use *entropy* and *information gain* to decide which question to ask first.

## Why this matters

Decision trees are the most *human-readable* model in the whole course — you can literally read the rules off the tree. They are also the one model in WIA1006 with real, hand-computable math (entropy and information gain), so the exam **loves** asking you to crank through the arithmetic. The course project uses a decision tree too, so this isn't just theory. Expect both conceptual questions (node types, pruning) and a worked entropy problem.

## Learning goals

- [ ] Name the three node types and which is the "first split"
- [ ] State the main objective of a decision tree in one phrase
- [ ] Compute **entropy** of a label set by hand, including using $\log_2 3 \approx 1.6$
- [ ] Compute **information gain** for an attribute and pick the best split
- [ ] Walk through the **ID3** algorithm step by step
- [ ] Handle continuous features with a threshold split
- [ ] Explain **pre-pruning vs post-pruning** and why we prune at all
- [ ] Match the three subtree-selection methods used in pruning

---

## 1. What a decision tree is

Imagine deciding whether to play badminton. You ask: *Is it sunny, cloudy, or rainy?* If sunny, you then ask *Is humidity high or low?* Each answer sends you down a branch until you reach a final decision: **Play** or **Don't play**. That chain of questions *is* a decision tree.

A tree has three kinds of nodes:

- **Root node** — the topmost node, the **very first question/split**. (In the badminton example, *Weather* is the root.)
- **Decision (internal) nodes** — any node that asks another question and splits the data again into segments.
- **Leaf nodes** — the dead-ends that give the **final answer**. A good leaf is **as homogeneous (pure) as possible** — ideally everything in it has the same label.

```
                [ Weather ]   ← root node (first split)
               /     |      \
          Sunny   Cloudy    Rainy
            |        |         |
       [Humidity] [Play=Yes] [Wind?]   ← decision nodes (Cloudy already pure → leaf)
        /     \              /     \
     High    Low         Strong   Weak
      No      Yes           No      Yes      ← leaf nodes (final answers)
```

> 💡 **Exam tip:** "Root node = first split" is asked verbatim (it's **True**). Don't overthink it.

> ⚠️ **Common trap:** A leaf is the *most* homogeneous group, not the *least*. The whole point of splitting is to push impurity down toward zero at the leaves.

## 2. The main objective

> **The goal of a decision tree is to split the data into _homogeneous_ subsets.**

That's it. "Homogeneous" means *all the same label*. Everything else — entropy, information gain, ID3 — is just machinery for choosing splits that make the groups as pure as possible.

It is **not** about normalizing features, reducing dimensionality, or minimizing training time. Those are distractor answers.

> 💡 **Exam tip:** If you see "What is the main objective of a decision tree?", the answer is **split data into homogeneous subsets**.

## 3. Entropy — measuring impurity

We need a number that says *"how mixed are the labels in this group?"* That number is **entropy**.

Intuition first: if a group is all "Pass", there's no uncertainty — entropy should be **0**. If a group is a perfect 50/50 mix of "Pass" and "Fail", uncertainty is maximal — entropy should be **1**. Entropy is the "surprise" left in the group.

The formula:

$$H(S) = -\sum_i p_i \log_2 p_i$$

where:
- $S$ is the set of examples,
- $p_i$ is the **fraction** of examples in class $i$ (e.g. fraction that passed),
- the sum runs over every class,
- $\log_2$ is log base 2 (so entropy is measured in *bits*).

Why the shape? The $-\log_2 p_i$ term is the "surprise" of seeing class $i$: rare classes (small $p_i$) are very surprising (large $-\log_2 p_i$). We weight each surprise by how often it happens ($p_i$) and add them up. The minus sign is just there because $\log_2$ of a fraction is negative, and we want entropy to come out positive.

**Two anchor values to memorize:**

| Group | Entropy |
|---|---|
| Completely pure (all one class) | $H = 0$ |
| Perfect 50/50 split | $H = 1$ |

Quick check of the 50/50 case: $H = -(\tfrac12\log_2\tfrac12 + \tfrac12\log_2\tfrac12) = -(\tfrac12\cdot(-1) + \tfrac12\cdot(-1)) = 1.$ ✓

And the pure case: a class with $p=1$ gives $-1\cdot\log_2 1 = -1\cdot 0 = 0.$ ✓ (We treat $0\log_2 0 = 0$.)

> 💡 **Exam tip:** Entropy is **the** metric for classification trees (it measures *randomness / impurity*). It is **not** Euclidean distance, **not** MSE, **not** cosine similarity — those are common distractors.

> ⚠️ **Common trap:** "Decision trees use Euclidean distance to split" is **False**. Distance is for things like k-NN/clustering. Trees split on entropy / information gain.

## 4. Information Gain — choosing the split

Entropy tells you how impure *one* group is. **Information gain (IG)** tells you how much *cleaner* the data gets *after* you split on an attribute.

$$IG(S, A) = H(S) \;-\; \underbrace{\sum_{v \in A} \frac{|S_v|}{|S|}\,H(S_v)}_{\text{weighted entropy after split}}$$

In words:

$$\text{Information Gain} = (\text{entropy before split}) - (\text{weighted entropy of the branches after split}).$$

Each branch's entropy is weighted by the **fraction of examples that fall into that branch** ($|S_v|/|S|$). That weighted sum is the **conditional entropy** $H(S \mid A)$ — "how mixed the groups remain after splitting on $A$."

A **good split** has:
- **high information gain** (it removes a lot of uncertainty),
- **low impurity** in the resulting groups,
- a **balanced** partition.

ID3's rule: **pick the attribute with the highest information gain.**

> 💡 **Exam tip:** "Characteristics of a good split = high info gain, low impurity, balanced partition" is **True**.

## 5. The ID3 algorithm, step by step

ID3 ("Iterative Dichotomiser 3") builds the tree **top-down** using entropy + information gain. The loop:

1. **Start** with all training data at the root.
2. **Compute entropy** of the current labels (how mixed they are).
3. **Compute information gain** for *each* candidate attribute.
4. **Choose the attribute with the highest IG** and split on it.
5. **Repeat** steps 2–4 on each resulting branch, stopping when a branch is pure (entropy 0) or you run out of attributes.

```
all data → entropy → IG for each attribute → pick max IG → split → recurse on branches
```

## 6. Worked example — "Will they pass ML?" (do this by hand)

We predict **Passed** (Yes/No) from **GPA** (High/Medium/Low) and **Studied** (Yes/No). Six students:

| # | GPA | Studied | Passed |
|---|---|---|---|
| 1 | High | No | No |
| 2 | High | Yes | Yes |
| 3 | Medium | No | Yes |
| 4 | Medium | Yes | Yes |
| 5 | Low | No | No |
| 6 | Low | Yes | Yes |

Totals: **4 Passed, 2 Failed** out of 6. Note the hint: $\log_2 3 \approx 1.6$.

### (a) Entropy of the target, $H(\text{Passed})$

$p(\text{Yes}) = \tfrac{4}{6} = \tfrac23$, $p(\text{No}) = \tfrac26 = \tfrac13$.

$$H(\text{Passed}) = -\left(\tfrac26\log_2\tfrac26 + \tfrac46\log_2\tfrac46\right) = -\left(\tfrac13\log_2\tfrac13 + \tfrac23\log_2\tfrac23\right)$$

Simplify using log rules ($\log_2\tfrac13 = -\log_2 3$ and $\log_2\tfrac23 = 1 - \log_2 3$):

$$H(\text{Passed}) = \log_2 3 - \tfrac23 \approx 1.6 - 0.667 \approx \boxed{0.92}$$

So before any split there are ~0.92 bits of uncertainty.

### (b) Split on GPA — entropy and IG

GPA has three values, **two students each**:

- **High** → {No, Yes} = 1 Yes, 1 No → $H = 1$ (perfect 50/50)
- **Medium** → {Yes, Yes} = 2 Yes, 0 No → $H = 0$ (pure!)
- **Low** → {No, Yes} = 1 Yes, 1 No → $H = 1$

Weight each branch by $\tfrac{2}{6} = \tfrac13$:

$$H(\text{Passed}\mid\text{GPA}) = \tfrac13(1) + \tfrac13(0) + \tfrac13(1) = \tfrac23 \approx 0.67$$

$$IG(\text{Passed}, \text{GPA}) = 0.92 - 0.67 = \boxed{0.26}$$

> *(The tutorial groups GPA as 50/50, 50/50, pure — giving $\tfrac13(1)+\tfrac13(1)+\tfrac13(0)=\tfrac23$. Same answer, $H\approx0.67$, $IG\approx0.26$, regardless of which branch is the pure one.)*

### (c) Split on Studied — entropy and IG

Studied splits into two groups of **three**:

- **Studied = No** → students 1, 3, 5 = {No, Yes, No} = 1 Yes, 2 No → $H = -(\tfrac13\log_2\tfrac13 + \tfrac23\log_2\tfrac23) = \log_2 3 - \tfrac23 \approx 0.92$
- **Studied = Yes** → students 2, 4, 6 = {Yes, Yes, Yes} = 3 Yes, 0 No → $H = 0$ (pure!)

Weight each branch by $\tfrac{3}{6} = \tfrac12$:

$$H(\text{Passed}\mid\text{Studied}) = \tfrac12(0.92) + \tfrac12(0) = \tfrac12\log_2 3 - \tfrac13 \approx 0.46$$

$$IG(\text{Passed}, \text{Studied}) = 0.92 - 0.46 = \boxed{0.46}$$

### Decision

| Split | Conditional entropy | Information gain |
|---|---|---|
| GPA | ≈ 0.67 | 0.26 |
| **Studied** | **≈ 0.46** | **0.46** |

**Studied has the higher information gain (0.46 > 0.26), so ID3 splits on Studied first.** Intuitively that's because "Studied = Yes" already gives a perfectly pure group (everyone passed), so it removes the most uncertainty.

> 💡 **Exam tip:** Memorize the three headline numbers from this problem: $H(\text{Passed})\approx0.92$, $IG_{\text{GPA}}=0.26$, $IG_{\text{Studied}}=0.46 \Rightarrow$ split on **Studied**.

## 7. Continuous (real-valued) features

Entropy/IG assume discrete branches, but temperature, length, etc. are continuous. The fix: **split by a threshold** into just two ranges.

For example, with **threshold = 30°C**:

```
            Temperature
           /            \
   Temp ≤ 30           Temp > 30
   (left branch)      (right branch)
```

The algorithm tries candidate thresholds and picks the one giving the best information gain — turning a continuous feature into a clean binary question (e.g. `length < 5` vs `length ≥ 5`).

## 8. Outputs aren't limited to Yes/No

The badminton tree gives a Boolean answer, but decision trees **extend naturally to multi-class** outputs (more than two labels) and even **real-valued outputs** (regression trees). Real-valued outputs are possible but less common in practice.

## 9. Pruning — fighting overfitting

Left alone, **ID3 tends to grow trees as large/deep as possible** — it keeps splitting until every leaf is pure. A giant tree *memorizes* the training data (including its noise) and generalizes badly. That's **overfitting**.

> ⚠️ **Subtle exam point:** "ID3 tends to produce trees as large as possible" is marked **False** on the midterm. The raw ID3 *behaviour* is indeed to grow greedily, but the statement is treated as false because a properly used / improved ID3 doesn't *aim* for the largest possible tree — we add stopping rules and pruning to keep it from doing that. Read the question's intent: the "correct practice" is a controlled-size tree.

**Pruning** = removing sub-nodes of a decision node (the opposite of splitting) to shrink the tree and improve generalization. Two flavours:

- **Pre-pruning** — stop growing *early*, during construction, when there's no longer enough data/evidence to justify another split.
- **Post-pruning** — grow the **full** tree first, then go back and **trim** subtrees that don't have enough supporting evidence / don't improve generalization.

> 💡 **Exam tip:** "To overcome overfitting, a decision tree can use pre-pruning or post-pruning" is **True**. Both combat overfitting.

**Improvements to ID3 over the years** were exactly two things: **handling missing attribute values** *and* **post-pruning trees**. (If a question lists these as options, pick the one with *both*.)

## 10. How post-pruning decides what to cut (subtree selection)

Three named methods — match each to its idea:

| Method | What it does |
|---|---|
| **Cross-validation** | Evaluate a subtree's usefulness using a separate **tuning set** |
| **Statistical test** | Find regularities that could just be **random chance** and dismiss them |
| **Minimum Description Length (MDL)** | Check the **complexity of the hypothesis** when deciding to prune |

> 💡 **Exam tip:** This is a *matching* question. Mnemonic: **tuning set → cross-validation**, **random chance → statistical test**, **complexity → MDL**.

## 11. Strengths & weaknesses

**Strengths**
- **Interpretable** — you can read the decision rules straight off the tree.
- **No feature scaling needed** — splits are threshold-based, so normalization doesn't matter (unlike distance-based models).
- Handle both categorical and continuous features, and multi-class outputs.
- Used in the course **project**.

**Weaknesses**
- **Overfit easily** if grown fully (hence pruning).
- **Unstable** — a small change in the data can produce a very different tree.

---

## Key formulas / facts at a glance

| Thing | Formula / Rule | Plain meaning |
|---|---|---|
| Entropy | $H(S) = -\sum_i p_i \log_2 p_i$ | Randomness / impurity of a group (0 = pure, 1 = 50/50) |
| Information gain | $IG = H(S) - \sum_v \frac{|S_v|}{|S|}H(S_v)$ | Drop in entropy caused by a split |
| ID3 rule | pick **highest IG** attribute | Greedy, top-down split selection |
| Good split | high IG, low impurity, balanced | What you want at each node |
| Objective | split into **homogeneous** subsets | The whole point of the tree |
| Continuous feature | threshold split (≤ t vs > t) | Turns a number into a yes/no question |
| $\log_2 3$ | $\approx 1.6$ | Needed for the "pass ML" worked example |
| Root node | the first / topmost split | — |
| Pruning | pre (stop early) / post (grow then trim) | Combats overfitting |
| ID3 improvements | missing-value handling + post-pruning | What was added over the years |

## Exam tips — quick recap

- Root node = the **first split** (True).
- Decision trees do **not** use Euclidean distance (False); they use **entropy** (the classification-tree metric).
- Entropy measures **randomness/impurity**; 0 when pure, 1 when 50/50.
- Main objective = **split into homogeneous subsets**.
- Good split = **high info gain, low impurity, balanced** (True).
- ID3 picks the attribute with the **highest information gain**.
- Memorize: $H(\text{Passed})\approx0.92$, $IG_{\text{GPA}}=0.26$, $IG_{\text{Studied}}=0.46$ → split on **Studied**.
- Pre- vs post-pruning both fight overfitting (True).
- ID3 improvements = **missing values + post-pruning**.
- Pruning matching: tuning set → cross-validation; random chance → statistical test; complexity → MDL.

## Self-check

1. What are the three node types, and which one is the "first split"?
2. Compute $H$ for a group of 8 examples that is 6 Yes / 2 No. (Hint: $\log_2 3 \approx 1.6$.)
3. In your own words, what does information gain measure, and what does ID3 do with it?
4. What's the difference between pre-pruning and post-pruning, and what problem do they solve?
5. How does a decision tree handle a continuous feature like temperature?

**→ Now test yourself with [`quiz.md`](./quiz.md).**
