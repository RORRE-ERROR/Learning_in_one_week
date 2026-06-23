# Chapter 1 — Foundations of Machine Learning

> **Day 1 of 7** · Prerequisites: none · Est. study time: ~60–75 min

**In one sentence:** Machine learning is the art of getting a computer to learn patterns *from data* instead of being told the rules by hand — and this chapter gives you the map (the paradigms, the workflow, and the vocabulary) that the rest of the week will fill in.

## Why this matters

Everything else in this course — linear regression, decision trees, neural networks, clustering — is just a *specific* way of doing the one thing this chapter describes: turning data into a model that makes predictions. If the big picture here is fuzzy, every later chapter feels like disconnected tricks. Get it solid now and the rest of the week clicks into place. The exam loves to test whether you can (a) name the right paradigm for a scenario, (b) order the workflow steps, and (c) use the vocabulary correctly.

## Learning goals

- [ ] Explain what ML is and when to use it *instead of* traditional programming.
- [ ] Tell apart the three paradigms — supervised, unsupervised, reinforcement — and give one example of each.
- [ ] Within supervised learning, distinguish **regression** from **classification**.
- [ ] Walk the end-to-end ML workflow from raw data to deployment.
- [ ] Use the core vocabulary correctly: features, target, train/test split, model, parameters, overfitting/underfitting, generalization.
- [ ] Read an evaluation metric (accuracy, precision, recall, F1, AUC-ROC) at an intuitive level.

---

## 1. What is machine learning?

**Intuition first.** In traditional programming, *you* write the rules. Want to flag spam? You sit down and write `if email contains "free money" then spam`. You are the brain; the computer just executes your rules.

Machine learning flips this around. Instead of writing the rules, you hand the computer a pile of **examples** ("here are 10,000 emails, each labelled spam or not-spam") and the computer figures out the rules *itself*. You write the learning procedure once; the data supplies the logic.

```
Traditional:   Rules + Data        →  computer  →  Answers
ML:            Data + Answers       →  computer  →  Rules  (a "model")
                                                    then: Rules + new Data → Answers
```

**Definition.** Machine learning is a set of methods that **automatically learn patterns from data** and use those patterns to make predictions or decisions on new, unseen data — without being explicitly programmed with the rules.

**When do you reach for ML instead of plain code?** When the rules are:

- **Too complex to write by hand** — nobody can write `if`-statements that recognise a cat in a photo.
- **Hidden in the data** — you suspect "ghosting" on a dating app is predictable from behaviour, but you don't know the exact formula.
- **Changing over time** — spam tactics evolve; a model retrained on fresh data adapts, hard-coded rules rot.

If a problem *can* be solved with a few clear hand-written rules (e.g. "tax = income × 0.2"), you don't need ML. Use the simplest tool that works.

> 💡 **Exam tip:** The dividing line is "*who writes the rules?*" In traditional programming a human writes them; in ML the algorithm derives them from data. Phrase it that way and you'll nail any "ML vs traditional programming" question.

> ⚠️ **Common trap:** ML does not mean "no programming". You still write code — to load data, preprocess it, choose and configure the algorithm, and evaluate results. What you *don't* write is the decision logic itself.

---

## 2. The three paradigms of machine learning

ML splits into three big families based on **what kind of feedback the algorithm gets while learning**.

### 2.1 Supervised learning — learning from labelled examples

**Intuition.** A teacher (the "supervisor") gives you flashcards with the answer on the back. You study input→answer pairs until you can predict the answer for a card you've never seen.

Here the data comes with **labels** — the correct answer for each example. The model learns the mapping from inputs to the known answer, then predicts answers for new inputs.

Supervised learning splits into two sub-types by *what kind of answer* you predict:

- **Regression** → predict a **continuous number**. *Example:* predict a house's price (RM 450,000), tomorrow's temperature, a person's age.
- **Classification** → predict a **category/class** from a fixed set. *Example:* spam vs not-spam, which of 10 digits a handwritten image shows, or — our running example — **will this dating-app user ghost their match or not?**

> 🏃 **Running example — the GhostBusters project.** The group project predicts whether a dating-app user will *ghost* their match (suddenly cut off all contact). Each user record has 19 behavioural/demographic **features** (app usage time, swipe-right ratio, messages sent, income bracket, …) and one **label**, `is_ghosted`, which is `1` (Ghosted) or `0` (Not Ghosted). Because the answer is one of exactly two categories, this is **supervised binary classification** — the workhorse example we'll return to all week.

### 2.2 Unsupervised learning — finding structure with no labels

**Intuition.** No teacher, no answer key. You're handed a pile of unsorted photos and asked to "group the ones that go together." You find structure on your own.

Here the data has **no labels**. The model's job is to discover hidden structure. Two main jobs:

- **Clustering** → automatically group similar examples. *Example:* a shop segments customers into "bargain hunters", "loyal regulars", "big spenders" without anyone pre-labelling them.
- **Dimensionality reduction** → squeeze many features into a few while keeping the important information. *Example:* **PCA** (Principal Component Analysis) compresses 100 correlated measurements into 2–3 summary numbers you can plot. (You'll see PCA again later in the week.)

### 2.3 Reinforcement learning — learning from rewards

**Intuition.** Training a dog. The dog (the **agent**) tries actions in the world (the **environment**); good actions earn treats (**rewards**), bad ones don't. Over many tries the agent learns a strategy that maximises its total reward.

> 📌 **This is literally midterm Q10:** *"Reinforcement learning involves agents that act on its environment to obtain ______"* → **rewards.**

In reinforcement learning there's no fixed answer key. The agent learns by **trial and error**, taking actions, observing the consequences, and adjusting to collect more reward over time. *Example:* a program learning to play chess by playing millions of games — winning gives reward, losing gives penalty — until it discovers strong strategies. Other examples: robots learning to walk, recommendation systems optimising long-term engagement.

| Paradigm | Data has labels? | Feedback signal | One-line example |
|---|---|---|---|
| **Supervised** | Yes (input + correct answer) | The correct label | Predict if a user will ghost (classification) |
| **Unsupervised** | No | None — find structure | Group customers into segments (clustering) |
| **Reinforcement** | No | **Rewards** from the environment | An agent learning to play chess |

> 💡 **Exam tip:** Classify by the *feedback*. Labels present → supervised. No labels, just "find groups/patterns" → unsupervised. An **agent + environment + reward** → reinforcement. The trio "agent / environment / reward" is the dead giveaway for RL.

> ⚠️ **Common trap:** "Continuous output = always regression" is wrong as a paradigm cue. Regression vs classification is a split *inside* supervised learning (continuous number vs category). Don't confuse that sub-split with the three-paradigm split.

---

## 3. The end-to-end ML workflow

Real ML projects follow the same pipeline almost every time. Memorise this order — the exam asks you to sequence it.

```
1. Data collection  →  2. Preprocessing  →  3. Model training  →  4. Evaluation  →  5. Deployment / Explanation
                                   ↑___________________________________|
                                   (loop back and improve)
```

1. **Data collection** — gather the raw examples. *GhostBusters:* 50,000 dating-app records, 19 features, from Kaggle.
2. **Preprocessing** — clean and prepare the data: handle missing values, encode categories into numbers, scale features to comparable ranges, and engineer new features. *This is where most of the real work lives.* (Whole chapter coming — **see Chapter on Data Preprocessing**.) *GhostBusters:* fills missing values, one-hot-encodes `gender`/`income_bracket`, scales numeric features with `StandardScaler`, and turns the text `interest_tags` into an `interest_count`.
3. **Model training** — feed the prepared **training set** to a learning algorithm so it can learn its **parameters**. *GhostBusters:* trains five models — Logistic Regression, Decision Tree, Random Forest, SVM, Neural Network. (Each gets its own chapter this week.)
4. **Evaluation** — measure how well the model does on a held-out **test set** it never saw during training, using metrics like accuracy and F1. (See §5 below, and the metrics chapter.)
5. **Deployment / Explanation** — put the model to use, and explain *why* it predicts what it predicts. *GhostBusters:* uses **SHAP** (Explainable AI) to surface which features drove each ghosting prediction.

The arrow looping back matters: ML is **iterative**. Poor evaluation results send you back to get more data or better preprocessing.

> 💡 **Exam tip:** A classic question scrambles these steps and asks for the correct order. Anchor on two facts: **preprocessing comes before training**, and **evaluation comes before deployment**. You can't train on dirty data, and you don't ship a model you haven't measured.

---

## 4. Core vocabulary (the words every later chapter assumes)

Learn these now; they recur on every page for the rest of the week.

- **Feature ($X$)** — an input variable describing an example; also called an attribute or predictor. *GhostBusters:* `message_sent_count`, `swipe_right_ratio`, etc. All features together form the input matrix, usually written $X$.
- **Target / label ($y$)** — the thing you're trying to predict. *GhostBusters:* `is_ghosted`. In supervised learning, every training example pairs an $X$ with its known $y$.
- **Model / hypothesis** — the learned function that maps inputs to a prediction: $\hat{y} = h(X)$. The hat on $\hat{y}$ means "predicted," to distinguish it from the true $y$.
- **Parameters** — the internal numbers the model *learns* from the training data (e.g. the weights/coefficients in a regression). Training = finding good parameter values.
- **Training set vs test set** — you **split** your data. The model *learns* from the training set; you *measure* it on the test set, which it has never seen. This is the only honest way to estimate how the model will do on new real-world data. *GhostBusters:* an 80/20 split, done **before** preprocessing is fitted, to prevent the test set leaking into training.

> 💡 **Exam tip:** The *primary* reason we split into train/test (midterm-style question) is **to evaluate the model on unseen data and detect/prevent overfitting** — not to "reduce dataset size" or "speed things up."

### 4.1 Overfitting vs underfitting (intuition only — depth comes later)

This is the central tension of all ML. Picture a student preparing for an exam:

- **Underfitting** = the student barely studied. The model is **too simple** to capture the real pattern — it does poorly on *both* the training data and the test data. (Like fitting a straight line to clearly curved data.)
- **Overfitting** = the student *memorised* the practice questions word-for-word instead of understanding. The model is **too complex** — it nails the training data but flops on the test data, because it learned noise and quirks instead of the real signal.

```
Underfit:  bad on train,  bad on test     (too simple)
Good fit:  good on train, good on test     (just right)  ← the goal
Overfit:   great on train, bad on test     (too complex, memorised noise)
```

- **Generalization** — the goal: a model that performs well on **new, unseen** data, not just the data it trained on. Overfitting is the enemy of generalization.
- **Bias vs variance (light touch).** Two sources of error that pull in opposite directions. **High bias** = the model is too rigid → underfitting. **High variance** = the model is too sensitive to the exact training data → overfitting. Good models balance the two. *(Full treatment beyond this chapter — for now just link bias↔underfit and variance↔overfit.)*

> ⚠️ **Common trap:** "High accuracy on the training set = great model." No! A model that overfits can score nearly 100% on training data and still be useless. **Only the test-set score tells you about generalization.**

---

## 5. Evaluation metrics — an intuitive orientation

You can't say a model is "good" without measuring it. Here's a light tour of the metrics the GhostBusters project uses — full detail comes in the metrics chapter; right now just build intuition. All four classification metrics below are built from the four outcomes of a binary prediction: **TP** (true positive), **TN** (true negative), **FP** (false positive — false alarm), **FN** (false negative — a miss).

- **Accuracy** = fraction of all predictions that are correct. Simple and intuitive — *but misleading when classes are imbalanced.* If only 10% of users ghost, a lazy model that always predicts "Not Ghosted" scores 90% accuracy while catching **zero** ghosters. (This actually happened in the project: ~90% accuracy, terrible F1.)

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

- **Precision** — of everyone the model *flagged* as ghosters, how many really ghosted? Punishes false alarms. $\;\text{Precision} = \dfrac{TP}{TP + FP}$
- **Recall** — of all the people who *actually* ghosted, how many did the model catch? Punishes misses. $\;\text{Recall} = \dfrac{TP}{TP + FN}$
- **F1-score** — the **harmonic mean** of precision and recall, giving a single balanced number. It's high only when *both* precision and recall are high, which is why GhostBusters uses it as the **primary ranking metric** under class imbalance. $\;F_1 = 2 \cdot \dfrac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
- **AUC-ROC** — measures how well the model **separates** the two classes across *all* decision thresholds. 1.0 = perfect separation, 0.5 = no better than a coin flip.

> 💡 **Exam tip:** When you hear "imbalanced classes," distrust accuracy and reach for **F1 / precision / recall**. The "always predict the majority class" trick is the canonical example of why accuracy alone lies.

---

## Key formulas / facts at a glance

| Thing | Formula / Rule | Plain meaning |
|---|---|---|
| ML vs traditional | Data + Answers → **Rules** | The algorithm derives the rules; the human doesn't |
| Supervised | Labelled data, predict $y$ | Regression = number; Classification = category |
| Unsupervised | No labels | Clustering + dimensionality reduction (e.g. PCA) |
| Reinforcement | agent + environment → **rewards** | Learn by trial and error to maximise reward (midterm Q10) |
| Workflow | Data → Preprocess → Train → Evaluate → Deploy | Preprocess *before* train; evaluate *before* deploy |
| Train/test split | learn on train, measure on test | Honest estimate of generalization; detects overfitting |
| Overfitting | great train, bad test | Too complex; memorised noise (high variance) |
| Underfitting | bad train, bad test | Too simple; missed the pattern (high bias) |
| Accuracy | $(TP+TN)/(\text{all})$ | Overall correctness; misleading if imbalanced |
| Precision | $TP/(TP+FP)$ | Of flagged positives, how many are real |
| Recall | $TP/(TP+FN)$ | Of real positives, how many were caught |
| F1 | $2\frac{PR}{P+R}$ | Balanced single score; use under imbalance |
| AUC-ROC | area under ROC curve | Class separability across all thresholds (0.5 = random) |

## Exam tips — quick recap

- "Who writes the rules?" — human = traditional programming; algorithm = ML.
- Classify a paradigm by its **feedback**: labels → supervised; none → unsupervised; **agent + environment + reward** → reinforcement.
- Regression vs classification is a split *inside* supervised (continuous number vs category) — don't confuse it with the three paradigms.
- RL keyword chain: **agent acts on environment to obtain rewards** (midterm Q10).
- Workflow order: preprocess **before** train; evaluate **before** deploy.
- The primary reason to split train/test is to **evaluate on unseen data and prevent overfitting**.
- Under class imbalance, **don't trust accuracy** — use F1 / precision / recall.

## Self-check

1. State the difference between traditional programming and machine learning in one sentence.
2. A bank wants to group its customers into segments it hasn't defined in advance. Which paradigm, and which task?
3. Reinforcement learning involves agents that act on their environment to obtain ____?
4. Put these in order: Evaluation, Preprocessing, Deployment, Model training, Data collection.
5. True or false: a model with 99% training accuracy is guaranteed to generalize well. Why?
6. Why does the GhostBusters project rank models by F1 instead of accuracy?

**→ Now test yourself with [`quiz.md`](./quiz.md).**
