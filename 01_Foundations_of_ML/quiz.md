# Chapter 1 Quiz — Foundations of Machine Learning

> Format mirrors the WIA1006 quiz/midterm: multiple-choice + true/false. Try it closed-book first. Answers + explanations are at the very bottom. (No Section C — this chapter has no heavy math.)

## Section A — Multiple Choice

**A1.** What is the key difference between traditional programming and machine learning?
- a. Machine learning never requires writing any code.
- b. In traditional programming a human writes the rules; in ML the algorithm learns the rules from data.
- c. Traditional programming uses data; machine learning does not.
- d. Machine learning is always more accurate than traditional programming.

**A2.** Reinforcement learning involves agents that act on its environment to obtain ______.
- a. answer
- b. rewards
- c. data
- d. clue

**A3.** A model is trained to predict the exact selling price (in RM) of a house from its size and location. What type of task is this?
- a. Classification
- b. Clustering
- c. Regression
- d. Dimensionality reduction

**A4.** Predicting whether a dating-app user will ghost their match (`is_ghosted` = 0 or 1) is an example of:
- a. Unsupervised clustering
- b. Supervised binary classification
- c. Reinforcement learning
- d. Supervised regression

**A5.** A retailer wants to automatically discover customer groups, with no predefined labels. Which paradigm best fits?
- a. Supervised learning
- b. Unsupervised learning
- c. Reinforcement learning
- d. Regression

**A6.** Which of the following is an example of dimensionality reduction?
- a. SMOTE
- b. Random sampling
- c. PCA
- d. One-hot encoding

**A7.** What is the PRIMARY reason for splitting a dataset into training and testing sets?
- a. To reduce the overall dataset size.
- b. To evaluate the model's performance on unseen data and prevent overfitting.
- c. To create multiple copies of the data for backup.
- d. To speed up data loading.

**A8.** Which is the correct order of the end-to-end ML workflow?
- a. Model training → Preprocessing → Data collection → Deployment → Evaluation
- b. Data collection → Preprocessing → Model training → Evaluation → Deployment
- c. Preprocessing → Data collection → Evaluation → Model training → Deployment
- d. Data collection → Model training → Preprocessing → Deployment → Evaluation

**A9.** In a dataset where only 10% of users ghost, a model that always predicts "Not Ghosted" still scores 90% accuracy but catches no ghosters. Which metric best exposes this failure?
- a. Accuracy
- b. F1-score (or precision/recall)
- c. Training time
- d. Number of features

**A10.** In ML vocabulary, the internal numbers a model *learns* from the training data (e.g. regression coefficients) are called its:
- a. Features
- b. Labels
- c. Parameters
- d. Metrics

**A11.** Which scenario is the best candidate for machine learning rather than traditional hand-written rules?
- a. Computing tax as 20% of income.
- b. Converting Celsius to Fahrenheit.
- c. Recognising a cat in a photograph.
- d. Sorting a list of numbers in ascending order.

**A12.** What does AUC-ROC measure?
- a. The model's training speed.
- b. The model's ability to distinguish between the classes across all decision thresholds.
- c. The number of parameters in the model.
- d. The fraction of missing values in the data.

## Section B — True / False

**B1.** In machine learning, the algorithm learns the decision rules from data rather than having a human program them explicitly. — (True / False)

**B2.** A model that achieves very high accuracy on the training set is guaranteed to generalize well to new data. — (True / False)

**B3.** Underfitting happens when a model is too simple and performs poorly on both the training set and the test set. — (True / False)

**B4.** Regression and classification are the two main types of *unsupervised* learning. — (True / False)

**B5.** Clustering is an unsupervised task because it groups examples without using predefined labels. — (True / False)

**B6.** Overfitting is associated with high variance, while underfitting is associated with high bias. — (True / False)

**B7.** Accuracy is always the most reliable metric, even when the classes are heavily imbalanced. — (True / False)

---

## ✅ Answer Key & Explanations

**A1 — (b)** The defining line is *who writes the rules*. Traditional programming: a human codes the logic. ML: the algorithm derives the logic from data. (a) is wrong — ML still involves plenty of code (loading, preprocessing, evaluation); you just don't hand-code the decision rules.

**A2 — (b) rewards.** This is midterm Q10. The signature of reinforcement learning is an **agent** acting on an **environment** to obtain **rewards**, learning by trial and error. "Data" and "answer" describe supervised-style inputs, not the RL feedback signal.

**A3 — (c) Regression.** The output is a continuous number (a price), so it's regression. Classification (a) would apply if the output were a category. Clustering and dimensionality reduction are unsupervised and don't predict a known target.

**A4 — (b) Supervised binary classification.** Each record has a known label (`is_ghosted`) — so it's supervised — and the label is one of exactly two categories (Ghosted / Not Ghosted) — so it's binary classification. It's not regression because the target is a category, not a continuous number.

**A5 — (b) Unsupervised learning.** No predefined labels and the goal is to *discover* groups → clustering, which lives under unsupervised learning. Supervised would need labelled examples; RL would need an agent/environment/reward setup.

**A6 — (c) PCA.** Principal Component Analysis compresses many features into fewer while preserving information — the textbook dimensionality-reduction method. SMOTE handles class imbalance, one-hot encoding turns categories into numbers, and sampling selects rows — none reduce dimensionality.

**A7 — (b)** The point of a held-out test set is an honest estimate of performance on data the model never saw, which lets you detect overfitting. Reducing size, backups, and load speed are not the reason. (This mirrors midterm Q26.)

**A8 — (b)** Data collection → Preprocessing → Model training → Evaluation → Deployment. Anchor on two rules: you must clean data **before** training, and you must measure a model **before** deploying it. Every other option violates one of those.

**A9 — (b) F1 (or precision/recall).** Under class imbalance, accuracy is misleading because the majority-class guess looks "good" while catching none of the minority class. F1 balances precision and recall and collapses toward 0 when the model misses all true positives — exactly why GhostBusters ranks by F1.

**A10 — (c) Parameters.** Parameters are the values the model learns during training (e.g. weights/coefficients). Features are the inputs ($X$), labels are the target ($y$), and metrics are how you score the model afterward.

**A11 — (c) Recognising a cat in a photograph.** The rules are far too complex to write by hand and are hidden in the pixel patterns — ideal for ML. The other three are simple, exact, deterministic rules better solved with ordinary code.

**A12 — (b)** AUC-ROC summarises how well the model separates the two classes across every possible decision threshold; 1.0 is perfect separation and 0.5 is random guessing. It says nothing about speed, parameter count, or missing data.

**B1 — True.** That's the essence of ML: the rules/patterns are learned from data, not hand-coded by a programmer.

**B2 — False.** High *training* accuracy can simply mean the model memorised the training data (overfitting). Only performance on the unseen **test set** indicates generalization.

**B3 — True.** Underfitting = model too simple to capture the pattern, so it does badly on training *and* test data. (Contrast overfitting: great on train, poor on test.)

**B4 — False.** Regression and classification are the two main types of **supervised** learning, not unsupervised. Unsupervised tasks include clustering and dimensionality reduction.

**B5 — True.** Clustering groups similar examples without any predefined labels, which is exactly what makes it unsupervised.

**B6 — True.** High variance (over-sensitivity to the training set) → overfitting; high bias (too rigid an assumption) → underfitting. Good models balance the two.

**B7 — False.** When classes are imbalanced, accuracy is misleading — a model can score high while ignoring the minority class entirely. Use F1, precision, and recall instead.
