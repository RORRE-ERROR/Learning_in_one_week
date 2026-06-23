# 📚 Machine Learning in One Week — WIA1006/WID3006

Welcome. This is a self-contained study set that takes you from **zero to exam-ready** in the
core topics of WIA1006 Machine Learning, built from your **tutorials**, the **previous
midterm/quiz**, and your **group project** (the GhostBusters ghosting predictor).

The goal is **understanding, not cramming.** Each chapter teaches one idea at a time —
intuition first, then the formula, then a worked example, then exam tips — so the concepts
actually stick.

---

## How this is organised

There are **10 chapters**, each in its own folder containing two files:

- **`notes.md`** — the lesson. Read this first, slowly, with a pen.
- **`quiz.md`** — test yourself *closed-book*, then check the answer key at the bottom. The
  quizzes reuse and adapt the **real previous-midterm questions** plus tutorial problems.

Plus two top-level files:

- **`00_START_HERE.md`** — this file (the map + the week plan).
- **`Review_Before_Final.md`** — the night-before cheat sheet. One page per topic of the
  highest-yield facts, formulas, and traps. Use it *after* you've done the chapters.

```
00_START_HERE.md                      ← you are here
01_Foundations_of_ML/                 notes.md  quiz.md
02_Data_Preprocessing/                notes.md  quiz.md
03_Linear_Regression_and_Gradient_Descent/   notes.md  quiz.md
04_Logistic_Regression/               notes.md  quiz.md
05_Decision_Trees/                    notes.md  quiz.md
06_Neural_Networks/                   notes.md  quiz.md
07_Support_Vector_Machines/           notes.md  quiz.md
08_Clustering_GMM_and_PCA/            notes.md  quiz.md
09_Bayes_Theorem_and_HMM/             notes.md  quiz.md
10_Python_NumPy_Pandas/               notes.md  quiz.md   ← "what's the output?" code quiz
Review_Before_Final.md                ← last-minute cheat sheet
```

---

## 🗓️ The 7-day plan

Each day = read the notes, then do the quiz the *same* day while it's fresh. Re-do quizzes you
scored < 80% on the next morning (spaced repetition beats one long sitting).

| Day | Chapters | Focus | Why this order |
|----|----------|-------|----------------|
| **1** | 1. Foundations + 2. Data Preprocessing | The map + the most-tested topic | Preprocessing was *the* biggest slice of the midterm. |
| **2** | 3. Linear Regression & Gradient Descent | Your first model + the optimisation engine | Gradient descent reappears in logistic regression & neural nets. |
| **3** | 4. Logistic Regression | Classification + the sigmoid | Builds directly on Day 2. |
| **4** | 5. Decision Trees | Entropy & information gain | Self-contained; lots of midterm questions. |
| **5** | 6. Neural Networks | Perceptron → backpropagation | Pulls together gradient descent + sigmoid from Days 2–3. |
| **6** | 7. SVM + 8. Clustering, GMM & PCA | Margins & unsupervised learning | Two tutorial-heavy topics. |
| **7** | 9. Bayes & HMM + 10. Python/NumPy/Pandas | Probability + your project's toolkit | Finish with the code you'll defend in the viva, then review. |
| **Night before** | `Review_Before_Final.md` | Skim everything | Don't learn new things — consolidate. |

> If you have **fewer than 7 days**, prioritise by midterm weight:
> **Data Preprocessing → Decision Trees → Regression (linear+logistic) → Neural Networks**,
> then the rest. Those four account for the bulk of the previous quiz.

---

## How to study each chapter (the method that works)

1. **Read the notes once** for the story — don't stop to memorise. Just follow the logic.
2. **Read again with a pen**, re-deriving each worked example yourself on paper. *If you can't
   reproduce a worked example, you don't know it yet.*
3. **Do the quiz closed-book.** Write your answers down before peeking.
4. **Mark it** against the answer key. For every miss, read the explanation and the relevant
   notes section again.
5. **Collect your misses** into the margin of `Review_Before_Final.md`.

> 💡 **Biggest exam-score lever:** be able to *reproduce the math by hand* — entropy &
> information gain (Ch 5), the cost-function / gradient-descent calculations (Ch 3), the sigmoid
> (Ch 4), and the HMM sequence probability (Ch 9). Conceptual MCQs you can reason out; the
> numeric ones you must have *practised*.

---

## A note on the source material

- The **lecture slides originally in this folder were for a different course (networking,
  WIA1005)** and have been set aside — they are not machine learning. This study set is built
  from the materials that *are* WIA1006 ML: your `Tutorials/`, the `old_question/` midterm
  screenshots, and the `project/` notebook.
- Quizzes are grounded in the **actual previous quiz** so the style matches what you'll face.
- Where a real exam question had a debatable keyed answer, the notes flag it explicitly (see
  Chapter 5, the ID3 "large trees" question) so you understand the intent rather than just
  memorising a possibly-imperfect key.

Good luck — work through it day by day and you'll walk in confident. 🎯
