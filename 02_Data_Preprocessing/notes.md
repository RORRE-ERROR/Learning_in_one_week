# Chapter 2 — Data Preprocessing

> **Day 1 of 7** (afternoon) · Prerequisites: Chapter 1 · Est. study time: ~75–90 min

**In one sentence:** Data preprocessing is the work of turning messy, raw, real-world data into a clean, consistent, numeric format that a machine learning model can actually learn from.

## Why this matters

This is the single most heavily tested topic on the WIA1006 midterm — and for good reason. In any real ML project, you spend far more time cleaning and shaping data than building models. The slogan to burn into memory is **"garbage in → garbage out"**: even the fanciest model produces nonsense if you feed it broken data. The project notebook you'll study (the GhostBusters dating-app classifier) spends an entire section on preprocessing *before* a single model is trained — that ordering is not an accident.

## Learning goals

- [ ] Explain *why* preprocessing exists and what problems raw data has
- [ ] Recite the four preprocessing steps **in the correct order**
- [ ] Handle missing values three ways and know the consequence of doing it wrong
- [ ] Handle noisy data with binning, clustering, regression (and know SMOTE is *not* for noise)
- [ ] Normalize / scale features and explain *why* it speeds up learning
- [ ] Encode categorical data with one-hot encoding
- [ ] Distinguish random vs stratified sampling, and when to stratify
- [ ] Split into train/test correctly and avoid **data leakage**
- [ ] Recognize dimensionality reduction as a preprocessing step (PCA, feature selection)

---

## 1. Why preprocess at all? (garbage in → garbage out)

Imagine you're a chef. You can be the best cook alive, but if your ingredients are rotten, mislabeled, and half-missing, the meal will be terrible. The model is the chef; the data is the ingredients. Preprocessing is washing, peeling, and measuring before you cook.

Raw data collected from the real world is almost never ready to use. Typical problems:

- **Missing values** — a user left a field blank, a sensor dropped a reading.
- **Outliers** — one user reports an income of 9,999,999; a typo or a freak case.
- **Noise** — random errors or meaningless jitter in measurements.
- **Meaningless data combinations** — fields that don't make sense together, duplicates, or features that carry no signal.
- **Inconsistent formats** — dates as "2024-01-01" in one column and "Jan 1 2024" in another; "Male"/"M"/"male" all meaning the same thing.

The *best description* of preprocessing's role is: **converting raw data into a reliable, clean format to enhance data quality and produce meaningful insights.** It is not "only for large datasets" and not "only about missing values and encoding" — those are too narrow.

> 💡 **Exam tip:** The "BEST describes the role of preprocessing" question wants the *broad* answer about data quality and meaningful insights (midterm Q38), not a narrow one-task answer.

> ⚠️ **Common trap (Q16):** "Standardized data formats that are universally compatible with ML algorithms" is **NOT** a problem of raw data — it's the *opposite* of a problem. The actual raw-data challenges are missing values, outliers, and meaningless combinations. The question asks which is NOT a challenge.

---

## 2. The four steps — and their order

Preprocessing has a standard pipeline. Memorize the order; the exam tests it directly:

> **Data cleaning → Data integration → Data transformation → Data reduction**

A mnemonic: **C·I·T·R** ("Clean It, Then Reduce"). Walk through it:

1. **Data cleaning** — fix the broken stuff first: handle missing values, smooth noise, remove/treat outliers. *This is always the first step.*
2. **Data integration** — merge data from multiple sources/tables into one coherent dataset (resolve duplicate columns, conflicting names).
3. **Data transformation** — reshape values into model-friendly form: normalization/scaling, encoding categories, aggregation.
4. **Data reduction** — shrink the data without losing meaning: dimensionality reduction (PCA), feature selection, sampling.

> 💡 **Exam tip (Q36):** "What is the first step in data preprocessing?" → **Data cleaning.** It comes before integration, transformation, and reduction. The logic: there's no point merging or transforming data that's still full of errors.

> ⚠️ **Common trap:** Don't confuse *transformation* (scaling, encoding) with *reduction* (PCA, feature selection). Scaling reshapes existing features; reduction removes or compresses them.

---

## 3. Handling missing values

Almost every real dataset has blanks. You have three broad options.

**Option A — Drop them.**
- Drop the *rows* with missing values, or drop a *column* if it's mostly empty.
- Simple, but you throw away data. Fine if only a tiny fraction is missing; dangerous if you'd lose a big chunk or bias the sample.

**Option B — Impute (fill in) them.** This is the usual choice.
- **Mean imputation** — fill with the column's average. Good for roughly symmetric numeric data.
- **Median imputation** — fill with the middle value. Better when there are outliers (the median isn't dragged around by a few huge values).
- **Mode imputation** — fill with the most frequent value. Used for *categorical* data (e.g., most common city).

**How the project does it.** In the GhostBusters notebook, missing values are filled with `SimpleImputer`:

```python
# numeric columns → fill blanks with the column median
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# categorical columns → fill blanks with a constant placeholder
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
```

Notice they chose **median** for numbers (robust to outliers) and a **constant** placeholder for categories.

> 💡 **Exam tip:** Match the imputation strategy to the data type — *mean/median* for numeric, *mode* for categorical.

> ⚠️ **Common trap (Q32):** What happens if you handle missing values *badly*? The answer is **inaccurate and faulty conclusions / inferences.** Bad imputation distorts the data's statistics, so every downstream result is built on sand. (The tempting wrong answers — "harder to deploy", "more compute", "less interpretable" — are real annoyances but miss the core danger: wrong conclusions.)

---

## 4. Handling noisy data

**Noise** = random error or variance in the data, not the true signal. Three classic smoothing techniques:

- **Binning** — sort values into buckets, then smooth within each bucket (e.g., replace every value in a bin by the bin's mean or median). Like rounding ages into "20s, 30s, 40s."
- **Clustering** — group similar points; points that fall outside any cluster are flagged as noise/outliers.
- **Regression** — fit a function (e.g., a line) to the data and use the fitted, smooth values in place of the noisy raw ones.

> 💡 **Exam tip (Q5):** Strategies to handle noisy data = **Binning, Clustering, Regression.** This is often a "select all that apply" question.

> ⚠️ **Common trap (Q5):** **SMOTE is NOT a noise-handling technique.** SMOTE (Synthetic Minority Over-sampling Technique) creates synthetic examples of a *minority class* to fix **class imbalance** — a completely different problem. If you see SMOTE listed as a noise option, do not select it.

---

## 5. Normalization / feature scaling

**The intuition.** Suppose one feature is *age* (range 18–70) and another is *annual income* (range 10,000–500,000). To a model that measures distances or sums weighted features, income's raw numbers *dwarf* age — not because income matters more, but because its numbers are simply bigger. Scaling fixes this so each feature gets a fair say.

**Why we do it (two reasons the exam loves):**
1. **Equal contribution to distance calculations.** Algorithms that use distances (KNN, K-means, SVM) or weighted sums are dominated by large-magnitude features unless you scale.
2. **Faster gradient descent.** When features share a comparable scale, the loss surface is more "round" and gradient descent converges faster and more stably (you avoid zig-zagging down a stretched valley).

**Two main methods:**

**Min–max normalization** — squashes a feature into a fixed range, usually $[0, 1]$ (or $[-1, 1]$):

$$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

where $x$ is the original value, $x_{\min}$ / $x_{\max}$ are the column's smallest/largest values, and $x'$ is the scaled value in $[0,1]$.

*Worked example.* Ages 20, 30, 50; so $x_{\min}=20$, $x_{\max}=50$.
- For $x = 30$: $x' = \frac{30 - 20}{50 - 20} = \frac{10}{30} = 0.33$.
- For $x = 20$: $x' = 0$. For $x = 50$: $x' = 1$. ✓

**Standardization (z-score)** — re-centers a feature to mean 0 and rescales to standard deviation 1:

$$z = \frac{x - \mu}{\sigma}$$

where $\mu$ is the column mean and $\sigma$ is the column standard deviation. A value exactly at the mean becomes 0; one standard deviation above becomes +1.

*Worked example.* Values with $\mu = 30$, $\sigma = 10$.
- For $x = 50$: $z = \frac{50 - 30}{10} = 2$ (two standard deviations above the mean).
- For $x = 30$: $z = 0$. ✓

**How the project does it.** The GhostBusters notebook scales all numeric features with `StandardScaler` (z-score):

```python
('scaler', StandardScaler())   # inside the numeric pipeline
```

> 💡 **Exam tip (Q23, Q39):** "Why normalize?" → **to scale features to a specific range (e.g., 0–1 or −1 to 1)** so that **all features contribute equally to distance calculations.** Both phrasings appear on the midterm — they're the same idea.

> ⚠️ **Common trap:** Normalization does *not* handle missing values, remove duplicates, or combine datasets — those are different steps. It only rescales numbers.

---

## 6. Encoding categorical data

Models do arithmetic; they can't multiply the word "Male." So categorical text must become numbers.

**One-hot encoding** — create one new binary (0/1) column per category. A `gender` column with values {Male, Female} becomes two columns `gender_Male`, `gender_Female`, each 0 or 1.

| gender | → | gender_Male | gender_Female |
|---|---|---|---|
| Male   |   | 1 | 0 |
| Female |   | 0 | 1 |

**Why one-hot and not just numbering** (Male=0, Female=1, Other=2)? Because plain integer labels imply a false *order and distance* — the model would think "Other" (2) is somehow "twice" Female (1) or greater than Male (0). One-hot avoids inventing fake ordering for categories that have none.

**How the project does it.** `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` is applied to all categorical columns (gender, income bracket, education level, etc.). `handle_unknown='ignore'` means: if a category appears at test time that wasn't seen in training, don't crash — just encode it as all zeros.

> 💡 **Exam tip:** One-hot encoding is the standard transform for *nominal* (unordered) categories. It lives in the **transformation** step.

---

## 7. Sampling: random vs stratified

Sometimes you take a *sample* of the data (to train faster, or to build train/test splits). How you sample matters.

- **Random sampling** — pick rows uniformly at random. Simple, but on **imbalanced** data it can leave you with too few (or zero) examples of a rare class purely by luck.
- **Stratified sampling** — first split the data into groups (*strata*) by class, then sample proportionally from each. This **guarantees every class is represented in proportion to the full dataset.**

**Why it matters here.** In the GhostBusters dataset, ghosting is the minority class — far fewer "Ghosted" users than "Not Ghosted." A purely random test set might contain almost no ghosters, making the evaluation meaningless. So the project splits with `stratify=y`:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y   # keep class ratios identical
)
```

> 💡 **Exam tip (Q28):** Stratified sampling is preferred because it **ensures equal (proportional) representation across all classes or groups.** That's the whole point — especially valuable with class imbalance.

---

## 8. Train/test split and data leakage

**Why split.** You can't grade an exam on the same questions you let students memorize. To know if a model really *learned* (rather than *memorized*), you must test it on data it has **never seen**. So you hold out a portion (commonly 20%) as a **test set**, train on the rest, and report performance on the test set.

> 💡 **Exam tip (Q26):** The PRIMARY reason to split into train/test is **to evaluate the model on unseen data and prevent overfitting.** Not "to reduce size" or "to create multiple versions."

**Data leakage** — the silent killer. Leakage is when information from the test set sneaks into training, giving a falsely optimistic score. The classic mistake: **fitting your scaler/imputer on the whole dataset before splitting.** Then the training data's scaling "knows" the test set's mean and range — that's cheating.

**The correct recipe** (and exactly what the project does):
1. **Split first** into train and test.
2. **`fit`** the scaler/imputer/encoder **on the training set only** (learn means, ranges, categories from train).
3. **`transform`** both train and test using those learned values.

```python
# Split BEFORE fitting to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(..., stratify=y)

X_train_raw = preprocessor.fit_transform(X_train)   # FIT on train
X_test_raw  = preprocessor.transform(X_test)         # only TRANSFORM test
```

Note the asymmetry: `fit_transform` on train, plain `transform` on test. That single discipline is what prevents leakage.

> ⚠️ **Common trap:** Scaling the full dataset and *then* splitting feels harmless but leaks test statistics into training. Always split first.

---

## 9. Data reduction: dimensionality reduction & feature selection

The fourth preprocessing step shrinks the data. Two flavours:

- **Feature selection** — *keep a subset* of the original features and drop the rest (e.g., remove low-information or redundant columns). The features you keep are unchanged.
- **Feature extraction / PCA** — *create new, fewer features* that are combinations of the originals. **Principal Component Analysis (PCA)** finds new axes that capture the most variance, letting you keep, say, 5 components instead of 50 columns. *(Full PCA math is in Chapter 8 — beyond today, optional for now.)*

Both **feature selection** and **PCA** are examples of **dimensionality reduction**.

**Why reduce dimensions?** Fewer, more relevant features mean less noise, faster training, and — importantly — **less overfitting.** A model with too many features can memorize quirks of the training data; trimming features helps it generalize.

> 💡 **Exam tip (Q29):** "Which is an example of dimensionality reduction?" → **PCA *and* feature selection** (both). Don't pick only one if "both" is an option. Sampling is *not* dimensionality reduction — it reduces *rows*, not *features/dimensions*.

> 💡 **Exam tip (Q17):** "Which preprocessing step helps avoid overfitting?" → **Feature selection.** (Reducing the number of features reduces the model's chance to memorize noise.) Note normalization, integration, and cleaning don't directly target overfitting.

---

## 10. Class imbalance and SMOTE (brief)

When one class massively outnumbers another (here: few ghosters, many non-ghosters), a model can score high "accuracy" just by always predicting the majority — while completely failing on the rare class. That's why the project's basic models hit ~90% accuracy but near-zero F1: they ignored the minority.

**SMOTE (Synthetic Minority Over-sampling Technique)** fixes this by *generating synthetic examples* of the minority class until the classes are balanced, so the model is forced to learn the rare pattern. The project applies SMOTE to the training data before fitting the neural network:

```python
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_final, y_train)
```

> ⚠️ **Common trap:** Re-stating section 4 because it's *that* heavily tested — **SMOTE is for class imbalance, not for noise.** Apply it to the *training set only* (never the test set), to keep evaluation honest.

---

## Key formulas / facts at a glance

| Thing | Formula / Rule | Plain meaning |
|---|---|---|
| Preprocessing order | Cleaning → Integration → Transformation → Reduction | "Clean It, Then Reduce"; cleaning is **first** |
| Why preprocess | garbage in → garbage out | quality of output is capped by quality of input |
| Missing values | drop, or impute with mean / median / mode | median for numbers w/ outliers; mode for categories |
| Noise handling | binning, clustering, regression | smooth/group out random error (NOT SMOTE) |
| Min–max scaling | $x' = \dfrac{x - x_{\min}}{x_{\max}-x_{\min}}$ | squash into $[0,1]$ |
| Standardization | $z = \dfrac{x - \mu}{\sigma}$ | mean 0, std 1 (z-score) |
| Why scale | equal contribution to distances; faster gradient descent | big-range features stop dominating |
| One-hot encoding | one 0/1 column per category | turn text categories into numbers w/o fake order |
| Stratified sampling | sample proportionally per class | equal representation across classes |
| Train/test split | evaluate on unseen data | detect/prevent overfitting |
| Avoid leakage | split first; `fit` scaler on **train only** | test stats must not touch training |
| Dimensionality reduction | PCA + feature selection | fewer features → less overfitting, faster |
| SMOTE | synthesize minority-class examples | fixes class imbalance (not noise) |

## Exam tips — quick recap

- First preprocessing step = **data cleaning** (order: Clean → Integrate → Transform → Reduce).
- "Best describes preprocessing" = converting raw data to a **clean, reliable format for meaningful insights** (broad answer).
- Raw-data challenges = missing values, outliers, meaningless combinations — *standardized universally-compatible formats is NOT one* (Q16).
- Bad missing-value handling → **inaccurate / faulty conclusions** (Q32).
- Noise → binning / clustering / regression; **SMOTE is not for noise** (Q5).
- Normalize/scale → features **contribute equally to distance calcs**, range like 0–1 or −1 to 1 (Q23, Q39).
- Stratified sampling → **equal/proportional representation across classes** (Q28).
- Train/test split → **evaluate on unseen data, prevent overfitting** (Q26); fit scaler on train only.
- Dimensionality reduction = **PCA *and* feature selection** (Q29); feature selection helps avoid overfitting (Q17).

## Self-check

1. Put the four preprocessing steps in order. Which one is always first?
2. You have a numeric column with a few extreme outliers and some blanks. Which imputation strategy is safest, and why?
3. Give the two reasons feature scaling helps a model. Which scaling formula gives mean 0 and std 1?
4. Why does the project pass `stratify=y` to `train_test_split`?
5. Explain data leakage in one sentence, and state the rule that prevents it.

**→ Now test yourself with [`quiz.md`](./quiz.md).**
