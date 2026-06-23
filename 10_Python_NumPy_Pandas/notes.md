# Chapter 10 — Python, NumPy & Pandas for Machine Learning

> **Day 7 of 7** (afternoon) · Prerequisites: Chapters 1–9 (they give you the ML *concepts*; this chapter gives you the *tools* that turn those concepts into running code) · Est. study time: ~75–90 min

**In one sentence:** This chapter teaches the small, practical slice of Python, NumPy, pandas, and scikit-learn that the GhostBusters group project actually uses — so you can read every code cell in your notebook and explain it out loud in the viva.

## Why this matters

Chapters 1–9 told you *what* logistic regression, decision trees, scaling, and F1-score mean. None of that runs by itself. The project notebook is ~30 code cells of Python that load a CSV, engineer features, build a preprocessing pipeline, train five models, and explain them with SHAP. In the oral/viva the examiner can point at **any line** and ask "what does this do and why?" This chapter walks through exactly those lines so you are never caught out by syntax.

## Learning goals

- [ ] Read and explain core Python: lists, list comprehensions, `lambda`, `.apply()`, f-strings, dicts.
- [ ] Use NumPy arrays: shape, dtype, broadcasting, `arange`, `ceil`, `unique`, `where`, `axis=0` vs `axis=1`.
- [ ] Use pandas: `read_csv`, `head`, `shape`, column selection, creating columns, `drop`, `value_counts`, `groupby().agg()`, `sort_values`, `isnull`.
- [ ] Understand the scikit-learn pipeline: `train_test_split(stratify=...)`, `StandardScaler`, `OneHotEncoder`, `SimpleImputer`, `ColumnTransformer`, `Pipeline`, `fit_transform` vs `transform`, `.fit/.predict/.predict_proba`, metrics, `GridSearchCV`.
- [ ] Walk through your own project notebook end-to-end and narrate each step.

---

## 1. Python essentials for ML

### 1.1 Lists vs arrays — what's the difference?

A **Python list** is a general container. It can hold anything (numbers, strings, other lists) and is flexible but slow for math.

```python
prices = [10, 20, 30]
prices * 2          # → [10, 20, 30, 10, 20, 30]   (repeats the list!)
```

A **NumPy array** holds one type of number and does *math element-by-element*:

```python
import numpy as np
arr = np.array([10, 20, 30])
arr * 2             # → array([20, 40, 60])   (multiplies each element)
```

> 💡 **Exam tip:** `list * 2` **repeats** the list; `np.array * 2` **doubles each number**. This is the single most common "what is the output?" trap.

### 1.2 List slicing

Slicing `[start:stop]` includes `start`, **excludes** `stop`.

```python
nums = [10, 20, 30, 40, 50]
nums[1:4]    # → [20, 30, 40]   (indices 1, 2, 3 — NOT 4)
nums[:2]     # → [10, 20]
nums[-1]     # → 50   (last element)
```

### 1.3 List comprehensions

A compact way to build a list from another sequence. Read it as "*give me `expr` for each `x` in `seq`*".

```python
squares = [x**2 for x in range(5)]
print(squares)      # → [0, 1, 4, 9, 16]
```

The project uses this pattern to build the EDA "gap" table:

```python
avg_ghosted = [ghosted[f].mean() for f in numeric_features]
```

That reads: "for each feature name `f` in `numeric_features`, compute the mean of that column among ghosters."

### 1.4 `lambda` + `.apply()` — the line everyone gets asked about

A **`lambda`** is a tiny one-line function with no name:

```python
square = lambda x: x * x
square(4)           # → 16
```

`.apply()` runs a function on **every value** of a pandas column. Now the exact project line, decoded piece by piece:

```python
df['interest_count'] = df['interest_tags'].apply(
    lambda x: len(x.split(',')) if isinstance(x, str) else 0
)
```

Read it inside-out, for one cell value `x`:

| Piece | Meaning |
|---|---|
| `df['interest_tags']` | the raw column of comma-separated strings, e.g. `"music,travel,gym"` |
| `.apply(lambda x: ...)` | run the lambda on **each** value `x` in that column |
| `isinstance(x, str)` | "is this value actually a string?" (guards against missing/`NaN` values) |
| `x.split(',')` | split the string on commas → a list, e.g. `["music","travel","gym"]` |
| `len(...)` | count how many items → `3` |
| `if ... else 0` | if `x` is a string, return the count; otherwise (missing) return `0` |
| `df['interest_count'] = ...` | store all the results as a brand-new column |

So `"music,travel,gym"` → `3`, and a missing value → `0`. This **feature engineering** turns messy text into one clean number the model can use.

> ⚠️ **Common trap:** Without the `isinstance(x, str)` guard, a missing value (`NaN`, which is a float) would crash on `.split` because floats have no `.split` method. The guard is not decoration — it prevents a runtime error.

### 1.5 f-strings (formatted strings)

Put an `f` before a string and embed expressions in `{}`. Format specifiers go after a colon.

```python
rate = 0.0987
print(f"Ghosting rate: {rate:.1%}")     # → Ghosting rate: 9.9%
print(f"Shape: {(2, 3)}")                # → Shape: (2, 3)
print(f"{1234567:,}")                    # → 1,234,567
```

Format mini-cheatsheet (all heavily used in the project's print statements):

| Spec | Input | Output | Meaning |
|---|---|---|---|
| `:.1%` | `0.1234` | `12.3%` | multiply by 100, 1 decimal, add `%` |
| `:.4f` | `0.9` | `0.9000` | fixed 4 decimals |
| `:,` | `50000` | `50,000` | thousands separator |
| `:.2f` | `3.14159` | `3.14` | 2 decimals |

> 💡 **Exam tip:** `:.1%` does the ×100 *for you*. `f"{0.1234:.1%}"` is `12.3%`, **not** `0.1%`. Mixing this up is a classic output question.

### 1.6 Dictionaries

A **dict** maps keys to values: `{key: value}`.

```python
params = {'n_estimators': 100, 'max_depth': 20}
params['max_depth']     # → 20
```

The project uses dicts for the GridSearch grid (`param_grid`) and for collecting model results (`all_models = {'Logistic Regression': (...), ...}`). `value_counts().to_dict()` turns a count table into a dict like `{0: 45000, 1: 5000}`.

---

## 2. NumPy

NumPy is the number-crunching engine under pandas and scikit-learn. Everything is an **ndarray** (n-dimensional array).

### 2.1 Creating arrays, `shape`, `dtype`

```python
import numpy as np
a = np.array([[1, 2, 3],
              [4, 5, 6]])
print(a.shape)    # → (2, 3)     2 rows, 3 columns
print(a.dtype)    # → int64      the element type
```

`shape` is a tuple `(rows, columns)`. You will read `X_train_final.shape` constantly to confirm "how many rows × how many features".

### 2.2 Vectorized arithmetic & broadcasting

Operations apply to every element at once — no loop needed. This is **vectorization**, and it's why NumPy is fast.

```python
arr = np.array([1, 2, 3])
print(arr * 2)        # → [2 4 6]
print(arr + 10)       # → [11 12 13]
print(arr ** 2)       # → [1 4 9]
```

**Broadcasting** is the rule that lets a small array stretch to fit a bigger one. The scalar `2` above is "broadcast" against every element. This is also exactly what `StandardScaler` does conceptually: `(column - mean) / std` applied to every value.

### 2.3 `np.arange` — like `range`, but an array

```python
np.arange(0, 10, 2)     # → array([0, 2, 4, 6, 8])
```

Same `start, stop, step` rule as slicing: **`stop` is excluded**. So it stops *before* 10. The project uses `np.arange(len(df_results))` to make x-positions for the comparison bar chart.

### 2.4 `np.ceil` — round up

```python
np.ceil(7 / 2)     # → 4.0    (3.5 rounded UP)
np.ceil(8 / 2)     # → 4.0    (4.0 stays 4.0)
```

The project uses it to figure out how many subplot rows it needs:

```python
n_rows = int(np.ceil(len(categorical_features) / n_cols))
```

With 8 features in 2 columns: `ceil(8/2) = 4` rows. If there were 9 features, `ceil(9/2) = 5` rows — you always round up so nothing is left without a slot.

### 2.5 `np.unique` — distinct sorted values

```python
np.unique(np.array([3, 1, 2, 1, 3, 3]))   # → array([1, 2, 3])
```

The project uses `len(np.unique(y_train)) == 2` to auto-detect "is this binary classification?" (two distinct labels → yes).

### 2.6 `np.where` — vectorized if/else

`np.where(condition, value_if_true, value_if_false)`:

```python
a = np.array([5, 12, 7, 20])
np.where(a > 10, 1, 0)     # → array([0, 1, 0, 1])
```

The project uses the single-argument form `np.where(y_pred == 1)[0]` to get the **indices** where the prediction is "Ghosted" (for the SHAP waterfall plot).

### 2.7 Indexing, slicing, boolean indexing

```python
a = np.array([10, 25, 30, 45, 5])
a[0]          # → 10            single element
a[1:3]        # → [25 30]       slice
a[a > 20]     # → [25 30 45]    boolean mask — keep only elements where True
```

**Boolean indexing** is everywhere in EDA: `df[df['is_ghosted'] == 1]` keeps only the ghoster rows.

### 2.8 `axis=0` vs `axis=1` — the one to really understand

For a 2-D array, `axis=0` goes **down the rows** (collapsing to one value per column), `axis=1` goes **across the columns** (one value per row).

```python
m = np.array([[1, 2],
              [3, 4],
              [5, 6]])
m.mean(axis=0)     # → [3. 4.]      mean of each COLUMN (down)
m.mean(axis=1)     # → [1.5 3.5 5.5]  mean of each ROW (across)
```

Mnemonic: **`axis=0` squashes the rows away → you get column stats.**

The SHAP code uses this directly:

```python
np.abs(shap_vals).mean(axis=0)
```

`shap_vals` has shape `(n_samples, n_features)`. Taking `axis=0` averages **down all samples** to get one mean-absolute-importance number **per feature** — exactly the "which feature matters most overall" summary.

```python
np.abs(np.array([[-1, 2], [3, -4]])).mean(axis=0)   # → [2. 3.]
# |−1|,|3| → mean 2 ;  |2|,|−4| → mean 3
```

> 💡 **Exam tip:** "Mean absolute SHAP per feature" = `np.abs(shap_vals).mean(axis=0)`. The `axis=0` is what makes it *per feature*. If you wrote `axis=1` you'd get one number per sample instead — wrong direction.

---

## 3. pandas

pandas gives you the **DataFrame** (a table, like a spreadsheet) and the **Series** (a single column). This is where your data lives before it becomes NumPy arrays for the model.

### 3.1 Loading and looking

```python
import pandas as pd
df = pd.read_csv('dating_app_behavior_dataset.csv')
df.head()        # first 5 rows
df.shape         # → (50000, 19)   rows, columns
```

`df.head()` is your first sanity check — does the data look right? `df.shape` confirms the size (50,000 records, 19 features in this project).

### 3.2 Selecting columns

```python
df['message_sent_count']         # one column → a Series
df[['gender', 'income_bracket']] # several columns → a DataFrame (note double [[ ]])
```

### 3.3 Creating a column — the target variable

```python
df['is_ghosted'] = (df['match_outcome'] == 'Ghosted').astype(int)
```

Decoded:
1. `df['match_outcome'] == 'Ghosted'` → a boolean Series: `True` where the outcome was "Ghosted", `False` otherwise.
2. `.astype(int)` → converts `True → 1`, `False → 0`.
3. Assign it as a new column `is_ghosted`.

This is how a messy multi-class column (`Ghosted`, `Mutual Match`, `Catfished`, …) becomes a clean **binary target** for classification.

```python
pd.Series([True, False, True, True]).astype(int)   # → [1, 0, 1, 1]
```

> 💡 **Exam tip:** `(some_condition).astype(int)` is the standard recipe for "make a 0/1 label from a True/False condition." `True` is 1, `False` is 0. And `bool_series.astype(int).sum()` counts how many were `True`.

### 3.4 Dropping columns

```python
df = df.drop(columns=['match_outcome'])
```

After building `is_ghosted` from `match_outcome`, the raw column is dropped (it would leak the answer). Likewise `interest_tags` is dropped after counting it into `interest_count`.

> ⚠️ **Common trap:** Leaving `match_outcome` in `X` would be **target leakage** — the model would "cheat" by reading the answer. Always drop the source of a derived target.

### 3.5 `value_counts()` and `value_counts(normalize=True)`

`value_counts()` tallies how many times each value appears (most frequent first):

```python
s = pd.Series(['a', 'b', 'a', 'c', 'a', 'b'])
s.value_counts()
# a    3
# b    2
# c    1
```

Add `normalize=True` to get **proportions** instead of counts:

```python
s.value_counts(normalize=True)
# a    0.500000   (3/6)
# b    0.333333   (2/6)
# c    0.166667   (1/6)
```

The project uses `value_counts()` to check class balance and `value_counts(normalize=True).to_dict()` to report the class distribution as fractions.

### 3.6 `groupby(...).agg(['mean', 'count'])`

`groupby` splits the data into groups, then `.agg()` computes statistics per group. This is the heart of the EDA ("who ghosts most?").

```python
df = pd.DataFrame({'g': ['x', 'x', 'y', 'y'], 'v': [1, 3, 10, 20]})
df.groupby('g')['v'].agg(['mean', 'count'])
#    mean  count
# g
# x   2.0      2     (mean of 1,3)
# y  15.0      2     (mean of 10,20)
```

In the project: `df.groupby(feat)['is_ghosted'].agg(['mean', 'count'])`. Because `is_ghosted` is 0/1, its **mean is the ghosting rate** for that group (e.g. mean 0.12 = 12% ghosted), and `count` is the group size.

> 💡 **Exam tip:** The **mean of a 0/1 column is a proportion/probability**. `df['is_ghosted'].mean()` = overall ghosting rate. This trick appears all over the EDA.

### 3.7 `sort_values`, `.mean()`, `.isnull().any()`

```python
df.sort_values('mean', ascending=False)   # rank groups, highest rate first
df['app_usage_time_min'].mean()           # average of a column
df.isnull().any()                          # per column: any missing values? → True/False
```

```python
pd.DataFrame({'a': [1, None], 'b': [2, 3]}).isnull().any()
# a     True     (has a missing value)
# b    False
```

`.isnull().any().any()` (two `.any()`s) collapses the whole table to a single True/False: "are there *any* missing values *anywhere*?"

### 3.8 `pd.Series` and `pd.DataFrame(...)`

You can build these by hand:

```python
pd.Series([1, 2, 3])                       # a single labelled column
pd.DataFrame({'feature': names, 'score': scores})   # a table from a dict of columns
```

The project wraps the scaled NumPy output back into a DataFrame so column names survive for SHAP:

```python
X_train_final = pd.DataFrame(X_train_raw, columns=all_feature_names)
```

---

## 4. The scikit-learn pipeline (ties back to Chapter 2)

This is the assembly line that turns raw `df` into a trained, evaluated model. Follow the order — it's the order in your notebook.

### 4.1 Split first: `train_test_split(..., stratify=y)`

```python
X = df.drop(columns=['is_ghosted'])
y = df['is_ghosted']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

- `test_size=0.2` → 80% train, 20% test.
- `random_state=42` → reproducible split (same every run).
- `stratify=y` → keep the **same class ratio** (≈ same % ghosted) in train and test. Crucial when classes are imbalanced — without it, the test set might get too few ghosters to evaluate fairly.

> ⚠️ **Common trap:** You split **before** fitting any scaler/encoder. If you scale the whole dataset first and then split, information from the test set leaks into training. Split → fit on train → transform both.

### 4.2 `SimpleImputer` — fill missing values

```python
SimpleImputer(strategy='median')                          # numbers → fill with column median
SimpleImputer(strategy='constant', fill_value='missing')  # categories → fill with "missing"
```

Models can't handle `NaN`. The imputer fills the gaps before scaling/encoding.

### 4.3 `StandardScaler` — put numbers on the same scale

Transforms each numeric column to **mean 0, standard deviation 1**: $z = \dfrac{x - \mu}{\sigma}$.

This stops a large-range feature (like `app_usage_time_min`, in hundreds) from dominating a small-range one (like `swipe_right_ratio`, between 0 and 1) just because of units. Linear models, SVM, and neural nets need this.

> 💡 **Exam tip:** After `StandardScaler`, each scaled feature has mean ≈ 0 and std ≈ 1. (True/False bait.)

### 4.4 `OneHotEncoder` — turn categories into numbers

A category like `gender ∈ {Male, Female, Other}` becomes separate 0/1 columns (`gender_Male`, `gender_Female`, `gender_Other`). Models do math on numbers, not text.

`handle_unknown='ignore'` means a category seen only at test time won't crash the model — it just becomes all zeros.

### 4.5 `Pipeline` and `ColumnTransformer`

A **`Pipeline`** chains steps so they run in order (impute → scale):

```python
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
```

A **`ColumnTransformer`** applies different pipelines to different columns — numbers get scaled, categories get one-hot encoded — in one object:

```python
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])
```

This is why the project's preprocessing is one tidy block instead of a mess of separate steps.

### 4.6 `fit_transform` vs `transform` — the data-leakage rule

```python
X_train_raw = preprocessor.fit_transform(X_train)   # LEARN params (means, categories) AND apply
X_test_raw  = preprocessor.transform(X_test)         # only APPLY the train-learned params
```

- **`fit_transform`** on **train**: learns the medians, means, stds, category lists *from the training data*, then transforms it.
- **`transform`** on **test**: reuses exactly those train-learned numbers. It does **not** look at the test set's own statistics.

> ⚠️ **Common trap (the big one):** Calling `fit_transform` on the **test set** is data leakage — the test set is supposed to simulate unseen data, so the model must never learn anything from it. Test set = `transform` only.

### 4.7 `.fit`, `.predict`, `.predict_proba`

```python
log_reg.fit(X_train_final, y_train)            # train
y_pred  = log_reg.predict(X_test_final)         # hard labels: 0 or 1
y_prob  = log_reg.predict_proba(X_test_final)[:, 1]   # probability of class 1 (Ghosted)
```

`predict` gives the class; `predict_proba` gives probabilities, and `[:, 1]` grabs the **second column = probability of the positive class** (Ghosted). You need probabilities for AUC-ROC.

### 4.8 Metrics

| Metric | Function | Plain meaning |
|---|---|---|
| Accuracy | `accuracy_score(y_test, y_pred)` | fraction correct overall (misleading when imbalanced) |
| F1-score | `f1_score(y_test, y_pred)` | harmonic mean of precision & recall — the project's main ranking metric |
| Precision | `precision_score(...)` | of predicted ghosters, how many really ghosted |
| Recall | `recall_score(...)` | of all real ghosters, how many we caught |
| AUC-ROC | `roc_auc_score(y_test, y_prob)` | ability to separate classes across all thresholds (uses probabilities) |

> 💡 **Exam tip:** With heavy class imbalance, **high accuracy can be a lie**. The project's basic models hit ~90% accuracy but tiny F1 — they just predicted "Not Ghosted" for almost everyone. That's why F1 is the ranking metric.

### 4.9 `GridSearchCV` — automatic hyperparameter tuning

```python
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=3, scoring='f1', n_jobs=-1)
grid_search.fit(X_train_final, y_train)
best_rf = grid_search.best_estimator_
```

It tries **every combination** in the grid (here 3 × 3 = 9 combos), evaluates each with **3-fold cross-validation** scored by **F1**, and keeps the best. `best_estimator_` is the winning, already-refitted model. `n_jobs=-1` uses all CPU cores.

---

## 5. Reading the project notebook (for your viva)

A spoken walkthrough you can give in 90 seconds, mapped to the notebook sections:

1. **Load (Sec 4):** `pd.read_csv` → `df`, 50,000 rows × 19 columns; `df.head()` to eyeball it.
2. **Make the target (Sec 5.1):** `df['is_ghosted'] = (df['match_outcome'] == 'Ghosted').astype(int)`, then drop `match_outcome` to avoid leakage.
3. **Feature engineering (Sec 5.2):** `interest_tags` (comma text) → `interest_count` via `.apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)`; drop the raw text.
4. **Preprocess (Sec 5.3):** split columns into numeric vs categorical; numeric → impute median + `StandardScaler`; categorical → impute "missing" + `OneHotEncoder`; combine in a `ColumnTransformer`.
5. **Split & transform (Sec 5.4):** `train_test_split(..., stratify=y)` **first**, then `fit_transform` on train and `transform` on test (no leakage); wrap back into DataFrames so feature names survive for SHAP.
6. **EDA (Sec 6):** `value_counts` for base rate; `groupby(feat)['is_ghosted'].mean()` for per-group ghosting rates; list comprehensions over `numeric_features` for the behavioural gap table.
7. **Models (Sec 7–8):** `.fit` / `.predict` / `.predict_proba` for Logistic Regression, Decision Tree, `GridSearchCV` Random Forest, SVM (SGDClassifier hinge loss), MLP (with SMOTE). `class_weight='balanced'` for imbalance.
8. **Compare (Sec 9):** build a results DataFrame, `sort_values('F1-Score')`, plot.
9. **Explain (Sec 11):** SHAP; `np.abs(shap_vals).mean(axis=0)` for per-feature importance; `np.where(y_pred == 1)[0]` to pick a ghosted sample for the local waterfall plot.

If the examiner points at a line, name (a) what data type goes in, (b) what comes out, (c) why it's there.

---

## Key formulas / facts at a glance

| Thing | Rule / Code | Plain meaning |
|---|---|---|
| List vs array `* 2` | list repeats, array doubles each element | classic output trap |
| Slice/`arange`/`range` | `stop` is **excluded** | `nums[1:4]` → indices 1,2,3 |
| `(cond).astype(int)` | `True→1`, `False→0` | make a 0/1 label |
| Mean of 0/1 column | `df['is_ghosted'].mean()` | the proportion / rate |
| `value_counts(normalize=True)` | counts ÷ total | proportions |
| `axis=0` | collapse rows → per-**column** stat | `mean(axis=0)` = column means |
| `axis=1` | collapse columns → per-**row** stat | one value per sample |
| `np.abs(s).mean(axis=0)` | per-feature mean abs SHAP | global importance |
| `f"{x:.1%}"` | ×100, 1 dp, `%` | `0.1234 → 12.3%` |
| `fit_transform` | learn + apply | **train only** |
| `transform` | apply learned params | **test** (prevents leakage) |
| `predict_proba(...)[:, 1]` | prob of class 1 | positive-class probability |
| StandardScaler | $z=(x-\mu)/\sigma$ | scaled mean 0, std 1 |
| `stratify=y` | preserve class ratio in split | fair test set under imbalance |
| `GridSearchCV` | try all param combos × CV | pick best hyperparameters |

## Exam tips — quick recap

- `list * 2` repeats; `np.array * 2` doubles each element.
- `stop` is always excluded in slices, `range`, and `np.arange`.
- `(condition).astype(int)` is the 0/1-label recipe; mean of a 0/1 column is a rate.
- `axis=0` gives per-**column** stats (squashes rows); `axis=1` gives per-**row** stats.
- `f"{0.1234:.1%}"` → `12.3%` (the `%` does the ×100).
- `fit_transform` on **train**, `transform` on **test** — never the reverse (leakage).
- High accuracy under class imbalance can be meaningless; F1 is the honest ranking metric here.

## Self-check

1. What does `[10, 20, 30, 40, 50][1:4]` print, and why isn't 50 in it?
2. Why does the `interest_count` lambda need `if isinstance(x, str) else 0`?
3. What is the difference between calling `fit_transform` and `transform`, and which goes on the test set?
4. For a 2-D array, does `.mean(axis=0)` give you one number per row or per column?
5. Why is the *mean* of the `is_ghosted` column the same thing as the ghosting rate?

**→ Now test yourself with [`quiz.md`](./quiz.md).**
