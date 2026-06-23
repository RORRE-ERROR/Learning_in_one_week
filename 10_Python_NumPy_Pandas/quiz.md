# Chapter 10 Quiz — Python, NumPy & Pandas for Machine Learning

> **Special format:** This quiz is **"What is the output of this code?"** style — exactly the way WIA1006 tests tooling knowledge. Read each snippet, predict the printed output (closed-book first), then check the answer key at the bottom. Every snippet is small, deterministic, and based on patterns from the GhostBusters project notebook.

---

## Section A — What is the output? (Multiple Choice / Fill-in)

**A1.** List slicing.
```python
nums = [10, 20, 30, 40, 50]
print(nums[1:4])
```
- a. `[10, 20, 30]`
- b. `[20, 30, 40]`
- c. `[20, 30, 40, 50]`
- d. `[10, 20, 30, 40]`

---

**A2.** `np.arange`.
```python
import numpy as np
print(np.arange(0, 10, 2))
```
- a. `[0 1 2 3 4 5 6 7 8 9]`
- b. `[2 4 6 8 10]`
- c. `[0 2 4 6 8]`
- d. `[0 2 4 6 8 10]`

---

**A3.** List vs array — broadcasting.
```python
import numpy as np
print([1, 2, 3] * 2)
print(np.array([1, 2, 3]) * 2)
```
- a. `[2, 4, 6]` then `[2 4 6]`
- b. `[1, 2, 3, 1, 2, 3]` then `[2 4 6]`
- c. `[1, 2, 3, 1, 2, 3]` then `[1 2 3 1 2 3]`
- d. `[2, 4, 6]` then `[1 2 3 1 2 3]`

---

**A4.** `.astype(int)` on a boolean Series (the target-variable recipe).
```python
import pandas as pd
s = pd.Series([True, False, True, True])
print(s.astype(int).tolist())
```
- a. `[1, 0, 1, 1]`
- b. `[True, False, True, True]`
- c. `[1, 1, 1, 1]`
- d. `[0, 1, 0, 0]`

---

**A5.** `np.where`.
```python
import numpy as np
a = np.array([5, 12, 7, 20])
print(np.where(a > 10, 1, 0))
```
- a. `[1 0 1 0]`
- b. `[0 1 0 1]`
- c. `[5 12 7 20]`
- d. `[12 20]`

---

**A6.** `value_counts()`.
```python
import pandas as pd
s = pd.Series(['a', 'b', 'a', 'c', 'a', 'b'])
print(s.value_counts())
```
- a. `a 3 / b 2 / c 1`
- b. `a 1 / b 2 / c 3`
- c. `a 0.5 / b 0.33 / c 0.17`
- d. `a 2 / b 2 / c 2`

(Order the counts top-to-bottom.)

---

**A7.** The `interest_count` lambda, on one value.
```python
x = "rock,music,travel"
print(len(x.split(',')))
```
- a. `1`
- b. `2`
- c. `3`
- d. `17`

---

**A8.** `axis=0` mean (the mean-absolute-SHAP pattern).
```python
import numpy as np
m = np.array([[1, 2],
              [3, 4],
              [5, 6]])
print(m.mean(axis=0))
```
- a. `[1.5 3.5 5.5]`
- b. `[3. 4.]`
- c. `[3.5]`
- d. `[2. 4. 6.]`

---

**A9.** Boolean indexing.
```python
import numpy as np
a = np.array([10, 25, 30, 45, 5])
print(a[a > 20])
```
- a. `[10 25 30 45 5]`
- b. `[False True True True False]`
- c. `[25 30 45]`
- d. `[10 5]`

---

**A10.** f-string percent formatting.
```python
rate = 0.1234
print(f"{rate:.1%}")
```
- a. `0.1%`
- b. `12.3%`
- c. `0.1234%`
- d. `12.34%`

---

**A11.** `np.abs(...).mean(axis=0)` — exactly the SHAP importance line.
```python
import numpy as np
v = np.array([[-1, 2],
              [ 3, -4]])
print(np.abs(v).mean(axis=0))
```
- a. `[1. 3.]`
- b. `[2. 3.]`
- c. `[0.5 -1.]`
- d. `[2.5 3.]`

---

**A12.** `groupby().agg(['mean', 'count'])` (per-group ghosting rate pattern).
```python
import pandas as pd
df = pd.DataFrame({'g': ['x', 'x', 'y', 'y'], 'v': [1, 3, 10, 20]})
print(df.groupby('g')['v'].agg(['mean', 'count']))
```
- a. `x: mean 2.0 count 2 ; y: mean 15.0 count 2`
- b. `x: mean 4.0 count 2 ; y: mean 30.0 count 2`
- c. `x: mean 1.0 count 1 ; y: mean 10.0 count 1`
- d. `x: mean 2.0 count 4 ; y: mean 15.0 count 4`

---

## Section B — True / False (pandas / sklearn behaviour)

**B1.** `fit_transform` should be called on the **test** set. — (True / False)

**B2.** After `StandardScaler`, each scaled numeric feature has mean ≈ 0 and standard deviation ≈ 1. — (True / False)

**B3.** `df['is_ghosted'].mean()` gives the overall ghosting *rate* (because the column is 0/1). — (True / False)

**B4.** `np.arange(0, 10, 2)` includes the value `10`. — (True / False)

**B5.** `predict_proba(X)[:, 1]` returns the predicted probability of the **positive** class (Ghosted). — (True / False)

**B6.** Setting `stratify=y` in `train_test_split` keeps roughly the same class proportions in the train and test sets. — (True / False)

**B7.** Under heavy class imbalance, a high accuracy score guarantees the model is good at finding the rare class. — (True / False)

---

## ✅ Answer Key & Explanations

**A1 — (b) `[20, 30, 40]`.** A slice `[1:4]` takes indices 1, 2, 3 — the `stop` index 4 is **excluded**, so `50` (index 4) is left out. (a) drops the start; (c) wrongly includes the stop.

**A2 — (c) `[0 2 4 6 8]`.** `np.arange(start, stop, step)` follows the same exclusive-stop rule as slicing: it counts 0, 2, 4, 6, 8 and **stops before 10**. (d) wrongly includes 10.

**A3 — (b) `[1, 2, 3, 1, 2, 3]` then `[2 4 6]`.** A Python **list** `* 2` *repeats* the list. A NumPy **array** `* 2` *doubles each element* (broadcasting). This contrast is the single most-tested idea in the chapter.

**A4 — (a) `[1, 0, 1, 1]`.** `.astype(int)` maps `True → 1` and `False → 0`. This is the exact recipe behind `df['is_ghosted'] = (df['match_outcome'] == 'Ghosted').astype(int)`.

**A5 — (b) `[0 1 0 1]`.** `np.where(cond, 1, 0)` puts 1 where the condition is True, 0 where False. Elements > 10 are `12` and `20` (positions 1 and 3) → `[0 1 0 1]`.

**A6 — (a) `a 3 / b 2 / c 1`.** `value_counts()` tallies occurrences and sorts most-frequent first: `a` appears 3×, `b` 2×, `c` 1×. (c) is what you'd get with `normalize=True` (proportions).

**A7 — (c) `3`.** `"rock,music,travel".split(',')` → `['rock', 'music', 'travel']`, and `len(...)` of that list is `3`. (d) `17` would be `len` of the raw string's characters — but `.split(',')` is called first.

**A8 — (b) `[3. 4.]`.** `axis=0` collapses **down the rows** → one mean per **column**. Column 0: `(1+3+5)/3 = 3`; column 1: `(2+4+6)/3 = 4`. (a) is `axis=1` (per-row means) — wrong direction.

**A9 — (c) `[25 30 45]`.** Boolean indexing keeps only the elements where the mask is `True`, i.e. those greater than 20: 25, 30, 45. (b) is the raw mask, not the filtered values.

**A10 — (b) `12.3%`.** The `%` format spec multiplies by 100 and appends `%`; `.1` keeps one decimal: `0.1234 → 12.3%`. The common wrong answer (a) `0.1%` forgets that `%` already does the ×100.

**A11 — (b) `[2. 3.]`.** Take absolute values → `[[1, 2], [3, 4]]`, then `mean(axis=0)` per column: `(1+3)/2 = 2`, `(2+4)/2 = 3`. This is literally `np.abs(shap_vals).mean(axis=0)` — mean absolute importance **per feature**.

**A12 — (a) `x: mean 2.0 count 2 ; y: mean 15.0 count 2`.** Group `x` has values 1 and 3 → mean 2.0, count 2; group `y` has 10 and 20 → mean 15.0, count 2. In the project, the grouped column is the 0/1 `is_ghosted`, so its mean is the ghosting rate per group.

**B1 — False.** `fit_transform` *learns* parameters (medians, means, category lists) from the data it sees. Doing that on the test set leaks test information into preprocessing. The test set gets `transform` only.

**B2 — True.** That is exactly what StandardScaler does: $z = (x - \mu)/\sigma$, giving each feature mean ≈ 0 and std ≈ 1.

**B3 — True.** A column of 0s and 1s has mean = (number of 1s)/(total) = the fraction that are 1 = the ghosting rate. This is used all over the EDA.

**B4 — False.** `np.arange` excludes the `stop` value. `np.arange(0, 10, 2)` = `[0 2 4 6 8]`; 10 is not included.

**B5 — True.** `predict_proba` returns a column per class; index `[:, 1]` selects the second column = probability of class 1 (the positive class, Ghosted). These probabilities feed `roc_auc_score`.

**B6 — True.** `stratify=y` preserves the class ratio across the split — essential when one class is rare, so the test set still contains enough of the minority (ghosters) to evaluate fairly.

**B7 — False.** Under imbalance a model can score high accuracy by predicting the majority class for almost everyone while catching none of the rare class. That's precisely why the project ranks models by **F1**, not accuracy.
