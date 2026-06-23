# Chapter 2 Quiz — Data Preprocessing

> Format mirrors the WIA1006 quiz/midterm: multiple-choice + true/false (+ one optional conceptual item). Try it closed-book first. Answers + explanations are at the very bottom.

## Section A — Multiple Choice

**A1.** What is the first step in data preprocessing?
- a. Data transformation
- b. Data reduction
- c. Data cleaning
- d. Data integration

**A2.** Which is **NOT** a typical challenge associated with raw data that necessitates preprocessing?
- a. Missing values
- b. Presence of outliers
- c. Meaningless data combinations
- d. Standardized data formats that are universally compatible with ML algorithms

**A3.** Which statement BEST describes the role of data preprocessing in a machine learning workflow?
- a. Exclusively for handling missing values and encoding categorical data
- b. Only necessary for large datasets
- c. Crucial for enhancing data quality and producing meaningful insights by converting raw data into a reliable, clean format
- d. Primarily for reducing computational cost

**A4.** What is a likely consequence of improper handling of missing values during preprocessing?
- a. Difficulties deploying the model
- b. Inaccurate and faulty conclusions and inferences
- c. Increased computational cost
- d. Reduced interpretability

**A5.** Which strategies can be used to handle **noisy** data? *(Select all that apply.)*
- a. Binning
- b. Clustering
- c. Regression
- d. SMOTE

**A6.** Why is normalization performed during preprocessing?
- a. To combine multiple datasets
- b. To handle missing values
- c. To scale features to a specific range (e.g., 0–1 or −1 to 1)
- d. To remove duplicate records

**A7.** What is the purpose of feature scaling in regression (and distance-based) models?
- a. To ensure all features contribute equally to distance calculations
- b. To reduce dimensionality
- c. To increase model complexity
- d. To handle missing values

**A8.** Why is stratified sampling preferred over random sampling in certain cases?
- a. Ensures equal (proportional) representation across all classes or groups
- b. Reduces computational complexity
- c. Improves model interpretability
- d. Eliminates noisy samples

**A9.** What is the PRIMARY reason for splitting a dataset into training and testing sets?
- a. To reduce the overall dataset size
- b. To evaluate the model's performance on unseen data and prevent overfitting
- c. To create multiple versions of the dataset
- d. To identify and correct data entry errors

**A10.** Which of these is an example of dimensionality reduction?
- a. Feature selection
- b. PCA
- c. PCA & feature selection
- d. Sampling

**A11.** Which preprocessing step most directly helps avoid overfitting?
- a. Normalization
- b. Data integration
- c. Data cleaning
- d. Feature selection

**A12.** *(New)* In the GhostBusters project, the numeric pipeline uses `SimpleImputer(strategy='median')` rather than the mean. What is the main reason median is preferred here?
- a. The median is faster to compute than the mean
- b. The median is robust to outliers, which would otherwise distort a mean-filled value
- c. The median works only on categorical data
- d. The mean cannot be used inside a scikit-learn pipeline

**A13.** *(New)* Which sequence is the correct order of the preprocessing steps?
- a. Reduction → Transformation → Integration → Cleaning
- b. Cleaning → Integration → Transformation → Reduction
- c. Integration → Cleaning → Reduction → Transformation
- d. Transformation → Cleaning → Integration → Reduction

**A14.** *(New)* The project calls `preprocessor.fit_transform(X_train)` but only `preprocessor.transform(X_test)`. Why fit on the training data only?
- a. The test set is too small to fit on
- b. To prevent data leakage — test-set statistics must not influence the scaling/imputation learned during training
- c. Because `transform` is faster than `fit_transform`
- d. It has no effect; either order gives the same result

## Section B — True / False

**B1.** SMOTE is a valid technique for smoothing noisy data. *(True / False)*

**B2.** Data cleaning is the very first step of the preprocessing pipeline, before integration, transformation, and reduction. *(True / False)*

**B3.** Standardization (z-score scaling) transforms a feature to have a mean of 0 and a standard deviation of 1. *(True / False)*

**B4.** One-hot encoding is preferred over simple integer labels for nominal categories because integer labels imply a false ordering/distance between categories. *(True / False)*

**B5.** Sampling (selecting a subset of rows) is a form of dimensionality reduction. *(True / False)*

**B6.** You should fit your scaler on the entire dataset *before* splitting into train and test, so that scaling is consistent. *(True / False)*

**B7.** Mode imputation is the natural choice for filling missing values in a *categorical* column. *(True / False)*

## Section C (optional) — Conceptual "What would you do?"

**C1.** You are handed a 50,000-row dating-app dataset to predict ghosting. Only about 10% of users ghost (class imbalance), several numeric columns have blanks and a few wild outliers, and there are text columns like `gender` and `education_level`. Outline the preprocessing steps you'd apply, in order, and justify each choice. (Use the project pipeline as your guide.)

---

## ✅ Answer Key & Explanations

**A1 — (c) Data cleaning.** The pipeline order is Cleaning → Integration → Transformation → Reduction. You clean errors first because there's no value in integrating or transforming data that is still broken.

**A2 — (d).** The question asks which is NOT a problem. Missing values, outliers, and meaningless combinations are all real raw-data problems. "Standardized formats universally compatible with ML algorithms" describes data that's *already ideal* — the opposite of a challenge.

**A3 — (c).** Preprocessing is broadly about turning raw data into a clean, reliable format for meaningful insights. (a) and (d) are too narrow (just one task each) and (b) is false — small datasets need preprocessing too.

**A4 — (b) Inaccurate and faulty conclusions and inferences.** Badly imputed values distort the data's statistics, so every downstream result is unreliable. The other options are minor side effects, not the core danger.

**A5 — (a), (b), (c).** Binning, clustering, and regression all smooth or group out random error. **SMOTE (d) is wrong** — it generates synthetic minority-class examples to fix *class imbalance*, not noise.

**A6 — (c).** Normalization rescales features into a specific range such as 0–1 or −1 to 1. It does not merge datasets, fill blanks, or remove duplicates — those are different steps.

**A7 — (a).** Scaling ensures large-magnitude features don't dominate distance/weighted-sum calculations, so every feature contributes fairly (and gradient descent converges faster). It does not reduce dimensionality or handle missing values.

**A8 — (a).** Stratified sampling splits by class first, then samples proportionally, guaranteeing each class is represented in the same ratio as the full data — crucial when one class is rare.

**A9 — (b).** The whole point of a held-out test set is to measure performance on data the model never saw, which reveals (and helps prevent) overfitting. Reducing size or making versions are not the purpose.

**A10 — (c) PCA & feature selection.** Both reduce the number of *features* (dimensions): feature selection keeps a subset of originals; PCA builds fewer new combined features. Sampling reduces *rows*, not dimensions, so it doesn't count.

**A11 — (d) Feature selection.** Fewer, more relevant features give the model less room to memorize noise, which reduces overfitting. Normalization, integration, and cleaning improve data quality but don't directly target overfitting.

**A12 — (b).** The median is unaffected by extreme outliers, whereas a few huge values would pull the mean upward and produce a misleading fill value. (Median works on numeric data, and both mean and median work fine inside pipelines.)

**A13 — (b) Cleaning → Integration → Transformation → Reduction.** Remember "Clean It, Then Reduce."

**A14 — (b).** Fitting the preprocessor on the full data (or on the test set) lets test-set statistics leak into training, inflating the score dishonestly. Correct discipline: split first, `fit` on train, then `transform` both. This is exactly what the project does.

**B1 — False.** SMOTE addresses class imbalance by synthesizing minority-class samples. Noise is handled by binning, clustering, or regression.

**B2 — True.** Cleaning is the documented first step, ahead of integration, transformation, and reduction.

**B3 — True.** That's the definition of the z-score transform: $z = (x - \mu)/\sigma$ gives mean 0, std 1.

**B4 — True.** Integer labels (0, 1, 2…) imply order and magnitude that nominal categories don't have. One-hot encoding gives each category its own independent 0/1 column, avoiding fake ordering.

**B5 — False.** Sampling reduces the number of *rows (records)*, not the number of *features (dimensions)*. Dimensionality reduction = PCA and feature selection.

**B6 — False.** This causes data leakage. Always split first, then fit the scaler on the training set only and transform both sets with it.

**B7 — True.** The mode (most frequent value) is the standard fill for categorical columns; mean/median are for numeric columns.

**C1 — Model answer.** A reasonable, project-aligned plan:
1. **Clean / handle missing values first.** Impute numeric blanks with the **median** (robust to the wild outliers) and categorical blanks with a constant placeholder or the mode. *(Cleaning step.)*
2. **Transform — encode categories.** Apply **one-hot encoding** to `gender`, `education_level`, etc., so text becomes numeric without implying false order. *(Transformation.)*
3. **Transform — scale numerics.** Apply **StandardScaler** (z-score) so features like usage-time and message-count contribute equally and gradient descent converges faster. *(Transformation.)*
4. **Split correctly with stratification.** Use `train_test_split(..., stratify=y)` so the rare ghosting class keeps its 10% proportion in both train and test. **Fit the scaler/imputer/encoder on the training set only**, then transform both, to avoid data leakage.
5. **Handle class imbalance.** Apply **SMOTE to the training set only** (or use `class_weight='balanced'`) so the model actually learns the minority "ghosted" pattern instead of always predicting the majority.
6. *(Optional reduction.)* Consider **feature selection / PCA** to drop redundant features, reducing overfitting and speeding up training.
