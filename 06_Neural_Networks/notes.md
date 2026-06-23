# Chapter 6 — Neural Networks (ANN / MLP)

> **Day 5 of 7** · Prerequisites: Ch 3 (gradient descent), Ch 4 (sigmoid / logistic regression) · Est. study time: ~75 min

**In one sentence:** A neural network is a stack of tiny "neurons" — each one does a weighted sum then squashes it through a non-linear function — wired into layers that learn to recognise patterns by adjusting their weights with backpropagation.

## Why this matters

Neural networks are where the course pivots from single models (a line, a tree) to a flexible function that can bend into almost any shape. In your GhostBusters project the MLP is one of the five models, trained with SMOTE to handle class imbalance. The exam leans heavily on **vocabulary and intuition** here: the biology analogy, what a perceptron computes, why you need an activation function, and the names of the training tricks (backprop, dropout, data augmentation). You rarely have to derive anything — you have to *understand and recognise*.

## Learning goals

- [ ] Map the biological neuron onto the artificial one (dendrite / cell body / axon / brain).
- [ ] Compute a perceptron's output: weighted sum $v$, then activation $f(v)$.
- [ ] Work the tutorial logic-gate problem and identify the gate (it's AND).
- [ ] Explain **why** a non-linear activation function is essential.
- [ ] Describe an MLP (input / hidden / output layers) and name its learning algorithm (backpropagation).
- [ ] Explain backpropagation at an intuitive level (chain rule, error flows backward, gradient descent updates).
- [ ] Recall what too-high a learning rate does, and what dropout and data augmentation are for.
- [ ] State what a well-trained ANN looks like.

---

## 1. Biological inspiration

The artificial neural network (ANN) is loosely modelled on the **mammalian brain**. You don't need neuroscience — you need the four-way analogy the exam tests (this is midterm Q8, a matching question).

Picture one brain cell. Signals come **in** through branching **dendrites**, get combined in the **cell body**, and the result is sent **out** along the **axon** to other neurons. The artificial neuron copies this flow exactly.

| Biological part | Artificial counterpart | What it does |
|---|---|---|
| **Dendrite** | **Input** ($x_i$) | Carries an incoming signal into the neuron |
| **Cell body** | **Summation + activation function** | Adds up the weighted inputs, then decides how strongly to fire |
| **Axon** | **Output** ($y$) | Sends the neuron's result onward |
| **Mammalian brain** | **Artificial neural network (ANN)** | The whole network of connected neurons |

> 💡 **Exam tip:** Memorise the matching as a sentence: *dendrites bring inputs in, the cell body sums-and-activates, the axon sends output out, and the whole brain is the whole network.* The trap is putting the activation function on the axon — it belongs to the **cell body**.

---

## 2. The perceptron (artificial neuron)

A single artificial neuron — the **perceptron** — does just two steps.

**Step 1 — Summation.** Multiply each input by its weight, add them up, and add a bias:

$$v = \sum_{i} w_i x_i + b$$

Expanded for two inputs:

$$v = w_1 x_1 + w_2 x_2 + b$$

- $x_i$ = the inputs (the signals coming in).
- $w_i$ = the **weights** — how much each input matters. A big weight means that input strongly influences the neuron.
- $b$ = the **bias** — a constant shift, like the intercept in a line. It lets the neuron fire even when all inputs are zero, by moving the decision threshold.
- $v$ = the **pre-activation** (sometimes called the net input). It's just a weighted vote.

**Step 2 — Activation.** Pass $v$ through an activation function $f$ to get the output:

$$y = f(v)$$

The activation decides whether — and how strongly — the neuron "fires". Section 3 covers the choices.

> ⚠️ **Common trap:** Weights and bias are *learned*; inputs are *given*. Don't confuse $w$ (a parameter the network tunes) with $x$ (the data).

### Worked example — the tutorial logic-gate problem

> Set $w_1 = 1$, $w_2 = 1$, $b = -1$. Inputs $x_1, x_2 \in \{-1, +1\}$. Use a **step** activation: output $1$ if $v \ge 0$, else $0$.

The summation function is:

$$v = w_1 x_1 + w_2 x_2 + b = (1)x_1 + (1)x_2 + (-1) = x_1 + x_2 - 1$$

Now compute $v$ and $y$ for all four input combinations. Do every step carefully:

- $(x_1,x_2) = (-1,-1)$: $\;v = -1 + (-1) - 1 = -3 \;\Rightarrow\; v < 0 \;\Rightarrow\; y = 0$
- $(x_1,x_2) = (-1,+1)$: $\;v = -1 + 1 - 1 = -1 \;\Rightarrow\; v < 0 \;\Rightarrow\; y = 0$
- $(x_1,x_2) = (+1,-1)$: $\;v = 1 + (-1) - 1 = -1 \;\Rightarrow\; v < 0 \;\Rightarrow\; y = 0$
- $(x_1,x_2) = (+1,+1)$: $\;v = 1 + 1 - 1 = +1 \;\Rightarrow\; v \ge 0 \;\Rightarrow\; y = 1$

**Truth table:**

| $x_1$ | $x_2$ | $v = x_1 + x_2 - 1$ | $y$ (output) |
|:---:|:---:|:---:|:---:|
| $-1$ | $-1$ | $-3$ | $0$ |
| $-1$ | $+1$ | $-1$ | $0$ |
| $+1$ | $-1$ | $-1$ | $0$ |
| $+1$ | $+1$ | $+1$ | $1$ |

**Which logic gate is it?** The neuron outputs $1$ **only when both inputs are "high" ($+1$)**, and $0$ otherwise. That is exactly the behaviour of an **AND gate**. (Read $+1$ as logical TRUE and $-1$ as logical FALSE: true AND true = true; everything else = false.)

**Plotting it** (goal 3 in the tutorial): plot the four points on the $(x_1, x_2)$ plane and label each by its output. The single point $(+1,+1)$ is class $y=1$; the other three are class $y=0$. A straight line — the line $x_1 + x_2 - 1 = 0$ — cleanly separates the lone "1" point from the three "0" points. That straightness is the key idea in the next section.

> 💡 **Exam tip:** If they hand you weights and a bias, just build the table mechanically: write $v$, plug in each row, apply the threshold. Then read off the pattern — *fires only when both inputs are high* = **AND**; *fires when at least one is high* = **OR**.

---

## 3. Activation functions

The activation function $f$ turns the raw weighted sum $v$ into the neuron's output. The course mentions three by name (midterm Q25 says an activation function "can take the sigmoid, step, or linear function" — that statement is **True**), plus two you should recognise.

| Function | Formula | Output range | Notes |
|---|---|---|---|
| **Step** | $f(v) = 1$ if $v \ge 0$, else $0$ | $\{0, 1\}$ | The classic perceptron threshold (used in §2). Hard on/off; not differentiable, so not used for gradient training. |
| **Sigmoid** | $f(v) = \dfrac{1}{1 + e^{-v}}$ | $(0, 1)$ | Smooth, squashes to a probability. You met this in Ch 4 (logistic regression). |
| **Linear** | $f(v) = v$ | $(-\infty, \infty)$ | "No squashing." Used at the output of *regression* networks. |
| **ReLU** | $f(v) = \max(0, v)$ | $[0, \infty)$ | The default in modern deep nets — cheap and trains fast. *(beyond the exam, good to recognise)* |
| **tanh** | $f(v) = \tanh(v)$ | $(-1, 1)$ | Like sigmoid but centred at 0. *(beyond the exam, good to recognise)* |

### Why we need an activation function (tutorial Q2)

This is the conceptual heart of the chapter, and a favourite exam question.

The reason is **non-linearity**. Each neuron's summation $v = \sum w_i x_i + b$ is a **linear** operation. If you stack linear operations on top of linear operations, the result is *still just one big linear operation* — the maths collapses. So a network with no (or only linear) activation functions can only ever represent a single straight-line / flat decision boundary, no matter how many layers you give it. It would be no more powerful than plain linear regression.

A **non-linear** activation (sigmoid, ReLU, tanh, step) breaks that collapse. It lets each layer bend the space, so stacking layers builds up genuinely complex, curved decision boundaries. **That is what lets a neural network learn non-linear patterns and separate data that isn't linearly separable.**

The canonical example is **XOR** (exclusive-or). XOR's four points cannot be split by any single straight line, so a single linear perceptron *cannot* learn it — but a multi-layer network with non-linear activations can.

> 💡 **Exam tip:** One-line answer to "why do we need an activation function?" → *To introduce non-linearity; without it, stacked layers collapse into a single linear model that can't learn non-linearly-separable patterns like XOR.*

> ⚠️ **Common trap:** "More layers" alone does **not** add power if the activations are linear. Depth only helps *because* of the non-linearity.

---

## 4. Multi-Layer Perceptron (MLP)

A single perceptron is weak (it can only draw one straight boundary — recall it can't do XOR). Wire many neurons into **layers** and you get a **Multi-Layer Perceptron**, the standard feed-forward neural network.

Three kinds of layer:

1. **Input layer** — one node per feature; just passes the data in. (In GhostBusters, this is the preprocessed feature vector.)
2. **Hidden layer(s)** — the workhorse neurons that learn intermediate patterns. There can be several. *Your project uses `hidden_layer_sizes=(64, 32)`: a first hidden layer of 64 neurons feeding a second of 32.*
3. **Output layer** — produces the final prediction (e.g. one sigmoid node giving $P(\text{ghosted})$).

Data flows **forward** input → hidden → output to make a prediction (the *forward pass*). Then the network learns by adjusting weights, using **backpropagation** as its learning algorithm (midterm Q18: "Multi-layer perceptron uses **backpropagation**").

> 💡 **Exam tip:** MLP's learning algorithm = **backpropagation**. Not reinforcement, not "unsupervised." It's supervised — it needs labelled targets to compute error.

---

## 5. Backpropagation

Backpropagation is how an MLP learns. You don't need the full derivation for the exam — you need the intuition and the three keywords (this is midterm Q33, a True/False that's **True**).

The idea, in plain terms:

1. **Forward pass.** Push an input through the network and get a prediction.
2. **Measure error.** Compare the prediction to the true label using a loss function.
3. **Propagate error backward.** Starting at the output, figure out how much *each weight* contributed to the error — working layer by layer **backward** toward the input. This blame-assignment uses the **chain rule** of calculus (the chain rule lets you multiply local gradients through each layer to get the gradient of the loss with respect to every weight).
4. **Update the weights.** Nudge each weight a little in the direction that reduces the error — this is **gradient descent** from Ch 3. Repeat over many examples / epochs, **iteratively**.

So the one-sentence definition the exam wants: *Backpropagation uses the **chain rule** to **propagate error backward** through the network and **update the weights iteratively** via gradient descent.* All three clauses are true.

> 💡 **Exam tip:** Keep the three keywords ready — **chain rule**, **error propagated backward**, **iterative weight updates**. The Q33 True/False bundles exactly these.

> ⚠️ **Common trap:** Backprop is *how gradients are computed and weights blamed*; gradient descent is *the rule that actually moves the weights*. They work together — backprop feeds gradient descent.

---

## 6. Learning rate

The **learning rate** ($\alpha$ or $\eta$) is the step size in gradient descent — how big a nudge each weight gets per update. This carries straight over from Ch 3, and the exam tests one failure mode (midterm Q34).

- **Too high** → the steps overshoot the minimum. The loss **oscillates or diverges** (blows up) instead of settling. This is the answer to Q34.
- **Too low** → training is stable but **painfully slow** to converge, and can get stuck.
- **Just right** → smooth, reasonably fast descent to a good minimum.

> 💡 **Exam tip:** "Learning rate too high" → **divergence / oscillation**. "Too low" → **slow convergence**. Don't swap them.

---

## 7. Regularisation: dropout & data augmentation

Big networks overfit easily — they memorise the training data instead of generalising. Two named techniques fight this.

**Dropout (midterm Q9).** During training, **randomly switch off (drop) a fraction of neurons** on each pass. The network can't lean on any single neuron, so it's forced to learn redundant, robust features. Net effect: **reduces overfitting**. (At test time all neurons are back on.)

- Q9 answer: dropout's main purpose is *"reduce overfitting by randomly disabling neurons."* Not "speed up training," not "normalise weights."

**Data augmentation (midterm Q11).** **Create modified copies of your training data** — for images that means rotating, flipping, cropping, adding noise; for other data it can mean small perturbations or synthetic samples. More varied data teaches the model to generalise rather than memorise, so **data augmentation improves generalization** (Q11 is **True**).

> 💡 **Exam tip:** Both dropout and data augmentation exist for the **same goal — better generalisation / less overfitting** — but by different means: dropout removes neurons, augmentation adds data variety. (SMOTE in your project is a cousin idea: it synthesises minority-class samples to rebalance the data before the MLP trains.)

---

## 8. What a well-trained ANN looks like

A network can fit the training data perfectly and still be useless on new data. So "well trained" is defined by **generalisation**, not training score alone (midterm Q21, **True**):

> A well-trained ANN has **low training error**, **low validation error**, and **good generalisation** to unseen data.

The pairing matters:

- Low training error **but** high validation error → **overfitting** (memorised the training set).
- High training error → **underfitting** (model too weak or undertrained).
- **Both** errors low and close → the sweet spot.

ANNs are also valued because they **handle noisy data and still produce generalised solutions** (midterm Q15, **True**) — they don't need perfectly clean inputs to find the underlying pattern.

> 💡 **Exam tip:** "Well-trained" = low train error **+** low validation error **+** generalises. If only training error is low, that's overfitting, not well-trained.

---

## Key formulas / facts at a glance

| Thing | Formula / Rule | Plain meaning |
|---|---|---|
| Summation (pre-activation) | $v = \sum_i w_i x_i + b$ | Weighted vote of the inputs plus a bias shift |
| Neuron output | $y = f(v)$ | Squash the vote through an activation |
| Step activation | $f(v)=1$ if $v\ge0$ else $0$ | Hard on/off threshold (perceptron) |
| Sigmoid | $f(v)=\frac{1}{1+e^{-v}}$ | Smooth squash to $(0,1)$, a probability |
| Biology map | dendrite→input · cell body→sum+activation · axon→output · brain→ANN | Q8 matching |
| MLP learning algo | **Backpropagation** | Q18 |
| Backprop = | chain rule + error backward + iterative weight updates | Q33 |
| Learning rate too high | Divergence / oscillation | Q34 |
| Dropout | Randomly disable neurons → less overfitting | Q9 |
| Data augmentation | Modified copies of data → better generalisation | Q11 |
| Well-trained ANN | Low train error + low validation error + generalises | Q21 |
| Why activation function | Adds non-linearity; else network = one linear model (can't do XOR) | Tutorial Q2 |

## Exam tips — quick recap

- Biology matching: activation lives in the **cell body**, not the axon.
- Given weights + bias, build the truth table mechanically; "fires only when both high" = **AND**.
- Why an activation function? **Non-linearity** — without it stacked layers collapse to one linear model and can't learn XOR.
- MLP's learning algorithm is **backpropagation** (supervised, not reinforcement/unsupervised).
- Backprop's three keywords: **chain rule, error backward, iterative updates**.
- Learning rate **too high → divergence/oscillation**; too low → slow.
- Dropout and data augmentation both fight **overfitting** (one drops neurons, one adds data).
- Well-trained = **low train + low validation error + good generalisation**.

## Self-check

1. Match each: dendrite, cell body, axon, mammalian brain → to their ANN counterparts.
2. For $w_1=1, w_2=1, b=-1$, step activation, why is the output $0$ for input $(+1,-1)$? Compute $v$.
3. In one sentence, why does a network need a non-linear activation function?
4. What are the three things backpropagation does (the Q33 keywords)?
5. Your friend says "my ANN has 100% training accuracy, so it's well trained." What's wrong with that claim?

**→ Now test yourself with [`quiz.md`](./quiz.md).**
