# Chapter 6 Quiz — Neural Networks (ANN / MLP)

> Format mirrors the WIA1006 quiz/midterm: multiple-choice + true/false + a worked perceptron problem.
> Try it closed-book first. Answers + explanations are at the very bottom.

## Section A — Multiple Choice

**A1.** In an artificial neuron, which biological part corresponds to the **summation function and the activation function**?
- a. Dendrite
- b. Axon
- c. Cell body
- d. Synapse

**A2.** What is the main purpose of using **dropout** during training?
- a. Improve gradient flow
- b. Increase training speed
- c. Reduce overfitting by randomly disabling neurons
- d. Normalize weights

**A3.** A multi-layer perceptron (MLP) uses which learning algorithm?
- a. Reinforcement learning
- b. Backpropagation
- c. Unsupervised learning
- d. K-means

**A4.** What happens when the **learning rate** is set too high?
- a. Better generalization
- b. Reduced model capacity
- c. Divergence or oscillation during training
- d. Slow convergence

**A5.** Why does a neural network need a **non-linear** activation function?
- a. To make training faster
- b. Without non-linearity, stacked layers collapse into a single linear model that can't learn non-linearly-separable patterns
- c. To reduce the number of weights
- d. To guarantee the loss reaches zero

**A6.** A perceptron has $w_1=1$, $w_2=1$, $b=-1$ and a step activation (output $1$ if $v\ge0$, else $0$), with inputs in $\{-1,+1\}$. Which logic gate does it implement?
- a. OR
- b. XOR
- c. AND
- d. NOT

**A7.** Which of the following can be used as an activation function?
- a. Only the sigmoid function
- b. The sigmoid, step, or linear function
- c. Only the step function
- d. None — activation functions are not used in neural networks

**A8.** In the summation $v = \sum_i w_i x_i + b$, what is the role of the bias $b$?
- a. It is one of the input features
- b. It scales the output to a probability
- c. It is a learned constant that shifts the firing threshold, letting the neuron activate even when inputs are zero
- d. It is the learning rate

**A9.** Which statement about **backpropagation** is correct?
- a. It propagates the input forward only and never computes error
- b. It uses the chain rule to propagate error backward and update weights iteratively
- c. It updates weights without any gradient information
- d. It is only used in unsupervised learning

**A10.** In the GhostBusters MLP, `hidden_layer_sizes=(64, 32)` means:
- a. The network has 64 input features and 32 outputs
- b. There are two hidden layers, the first with 64 neurons and the second with 32
- c. The model trains for 64 epochs then 32 epochs
- d. The learning rate is 64 and the batch size is 32

**A11.** Which best describes a **well-trained** ANN?
- a. Zero training error, regardless of validation error
- b. High training error but low validation error
- c. Low training error, low validation error, and good generalization
- d. Low validation error only, training error ignored

**A12.** What is the purpose of **data augmentation**?
- a. To delete noisy samples from the dataset
- b. To create modified copies of the data to improve generalization
- c. To reduce the number of features
- d. To increase the learning rate over time

## Section B — True / False

**B1.** In the biological-to-artificial neuron mapping, the **axon corresponds to the output** and the **mammalian brain corresponds to the artificial neural network**. — (True / False)

**B2.** An activation function can take the sigmoid function, step function, or linear function. — (True / False)

**B3.** Backpropagation uses the chain rule, propagates error backward, and updates weights iteratively. — (True / False)

**B4.** Adding more layers to a network always increases its representational power, even if every activation function is linear. — (True / False)

**B5.** Data augmentation will improve generalization. — (True / False)

**B6.** An ANN can process noisy data and still provide generalized solutions. — (True / False)

**B7.** Dropout is mainly used to increase training speed. — (True / False)

**B8.** A single perceptron (no hidden layer) can learn the XOR function. — (True / False)

## Section C — Worked Problem (Perceptron / Logic Gate)

**C1.** A perceptron has weights $w_1 = 1$, $w_2 = 1$ and bias $b = -1$. The inputs take values $x_1, x_2 \in \{-1, +1\}$. The activation is a **step** function: output $y = 1$ if $v \ge 0$, otherwise $y = 0$.

1. Write the summation equation $v$, including its expanded form.
2. Complete the truth table for all four input combinations (compute $v$ and $y$).
3. Describe how you would plot the inputs $(x_1, x_2)$ against the output $y$.
4. Identify which **logic operator** this network implements.

---

## ✅ Answer Key & Explanations

**A1 — (c) Cell body.** The cell body is where incoming signals are combined (summation) and the firing decision (activation) is made. Dendrites only bring inputs *in* (a); the axon only sends output *out* (b). Putting the activation on the axon is the classic trap.

**A2 — (c) Reduce overfitting by randomly disabling neurons.** Dropout switches off a random subset of neurons each training pass, forcing the network to learn robust, redundant features instead of leaning on any single neuron. It is not primarily about speed (b) or weight normalisation (d).

**A3 — (b) Backpropagation.** MLPs are trained by backpropagation, a supervised algorithm that needs labelled targets. Reinforcement learning (a) and unsupervised learning (c) are different paradigms entirely.

**A4 — (c) Divergence or oscillation during training.** Too large a step overshoots the minimum, so the loss bounces around or blows up. *Slow* convergence (d) is the symptom of a learning rate that's too *low* — don't swap them.

**A5 — (b).** Each neuron's weighted sum is a linear operation; composing linear operations gives another linear operation. A non-linear activation breaks this collapse so layers can build curved boundaries and separate data like XOR. (a), (c), (d) are not the reason — depth without non-linearity adds no power.

**A6 — (c) AND.** Build the table: $v = x_1 + x_2 - 1$. Only $(+1,+1)$ gives $v=1\ge0\Rightarrow y=1$; the other three rows give $y=0$. Firing only when *both* inputs are high is exactly AND. OR (a) would also fire when only one input is high; XOR (b) is not even linearly separable, so no single perceptron can do it.

**A7 — (b) The sigmoid, step, or linear function.** All three are valid activation functions (linear is used for regression outputs; step for the classic perceptron; sigmoid for probabilities). Saying only one is allowed (a, c) is too narrow, and (d) is plainly false.

**A8 — (c).** The bias is a learned constant (like a line's intercept) that shifts where the neuron's threshold sits, allowing it to fire even when all inputs are zero. It is not an input feature (a), not a probability scaler (b), and not the learning rate (d).

**A9 — (b).** Backprop's definition: chain rule + error propagated backward + iterative weight updates (via gradient descent). (a) describes only a forward pass; (c) ignores gradients; (d) is the wrong paradigm.

**A10 — (b).** `hidden_layer_sizes=(64, 32)` specifies two hidden layers — 64 neurons then 32 — between the input and output layers. The tuple does not set input/output counts (a), epochs (c), or learning rate/batch size (d).

**A11 — (c).** "Well-trained" requires **both** low training error and low validation error, which together signal good generalisation. Zero training error with poor validation (a) is overfitting; high training error (b) is underfitting; ignoring training error (d) misses the picture.

**A12 — (b).** Data augmentation creates modified copies of the training data (rotations, flips, noise, small perturbations) so the model sees more variety and generalises better. It does not delete data (a), reduce features (c), or change the learning rate (d).

---

**B1 — True.** The standard Q8 mapping: dendrite→input, cell body→summation+activation, **axon→output**, **mammalian brain→ANN**. Both clauses are correct.

**B2 — True.** Sigmoid, step, and linear are all legitimate activation functions (ReLU and tanh are common too). This is midterm Q25.

**B3 — True.** All three clauses hold: backprop applies the chain rule, sends error backward through the layers, and updates weights iteratively over many passes. This is midterm Q33.

**B4 — False.** Depth only adds power *because of* non-linear activations. If every activation is linear, the entire stack reduces to a single linear transformation no matter how many layers — same power as one linear model.

**B5 — True.** More varied training data teaches the model the underlying pattern rather than memorising specifics, improving generalisation. This is midterm Q11.

**B6 — True.** Tolerating noise and still producing generalised solutions is a recognised strength of ANNs. This is midterm Q15.

**B7 — False.** Dropout's purpose is to **reduce overfitting** by randomly disabling neurons, not to speed up training. (Compare A2.)

**B8 — False.** XOR is not linearly separable, and a single perceptron can only draw one straight boundary. You need at least one hidden layer (an MLP) with non-linear activations to learn XOR.

---

**C1 — Full solution.**

**1. Summation equation.** With $w_1 = w_2 = 1$ and $b = -1$:
$$v = \sum_i w_i x_i + b = w_1 x_1 + w_2 x_2 + b = (1)x_1 + (1)x_2 + (-1) = x_1 + x_2 - 1$$

**2. Truth table.** Compute $v$ for each row, then apply the step rule ($y=1$ if $v\ge0$, else $0$):

| $x_1$ | $x_2$ | $v = x_1 + x_2 - 1$ | $y$ |
|:---:|:---:|:---:|:---:|
| $-1$ | $-1$ | $-1-1-1 = -3$ | $0$ |
| $-1$ | $+1$ | $-1+1-1 = -1$ | $0$ |
| $+1$ | $-1$ | $1-1-1 = -1$ | $0$ |
| $+1$ | $+1$ | $1+1-1 = +1$ | $1$ |

**3. Plot.** Place the four input points on the $(x_1, x_2)$ plane: $(-1,-1)$, $(-1,+1)$, $(+1,-1)$ all belong to class $y=0$, and the single point $(+1,+1)$ belongs to class $y=1$. The decision line $x_1 + x_2 - 1 = 0$ separates the lone $y=1$ point from the three $y=0$ points — showing the problem is linearly separable.

**4. Logic operator.** The neuron outputs $1$ **only when both inputs are high ($+1$)** and $0$ otherwise. Reading $+1$ as TRUE and $-1$ as FALSE, this is the **AND** gate.
