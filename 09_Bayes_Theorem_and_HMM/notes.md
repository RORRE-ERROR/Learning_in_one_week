# Chapter 9 — Bayes' Theorem & Hidden Markov Models

> **Day 7 of 7** (morning) · Prerequisites: basic probability · Est. study time: ~75 min

**In one sentence:** Bayes' theorem tells you how to flip a conditional probability around to update a belief when new evidence arrives, and Hidden Markov Models use that same probabilistic thinking to reason about sequences where the thing you care about is hidden behind what you can observe.

## Why this matters

This is the probabilistic corner of the course. Bayes' theorem underpins the Naive Bayes classifier (still a workhorse for spam filtering and text classification), and HMMs are the classic model for sequence data — speech, handwriting, biological sequences. The exam loves this chapter because it is mostly clean arithmetic: if you can plug into a formula carefully, you can score every mark. The two guaranteed-style questions are the **weather-sequence probability** and the **expected-duration formula**.

## Learning goals

- [ ] State Bayes' theorem and name each piece: prior, likelihood, evidence, posterior.
- [ ] Work a disease-test / spam Bayes problem from scratch.
- [ ] Explain the "naive" assumption in Naive Bayes and why it helps text classification.
- [ ] State the Markov property and read a state transition matrix.
- [ ] Name the three elements of an HMM, $M=(A,B,\pi)$, and what each one means.
- [ ] Run the forward algorithm (initialise → recurse → terminate) and say why it beats brute force.
- [ ] Compute the probability of a specific weather sequence from a transition matrix.
- [ ] Derive and use the expected-duration formula $\frac{1}{1-a_{ii}}$.

---

## 1. Bayes' theorem

### Intuition first

You have a belief about how likely something is. Then you observe some evidence. Bayes' theorem is the rule for **updating** that belief in light of the evidence. It is the mathematics of changing your mind by exactly the right amount.

The trick it performs is **flipping the condition**. Often you know $P(\text{evidence} \mid \text{cause})$ — e.g. how often a test reads positive *when you actually have the disease* — but what you really want is $P(\text{cause} \mid \text{evidence})$ — *given a positive test, do I actually have the disease?* Those two numbers are not the same, and Bayes connects them.

### The formula

$$P(A\mid B) = \frac{P(B\mid A)\,P(A)}{P(B)}$$

Name every piece (memorise these words — the exam asks them directly):

| Piece | Symbol | Plain meaning |
|---|---|---|
| **Prior** | $P(A)$ | What you believed about $A$ *before* seeing the evidence. |
| **Likelihood** | $P(B\mid A)$ | How probable the evidence $B$ is *if* $A$ were true. |
| **Evidence** | $P(B)$ | How probable the evidence is overall (the normaliser). |
| **Posterior** | $P(A\mid B)$ | Your updated belief in $A$ *after* seeing the evidence. |

A handy slogan: **posterior $\propto$ likelihood $\times$ prior**. The evidence $P(B)$ in the denominator just rescales things so the probabilities add to 1.

Computing the evidence often uses the **law of total probability**:
$$P(B) = P(B\mid A)P(A) + P(B\mid \neg A)P(\neg A).$$

### Worked example — a disease test

Suppose a disease affects **1%** of people. A test is decent but imperfect:

- If you **have** the disease, it reads positive 99% of the time: $P(+\mid D)=0.99$.
- If you **don't** have it, it still reads positive 5% of the time (a false alarm): $P(+\mid \neg D)=0.05$.

You test positive. What is the chance you actually have the disease, $P(D\mid +)$?

**Step 1 — write down the knowns.**
$$P(D)=0.01,\quad P(\neg D)=0.99,\quad P(+\mid D)=0.99,\quad P(+\mid \neg D)=0.05.$$

**Step 2 — compute the evidence** $P(+)$ via total probability:
$$P(+) = (0.99)(0.01) + (0.05)(0.99) = 0.0099 + 0.0495 = 0.0594.$$

**Step 3 — apply Bayes:**
$$P(D\mid +) = \frac{P(+\mid D)\,P(D)}{P(+)} = \frac{(0.99)(0.01)}{0.0594} = \frac{0.0099}{0.0594} = 0.1667.$$

So even after a positive result, there is only about a **17%** chance you have the disease. That feels wrong until you realise the disease is *rare* — there are simply far more healthy people generating false positives than sick people generating true positives. This is the famous **base-rate** effect.

> 💡 **Exam tip:** When they give you $P(+\mid D)$ and ask for $P(D\mid +)$, that is your signal to use Bayes. Always build the denominator $P(B)$ with total probability unless it's handed to you.

> ⚠️ **Common trap:** Confusing $P(B\mid A)$ with $P(A\mid B)$. "99% accurate" does NOT mean "99% chance you're sick if positive." The prior (base rate) matters enormously.

---

## 2. Naive Bayes classifier (brief)

Naive Bayes uses Bayes' theorem to classify. For a class $C$ and features $x_1,\dots,x_n$:
$$P(C\mid x_1,\dots,x_n) \propto P(C)\prod_{i=1}^{n} P(x_i\mid C).$$

That product is where the **"naive"** comes from: it assumes the features are **conditionally independent given the class**. In reality words in an email are *not* independent, but pretending they are makes the math trivial — you just multiply per-word probabilities instead of estimating an impossibly huge joint distribution.

**Why it's used for text/spam:** an email has thousands of possible words (features). Estimating how every combination co-occurs is hopeless. The naive assumption lets you estimate one simple number per word per class ("how often does *free* appear in spam?") and multiply them. It is fast, needs little data, and works surprisingly well even though the independence assumption is technically false.

> 💡 **Exam tip:** If asked *why* Naive Bayes is "naive," answer: it assumes features are conditionally independent given the class.

---

## 3. The Markov property & Markov chains

### The Markov property

> The future depends only on the present, not on the past.

Formally, the probability of the next state depends only on the current state, not on the whole history:
$$P(q_{t+1}\mid q_t, q_{t-1}, \dots, q_1) = P(q_{t+1}\mid q_t).$$

This is "memorylessness." Knowing today's weather is enough to predict tomorrow — the weather from last week adds nothing once you know today.

### State transition matrix $A$

A Markov chain is described by a matrix $A=[a_{ij}]$ where
$$a_{ij} = P(q_{t+1}=S_j \mid q_t = S_i)$$
is the probability of moving from state $S_i$ to state $S_j$. Each **row sums to 1**, because from any state you must go *somewhere* next.

> ⚠️ **Common trap:** Rows sum to 1, not columns. $a_{ij}$ goes **from row $i$ to column $j$**. Read the index order carefully or you'll grab the wrong number.

---

## 4. Hidden Markov Model: $M=(A, B, \pi)$

In a plain Markov chain you can *see* the states. In a **Hidden** Markov Model the states are hidden — you only see **observations** that the hidden states emit. Example: the weather (hidden) makes someone carry an umbrella or not (observed). You infer the weather from the umbrella.

An HMM is fully specified by three things, $M=(A, B, \pi)$:

- **$A$ — State transition probability matrix.** How hidden states move from one to the next.
$$a_{ij} = P(q_{t+1}=S_j \mid q_t = S_i)$$

- **$B$ — Observation (emission) probability matrix.** How likely each observable symbol $v_k$ is, given the hidden state $S_j$.
$$b_j(k) = P(v_k \text{ at } t \mid q_t = S_j)$$

- **$\pi$ — Initial state distribution.** The probability the chain *starts* in each state.
$$\pi_i = P(q_1 = S_i)$$

Implicit in the model: a finite set of hidden states $S=\{S_1,\dots,S_N\}$ and a finite set of observation symbols $V=\{v_1,\dots,v_M\}$.

> 💡 **Exam tip:** A classic question hands you $A$, $B$, $\pi$ scrambled and asks you to match each to its name/role. Lock in: **A = transitions (state→state), B = emissions (state→observation), π = start.**

---

## 5. The forward algorithm — computing $P(O\mid M)$

### The problem

Given an observation sequence $O = O_1 O_2 \dots O_T$, how likely was it under the model $M$? The naive way is to sum over **every possible hidden-state path** — there are $N^T$ of them. For even modest sequences that explodes exponentially. The forward algorithm gets the exact same answer in time proportional to $N^2 T$ by reusing partial results (dynamic programming).

### The forward variable

$$\alpha_t(i) = P(O_1 \dots O_t,\; q_t = S_i \mid M)$$

In words: the probability of having seen the first $t$ observations **and** being in state $S_i$ right now. Build this up step by step.

### The three steps

**1. Initialisation** (at $t=1$):
$$\alpha_1(i) = \pi_i\, b_i(O_1), \quad 1\le i\le N.$$
Start in state $i$ (prob $\pi_i$) and emit the first observation (prob $b_i(O_1)$).

**2. Recursion** (for $t=1,\dots,T-1$):
$$\alpha_{t+1}(j) = \left[\sum_{i=1}^{N} \alpha_t(i)\, a_{ij}\right] b_j(O_{t+1}).$$
To be in state $j$ at time $t+1$: sum over every state $i$ you could have come from (each weighted by the transition $a_{ij}$), then multiply by the chance of emitting the next observation from $j$.

**3. Termination:**
$$P(O\mid M) = \sum_{i=1}^{N} \alpha_T(i).$$
Add up the probabilities of ending in each state — that covers all paths.

**Intuition:** the $\alpha$ values are like a running tally. Instead of re-walking every full path, you carry forward "the total probability of all paths that land here" and extend it one step at a time. That's why exponential enumeration collapses into a cheap loop.

> 💡 **Exam tip:** Know the three step names — **initialisation, recursion, termination** — and which formula belongs to each. The recursion's shape (sum of incoming, then times the emission) is the bit they test.

---

## 6. Where HMMs are used

- **Speech recognition** — the canonical application. The words/phonemes are hidden states; the audio features are observations.
- **Sequence modeling** generally — part-of-speech tagging, handwriting recognition, gene/DNA sequence analysis, time-series with hidden regimes.

> ⚠️ **Common trap:** Vague answers like "understanding the real world" are distractors. The precise, defensible HMM application is **speech recognition / sequence modeling**.

---

## 7. WORKED EXAMPLE — Markov weather sequence

States: $S_1$ = rainy, $S_2$ = cloudy, $S_3$ = sunny. Transition matrix (rows = today, columns = tomorrow):

$$A = \begin{bmatrix} 0.4 & 0.3 & 0.3 \\ 0.2 & 0.6 & 0.2 \\ 0.1 & 0.1 & 0.8 \end{bmatrix}$$

So for instance $a_{33}=0.8$ (sunny → sunny), $a_{31}=0.1$ (sunny → rainy).

**Question:** Day 1 is sunny ($S_3$). What is the probability the next 7 days are
**sun → sun → rain → rain → sun → cloudy → sun**?

**Step 1 — list the transitions in order.** Starting from $S_3$ on day 1:

| Transition | From → To | Entry |
|---|---|---|
| day1→2 | sunny→sunny | $a_{33}=0.8$ |
| day2→3 | sunny→sunny | $a_{33}=0.8$ |
| day3→4 | sunny→rainy | $a_{31}=0.1$ |
| day4→5 | rainy→rainy | $a_{11}=0.4$ |
| day5→6 | rainy→sunny | $a_{13}=0.3$ |
| day6→7 | sunny→cloudy | $a_{32}=0.1$ |
| day7→8 | cloudy→sunny | $a_{23}=0.2$ |

**Step 2 — by the Markov property, multiply them** (each step depends only on the previous state):
$$P = a_{33}\cdot a_{33}\cdot a_{31}\cdot a_{11}\cdot a_{13}\cdot a_{32}\cdot a_{23}$$
$$P = 0.8\cdot 0.8\cdot 0.1\cdot 0.4\cdot 0.3\cdot 0.1\cdot 0.2$$

**Step 3 — multiply carefully:**
$$0.8\cdot 0.8 = 0.64$$
$$0.64\cdot 0.1 = 0.064$$
$$0.064\cdot 0.4 = 0.0256$$
$$0.0256\cdot 0.3 = 0.00768$$
$$0.00768\cdot 0.1 = 0.000768$$
$$0.000768\cdot 0.2 = 0.0001536$$

$$\boxed{P = 0.0001536}$$

> 💡 **Exam tip:** Day 1 is *given*, so you do NOT multiply by an initial probability — you start counting from the *first transition*. Seven days ahead means seven transitions.

---

## 8. Expected duration in a state

**Question:** Once the chain is in state $i$, what is the probability it stays for exactly $d$ days, and what is the expected number of days?

**Probability of exactly $d$ days.** To stay exactly $d$ days you need $d-1$ self-transitions (stay, stay, …) and then exactly one transition *out*:
$$P(d) = (a_{ii})^{d-1}(1 - a_{ii}).$$
This is a **geometric distribution**: success ("leave") with probability $1-a_{ii}$ on each day.

**Expected duration.** Take the expected value of $d$:
$$E[d] = \sum_{d=1}^{\infty} d\,(a_{ii})^{d-1}(1-a_{ii}) = (1-a_{ii})\sum_{d=1}^{\infty} d\,(a_{ii})^{d-1}.$$
Using the standard series identity $\sum_{d=1}^{\infty} d\,x^{d-1} = (1-x)^{-2}$ for $|x|<1$ (with $x=a_{ii}$):
$$E[d] = (1-a_{ii})\cdot (1-a_{ii})^{-2} = \frac{1}{1-a_{ii}}.$$

$$\boxed{E[d] = \dfrac{1}{1 - a_{ii}}}$$

**Intuition:** $1-a_{ii}$ is the per-day chance of leaving. If you leave with probability $0.2$ each day, you expect to stay $1/0.2 = 5$ days — exactly like expecting 5 coin-style trials before a 1-in-5 event happens. Stickier states (larger $a_{ii}$) have smaller leave-probability and thus longer expected stays.

**Quick check — sunny** ($a_{33}=0.8$): $E[d] = \frac{1}{1-0.8} = \frac{1}{0.2} = 5$ days.

> 💡 **Exam tip:** Memorise both: $P(d)=(a_{ii})^{d-1}(1-a_{ii})$ and $E[d]=\frac{1}{1-a_{ii}}$. The $d-1$ exponent (not $d$) is the detail they check.

---

## Key formulas / facts at a glance

| Thing | Formula / Rule | Plain meaning |
|---|---|---|
| Bayes' theorem | $P(A\mid B)=\dfrac{P(B\mid A)P(A)}{P(B)}$ | Flip a conditional to update a belief |
| Posterior shape | posterior $\propto$ likelihood $\times$ prior | Evidence just normalises |
| Total probability | $P(B)=P(B\mid A)P(A)+P(B\mid\neg A)P(\neg A)$ | Build the evidence/denominator |
| Naive Bayes | $P(C\mid \mathbf{x})\propto P(C)\prod_i P(x_i\mid C)$ | Features independent given class |
| Markov property | $P(q_{t+1}\mid q_t,\dots,q_1)=P(q_{t+1}\mid q_t)$ | Future depends only on present |
| Transition matrix | $a_{ij}=P(q_{t+1}=S_j\mid q_t=S_i)$, rows sum to 1 | State → state probabilities |
| HMM elements | $M=(A,B,\pi)$ | A=transitions, B=emissions, π=start |
| Emission | $b_j(k)=P(v_k\mid q_t=S_j)$ | State → observation probability |
| Forward init | $\alpha_1(i)=\pi_i b_i(O_1)$ | Start + first emission |
| Forward recursion | $\alpha_{t+1}(j)=\big[\sum_i \alpha_t(i)a_{ij}\big]b_j(O_{t+1})$ | Sum incoming, times emission |
| Forward termination | $P(O\mid M)=\sum_i \alpha_T(i)$ | Add up all ending states |
| Stay exactly $d$ days | $P(d)=(a_{ii})^{d-1}(1-a_{ii})$ | Geometric distribution |
| Expected duration | $E[d]=\dfrac{1}{1-a_{ii}}$ | 1 / per-day leave probability |

## Exam tips — quick recap

- Bayes signal: given $P(+\mid D)$, asked for $P(D\mid +)$. Build $P(B)$ with total probability.
- Don't confuse $P(A\mid B)$ with $P(B\mid A)$; the prior/base rate matters.
- "Naive" = features conditionally independent given the class.
- Transition matrix **rows** sum to 1; $a_{ij}$ is from $i$ to $j$.
- Match HMM pieces: **A** transitions, **B** emissions, **π** initial.
- Forward algorithm: initialisation → recursion → termination; it avoids exponential $N^T$ enumeration.
- Weather sequence: day 1 given ⇒ no initial term; 7 days ⇒ 7 transitions multiplied.
- Duration: exponent is $d-1$; expected stay is $\frac{1}{1-a_{ii}}$.

## Self-check

1. In Bayes' theorem, which term is the "evidence" and how do you usually compute it?
2. Why does a positive test for a rare disease still leave a low probability of being sick?
3. What exactly does the word "naive" refer to in Naive Bayes?
4. State the Markov property in one sentence and say which axis of $A$ sums to 1.
5. Write the forward-algorithm recursion and explain why it's faster than brute force.
6. From the weather matrix, compute $P$ of sun→rain→rain given day 1 is sunny.
7. If $a_{ii}=0.6$, what is the expected duration in state $i$?

**→ Now test yourself with [`quiz.md`](./quiz.md).**
