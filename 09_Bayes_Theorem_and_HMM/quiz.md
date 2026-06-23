# Chapter 9 Quiz — Bayes' Theorem & Hidden Markov Models

> Format mirrors the WIA1006 quiz/midterm: multiple-choice + true/false + worked problems.
> Try it closed-book first. Answers + explanations are at the very bottom.

## Section A — Multiple Choice

**A1.** In $P(A\mid B)=\dfrac{P(B\mid A)P(A)}{P(B)}$, which term is the **prior**?
- a. $P(A\mid B)$
- b. $P(B\mid A)$
- c. $P(A)$
- d. $P(B)$

**A2.** Which term is the **posterior**?
- a. $P(A\mid B)$
- b. $P(B\mid A)$
- c. $P(A)$
- d. $P(B)$

**A3.** The denominator $P(B)$ in Bayes' theorem is called the:
- a. likelihood
- b. prior
- c. evidence
- d. posterior

**A4.** What does the "naive" in Naive Bayes refer to?
- a. The algorithm ignores the prior probability
- b. Features are assumed conditionally independent given the class
- c. It only works on two classes
- d. It assumes all classes are equally likely

**A5.** The Markov property states that:
- a. The next state depends on all previous states equally
- b. The next state depends only on the current state
- c. Every state is equally likely
- d. Observations are independent of states

**A6.** In a state transition matrix $A$ where $a_{ij}=P(q_{t+1}=S_j\mid q_t=S_i)$, which is true?
- a. Each column sums to 1
- b. Each row sums to 1
- c. The whole matrix sums to 1
- d. The diagonal sums to 1

**A7.** In an HMM $M=(A,B,\pi)$, the matrix $B$ represents:
- a. State-to-state transition probabilities
- b. The initial state distribution
- c. The probability of an observation given a hidden state
- d. The number of hidden states

**A8.** In an HMM, $\pi$ represents:
- a. The emission probabilities
- b. The transition probabilities
- c. The probability the model starts in each state
- d. The final state probabilities

**A9.** The forward algorithm is used to:
- a. Find the single most likely hidden state sequence
- b. Compute $P(O\mid M)$, the likelihood of an observation sequence
- c. Learn the parameters $A$, $B$, $\pi$ from data
- d. Sort the states by probability

**A10.** The forward initialisation step is:
- a. $\alpha_1(i)=\pi_i\, b_i(O_1)$
- b. $\alpha_1(i)=a_{ii}\, b_i(O_1)$
- c. $\alpha_1(i)=\sum_j \pi_j a_{ji}$
- d. $\alpha_1(i)=\pi_i$

**A11.** Why is the forward algorithm preferred over naively summing over all hidden-state paths?
- a. It gives a different, more accurate answer
- b. It avoids the exponential $N^T$ cost by reusing partial sums
- c. It does not need the transition matrix
- d. It works without any observations

**A12.** A canonical, rigorously correct application of HMMs is:
- a. Speech recognition / sequence modeling
- b. "Understanding of the real world"
- c. Sorting numbers
- d. Image compression only

## Section B — True / False

**B1.** "99% test accuracy" guarantees a 99% chance of being sick after a positive result. — (True / False)

**B2.** Bayes' theorem lets you compute $P(A\mid B)$ from $P(B\mid A)$, $P(A)$ and $P(B)$. — (True / False)

**B3.** In the forward recursion, $\alpha_{t+1}(j)=\big[\sum_i \alpha_t(i)a_{ij}\big]b_j(O_{t+1})$. — (True / False)

**B4.** In a Markov chain, the columns of the transition matrix must each sum to 1. — (True / False)

**B5.** The probability of staying in state $i$ for exactly $d$ days is $(a_{ii})^{d}(1-a_{ii})$. — (True / False)

**B6.** In an HMM the hidden states are directly observed. — (True / False)

**B7.** The forward termination step is $P(O\mid M)=\sum_{i=1}^{N}\alpha_T(i)$. — (True / False)

## Section C — Worked Problems

Use the weather model with states $S_1$=rainy, $S_2$=cloudy, $S_3$=sunny and
$$A=\begin{bmatrix}0.4&0.3&0.3\\0.2&0.6&0.2\\0.1&0.1&0.8\end{bmatrix}$$

**C1.** Day 1 (t=1) is **sunny** ($S_3$). Find the probability that the next 7 days are
**sun → sun → rain → rain → sun → cloudy → sun**. Show every step.

**C2.** Day 1 is **cloudy** ($S_2$). Find the probability the next 4 days are
**cloudy → sunny → sunny → rainy**. Show your steps.

**C3.** (a) Once in state $i$, write the probability of staying exactly $d$ days, and derive the expected duration. (b) Apply both to the **cloudy** state ($a_{22}=0.6$): the probability of staying exactly 2 days, and the expected duration.

---

## ✅ Answer Key & Explanations

**A1 — (c)** The prior $P(A)$ is your belief about $A$ before evidence. $P(B\mid A)$ is the likelihood, $P(A\mid B)$ the posterior, $P(B)$ the evidence.

**A2 — (a)** The posterior $P(A\mid B)$ is the updated belief *after* observing $B$ — the whole point of the theorem.

**A3 — (c)** $P(B)$ is the evidence (the normaliser), usually built via total probability so the posterior sums to 1.

**A4 — (b)** "Naive" = the assumption that features are conditionally independent given the class. It's an approximation that makes the math tractable; (a)/(c)/(d) are not what the term means.

**A5 — (b)** The Markov property is memorylessness: the next state depends only on the current state, not the full history. (d) describes a different (emission) independence, not the Markov property.

**A6 — (b)** Rows sum to 1: from any state $S_i$ the chain must transition to *some* state next, so the outgoing probabilities total 1. Columns generally do not.

**A7 — (c)** $B$ holds emission probabilities $b_j(k)=P(v_k\mid q_t=S_j)$ — observation given hidden state. (a) is $A$, (b) is $\pi$.

**A8 — (c)** $\pi_i=P(q_1=S_i)$ is the initial state distribution — where the chain starts.

**A9 — (b)** The forward algorithm computes the likelihood $P(O\mid M)$ of an observation sequence. Finding the best path (a) is the Viterbi algorithm; learning parameters (c) is Baum-Welch.

**A10 — (a)** Initialisation is $\alpha_1(i)=\pi_i b_i(O_1)$: start in state $i$ then emit the first observation.

**A11 — (b)** Brute force sums over $N^T$ paths (exponential). The forward algorithm reuses partial sums (dynamic programming) for the *same exact answer* at cost $\sim N^2 T$. So (a) is wrong — the answer is identical, just computed cheaply.

**A12 — (a)** Speech recognition / sequence modeling is the precise, defensible application. "Understanding of the real world" (b) is a vague distractor.

**B1 — False.** This confuses $P(+\mid D)$ with $P(D\mid +)$. For a rare disease, false positives from the large healthy population dominate, so the posterior can be far below 99% (our worked example gave ~17%).

**B2 — True.** That is exactly what Bayes' theorem does — it inverts the conditional using prior and evidence.

**B3 — True.** That is the recursion: sum over all states you could have come from (weighted by $a_{ij}$), then multiply by the emission for the next observation.

**B4 — False.** It's the **rows** that sum to 1, not the columns.

**B5 — False.** The correct exponent is $d-1$: you need $d-1$ stays then one departure, giving $(a_{ii})^{d-1}(1-a_{ii})$.

**B6 — False.** In an HMM the states are **hidden**; you observe only the emissions and infer the states. (A plain Markov chain has visible states.)

**B7 — True.** Termination sums the final forward variables over all states to get $P(O\mid M)$.

---

**C1 — Worked solution.**

Day 1 is given as sunny, so we start counting from the first transition (no initial-probability factor). Seven future days = seven transitions. Reading $A$ (rows = today):

| Step | From → To | Entry |
|---|---|---|
| 1 | sunny→sunny | $a_{33}=0.8$ |
| 2 | sunny→sunny | $a_{33}=0.8$ |
| 3 | sunny→rainy | $a_{31}=0.1$ |
| 4 | rainy→rainy | $a_{11}=0.4$ |
| 5 | rainy→sunny | $a_{13}=0.3$ |
| 6 | sunny→cloudy | $a_{32}=0.1$ |
| 7 | cloudy→sunny | $a_{23}=0.2$ |

By the Markov property, multiply:
$$P = a_{33}a_{33}a_{31}a_{11}a_{13}a_{32}a_{23} = 0.8\cdot0.8\cdot0.1\cdot0.4\cdot0.3\cdot0.1\cdot0.2$$
Multiply step by step: $0.8\cdot0.8=0.64$; $\times0.1=0.064$; $\times0.4=0.0256$; $\times0.3=0.00768$; $\times0.1=0.000768$; $\times0.2=0.0001536$.
$$\boxed{P = 0.0001536}$$

**C2 — Worked solution.**

Day 1 is given as cloudy ($S_2$). Four future days = four transitions:

| Step | From → To | Entry |
|---|---|---|
| 1 | cloudy→cloudy | $a_{22}=0.6$ |
| 2 | cloudy→sunny | $a_{23}=0.2$ |
| 3 | sunny→sunny | $a_{33}=0.8$ |
| 4 | sunny→rainy | $a_{31}=0.1$ |

$$P = a_{22}a_{23}a_{33}a_{31} = 0.6\cdot0.2\cdot0.8\cdot0.1$$
Step by step: $0.6\cdot0.2=0.12$; $\times0.8=0.096$; $\times0.1=0.0096$.
$$\boxed{P = 0.0096}$$

**C3 — Worked solution.**

**(a)** Staying exactly $d$ days means $d-1$ consecutive self-transitions then one departure:
$$P(d) = (a_{ii})^{d-1}(1-a_{ii}).$$
This is geometric. The expected duration:
$$E[d]=\sum_{d=1}^{\infty} d\,(a_{ii})^{d-1}(1-a_{ii}) = (1-a_{ii})\sum_{d=1}^{\infty} d\,(a_{ii})^{d-1}.$$
Using $\sum_{d=1}^{\infty} d\,x^{d-1}=(1-x)^{-2}$ with $x=a_{ii}$:
$$E[d]=(1-a_{ii})(1-a_{ii})^{-2}=\frac{1}{1-a_{ii}}.$$

**(b)** Cloudy has $a_{22}=0.6$.
- Stay exactly 2 days: $P(2)=(0.6)^{2-1}(1-0.6)=0.6\times0.4=\boxed{0.24}$.
- Expected duration: $E[d]=\dfrac{1}{1-0.6}=\dfrac{1}{0.4}=\boxed{2.5\text{ days}}$.
