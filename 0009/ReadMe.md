# Searching for Activation Functions

## Overview

Today we are talking about the paper "Searching for Activation Functions" by Prajit Ramachandran, Barret Zoph, and Quoc V. Le from Google Brain.

At the heart of every deep learning model is an activation function — formula that helps the model learn from input data and of course the most popular one we all know is ReLU but this paper explores a bold idea:
> Can we use automatic search (not hand crafted design) to discover better activation functions?

The answer is yes and this approach led to a new function called *Swish* which performs better than ReLU in many cases.

---

### What is Swish?

The **Swish activation function** is defined as:

f(x) = x * sigmoid(βx)

- `sigmoid(z) = 1 / (1 + exp(-z))`
- `β` is a constant or trainable parameter.

### What the Researchers Did
Designed a search space of possible activation functions using combinations of simple math operations. they used two techniques:

- Exhaustive search (for small spaces).

- Reinforcement learning (for large spaces) with an RNN controller to generate and test new functions.

And for testing : they trained a small model and evaluated its accuracy, after that they have found top-performing functions — the best one was Swish


### Key Observations
- Simple functions often outperform complex ones.

- Swish is smooth, non-monotonic, and trainable — giving it an edge over ReLU.

- Periodic functions (like sin or cos) showed potential for future exploration.

- Swish generalizes well across different models and tasks — like image classification and machine translation.


### Reference

If you're interested in the full paper, you can find it on arXiv:
"Searching for Activation Functions"
arXiv: 1710.05941v2