# Generative Adversarial Nets

## What is a GAN?

A Generative Adversarial Network (GAN) is a framework for training two competing neural networks:

- **Generator (G)**: takes random noise (z) and produces fake data (like images)
- **Discriminator (D)**: takes a sample and decides if it is real (from the training data) or fake (from G)

They are trained in a game:
- G tries to fool D
- D tries to correctly detect real vs fake

Over time, G gets better at generating real-looking data, and D gets better at detecting.

---

## GAN Objective Function 

The core of GAN training is the **value function** V(G, D), which defines the adversarial game between the generator and the discriminator.

The original **minimax criterion** is:

V(G, D) = E_x[log(D(x))] + E_z[log(1 - D(G(z)))]

- The **Discriminator D** tries to maximize this value:
  - D(x) should be close to 1 for real data
  - D(G(z)) should be close to 0 for fake data

- The **Generator G** tries to minimize this value:
  - It wants D(G(z)) to be close to 1
  - So that fake data is classified as real

In practice, instead of minimizing:
  log(1 - D(G(z)))

We often maximize:
  log(D(G(z)))

Because it gives **stronger gradients** when G is still weak early in training.

---

### Global Optimum of the Criterion

At the ideal point where:
  p_g(x) = p_data(x)

The discriminator becomes:
  D(x) = 0.5

And the value function becomes:
  V(G, D) = log(4)

This is the **global optimum** — it means the generator has learned the real data distribution perfectly.

---

### Based on Jensen–Shannon Divergence

The criterion is mathematically equivalent to minimizing the **Jensen–Shannon divergence** between p_data(x) and p_g(x), a measure of how different two distributions are.

- When JSD = 0 → p_g = p_data
- So the generator wins the game by becoming indistinguishable from the real distribution

---

## Key Insights

- The GAN reaches a **global optimum** when:
  p_g(x) = p_data(x)
  → The generator distribution matches the real data distribution

- At this point:
  D(x) = 0.5 for all x
  → Discriminator cannot tell real from fake

- The loss function (value function) used during training is:

  V(G, D) = E_x[log(D(x))] + E_z[log(1 - D(G(z)))]

- GAN training is based on **Jensen–Shannon divergence** between real and fake data distributions.

---

## Advantages of GANs

- No Markov Chains required
- No explicit probability calculation needed
- Easy to sample from (just run G(z))
- Backpropagation is enough for training
- Can generate sharp, realistic images
- Supports any differentiable function


---

## Challenges in GANs

- No explicit form of p(x), so evaluating performance is hard
- No direct likelihood estimation
- Training is unstable (needs balance between G and D)
- Mode collapse: G may generate the same outputs repeatedly


---

## Paper

"Generative Adversarial Nets" by Ian Goodfellow et al., 2014  
[Read the full paper on arXiv](https://arxiv.org/abs/1406.2661)


