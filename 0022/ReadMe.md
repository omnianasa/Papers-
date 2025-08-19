# Least Squares Generative Adversarial Networks

> **Paper**: *Least Squares Generative Adversarial Networks (LSGANs)*  
> **Authors**: Xudong Mao, Qing Li, Haoran Xie, Raymond Y.K. Lau, Zhen Wang, Stephen Paul Smolley  
> **arXiv**: [1611.04076v3](https://arxiv.org/pdf/1611.04076)  
> **Goal**: Make GAN training **more stable** and produce **higher quality images** by replacing the discriminator’s cross-entropy loss with a **least squares** loss

---

## 1) What & Why

- **Regular GAN** uses a sigmoid + cross-entropy loss for the discriminator. This often causes **vanishing gradients** for the generator when fake samples are already on the “real” side of the boundary but still **far** from real data. Training can stall.
- **LSGAN** replaces that loss with **least squares** (L2). This **penalizes** samples even when they are on the correct side but **far** from real data, giving the generator useful gradients → **more stable training** + **better image quality**. 
- Minimizing the LSGAN objective corresponds to minimizing a Pearson X^2  divergence (under simple label settings). 

---

## 2) Background: Regular GAN (Goodfellow et al.)

We train a **discriminator** \(D\) and a **generator** \(G\) together.

- **Data**: $$(x \sim p_{data}(x))$$
- **Noise** (Gaussian or uniform): $$(z \sim p_z(z))$$ 
- **Generator**: $$(G(z; \theta_g))$$
- **Discriminator**  outputs a scalar (probability real): $$(D(x; \theta_d))$$

**Minimax objective:**
$$
\min_G \max_D \; V_{\text{GAN}}(D,G)
= \mathbb{E}_{x \sim p_{data}}[\log D(x)]
+ \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))].
$$
This objective is linked to the **Jensen–Shannon** divergence at optimum. 

### The problem: Vanishing gradients for \(G\)
When fake samples already land on the "real" side of \(D\)'s boundary, the cross-entropy loss gives almost **no error** even if they are **far** from the true data manifold. Generator updates become weak. Training can saturate or collapse. (See Fig. 1 in paper.) 

---

## 3) LSGAN: Replace cross-entropy with least squares

We view \(D\) as a regressor to target labels:

- **Label scheme**: real -> fake b-> a
- **Generator target**: make D(G(z)) -> c

**Discriminator objective:**
$$
\min_D \; V_{\text{LSGAN}}(D)=
\frac{1}{2}\,\mathbb{E}_{x \sim p_{data}} \big[(D(x)-b)^2\big]
+ \frac{1}{2}\,\mathbb{E}_{z \sim p_z} \big[(D(G(z))-a)^2\big].
$$

**Generator objective:**
$$
\min_G \; V_{\text{LSGAN}}(G)=
\frac{1}{2}\,\mathbb{E}_{z \sim p_z} \big[(D(G(z)) - c)^2\big].
$$

**Key intuition**: With L2, **even correctly classified** fakes get a penalty if they are **far** from the target score. This keeps gradients **alive** and pulls fakes **toward** the decision boundary (and thus toward the real manifold). 

---

## 4) Choosing the labels \(a, b, c\)

Two common choices (both work well in practice)

### (A) 0–1 coding with “make fakes look real”
- Set \(a=0\), \(b=1\), \(c=1\)

Then:
$$
\min_D \; \frac{1}{2}\,\mathbb{E}_{x \sim p_{data}}[(D(x)-1)^2]
+ \frac{1}{2}\,\mathbb{E}_{z \sim p_z}[D(G(z))^2],
$$
$$
\min_G \; \frac{1}{2}\,\mathbb{E}_{z \sim p_z}[(D(G(z))-1)^2].
$$

### (B) Symmetric coding linked to Pearson \(\chi^2\)
- Set \(a=-1\), \(b=1\), \(c=0\)

Then:
$$
\min_D \; \frac{1}{2}\,\mathbb{E}_{x \sim p_{data}}[(D(x)-1)^2]
+ \frac{1}{2}\,\mathbb{E}_{z \sim p_z}[(D(G(z))+1)^2],
$$
$$
\min_G \; \frac{1}{2}\,\mathbb{E}_{z \sim p_z}[D(G(z))^2].
$$

> In experiments, (A) and (B) behave similarly. Many implementations use (A). 

---

## 5) Optimal \(D\) and the Pearson X^2 connection (math overview)

Consider a slightly extended generator objective that includes a real-data term (this does not change \(G\)’s optimum because it is constant w.r.t. \(G\)’s parameters): 

$$
\min_D \; \frac{1}{2}\,\mathbb{E}_{x \sim p_{data}}[(D(x)-b)^2]
+ \frac{1}{2}\,\mathbb{E}_{z \sim p_z}[(D(G(z))-a)^2],
$$
$$
\min_G \; \frac{1}{2}\,\mathbb{E}_{x \sim p_{data}}[(D(x)-c)^2]
+ \frac{1}{2}\,\mathbb{E}_{z \sim p_z}[(D(G(z))-c)^2].
$$

Let $$ (p_d(x)=p_{data}(x)) $$ and $$(p_g(x))$$ the model distribution.  
For fixed \(G\), the optimal discriminator is a **weighted average**: :contentReference[oaicite:10]{index=10}
$$
D^*(x) = \frac{b\,p_d(x) + a\,p_g(x)}{p_d(x)+p_g(x)}.
$$

Plug \(D^*\) into the generator criterion:
$$
2C(G) = \mathbb{E}_{x \sim p_d}\big[(D^*(x) - c)^2\big]
+ \mathbb{E}_{x \sim p_g}\big[(D^*(x)-c)^2\big].
$$

Algebra gives:
$$
2C(G) = \int \frac{\Big((b-c)\,p_d(x) + (a-c)\,p_g(x)\Big)^2}{p_d(x)+p_g(x)} \, dx.
$$

If we choose labels such that
$$
b - c = 1, \qquad b - a = 2,
$$
then
$$
2C(G) = \int \frac{\big(2p_g(x) - (p_d(x)+p_g(x))\big)^2}{p_d(x)+p_g(x)} \, dx
= \chi^2_{\text{Pearson}}\!\big(p_d + p_g \,\Vert\, 2p_g\big).
$$

So minimizing LSGAN’s objective **minimizes a Pearson \(\chi^2\)** divergence between $$(p_d + p_g)  and  (2p_g)$$. This gives a divergence-minimization view similar to other f-GAN analyses. 

---

## 6) Why least squares helps (geometry & gradients)

- **Geometry**: With L2, samples far from their target score get **quadratically** larger penalties. Even if a fake is on the real side, if it is **far**, it still gets a **strong** push toward the boundary → closer to the real manifold. 
- **Gradients**: Sigmoid cross-entropy saturates for large logits (flat loss), giving tiny gradients. L2 is flat only at the target point → **more reliable gradients** during training. 

---

## 7) Architectures (simple summary)

### Scene image generator/discriminator (112×112)
- Generator: deconvs (upsampling) with BN and ReLU; final conv to image.  
- Discriminator: conv stack with LeakyReLU; final linear to scalar; **least squares** loss.

### Many-class conditional model (e.g., 3,740 Chinese characters)
- Use a **linear mapping** \(\Phi(y)\) to compress huge one-hot labels to a small vector.
- Concatenate $$ (\Phi(y)) into (G) and (D) $$ at chosen layers.
- Objectives (0–1 coding):
  $$
  \min_D \; \tfrac12 \mathbb{E}_{x \sim p_{data}}[(D(x \mid \Phi(y)) - 1)^2]
  + \tfrac12 \mathbb{E}_{z \sim p_z}[D(G(z) \mid \Phi(y))^2],
  $$
  $$
  \min_G \; \tfrac12 \mathbb{E}_{z \sim p_z}[(D(G(z) \mid \Phi(y)) - 1)^2].
  $$



---




