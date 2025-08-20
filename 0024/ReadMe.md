# CycleGAN — Paper-Centric README

A practical, paper-focused guide to **Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks** (CycleGAN) by Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros (arXiv:1703.10593). This README distills the paper’s **intuition, math, architecture, training recipe, experiments, and limitations**—ready to drop into a project repo.

> Goal: Learn mappings between two visual domains (e.g., horses ↔ zebras, summer ↔ winter, Monet ↔ photos) **without paired supervision**.

---

## 1) Problem Setup

We have two domains:
- **X**: source domain samples, e.g., horses (unpaired)  
- **Y**: target domain samples, e.g., zebras (unpaired)

We want two mappings:
- **G: X → Y**  
- **F: Y → X**

And two discriminators:
- **D_Y** to distinguish real **y ∈ Y** from **G(x)**  
- **D_X** to distinguish real **x ∈ X** from **F(y)**  

**Key challenge:** With adversarial loss alone, many mappings can match the target distribution but **not** preserve per-image content (mode collapse / permutation ambiguity).  

**Key idea:** Enforce **cycle consistency** so that translating forward and then back returns the original: **F(G(x)) ≈ x** and **G(F(y)) ≈ y**.  

---

## 2) Mathematical Objective

### 2.1 Adversarial Loss (per domain)
For **G: X → Y** with discriminator **D_Y** (original GAN form):

```math
\mathcal{L}_{	ext{GAN}}(G, D_Y; X, Y) =
\mathbb{E}_{y\sim p_{	ext{data}}(y)}[\log D_Y(y)] +
\mathbb{E}_{x\sim p_{	ext{data}}(x)}[\log(1 - D_Y(G(x)))]
```

Similarly for **F: Y → X** with **D_X**:

```math
\mathcal{L}_{	ext{GAN}}(F, D_X; Y, X)
```

> **Practical note (as in the paper):** they adopt **Least Squares GAN (LSGAN)** for stability. With LSGAN, the generator minimizes  
> ```math
> \mathbb{E}[(D(G(x)) - 1)^2]
> ```  
> and the discriminator minimizes  
> ```math
> \mathbb{E}[(D(y)-1)^2] + \mathbb{E}[D(G(x))^2]
> ```

---

### 2.2 Cycle Consistency (L1)

```math
\mathcal{L}_{	ext{cyc}}(G,F) =
\mathbb{E}_{x\sim p_{	ext{data}}(x)}/big[\|F(G(x)) - x\|_1/big] +
\mathbb{E}_{y\sim p_{	ext{data}}(y)}/big[\|G(F(y)) - y\|_1/big]
```

**Why L1?**
- Promotes **sharp, stable** reconstructions  
- Less sensitive to outliers than L2; empirically reduces blur and works better for per-image consistency  

---

### 2.3 Identity Mapping Loss (optional but useful)

```math
\mathcal{L}_{	ext{id}}(G,F) =
\mathbb{E}_{y\sim p_{	ext{data}}(y)}/big[\|G(y) - y\|_1/big] +
\mathbb{E}_{x\sim p_{	ext{data}}(x)}/big[\|F(x) - x\|_1/big]
```

---

### 2.4 Full Objective

```math
\mathcal{L}(G,F,D_X,D_Y) =
\mathcal{L}_{	ext{GAN}}(G, D_Y; X, Y) +
\mathcal{L}_{	ext{GAN}}(F, D_X; Y, X) +
\lambda_{	ext{cyc}}\,\mathcal{L}_{	ext{cyc}}(G,F) +
\lambda_{	ext{id}}\,\mathcal{L}_{	ext{id}}(G,F)
```

Training solves:

```math
G^*,F^* = \arg\min_{G,F}\;\max_{D_X,D_Y}\; \mathcal{L}
```

**Typical weights:**  
```math
\lambda_{	ext{cyc}} = 10,\;\;\lambda_{	ext{id}} \in \{0, 0.5\lambda_{	ext{cyc}}\}
```

---

## 3) Architecture
- **Generators (G, F):** ResNet-based image-to-image nets  
- **Discriminators (D_X, D_Y):** 70×70 PatchGAN  

---

## 4) Training Recipe (from paper)

- Optimizer: Adam (lr=2e-4, batch=1)  
- Schedule: 100 epochs constant lr, 100 epochs linear decay  
- GAN loss: LSGAN  
- Replay buffer: ~50 generated samples  
- Preprocessing: resize & crop to 256×256  

(Pseudocode kept same as before.)

---

## 5) Experiments & Findings
- Datasets: Cityscapes, Maps↔Aerial, Horse↔Zebra, Apple↔Orange, Style transfer, Season transfer  
- Metrics: AMT tests, FCN scores  
- Ablations: remove GAN → blurry, remove cycle → collapse  
- Qualitative: works well for appearance/style, struggles with geometry  

---

## 6) Why L1 for Cycle Consistency?
- Pixel-wise fidelity  
- Robust & sharp vs L2  
- Stable vs adversarial cycle  

---

## 7) Practical Tips
- Start with 256×256  
- Use identity loss for color constancy  
- Keep replay buffer  
- Monitor GAN & cycle losses  

---

## 8) Limitations
- Weak on geometric changes  
- Label permutation ambiguity  
- Gap to paired methods remains  

---

## 9) References
- Zhu et al. *CycleGAN*, arXiv:1703.10593  
- Isola et al. *pix2pix*, CVPR 2017  
- Mao et al. *LSGAN*, CVPR 2017  
- Johnson et al. *Perceptual Losses*, ECCV 2016  
