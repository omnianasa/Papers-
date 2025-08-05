# DCGAN:  UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS

This paper introduces **DCGAN**, a class of Convolutional Neural Networks designed for stable training of Generative Adversarial Networks (GANs). The original paper is:  
**"Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" (Radford et al., 2015)**

---

## Overview

DCGANs are a type of GAN that combine the idea of generative modeling with CNNs, allowing models to generate realistic images **from noise** and **learn useful image features** without needing labels.

The goal of this architecture is to:
- Generate high quality images from random vectors.
- Learn visual representations useful for other tasks like classification.
- Train stably across diverse datasets without collapsing or oscillation.

---

## Generator Architecture

The **generator** takes a random vector `Z` (typically 100-dimensional) and produces a synthetic image through a series of convolutional layers.

### Steps:
1. **Input**: A latent vector `Z` drawn from a uniform distribution.
2. **Fully connected projection** to reshape `Z` into a small feature map with multiple channels.
3. **Fractionally-strided convolutions** (sometimes called "deconvolutions") are used to upsample the image gradually to the desired size (e.g., 64x64).
4. **Batch Normalization** is applied after each layer to stabilize learning.
5. **ReLU Activation** is used in all layers except the output.
6. **Tanh Activation** in the output ensures pixel values are bounded between [-1, 1].

### Why this setup?

- **Fractional-stride convs** help the model learn to upsample spatially rather than use heuristics like nearest-neighbor.
- **No fully connected layers** after the initial projection encourages spatial structure.
- **ReLU and Tanh** help with gradient flow and pixel range stabilization.

---

## Discriminator Architecture

The **discriminator** receives an image (real or generated) and learns to distinguish between the two.

### Steps:
1. **Input**: Image (e.g., 64x64 RGB)
2. **Strided Convolutions** are used to downsample spatial dimensions.
3. **Leaky ReLU** activations are used after each layer.
4. **Batch Normalization** is applied, except at the input layer.
5. The final layer outputs a single scalar through a **Sigmoid**, representing the probability the image is real.

### Why this setup?

- **Strided convolutions** are used instead of pooling, giving the network more control over feature extraction.
- **Leaky ReLU** prevents dying neurons, which helps especially in the discriminator.
- **BatchNorm** improves gradient flow and speeds up convergence.

---

## Design Principles for Stability

The paper identifies the following guidelines for stable DCGAN training:

- Replace pooling with **strided convolutions** (Discriminator) and **fractional-strided convolutions** (Generator).
- Use **batch normalization** in both networks (except at key input/output points).
- Remove **fully connected layers** in deeper networks.
- Use **ReLU** in the generator (except the output).
- Use **LeakyReLU** in the discriminator.

---

## Training Setup

- Optimizer: **Adam**
  - Learning rate: `0.0002`
  - Beta1 (momentum): `0.5`
- No image augmentation.
- Weight initialization: normal distribution `N(0, 0.02)`
- Images are scaled to the range `[-1, 1]` (because of Tanh)

### Datasets used:

- **LSUN Bedrooms** (~3 million images)
- **Imagenet-1k**
- **Custom Faces Dataset** (scraped from web)

---

## Evaluating Representation Learning

Although DCGANs are trained in an **unsupervised** way, the learned **features from the discriminator** can be reused for **supervised tasks**.

### Experiments:

- On **CIFAR-10**, DCGAN features + Linear SVM achieves **82.8% accuracy**, outperforming other unsupervised methods like K-Means.
- On **SVHN**, DCGAN + Linear SVM gives **22.48% test error**, beating many semi-supervised baselines.

---

## Understanding the Model Internals

The authors explored the internal structure of the learned representations:

- **Walking in the latent space (Z)** results in smooth transformations (e.g., a room gaining a window gradually).
- **Discriminator filters** activate on real-world concepts (e.g., beds, windows), showing that it learns semantic features.
- **Vector arithmetic** in the latent space (Z) can achieve meaningful edits (e.g., "smiling face - neutral face + glasses").

---

## Key Contributions

- A well-defined architecture for stable GAN training using CNNs.
- Empirical evidence that unsupervised GANs can learn meaningful features.
- Demonstration that learned features are transferable to supervised tasks.
- Introduction of semantic vector operations in the latent space.

---

## Limitations & Future Work

- **Mode collapse** still occurs with prolonged training.
- Further work is needed to extend to other domains (e.g., video, audio).
- Exploring structure of the latent space may help build more controllable generative models.

---

## Reference

Radford, A., Metz, L., & Chintala, S. (2015).  
Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.  
[arXiv:1511.06434v2](https://arxiv.org/abs/1511.06434)
