# Image Classification with Very Deep CNN (Inspired by VGG - ICLR 2015)

## Introduction

This section-mini-project is inspired by the foundational paper **"Very Deep Convolutional Networks for Large-Scale Image Recognition"** by Karen Simonyan and Andrew Zisserman, presented at ICLR 2015. Known as the VGG network, it demonstrated that increasing the depth of convolutional neural networks using small 3×3 filters leads to significant improvements in classification accuracy. VGG models achieved top results in the ImageNet LSVRC-2014 competition, securing second place in classification with a top-5 test error of just 6.8%.

In this section-project, we build and train a simplified version of VGG16 using PyTorch, apply it to a small dataset, and explore the key principles that made VGG a breakthrough in deep learning.

---

## Architecture Overview

VGG is a deep convolutional neural network characterized by its simplicity and depth:

- Uses only **3×3 convolutional filters** (stride 1, padding 1)
- Max-pooling with 2×2 windows, stride 2
- No Local Response Normalization (unlike AlexNet)
- Stacks multiple conv layers before each pooling layer
- Followed by 3 fully connected layers at the end

### Detailed Layer Structure (VGG-16)

| Layer Type         | Filters / Units         | Filter Size / Notes               |
|--------------------|--------------------------|------------------------------------|
| Input              | 224×224×3                | RGB image                         |
| Conv1_1            | 64 filters               | 3×3, stride 1, padding 1           |
| Conv1_2            | 64 filters               | 3×3                                |
| Max Pool           | -                        | 2×2, stride 2                      |
| Conv2_1            | 128 filters              | 3×3                                |
| Conv2_2            | 128 filters              | 3×3                                |
| Max Pool           | -                        | 2×2                                |
| Conv3_1            | 256 filters              | 3×3                                |
| Conv3_2            | 256 filters              | 3×3                                |
| Conv3_3            | 256 filters              | 3×3                                |
| Max Pool           | -                        | 2×2                                |
| Conv4_1            | 512 filters              | 3×3                                |
| Conv4_2            | 512 filters              | 3×3                                |
| Conv4_3            | 512 filters              | 3×3                                |
| Max Pool           | -                        | 2×2                                |
| Conv5_1            | 512 filters              | 3×3                                |
| Conv5_2            | 512 filters              | 3×3                                |
| Conv5_3            | 512 filters              | 3×3                                |
| Max Pool           | -                        | 2×2                                |
| FC6                | 4096 units               | Dropout (0.5)                      |
| FC7                | 4096 units               | Dropout (0.5)                      |
| FC8                | 1000 output classes      | Softmax                            |

<br>
<p align="center">
  <img src="https://www.researchgate.net/publication/356975676/figure/fig7/AS:1100460064804886@1639381533469/Basic-architecture-of-VGG16-model.jpg" width="600"><br>
  <strong>Figure:</strong> Basic architecture of the VGG-16 model  
  <br>
  <em>Source: ResearchGate (from the paper "COVIDetection-Net: A tailored COVID-19 detection from chest radiography images using deep learning")</em>
</p>
<br>

---

## Key Ideas & Techniques

### 1. Deep Architecture with Small Filters

Instead of using large filters (e.g., 7×7 or 11×11), VGG uses only 3×3 filters throughout the convolutional layers. Stacking two 3×3 layers gives the receptive field of 5×5, and stacking three gives 7×7 — but with fewer parameters and more non-linearity. This deep stack of small filters is more efficient and expressive.

### 2. Homogeneous Architecture

All convolutional layers use the same filter size (3×3) and padding (1). Pooling layers are all max-pooling with 2×2 window and stride 2. This clean design makes VGG easy to scale and implement.

### 3. ReLU Activation

Like AlexNet, VGG uses the ReLU activation function to introduce non-linearity and speed up training:

\[
f(x) = \max(0, x)
\]

### 4. No Local Response Normalization (LRN)

Unlike AlexNet, VGG avoids LRN as it was shown to increase memory usage without improving accuracy on ImageNet.

### 5. Fully Connected Layers with Dropout

The two fully connected layers at the end of the network have 4096 units each. Dropout is applied during training with a rate of 0.5 to prevent overfitting.

### 6. Multi-Scale Training & Testing

During training, images were randomly rescaled (scale jittering) to simulate multiple object sizes. At test time, the model is applied at multiple scales and predictions are averaged to improve robustness.

### 7. Training Setup

Training was done using SGD with:
- Batch size = 256
- Learning rate = 0.01
- Momentum = 0.9
- Weight decay = 5e-4
- Dropout = 0.5 on FC layers
- Learning rate decreased by 10× when validation accuracy plateaued

Models were trained on multiple GPUs (up to 4), and initialization was carefully handled to allow deeper networks to converge.

---

## Training Results

Below are the training losses over 10 epochs for our simplified VGG model:

| Epoch | Training Loss |
|-------|----------------|
|   1   |     2.8732     |
|   2   |     3.0151     |
|   3   |     2.8734     |
|   4   |     2.8183     |
|   5   |     3.0160     |
|   6   |     3.2149     |
|   7   |     2.5706     |
|   8   |     2.4398     |
|   9   |     2.9295     |
|  10   |     3.2723     |

**Validation Accuracy**: **63.01%**  
**Test Accuracy**: **79.41%**  

Noting that our model is trained on a small image classification dataset (~100 images) split into 80% training, 10% validation, and 10% testing. 
---

## Citation

If you are interested in the original research, you can cite it as:

> Karen Simonyan and Andrew Zisserman.  
> *Very Deep Convolutional Networks for Large-Scale Image Recognition*.  
> International Conference on Learning Representations (ICLR), 2015.  
> [arXiv:1409.1556](https://arxiv.org/abs/1409.1556)

---
