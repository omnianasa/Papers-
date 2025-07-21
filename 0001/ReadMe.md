# Image Classification with Deep CNN (Inspired by AlexNet - NIPS 2012)

## Introduction

This project is inspired by the landmark paper **"ImageNet Classification with Deep Convolutional Neural Networks"** by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, presented at NIPS 2012. The model, later known as AlexNet, and won the ImageNet LSVRC-2012 competition with a top-5 test error rate of 15.3%, far ahead of previous methods.

In this section-project, we build and train a simplified version of AlexNet using PyTorch, apply it to a small dataset, and confidently say — we’ve nailed the key concepts and ideas behind the original paper.

---

## Architecture Overview

AlexNet is a deep convolutional neural network with the following key characteristics:

- 5 convolutional layers
- 3 fully connected layers
- ReLU activation for non-linearity
- Dropout for regularization
- Overlapping max pooling
- Local Response Normalization (LRN)
- Trained on 2 GPUs in parallel

### Detailed Layer Structure

| Layer Type         | Filters / Units         | Filter Size / Notes           |
|--------------------|--------------------------|--------------------------------|
| Input              | 224×224×3                | RGB image                     |
| Conv1              | 96 filters               | 11×11, stride 4               |
| Max Pool + LRN     | -                        | 3×3 pooling, stride 2         |
| Conv2              | 256 filters              | 5×5                           |
| Max Pool + LRN     | -                        | 3×3 pooling, stride 2         |
| Conv3              | 384 filters              | 3×3                           |
| Conv4              | 384 filters              | 3×3                           |
| Conv5              | 256 filters              | 3×3                           |
| Max Pool           | -                        | 3×3 pooling, stride 2         |
| FC6                | 4096 neurons             | Dropout (0.5)                 |
| FC7                | 4096 neurons             | Dropout (0.5)                 |
| FC8                | 1000 output classes      | Softmax                       |

<br>
<p align="center">
  <img src="images/modelStructure.jpg" width="300"><br>
  <strong>Figure:</strong>Basic AlexNet Model Structure
</p>
<br>

## Key Ideas & Techniques

### 1. ReLU Activation
Used instead of sigmoid or tanh. Much faster training:

\[
f(x) = \max(0, x)
\]

### 2. Local Response Normalization (LRN)
Promotes competition between neuron outputs in nearby locations:

\[
b_{x,y}^{i} = a_{x,y}^{i} \Bigg/ \left( k + \alpha \sum_{j=max(0,i-n/2)}^{min(N-1,i+n/2)} (a_{x,y}^{j})^2 \right)^{\beta}
\]

Used after Conv1 and Conv2.

###  3. Overlapping Max Pooling
Reduces feature map size, improves generalization. Pooling window: 3×3 with stride 2 (so windows overlap).

### 4. Dropout
Prevents overfitting by randomly dropping neurons during training:

- Applied to FC6 and FC7
- Dropout rate = 0.5

###  5. Data Augmentation
Two types used:
- Random crops and horizontal flips
- PCA color augmentation

### 6. Multi-GPU Training
Used 2 GPUs to split the model and increase model size without memory overflow.

---

## Training Results

Below are the training losses over 10 epochs:

| Epoch | Training Loss |
|-------|----------------|
|   1   |     2.7858     |
|   2   |     2.7559     |
|   3   |     2.7432     |
|   4   |     2.7310     |
|   5   |     2.6053     |
|   6   |     2.4118     |
|   7   |     2.7366     |
|   8   |     2.5575     |
|   9   |     2.4344     |
|  10   |     2.7107     |

**Validation Accuracy**: **69.12%**  
**Test Accuracy**: **71.88%**
Noting that we are dealing with so small sized dataset of ~100 images splitted into 80% train, 10% val and 10% test. you can view sample of it in the figure below:
<br>
<p align="center">
  <img src="images/cats.jpg" width="300"><br>
  <strong>Model Structure</strong> Dog
</p>
<br>

## Citation

If you are interested in the original research, you can cite it as:

> Karen Simonyan and Andrew Zisserman.  
> *Very Deep Convolutional Networks for Large-Scale Image Recognition*.  
> International Conference on Learning Representations (ICLR), 2015.  
> [arXiv:1803.01164](https://arxiv.org/abs/1803.01164)

---










