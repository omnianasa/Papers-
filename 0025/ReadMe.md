
# YOLO v1 – You Only Look Once: Unified, Real-Time Object Detection

## Overview
YOLO reframes object detection as a single regression problem—predicting bounding boxes and class probabilities directly from full images in one evaluation using a single neural network. This unified approach delivers **real-time performance**, with the base model reaching **~45 FPS** and **Fast YOLO** achieving **~155 FPS** ([arxiv.org](https://arxiv.org/abs/1506.02640?utm_source=chatgpt.com))

---

## Key Ideas

### 1. Single-Pass Detection
Instead of using multi-stage pipelines (like sliding windows or region proposals), YOLO treats detection as regression. It resizes input images to **448×448**, feeds them through a CNN, and applies **non-max suppression** to finalize predictions 

### 2. Grid-Based Prediction
YOLO divides the image into an **S×S** grid. Each grid cell predicts:
- **B bounding boxes**, each described by (x, y, w, h, confidence)
- **C class probabilities**
All predictions form a tensor of shape **S × S × (B×5 + C)** 

### 3. Confidence Score
Each bounding box has a confidence score defined as:
```math
confidence = Pr(Object) × IoU_{pred}^{truth}
```
Indicating both the likelihood that an object exists and how well the box fits ([zenn.dev](https://zenn.dev/yuto_mo/articles/9e216773dd321a?utm_source=chatgpt.com)).

### 4. Loss Function
YOLO’s loss combines multiple components:
```math
Loss = λ_coord ∑_{i=0}^{S^2} ∑_{j=0}^{B} 𝟙_{ij}^{obj} [(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2]
     + λ_coord ∑_{i=0}^{S^2} ∑_{j=0}^{B} 𝟙_{ij}^{obj} [√w_i - √\hat{w}_i)^2 + (√h_i - √\hat{h}_i)^2]
     + ∑_{i=0}^{S^2} ∑_{j=0}^{B} 𝟙_{ij}^{obj} (C_i - \hat{C}_i)^2
     + λ_noobj ∑_{i=0}^{S^2} ∑_{j=0}^{B} 𝟙_{ij}^{noobj} (C_i - \hat{C}_i)^2
     + ∑_{i=0}^{S^2} 𝟙_{i}^{obj} ∑_{c \in classes} (p_i(c) - \hat{p}_i(c))^2
```
- `λ_coord = 5` emphasizes localization error
- `λ_noobj = 0.5` reduces penalty for boxes without objects
Bounding box dimensions use square root to emphasize small errors ([arxiv.org](https://arxiv.org/pdf/1506.02640)).

---

## Experiments & Results

### Datasets & Setup
- Trained on **PASCAL VOC 2007 / 2012**
- **Batch size**: 64, **momentum**: 0.9, **decay**: 0.0005
- **LR schedule**: warm-up from 1e-3 to 1e-2, then 1e-2 (75 epochs), 1e-3 (30), 1e-4 (30)
- Used data augmentation and dropout (0.5) ([arxiv.org](https://arxiv.org/pdf/1506.02640)).

### Performance
- **YOLO base**: ~45 FPS, **Fast YOLO**: ~155 FPS
- Outperforms other real-time methods with over **2× mAP** 
- Generalizes better; outperforms DPM and R-CNN on artwork datasets 

### Boosting Fast R‑CNN
Combining YOLO with Fast R‑CNN yields a **+3.2% mAP improvement (to 75.0%)**, while combining multiple Fast R‑CNN models only adds +0.3–0.6%. This is because YOLO and R‑CNN make **different types of errors**, making them complementary 
---

## Pros & Limitations

### Advantages
- Unified, end-to-end trainable model
- Extremely fast, suitable for real-time applications
- Global reasoning reduces false positives
- Strong generalization to unseen domains

### Limitations
- Spatial constraints: each cell predicts only 2 boxes & 1 class → struggles with multiple small objects
- Coarse features due to downsampling hurt localization, especially for small objects
- Loss function penalizes large and small boxes equally, which may be suboptimal 
---

## Why It Matters
YOLO redefined object detection by offering a **simple, fast, and accurate** alternative to slow, complex pipelines. Its impact continues in subsequent versions (YOLOv2, v3, …). This original version paved the way for modern real-time detection in AI systems.

---

## References
- Redmon *et al.*, **You Only Look Once: Unified, Real-Time Object Detection**, CVPR 2016 / arXiv:1506.02640 