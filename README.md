# Model: bplane-small

Technical documentation for a lightweight military aircraft detection and classification model, designed for high-dynamic aerial scenarios and real-world operational footage.

<img src="assets/bplane_demo.gif" alt="bplane-small demo" width="100%">

### Operational Demonstration
The demonstration above highlights the model’s behavior under rapid viewpoint changes, high-G maneuvers, and partial silhouette deformation. Temporal stabilization and class voting are applied to maintain consistent identity assignment across consecutive frames.

### Project Overview
BPlane-Small-v1 is a computer vision framework built on the YOLO architecture, optimized for the detection and classification of military aircraft in unconstrained environments. The model is designed to operate on real surveillance-style imagery, where scale variation, motion blur, and non-ideal viewing angles are common. The system supports fine-grained airframe classification across multiple fighter and support platforms, while retaining a general Military Aircraft fallback class for ambiguous or previously unseen silhouettes.

### Technical Specifications
The BPlane-Small-v1 architecture utilizes a specialized YOLO backbone optimized for high-frequency aerial inference. Operating at a native resolution of 1024×1024 with support for dynamic inference, the model achieves a peak mAP@50 of 0.805. This balance of speed and precision is tailored for hardware-constrained environments, delivering robust localization accuracy. The system is exported in both PyTorch (.pt) and ONNX (.onnx) formats to ensure seamless integration across edge computing and cloud-based deployment pipelines.

| Category | Parameter | Specification / Result |
| :--- | :--- | :--- |
| **Architecture** | Base Framework | YOLOv11s (Small) |
| **Input** | Native Resolution | 1024 x 1024 px |
| **Training** | Total Epochs | 100 |
| **Accuracy** | mAP@50 | 0.805 |
| **Precision** | mAP@50-95 | 0.657 |
| **Reliability** | Precision (B) | 0.814 |
| **Sensitivity** | Recall (B) | 0.701 |
| **Deployment** | Export Formats | PyTorch (.pt), ONNX (.onnx) |

> **Technical Note on Localization:** The recorded **mAP@50-95 of 0.657** indicates high structural fidelity in bounding box placement. This level of precision is achieved through 1024px native inference, which minimizes feature loss for small-scale silhouettes at long ranges.

### Training Dynamics & Data Synthesis
The following training batch illustrates the augmentation strategies used to improve model robustness, including mosaic compositions and color space jittering to simulate varying atmospheric conditions.

<img src="assets/train_batch10890.jpg" alt="Training Batch Samples" width="100%">

### Ground Truth Label Visualization
The images below show annotated validation samples with ground truth bounding boxes. They provide insight into dataset quality, annotation consistency, and scale variation across different aircraft classes.

<p align="center">    
  <img src="assets/val_batch0_labels.jpg" width="49%">    
  <img src="assets/val_batch1_labels.jpg" width="49%">    
</p>

### Dataset Composition & Label Statistics
The following figure summarizes class frequency and bounding box distributions generated automatically during training. It highlights both class imbalance and object scale diversity, which directly influence classification stability.

<img src="assets/labels.jpg" alt="YOLO label statistics" width="80%">

> Notably, fighter-class aircraft exhibit significant scale and aspect-ratio variance, reinforcing the need for robust multi-scale feature extraction.

### Dataset Statistics (Quantitative)

The dataset consists of **11,006 images** with a total of **19,754 annotated aircraft instances**, resulting in an average of **1.79 objects per image**.

#### Split Configuration

| Split | Images | Ratio |
|-------|--------|-------|
| Train | 7,700  | 70%   |
| Validation | 2,197 | 20% |
| Test | 1,109 | 10% |
| **Total** | **11,006** | **100%** |

#### Class Distribution 

| Class | Instances | % of Total |
|-------|-----------|------------|
| f16 | 2,692 | 13.63% |
| f18 | 2,236 | 11.32% |
| f35 | 1,905 | 9.64% |
| f15 | 1,948 | 9.86% |
| c130 | 1,771 | 8.97% |
| j20 | 913 | 4.62% |
| ef2000 | 1,042 | 5.27% |
| rafale | 1,010 | 5.11% |
| a10 | 930 | 4.71% |
| c2 | 846 | 4.28% |
| aircraft (fallback) | 4,461 | 22.57% |
| **Total** | **19,754** | **100%** |

The fallback **aircraft** class intentionally absorbs ambiguous silhouettes to reduce cross-class misclassification between visually similar fighter platforms.

### Statistical Analysis & Performance
Overall performance indicates strong reliability for transport and utility aircraft, as well as clear differentiation between most air superiority fighters. Misclassifications tend to occur under extreme maneuvering or when aircraft silhouettes deviate significantly from canonical viewpoints.

### Normalized Confusion Matrix
<img src="assets/confusion_matrix_normalized.png" alt="Confusion Matrix" width="100%">

### F-16 Detection Case Study
Among all classes, the F-16 exhibits the highest variability in detection confidence. This behavior is primarily driven by rapid silhouette transitions during high-G turns and aggressive roll angles.

* **Classification Accuracy:** 61%
* **Background Misclassification:** 13% (Typically during extreme banking or high-alpha maneuvers)
* **Generalization Loss:** 8% (Fallback to general “Aircraft” class under low-visibility conditions)
* **Cross-Model Confusion:** Minor overlap with F-18 (6%) and Rafale (4%)

### Training Progress & Metric Stabilization
Training metrics demonstrate stable convergence across localization, classification, and distribution focal loss (DFL). Precision–recall behavior indicates consistent generalization on the validation set.

<img src="assets/results.png" alt="Training Results" width="100%">

### Precision–Recall (PR) Curve
<img src="assets/BoxPR_curve.png" alt="Precision-Recall Curve" width="100%">

### Target Classification List
The dataset is composed exclusively of military aviation assets and includes the following target categories:

* **Air Superiority & Multi-Role:** F-15, F-16, F-18, F-35, J-20, EF2000, Rafale
* **Close Air Support:** A-10
* **Airlift & Utility:** C-130, C-2
* **General Category:** Military Aircraft (Fallback class for unspecified or ambiguous platforms)

### Future Development Objectives
To further enhance operational reliability, future updates will focus on expanding F-16 samples from ventral and dorsal viewpoints to mitigate background-induced misclassification. Additionally, architecture scaling to medium-sized YOLO variants is being evaluated to improve feature separation in dense or cluttered scenes, alongside refined silhouette discrimination for delta-wing configurations.

### License & Attribution
This project is licensed under the **Apache License 2.0**. It is developed using the **Ultralytics YOLOv11** framework, which is subject to the **AGPL-3.0 License**. While the specific model weights and documentation provided in this repository are open for use under the terms of Apache 2.0, any commercial deployment or redistribution involving the underlying framework must comply with Ultralytics' licensing requirements. 

Copyright © 2026 Bulut M.

---

### About the Developer

I am a 15-year-old **Backend Developer** with a passion for building efficient systems. This model was born out of a sudden burst of motivation while I was developing an `npmjs` module; I decided to see how far I could push a lightweight YOLO architecture within a **48-hour sprint**. 

By curating high-quality datasets from **Kaggle** and **Roboflow** and focusing on rapid hyperparameter tuning, I was able to stabilize the model at **0.805 mAP50** over a single weekend. While I utilized AI to help structure this professional documentation, the entire technical execution—from data merging to training and optimization—is a result of my personal initiative. This project serves as a "Proof of Work" for my ability to rapidly adapt to new domains and deliver high-performance results under tight constraints.

**Feel free to reach out for collaboration or inquiries via the icons or email below:**
**Email:** bulutmuf@criai.art

<p align="left">
  <a href="https://discord.gg/criai" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/discord.svg" alt="bulutdiscord" height="30" width="40" /></a>
<a href="https://twitter.com/bulutdevs" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/twitter.svg" alt="bulutdevs" height="30" width="40" /></a>
<a href="https://linkedin.com/in/bulutmuftuoglu" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" alt="bulutmuftuoglu" height="30" width="40" /></a>
<a href="https://instagram.com/bulutmuf" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/instagram.svg" alt="bulutdevs" height="30" width="40" /></a>
</p>
