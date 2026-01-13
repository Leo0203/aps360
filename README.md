# Leaf Disease Detection (APS360 Project)

A deep learning project focused on automated plant disease recognition from leaf images, developed as part of **APS360 – Applied Fundamentals of Deep Learning** at the **University of Toronto**.

This project explores multiple computer vision architectures to improve model accuracy, robustness, and generalization, simulating a real-world agricultural AI application for early disease detection.

---

## 📌 Project Overview

Early identification of plant diseases is critical for improving crop yield and reducing economic loss in agriculture. This project aims to build an intelligent image classification system capable of recognizing plant leaf diseases from raw images using modern deep learning techniques.

**Goal:**
Classify plant leaf diseases using computer vision and neural networks.

**Approach:**

* Built CNN-based image classification models
* Applied transfer learning using pretrained architectures
* Performed systematic experimentation and evaluation
* Focused on robustness and generalization for real-world deployment scenarios

---

## 🧠 Model Architecture

**Final Architecture:**
**ResNet18 + CBAM + Soft Voting Ensemble**

The final model is an ensemble architecture combining:

* A **ResNet18 backbone** for strong feature extraction
* A **Convolutional Block Attention Module (CBAM)** to enhance spatial and channel-wise attention
* **Soft Voting Ensemble Learning** to aggregate predictions from multiple trained models and improve robustness and generalization

**Key Components:**

* ResNet18 backbone
* CBAM attention module
* Ensemble learning via soft voting

### Architecture Diagram

**Architecture (ResNet18 + CBAM + Soft Voting)**

![Architecture Diagram](./final_model_struct.png)

---

## 🛠 Technical Stack

* **Languages:** Python
* **Frameworks:** PyTorch
* **Core Topics:**

  * Convolutional Neural Networks (CNNs)
  * Transfer Learning
  * Attention Mechanisms (CBAM)
  * Ensemble Learning
  * Model Evaluation & Experimentation
* **Tools:**

  * NumPy, Matplotlib
  * Jupyter Notebook
  * Git / GitHub

---

## 📈 Key Contributions

* Designed and trained custom CNN architectures for image classification
* Implemented **ResNet18 enhanced with CBAM attention**
* Built an **ensemble pipeline using soft voting** to improve generalization
* Conducted systematic experiments across architectures and hyperparameters
* Analyzed performance using validation metrics and error patterns
* Produced technical documentation and reports as part of course deliverables

---

## 🎯 Project Significance

This project simulates a realistic AI application in agriculture:

* Demonstrates how deep learning can support sustainable farming
* Highlights the importance of robustness in real-world datasets
* Reflects a full ML workflow: design → training → evaluation → iteration

It also strengthened practical skills in:

* Deep learning system design
* Research-style experimentation
* Debugging training pipelines
* Communicating technical results clearly

---

## 👤 Author

**Shilin Ma**
Computer Engineering @ University of Toronto
GitHub: [https://github.com/Leo0203](https://github.com/Leo0203)
LinkedIn: [https://linkedin.com/in/shilin-ma-0b8785343](https://linkedin.com/in/shilin-ma-0b8785343)
Course: APS360 – Applied Fundamentals of Deep Learning

---

## 📎 Note

This project was completed as part of academic coursework.
Code is shared for demonstration of technical ability and learning outcomes.
