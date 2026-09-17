# Federated-Skin-Disease-Classification-under-Non-IID-Data-Using-Model-Contrastive-Learning


## 📌 Overview

This project proposes a **privacy-preserving federated learning framework for skin disease classification under Non-IID data distributions**.

Skin disease classification models generally require large and diverse collections of dermoscopic images. However, individual hospitals and dermatology clinics may have limited datasets, and the distribution of diseases can vary significantly between institutions.

Centralizing medical images can improve model training, but transferring sensitive medical images between institutions introduces **privacy and security concerns**.

To address this problem, this project will investigate **Federated Learning (FL)**, where multiple clients collaboratively train a shared model while keeping their raw medical images locally.

The primary research focus is the **Non-IID data problem** in Federated Learning. Differences in disease distributions across clients can cause local models to learn different patterns and drift away from the global model. This project proposes the use of **Model-Contrastive Learning** to reduce the effect of client drift and improve federated model performance.

---

## 🎯 Problem Statement

In a conventional centralized machine learning setup, medical images from different hospitals or clinics are collected and transferred to a central server for training.

This approach presents two major challenges:

1. **Privacy:** Medical images are sensitive and may contain patient-related information.
2. **Data Heterogeneity:** Different institutions may have significantly different distributions of skin diseases.

Federated Learning addresses the privacy challenge by keeping raw data on individual clients. However, when client datasets are **Non-IID**, local models can diverge from the global model, resulting in reduced performance and slower or unstable convergence.

Therefore, this project aims to investigate:

> **How can Model-Contrastive Learning improve skin disease classification in Federated Learning when client data distributions are Non-IID?**

---

## 💡 Proposed Approach

The proposed framework will simulate multiple healthcare institutions as federated clients.

A publicly available dermoscopic image dataset, such as **HAM10000**, will be distributed among multiple clients with heterogeneous disease distributions.

Each client will:

1. Receive its local subset of skin disease images.
2. Train a local deep learning classification model.
3. Keep the raw images locally.
4. Generate model updates from local training.
5. Send model parameters/updates to the central server.

The central server will aggregate the client model updates without accessing the raw medical images.

The proposed **Model-Contrastive Learning** mechanism will encourage locally learned representations to remain consistent with the global model while still allowing clients to learn from their local data.

**FedProx** will also be implemented as a comparison approach under the same Non-IID conditions.

---

## 🏗️ Proposed System Architecture

```text
                  ┌──────────────────────────┐
                  │      Central Server      │
                  │                          │
                  │ Global Model             │
                  │ Model Aggregation        │
                  └────────────┬─────────────┘
                               │
                  Global Model │
                               ▼
        ┌──────────────────────────────────────────┐
        │              Federated Clients           │
        └──────────────────────────────────────────┘

       ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
       │   Client 1   │  │   Client 2   │  │   Client 3   │
       │              │  │              │  │              │
       │ Local Data   │  │ Local Data   │  │ Local Data   │
       │              │  │              │  │              │
       │ Skin Images  │  │ Skin Images  │  │ Skin Images  │
       │      ↓       │  │      ↓       │  │      ↓       │
       │ Local Model  │  │ Local Model  │  │ Local Model  │
       └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
              │                 │                 │
              └───────── Model Updates ──────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │ Model-Contrastive        │
                  │ Learning / Aggregation   │
                  └──────────────────────────┘
```

### Key Principle

**Raw medical images remain on the individual clients.**

Only model parameters or updates will be communicated with the central server.

---

## 🔬 Research Focus

The project will primarily investigate the following research aspects:

### 1. Federated Learning

Multiple clients collaboratively train a shared model without transferring their raw datasets to a central location.

### 2. Non-IID Data

Client datasets will have different disease distributions to simulate realistic heterogeneity between healthcare institutions.

For example:

```text
Client 1 → Mostly Melanoma + Melanocytic Nevi

Client 2 → Mostly Basal Cell Carcinoma + Benign Keratosis

Client 3 → Mostly Melanoma + Actinic Keratosis
```

The exact distribution will be determined during experimentation.

### 3. Client Drift

Because different clients observe different data distributions, their local models may learn different patterns.

This can cause local models to move away from the global model during training.

### 4. Model-Contrastive Learning

The proposed approach will investigate whether contrastive objectives between local and global model representations can reduce the effects of client drift under Non-IID conditions.

### 5. FedProx Comparison

FedProx will be implemented as a baseline/comparison method to evaluate the proposed approach under the same experimental conditions.

---

## 🧠 Model

A **ResNet-based image classification model** is planned for the initial implementation.

The model will be trained independently by each federated client using its local data.

The exact ResNet variant and training configuration will be finalized during implementation and experimentation.

---

## 📊 Dataset

The project plans to use a publicly available dermoscopic skin-image dataset such as:

### HAM10000

HAM10000 contains dermoscopic images representing multiple common pigmented skin lesions.

The dataset will be partitioned among simulated federated clients to create heterogeneous **Non-IID distributions**.

> Dataset selection and client-partitioning strategy will be finalized during the implementation stage.

---

## ⚙️ Experimental Plan

The project will compare different federated learning approaches under controlled Non-IID conditions.

### Planned approaches

```text
                 Federated Skin Disease Classification
                              │
                ┌─────────────┴─────────────┐
                │                           │
             FedProx              Model-Contrastive FL
                │                           │
                └─────────────┬─────────────┘
                              │
                    Same Non-IID Conditions
                              │
                              ▼
                     Performance Comparison
```

The comparison will investigate whether the model-contrastive approach can provide improved performance and more stable training compared with the baseline approach.

---

## 📈 Evaluation Metrics

The following metrics are planned for evaluation:

* **Accuracy**
* **Precision**
* **Recall**
* **Macro-F1 Score**
* **Per-class performance**
* **Convergence behavior**

Particular attention will be given to the performance of **underrepresented disease categories**, since Non-IID distributions may affect different classes differently.

---

## 🔐 Privacy and Scope

This project focuses on the federated learning aspect of privacy-preserving medical image classification.

### Included

* Federated Learning
* Non-IID client distributions
* Skin disease image classification
* Model-Contrastive Learning
* FedProx comparison
* Simulated federated clients
* Performance evaluation

### Not Included

* Real-world hospital deployment
* Clinical validation
* Diagnosis or treatment recommendations
* Real patient data
* Differential privacy
* Real-world deployment across hospitals

The initial implementation will use **simulated federated clients and publicly available data**.

---

## 🛠️ Planned Technology Stack

The exact stack will be finalized during implementation.

### Machine Learning

* Python
* PyTorch
* Torchvision
* NumPy
* Pandas
* Scikit-learn

### Federated Learning

A suitable Federated Learning framework will be evaluated during implementation, such as:

* Flower
* PyTorch-based custom federated simulation

### Visualization & Evaluation

* Matplotlib
* Seaborn
* Scikit-learn

### Development

* Git
* GitHub
* Jupyter Notebook / VS Code

---

## 📁 Planned Repository Structure

```text
federated-skin-disease-classification/
│
├── README.md
│
├── data/
│   └── README.md
│
├── src/
│   ├── models/
│   ├── clients/
│   ├── server/
│   ├── aggregation/
│   └── training/
│
├── experiments/
│   ├── fedprox/
│   └── model_contrastive/
│
├── notebooks/
│
├── results/
│
├── requirements.txt
│
└── .gitignore
```

> The repository structure is currently a proposed structure and will be updated as development progresses.


---

## 📚 Key Research References

### 1. PCRFed — Model-Contrastive Learning

Liu, S., Zhang, R., Fang, M., et al. (2025).
**“PCRFed: Personalized Federated Learning with Contrastive Representation for Non-Independently and Identically Distributed Medical Image Segmentation.”**
*Visual Computing for Industry, Biomedicine, and Art.*

This work is particularly relevant to the project's focus on contrastive representation learning under Non-IID medical imaging data.

### 2. Federated Learning for Retinal Disease Detection

Gulati, S., Guleria, K., Goyal, N., AlZubi, A. A., & Castilla, Á. K. (2024).
**“A Privacy-Preserving Collaborative Federated Learning Framework for Detecting Retinal Diseases.”**
*IEEE Access.*

This work provides relevant background on privacy-preserving federated medical-image classification and experiments involving heterogeneous/Non-IID data and FedProx.

### 3. Federated Learning for Medical Image Analysis

Guan, H., Yap, P.-T., Bozoki, A., & Liu, M. (2024).
**“Federated Learning for Medical Image Analysis: A Survey.”**
*Pattern Recognition, 151, 110424.*

This survey provides broader background on Federated Learning for medical image analysis and the challenges associated with heterogeneous data.

---

## 👥 Project Team

| S. No. | Team Member                 |
| -----: | --------------------------- |
|      1 | Polineni Lakshmi Pravallika |
|      2 | Rachakonda Jayasree         |
|      3 | Kathari Meghana             |
|      4 | Vellala Harshith Kumar      |

---

## 👨‍🏫 Project Guide

**Mr. Ch. Vijayananda Ratnam**

---

## ⭐ Future Work

Potential future extensions may include:

* More realistic client heterogeneity
* Additional federated learning algorithms
* Personalized federated learning
* Differential privacy
* Secure aggregation
* Larger and more diverse medical datasets
* Real-world multi-institutional validation

---

## 📌 Project Objective

The central objective of this project is to investigate whether **Model-Contrastive Learning can reduce the negative effects of Non-IID data in Federated Skin Disease Classification**, while preserving the fundamental privacy advantage of keeping medical images on local clients.

