
# Towards Real-Time Machine Learning Approximations of AC Optimal Power Flow
[![DOI](https://zenodo.org/badge/964373548.svg)](https://doi.org/10.5281/zenodo.15193155)

This repository provides open-source code and experiments accompanying the paper:

> **"Towards Real-Time Machine Learning Approximations of AC Optimal Power Flow"**  
> *IEEE Conference on Artificial Intelligence (CAI), 2025*

## 📄 Overview

This project explores how machine learning can approximate solutions to the **Alternating Current Optimal Power Flow (ACOPF)** problem — a fundamental but computationally intensive task in power system operation. The work compares **Neural Networks (NNs)**, **Gaussian Mixture Models (GMMs)**, and **Linear Regression (LR)** across IEEE standard test cases.

The goal is to provide insights into the feasibility, accuracy, and runtime of data-driven OPF surrogates for **near real-time applications**, especially under varying grid sizes and complexity.

## 🚀 Features

- ✅ ACOPF solved using [pandapower](https://www.pandapower.org/) on multiple IEEE benchmark networks
- ✅ Clean modular codebase with support for:
  - Neural Network model training and tuning
  - GMM clustering with error estimation
  - Baseline Linear Regression
- ✅ Hyperparameter grid search with early stopping
- ✅ Evaluation across five test systems (30-bus to GB-reduced)
- ✅ Extensible framework for future researchers

## 📘 Paper Summary

**Main Contributions:**

- A reproducible pipeline to extract input-output mappings from ACOPF solutions:  
(P_d, Q_d) → (P_g, V_g)

- Comparative analysis of NN, GMM, and LR across 5 power networks:
  - 🏆 NNs perform best in large/highly nonlinear grids
  - 📈 LR is surprisingly strong in small systems
  - ⚠️ GMMs struggle with high-dimensional data

- MSE benchmarking on scaled data  
- Discussion of **feasibility, generalization**, and **real-time deployment risks**

📄 Full paper: (link to proceedings will be added, now use this [DOI](https://doi.org/10.5281/zenodo.15193155))

## 🧪 Example Test Systems

| Case              | Buses | Generators | Description                        |
|------------------|-------|------------|------------------------------------|
| `case30`         | 30    | 6          | Classic IEEE 30-bus system         |
| `case_ieee30`    | 30    | 6          | Variant with minor topology tweaks|
| `case39`         | 39    | 10         | New England benchmark              |
| `case118`        | 118   | 54         | Large IEEE network                 |
| `GBreducednetwork` | ~150 | Many       | UK transmission-level abstraction  |

## 🧠 Model Architecture

The neural network model learns a mapping from load demands to optimal generator setpoints using a feedforward structure.

The core approximation learned is:

(P_d, Q_d) → (P_g, V_g)

Where:
- \( P_d, Q_d \) are the active and reactive power demands at load buses (inputs)
- \( P_g, V_g \) are the generator real power outputs and voltage magnitudes at generator buses (outputs)

Each architecture is tuned using grid search on:

- **Learning rate**
- **Batch size**
- **Hidden layer width/depth**
- **Dropout rate**
- **Patience for early stopping**

---

## 📌 Known Limitations

- Only a single ACOPF sample is used per case → no time-series/load diversity  
- No physics-aware features beyond load inputs  
- NN feasibility violations possible without post-processing  
- GMM scalability limited in high-dimensional settings  

---

## 🧩 Citation

If you use this codebase or find it helpful in your work, please cite it using the `CITATION.cff` file located in the root of this repository.

To automatically generate a citation, you can use GitHub's "Cite this repository" feature or reference the metadata in [`CITATION.cff`](./CITATION.cff).


## 🤝 Acknowledgments

This work was supported by the grant **SGS24/093/OHK5/2T/13** and the **CTU Distinguished co-supervisor grant**.

We gratefully acknowledge the contributions of the open-source community, whose tools made this research possible:

- [pandapower](https://www.pandapower.org/) — for flexible and user-friendly power system modeling
- [PyTorch](https://pytorch.org/) — for building and training machine learning models
- [scikit-learn](https://scikit-learn.org/) — for baseline models and preprocessing utilities

We also thank the maintainers of the IEEE benchmark networks for providing standardized test cases used in this study.




