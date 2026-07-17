# Program Equivalence & Finding Counterexamples via MLP

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-ee4c2c?logo=pytorch)
![Z3](https://img.shields.io/badge/Z3_Solver-Formal_Verification-lightgrey)
![Academic](https://img.shields.io/badge/Research-TU%2Fe-red)

## Overview
This repository explores the application of Machine Learning, specifically a Multi-Layer Perceptron (MLP), to identify counterexamples in zero-equivalent straight-line programs. Determining whether a program consistently evaluates to zero is a fundamental problem in software verification. Traditional formal tools like SAT/SMT solvers guarantee absolute mathematical correctness but inevitably suffer from exponential time complexity due to the state space explosion problem. 

To address this, this project implements an MLP-based heuristic model that processes symbolic, tokenized program representations to directly predict the input variable assignments (counterexamples) required to disprove zero-equivalence.

## Key Results
* **Speed & Scalability:** The Machine Learning approach executes in approximately 0.40 ms, making it over 6 times faster than the exhaustive baseline search. It operates at a constant $\mathcal{O}(1)$ inference time.
* **Success Rate:** The model achieves a 52.10% functional verification success rate on completely unseen code.
* **Dynamic Confidence Routing:** The network's internal confidence profiles provide a statistically rigorous indicator of empirical verification success. By enforcing a >90% confidence threshold, functional verification accuracy surges to 74.45%. This allows the MLP to act as a highly reliable "first-pass" filter before offloading ambiguous cases to SMT solvers.
* **Error Analysis:** In cases of functional failure, the model still achieves an average of 8.46 out of 10 correct bits, with 57.60% of failed programs missing the valid counterexample vector by just a single bit.

## Model Architecture
The architecture is built using PyTorch and is designed to process tokenized straight-line programs globally:
1. **Embedding Layer:** Maps discrete tokens into 16-dimensional continuous vectors.
2. **Hidden Layers:** A fully connected layer mapping 640 inputs to 128 neurons, followed by a second layer mapping 128 inputs to 64 neurons.
3. **Activation & Regularization:** Utilizes ReLU activation functions and Dropout layers ($p=0.3$) to combat overfitting.
4. **Output Layer:** Maps the 64 features to a 10-dimensional output space, representing the independent predicted states of the 10 input variables.

## Repository Structure

* **`dataset_loader.py`**: Handles the tokenization of symbolic program code into numerical tensors for the neural network.
* **`model.py`**: Contains the PyTorch implementation of the `CounterexampleMLP` architecture
* **`train.py`**: The main training loop utilizing the Adam optimizer and Binary Cross-Entropy with Logits Loss, featuring early stopping based on validation loss.
* **`synthetic_data_gen.py`**: Generates unique straight-line arithmetic programs and leverages the Z3 Theorem Prover to systematically find ground-truth counterexamples.
* **`evaluate.py`**: Measures the model's success rate and execution time against a random baseline and systematic search.
* **`statistical_analysis.py`**: Computes Pearson and Spearman correlation coefficients to evaluate the relationship between the network's predictive confidence and its empirical accuracy.
* **`error_analysis.py`**: Conducts a bit-by-bit divergence analysis on failed instances to understand the network's internal ambiguity.
* **`plot_loss.py`**: Utility script to visualize the training and validation loss curves (Divergence Analysis).

## Dataset Construction
Due to the absence of standard public benchmarks, a custom parameterized synthetic data generator was developed. The script generates exactly 10,000 strictly unique straight-line programs, ensuring no data leakage occurs between the training (80%), validation (10%), and testing (10%) splits. Arithmetic edge cases, such as division by zero and integer overflow, are explicitly constrained during generation.
