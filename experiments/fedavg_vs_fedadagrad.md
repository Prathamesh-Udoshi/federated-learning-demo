# FedAvg vs FedAdagrad: An Experimental Comparison

## Overview
In this experiment, we compare two popular aggregation strategies for Federated Learning: **FedAvg** and **FedAdagrad**.

### 1. FedAvg (Federated Averaging)
**FedAvg** is the canonical algorithm for Federated Learning (proposed by McMahan et al., 2017). 
*   **Mechanism**: The server distributes the global model to selected clients. Clients train locally (SGD) and send back their weight updates. The server simply averages these updates (weighted by the number of samples each client possesses) to update the global model.
*   **Pros**: Simple, robust, low communication overhead per round (relative to sending gradients).
*   **Cons**: Can struggle with non-IID data (heterogeneous data distributions) or when client updates have very different magnitudes.

### 2. FedAdagrad
**FedAdagrad** is an adaptive optimization algorithm applied at the server level (part of the **FedOpt** family).
*   **Mechanism**: Instead of just averaging the updates, the server treats the aggregated update as a "pseudo-gradient". It uses the Adagrad optimizer logic to update the global model.
*   **Key Feature**: It adapts the server-side learning rate for each parameter based on potential historical updates. Frequent updates to a parameter reduce its learning rate, while infrequent updates increase it.
*   **Use Case**: Particularly effective when data is sparse or heterogeneous, offering better convergence stability in complex scenarios.

## Metrics
We track two key metrics over 5 rounds of simulation:
1.  **Centralized Validation Loss**: The cross-entropy loss of the global model on the server's validation set.
2.  **Centralized Accuracy**: The percentage of correctly classified images on the server's validation set.

## Observed Results
*   **FedAvg**: Demonstrated steady, stable convergence throughout the training rounds.
*   **FedAdagrad**: Although FedAdagrad initially converged faster than FedAvg, it became unstable after round 3, with evaluation loss increasing sharply. This behavior is attributed to accumulated noisy gradients at the server and sensitivity to learning rate. In contrast, FedAvg demonstrated more stable convergence.

### Anomalous "Correctness" in Instability
An interesting phenomenon was observed during manual testing: despite FedAdagrad having higher loss and instability, it correctly classified a difficult test image (a cat) that both the Centralized Model and FedAvg misclassified as a Frog.
*   **Hypothesis**: The adaptive boosting of rare features (e.g., pointed ears) in FedAdagrad may have preserved specific signal detectors that were "averaged out" in FedAvg, even while the overall model decision boundary became noisy. This highlights the "Loss vs. Accuracy" disconnect—a high-loss model can still be right on specific instances due to high uncertainty/entropy spreading.

### Implementation Insight
*   **Initialization**: FedOpt strategies require explicit initialization of global model parameters, unlike FedAvg, which can lazily initialize from a client. This reflects the need for optimizer state tracking at the server side.

## Conclusion
(To be filled after running `run_simulation.py`)
Typically, adaptive methods like FedAdagrad outperform vanilla FedAvg in large-scale, heterogeneous settings, though for simple IID CIFAR-10 tasks, FedAvg is often sufficient and extremely competitive.
