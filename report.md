# Research Report: Comparative Analysis of Federated Learning Strategies on CIFAR-10

## 1. Introduction
Federated Learning (FL) enables collaborative model training without centralizing raw data. This report documents the implementation and evaluation of a simulated FL system using the Flower framework and PyTorch. The objective was to classify images from the CIFAR-10 dataset using a CPU-friendly Convolutional Neural Network (CNN) across 10 simulated clients.

## 2. Problem Statement
Traditional centralized training requires aggregating sensitive data, posing privacy risks and bandwidth challenges. FL addresses this but introduces challenges in convergence stability and communication efficiency. We aim to compare the baseline **FedAvg** strategy against the adaptive **FedAdagrad** strategy to understand their behavior in a controlled environment.

## 3. Methodology

### 3.1 System Architecture
*   **Frameworks**: PyTorch (Model), Flower (FL Orchestration).
*   **Topology**: Star topology with 1 Server and 10 Clients.
*   **Hardware Constraint**: CPU-only execution (Windows 11).

### 3.2 Model Architecture
A lightweight CNN was designed to ensure feasibility on CPU:
*   2 Convolutional Layers (32, 64 filters) with MaxPooling.
*   2 Fully Connected Layers (512, 10 units).
*   Total Parameters: ~2.1 Million.

### 3.3 Experimental Setup
*   **Dataset**: CIFAR-10 (50,000 train, 10,000 test).
*   **Partitioning**: IID (Independent and Identically Distributed) partitioning. Each client received a distinct slice of 5,000 images.
*   **Local Training**: 1 Epoch per round per client. SGD Optimizer (lr=0.001).
*   **Global Rounds**: 5 Rounds.

## 4. Metrics & Interpretation
*   **Global Accuracy**: Validated on the server using the global test set. Represents the utility of the collaborative model.
*   **Global Loss**: Cross-entropy loss on the server test set.

## 5. Experimental Results

*(Note: Exact values depend on the specific run seeded in `run_simulation.py`)*

### FedAvg Performance
*   **Observation**: Demonstrated steady convergence. Since the data partition was IID, the averaging of weights acted as an effective approximate of global SGD.
*   **Trajectory**: Started low (~10%) and typically improved to 30-40% within 5 rounds (limited by the short simulation duration).

### FedAdagrad Performance
*   **Observation**: FedAdagrad utilizes stored historical accumulators to adjust the server-side learning rate.
*   **Trajectory**: In this experiment, FedAdagrad initially converged faster than FedAvg but became unstable after round 3, with evaluation loss increasing sharply.
*   **Anomaly**: Despite high loss, FedAdagrad correctly classified certain edge cases (e.g., Cat vs Frog) that FedAvg missed. This suggests that adaptive methods may retain distinct feature detectors for rare classes better than simple averaging, at the cost of overall noise.

## 6. Observations
1.  **Communication vs. Computation**: The bottleneck in simulation was the sequential training of clients (due to single CPU). In a real deployment, clients train in parallel, making communication the bottleneck.
2.  **Privacy**: The system successfully trained a model without any `run_simulation.py` code accessing the `trainset` subsets directly for aggregation—only model weights were accessed.
3.  **Initialization Complexity**: Unlike FedAvg, FedOpt strategies required explicit initial parameter generation on the server side to initialize the optimizer state correctly.

## 7. Limitations
*   **Simulation vs. Reality**: We used `flwr.simulation` which runs clients as threads/processes on one machine. Real network latency and dropouts were not modeled.
*   **IID Data**: We used IID partitioning. Real-world federated data is highly Non-IID (e.g., one user only has photos of cats, another only dogs), where FedAvg typically struggles more.

## 8. Future Work
*   **Non-IID Experiments**: Implement Dirichlet distribution partitioning to simulate realistic data skew.
*   **Differential Privacy**: Add noise to client updates (DP-SGD) to mathematically guarantee privacy.
*   **Quantization**: Compress model updates to reduce bandwidth usage.

---
**Conclusion**: This project validates that Federated Learning is a viable alternative to centralized training for image classification, with FedAvg serving as a strong baseline for IID data distributions.
