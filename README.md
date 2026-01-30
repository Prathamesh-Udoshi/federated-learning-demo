# Federated Learning Demo with CIFAR-10

Welcome to the **Federated Learning Demo**! This project is an educational showcase designed to introduce you to the concepts of Federated Learning (FL) using **PyTorch** and **Flower (Flwr)**. 

Targeted at developers and students, this repository demonstrates how to move from traditional "Centralized" machine learning to a privacy-preserving "Decentralized" approach.

---

## 1. Project Overview

### What is Federated Learning?
Federated Learning is a machine learning technique where a model is trained across multiple decentralized edge devices or servers holding local data samples, **without exchanging their data**. 

### Real-World Use Cases
*   **Healthcare**: Hospitals collaborate to train a cancer detection model without sharing sensitive patient records.
*   **Mobile Keyboards (Gboard/Apple)**: Your phone learns your typing habits locally and sends only anonymous updates to the cloud to improve the global autocorrect model.
*   **Financial Services**: Banks identify fraud patterns collaboratively without exposing transaction details.

---

## 2. Centralized vs. Federated Training

| Feature | Centralized Training (Standard) | Federated Training |
| :--- | :--- | :--- |
| **Data Location** | All data is collected in one server/cloud. | Data remains on the client's device (Local). |
| **Privacy** | Low (Data must be shared). | High (Data never leaves the device). |
| **Computation** | Server handles 100% of the load. | Computation is distributed across clients. |
| **Bandwidth** | High (Upload all raw data). | Low/Medium (Upload only model weights). |

---

## 3. Metrics Explained

In this project, you will see three main metrics:

*   **Training Loss (Local)**: How well a single client's model is learning from its *own* local data. This guides the local optimization (SGD).
*   **Evaluation Loss (Global)**: Calculated on the server using a held-out test set. This measures how "generalizable" the global aggregated model is. Lower is better.
*   **Accuracy**: The percentage of correct predictions. 
    *   *Note*: In FL, it's common for Accuracy to fluctuate in early rounds as the global model tries to find a "middle ground" between diverse client updates.

---

## 4. Project Structure

This repository is organized to simulate a real-world FL environment:

*   `local_training/`: Contains the baseline centralized training script (`local_train_demo.py`). Run this first to understand the "traditional" way.
*   `federated_training/`: The core FL logic.
    *   `client_app.py`: The code running on the "edge" devices (simulated). It loads data and trains locally.
    *   `server_app.py`: Logic for the central server (Strategy, Aggregation).
    *   `run_simulation.py`: A script that launches the server and 10 simulated clients on your machine.
*   `experiments/`: Documentation and results comparing different FL strategies (FedAvg vs. FedAdagrad).
*   `plots/`: Visualizations of the training progress.

---

## 5. How to Run

### Prerequisites
*   Windows 11 (or Linux/macOS)
*   Python 3.8+
*   Internet connection (to download CIFAR-10)

### Installation
1.  Create a virtual environment:
    ```bash
    python -m venv venv
    .\venv\Scripts\Activate
    ```
2.  Install dependencies:
    ```bash
    pip install torch torchvision flwr matplotlib
    ```

### Step 1: Run Centralized Baseline
Train a model normally to see what "good" performance looks like.
```bash
python local_training/local_train_demo.py
```
*Expected Output:* ~50-60% accuracy after 2 epochs.

### Step 2: Run Federated Simulation
Run the FL simulation which tests both FedAvg and FedAdagrad.
```bash
python federated_training/run_simulation.py
```
*This will:*
*   Simulate 10 clients.
*   Run 5 rounds of FedAvg.
*   Run 5 rounds of FedAdagrad.
*   Save results to `experiments/results.json`.
*   Generate `plots/accuracy_vs_round.png`.

---

## 6. Results & Learnings

After running the simulation, check `plots/accuracy_vs_round.png`.

*   **Convergence Speed**: Federated Learning typically converges slower than centralized training because the optimizer takes "steps" based on averaged weights rather than pure gradients, and data is often non-IID.
*   **Stability**: FedAdagrad showed initial fast convergence but became unstable after round 3 in this experiment. This highlights the sensitivity of server-side adaptive optimizers to hyperparameter tuning (learning rate) compared to the robust FedAvg.

### Key Takeaways
1.  **FedOpt Initialization**: FedOpt strategies (like FedAdagrad) require explicit initialization of global parameters to track optimizer state, whereas FedAvg can initialize lazily.
2.  **Privacy-First Design**: We built a system where raw images never left the `client_app.py`.
2.  **Trade-offs**: We sacrificed some simplicity and convergence speed for privacy and distributed capability.
3.  **Scalability**: This architecture can theoretically scale to millions of devices.

---

## 7. Who Should Use This?
This architecture is ideal for scenarios where **data privacy is paramount** or **data transfer is expensive**. 
If you are building for Hospitals, Banks, or Edge IoT devices, Federated Learning is your tool of choice.
