import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# =============================================================================
# 1. Configuration & Setup
# =============================================================================
# Set device to CPU as requested (can be changed to 'cuda' if available)
DEVICE = torch.device("cpu")
BATCH_SIZE = 32
EPOCHS = 2  # Small number of epochs for CPU-friendly demo
LEARNING_RATE = 0.001

print(f"Using device: {DEVICE}")

# =============================================================================
# 2. Data Preparation (Centralized)
# =============================================================================
# In centralized training, we have access to the entire dataset on one machine.

def load_data():
    """
    Load CIFAR-10 dataset with basic normalization.
    """
    print("Loading CIFAR-10 data...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Download train and test sets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    return trainloader, testloader

# =============================================================================
# 3. Model Definition
# =============================================================================
class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network for CIFAR-10.
    Structure:
    - Conv Layer 1 (3 -> 32 channels) + ReLU + MaxPool
    - Conv Layer 2 (32 -> 64 channels) + ReLU + MaxPool
    - Fully Connected Layer 1 (Flat features -> 512) + ReLU
    - Output Layer (512 -> 10 classes)
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# =============================================================================
# 4. Training & Evaluation Functions
# =============================================================================
def train(net, trainloader, epochs):
    """Train the network on the training set."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    print(f"Starting training for {epochs} epochs...")
    net.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] Train Loss: {running_loss / 100:.3f}")
                running_loss = 0.0
    print("Training finished.")

def evaluate(net, testloader):
    """Evaluate the network on the test set."""
    criterion = nn.CrossEntropyLoss()
    net.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = val_loss / len(testloader)
    
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.2f}%")
    
    # EXPLANATION OF METRICS:
    # Validation Loss: Measures how 'wrong' the model's predictions are on new data. Lower is better.
    # Accuracy: Percentage of correctly classified images. Higher is better.
    # In centralized training, these metrics represent the model's performance on a held-out test set
    # after learning from the complete training dataset.

# =============================================================================
# 5. Main Execution
# =============================================================================
if __name__ == "__main__":
    # 1. Load Data
    train_loader, test_loader = load_data()

    # 2. Instantiate Model
    net = SimpleCNN().to(DEVICE)

    # 3. Train
    # Train Loss: The error on the data the model is currently learning from.
    train(net, train_loader, EPOCHS)

    # 4. Evaluate
    # Validation Loss/Accuracy: Performance on unseen data.
    evaluate(net, test_loader)

    # 5. Save Model
    save_path = "local_model.pt"
    torch.save(net.state_dict(), save_path)
    print(f"Model saved to {save_path}")
