import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import flwr as fl
from collections import OrderedDict

# =============================================================================
# 1. Model Definition (Same as local for consistency)
# =============================================================================
class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network for CIFAR-10.
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
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# =============================================================================
# 2. Data Helper Functions
# =============================================================================
def load_data():
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # using download=True might be racy in concurrent clients, but fine for simulation
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return trainset, testset

def load_partition(partition_id, num_clients, batch_size):
    """
    Simulate a data partition for a specific client.
    We split the 50,000 training images into `num_clients` chunks.
    """
    trainset, testset = load_data()
    
    # Simple IID partitioning (sequential split)
    # In a real scenario, data would be non-IID and naturally distinct.
    num_train = len(trainset)
    split_size = num_train // num_clients
    indices = list(range(num_train))
    
    start_idx = partition_id * split_size
    end_idx = start_idx + split_size
    
    partition_indices = indices[start_idx:end_idx]
    
    train_partition = Subset(trainset, partition_indices)
    
    # Clients usually have their own validation set, here we use a subset of the global test set 
    # or just the global test set for simplicity (though purely local eval is better).
    # For this demo, let's give each client a small local validation set derived from the test set or train set.
    # To keep it simple and standard: use a subset of the global test set for local evaluation if needed.
    # However, standard FL often aggregates metrics from 'fit' (training loss) and 'evaluate' (val loss).
    
    trainloader = DataLoader(train_partition, batch_size=batch_size, shuffle=True)
    
    # We pass the full test set for local evaluation in this simple demo, 
    # but strictly speaking, clients should only have their own data.
    # To be more realistic, let's slice the test set too.
    num_test = len(testset)
    test_split_size = num_test // num_clients
    test_start = partition_id * test_split_size
    test_end = test_start + test_split_size
    test_partition = Subset(testset, range(test_start, test_end))
    valloader = DataLoader(test_partition, batch_size=batch_size, shuffle=False)
    
    return trainloader, valloader

# =============================================================================
# 3. Flower Client
# =============================================================================
class CifarClient(fl.client.NumPyClient):
    def __init__(self, partition_id, num_clients):
        self.device = torch.device("cpu")  # CPU-only as requested
        self.net = SimpleCNN().to(self.device)
        self.trainloader, self.valloader = load_partition(partition_id, num_clients, batch_size=32)

    def get_parameters(self, config):
        """Return model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        """Load model weights from a list of NumPy ndarrays."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """
        Train the model on the local dataset.
        This is the core of the FL process on the client side.
        """
        # 1. Update local model with global parameters
        self.set_parameters(parameters)
        
        # 2. Local Training (e.g., 1 epoch per round for speed)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        
        self.net.train()
        running_loss = 0.0
        samples = 0
        
        # We assume 1 epoch of local training per round
        for inputs, labels in self.trainloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            samples += inputs.size(0)
            
        avg_loss = running_loss / samples
        
        # 3. Return updated parameters and metrics
        # The data never leaves the client; only the weight updates (parameters) do.
        return self.get_parameters(config={}), samples, {"train_loss": avg_loss}

    def evaluate(self, parameters, config):
        """
        Evaluate the model on the local dataset.
        """
        self.set_parameters(parameters)
        criterion = nn.CrossEntropyLoss()
        
        self.net.eval()
        loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.valloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.net(inputs)
                loss += criterion(outputs, labels).item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        avg_loss = loss / total
        
        return float(avg_loss), total, {"accuracy": float(accuracy)}
