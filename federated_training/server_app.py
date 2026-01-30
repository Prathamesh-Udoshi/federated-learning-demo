import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional

# Import model from client_app to avoid duplication in file, 
# although in a real distributed system server might not have the model code 
# if it's generic, but usually it needs to know the architecture for initial parameters.
try:
    from client_app import SimpleCNN, load_data
except ImportError:
    # If running from a different directory (e.g. root), adjust import
    try:
        from federated_training.client_app import SimpleCNN, load_data
    except ImportError:
        # Fallback if imports fail (should not happen if generic paths correct)
        pass

DEVICE = torch.device("cpu")

def get_evaluate_fn(testloader):
    """
    Return an evaluation function for server-side evaluation.
    This runs after every round to check the global model's performance on held-out data.
    """
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        
        # Instantiate model
        net = SimpleCNN().to(DEVICE)
        
        # Load parameters
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        
        # Evaluate
        criterion = nn.CrossEntropyLoss()
        net.eval()
        loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = net(inputs)
                loss += criterion(outputs, labels).item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        avg_loss = loss / total
        
        # Print metrics for the user to see progress
        print(f"--- Round {server_round} eval: Loss {avg_loss:.4f}, Accuracy {accuracy:.4f}")
        
        return avg_loss, {"accuracy": accuracy}

    return evaluate

def get_strategy(strategy_name: str, testloader: DataLoader):
    """
    Factory function to get the Flower strategy.
    
    Args:
        strategy_name: 'fedavg' or 'fedadagrad'
        testloader: DataLoader for server-side evaluation
    """
    model = SimpleCNN()
    params = fl.common.ndarrays_to_parameters(
        [val.cpu().numpy() for _, val in model.state_dict().items()]
    )
    # Common settings
    # fraction_fit=1.0 means sample 100% of available clients for training (simulating small cohort)
    # min_fit_clients=2 means at least 2 clients participate per round
    
    evaluate_fn = get_evaluate_fn(testloader)
    
    if strategy_name.lower() == "fedavg":
        # FedAvg: The baseline aggregation strategy. Average weights weighted by number of samples.
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,  
            fraction_evaluate=1.0,
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
            evaluate_fn=evaluate_fn,  # Server-side evaluation
            initial_parameters=None,
        )
    elif strategy_name.lower() == "fedadagrad":
        # FedAdagrad: An adaptive strategy useful when clients have heterogeneous data.
        # It adapts the server-side learning rate based on updates history.
        
        # We need initial parameters for FedOpt strategies usually, 
        # but Flower can handle it if we pass them to start_simulation. 
        # Here we configure the strategy object.
        strategy = fl.server.strategy.FedAdagrad(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
            evaluate_fn=evaluate_fn,
            initial_parameters=params, # Will be set in simulation
            eta=0.1, # Server-side learning rate
            eta_l=0.1, # Client-side learning rate (implicitly handled by client optimizer)
            tau=0.01,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
        
    return strategy
