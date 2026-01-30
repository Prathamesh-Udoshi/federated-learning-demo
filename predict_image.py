import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import sys
import os

# Define the Model Architecture (Must match training)
class SimpleCNN(nn.Module):
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

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def load_image(image_path):
    """Load and preprocess image."""
    transform = transforms.Compose([
        transforms.Resize(32),        # Resize shortest side to 32
        transforms.CenterCrop(32),    # Crop center 32x32 (avoids stretching)
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)  # Add batch dimension
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def predict(model, image, device):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence = probs[0][predicted].item()
        return CLASSES[predicted.item()], confidence

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_image.py <path_to_image>")
        return

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    device = torch.device("cpu")
    
    # 1. Load Local Model
    print("\n--- Local Centralized Model ---")
    local_path = "local_model.pt"
    if os.path.exists(local_path):
        local_net = SimpleCNN().to(device)
        local_net.load_state_dict(torch.load(local_path, map_location=device))
        
        img_tensor = load_image(image_path)
        if img_tensor is not None:
            cls, conf = predict(local_net, img_tensor, device)
            print(f"Prediction: {cls} ({conf*100:.2f}%)")
    else:
        print(f"Model not found at {local_path}. Please run local_training/local_train_demo.py first.")

    # 2. Load Federated Model (FedAvg)
    print("\n--- Federated Model (FedAvg) ---")
    fl_path = "federated_model_FedAvg.pt"
    if os.path.exists(fl_path):
        fl_net = SimpleCNN().to(device)
        fl_net.load_state_dict(torch.load(fl_path, map_location=device))
        
        img_tensor = load_image(image_path)
        if img_tensor is not None:
            cls, conf = predict(fl_net, img_tensor, device)
            print(f"Prediction: {cls} ({conf*100:.2f}%)")
    else:
        print(f"Model not found at {fl_path}. Please run federated_training/run_simulation.py first.")

    # 3. Load Federated Model (FedAdagrad) - Optional check
    fl_ada_path = "federated_model_FedAdagrad.pt"
    if os.path.exists(fl_ada_path):
         print("\n--- Federated Model (FedAdagrad) ---")
         fl_net_ada = SimpleCNN().to(device)
         fl_net_ada.load_state_dict(torch.load(fl_ada_path, map_location=device))
         img_tensor = load_image(image_path)
         if img_tensor is not None:
            cls, conf = predict(fl_net_ada, img_tensor, device)
            print(f"Prediction: {cls} ({conf*100:.2f}%)")

if __name__ == "__main__":
    main()
