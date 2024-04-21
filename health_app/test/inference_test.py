import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
import intel_extension_for_pytorch as ipex

# Constants
MODEL_PATH = "/home/health_app/checkpoints/checkpoint_epoch_3.pth"
TEST_DATA_PATH = "/home/health_app/test/Testing"
BATCH_SIZE = 4

# Transformations
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the model
device = "cpu"
print(f"DEVICE USED: {device}")
checkpoint = torch.load(MODEL_PATH, map_location=device)
model = torchvision.models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 4)  # Adjust for 4 classes
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()
#model = ipex.optimize(model, dtype=torch.float32)
model = model.to(dtype=torch.bfloat16)
# Dataset and DataLoader
test_dataset = datasets.ImageFolder(root=TEST_DATA_PATH, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Function to perform inference and calculate accuracy
def evaluate_model(model, data_loader):
    correct = 0
    total = 0
    with torch.cpu.amp.autocast():
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def infer(image_path):
    # Load and transform the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class_idx = predicted.item()

    # Map predicted index to class name
    class_names = test_dataset.classes  # Assuming test_dataset is globally available
    predicted_class = class_names[predicted_class_idx]

    return predicted_class


# Evaluate the model
if __name__ == "__main__":
    accuracy = evaluate_model(model, test_loader)
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')
