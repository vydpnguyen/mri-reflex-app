import os
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch import nn
import intel_extension_for_pytorch as ipex

from torch.utils.data import random_split

from torch.optim.lr_scheduler import ReduceLROnPlateau



# Configuration variables
LR = 2e-5
DATA = 'Training'
NUM_EPOCHS = 8
EARLY_STOPPING_PATIENCE = 3
DEBUG_MODE = False  # Set to True to enable debug mode
CHECKPOINT_INTERVAL=1

# Configuration for periodic checkpointing
CHECKPOINT_DIR = '/home/health_app/checkpoints'
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Assuming DATA contains all training data
# Dataset and DataLoader setup
train_dataset = datasets.ImageFolder(root=DATA, transform=transform)
total_size = len(train_dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)


# Model setup
model = torchvision.models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)  # Adjust for 4 classes
model.to("cpu")

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
#scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

model, optimizer = ipex.optimize(model=model, optimizer=optimizer, dtype=torch.bfloat16)  # intel optimizations

def validate_model(model, data_loader):
    model.eval()
    total = 0
    correct = 0
    with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to("cpu"), target.to("cpu")
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        accuracy = 100 * correct / total
    model.train()
    return accuracy

def calculate_validation_loss(model, data_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_samples = 0
    with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
        with torch.no_grad():  # Turn off gradients for validation, saves memory and computations
            for data, target in data_loader:
                data, target = data.to("cpu"), target.to("cpu")  # Move data to the appropriate device
                outputs = model(data)
                loss = criterion(outputs, target)
                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)
    
    model.train()  # Set the model back to training mode
    avg_loss = total_loss / total_samples
    return avg_loss

# Training loop with metrics and early stopping
min_loss = float('inf')
no_improve_epoch = 0

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to("cpu"), target.to("cpu")
        with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        epoch_loss += loss.item()
        print(f"Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}")
    
    avg_epoch_loss = epoch_loss / len(train_loader)
    val_accuracy = validate_model(model, val_loader)
    val_loss = calculate_validation_loss(model, val_loader, criterion)
    print(f"Epoch {epoch}: Train Loss: {avg_epoch_loss}, Valid Loss: {val_loss}, Validation Accuracy: {val_accuracy}%")
    
    if val_loss < min_loss:
        min_loss = val_loss
        no_improve_epoch = 0
    else:
        no_improve_epoch += 1
        if no_improve_epoch >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered due to no improvement")
            break
    
    scheduler.step(val_loss)
    if epoch % CHECKPOINT_INTERVAL == 0:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
print("Training finished.")
