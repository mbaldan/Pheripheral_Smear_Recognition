import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
import os
import json

def create_resnet_model(num_classes):
    """Creates a ResNet50 model with custom classification layers."""
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False  # Freeze base layers

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model

def train_resnet_model(train_dir, val_dir, batch_size, epochs, device):
    """Trains the ResNet50 model."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(train_dataset.classes)
    model = create_resnet_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.0001) # Optimize only the fully connected layer

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        val_loss = 0.0
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}, Val Acc: {100*correct/total}%")

    torch.save(model.state_dict(), 'blood_cell_resnet_torch.pth')
    print("ResNet model saved as blood_cell_resnet_torch.pth")

    class_indices = train_dataset.class_to_idx
    with open('class_indices_torch.json', 'w') as f:
        json.dump(class_indices, f)
    print(f"Class Indices saved to class_indices_torch.json: {class_indices}")

if __name__ == "__main__":
    train_dir = 'path/to/your/train_data'
    val_dir = 'path/to/your/val_data'
    batch_size = 32
    epochs = 15
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_resnet_model(train_dir, val_dir, batch_size, epochs, device)