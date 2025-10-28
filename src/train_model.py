import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.utils.data import DataLoader

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Dataset and dataloader
data_dir = "../data/selected_foods"
train_data = datasets.ImageFolder(data_dir, transform=transform)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

# Load pretrained model
model = models.efficientnet_b0(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

num_classes = len(train_data.classes)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# Training loop
for epoch in range(5):  # small number for quick fine-tune
    running_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {running_loss/len(train_loader):.4f}")

# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'class_to_idx': train_data.class_to_idx
}, "../models/food_classifier.pt")

print("✅ Model trained and saved!")
