from torchvision import models
import torch.nn as nn
from loadingData import *
import torch
model = models.vgg16(num_classes=2)
for param in model.features.parameters():
    param.requires_grad = False
model.classifier[6] = nn.Linear(4096, 2)  
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(10): 
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")
def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100*correct/total

val_accuracy = evaluate(model, val_loader)
print(f"val Accuracy: {val_accuracy:.2f}%")
test_accuracy = evaluate(model, test_loader)
print(f"test Accuracy: {test_accuracy:.2f}%")
torch.save(model.state_dict(), "vgg_cats_dogs.pth")