from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()])

train_dataset = datasets.ImageFolder("cats_and_dogs/train", transform=transform)
val_dataset   = datasets.ImageFolder("cats_and_dogs/val", transform=transform)
test_dataset  = datasets.ImageFolder("cats_and_dogs/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32)
test_loader  = DataLoader(test_dataset, batch_size=32)
