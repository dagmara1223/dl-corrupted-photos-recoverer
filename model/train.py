import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from data_loader import JPGClassificationDataset
from architecture import JPGClassifier

def train():
    EPOCHS = 10
    BATCH_SIZE = 16
    LR = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Trening na: {DEVICE}")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train_ds = JPGClassificationDataset(root_dir="dataset/ludzie", split='train', transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = JPGClassifier().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).view(-1, 1)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_acc = 100 * correct / total
        print(f"Epoka [{epoch+1}/{EPOCHS}] - Loss: {running_loss/len(train_loader):.4f}, Acc: {epoch_acc:.2f}%")

    torch.save(model.state_dict(), "model/jpg_classifier.pth")
    print("Model zapisany jako jpg_classifier.pth")

    dummy_input = torch.randn(1, 3, 256, 256).to(DEVICE)
    torch.onnx.export(model, dummy_input, "model/jpg_classifier.onnx", 
                      input_names=['input'], output_names=['output'])
    print("Model wyeksportowany do jpg_classifier.onnx")

if __name__ == "__main__":
    train()