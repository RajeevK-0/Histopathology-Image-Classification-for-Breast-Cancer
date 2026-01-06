import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import get_model
from sklearn.metrics import accuracy_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data"
EPOCHS = 10
LR = 1e-4

def train():
    train_loader, val_loader = get_dataloaders(DATA_DIR)
    model = get_model(num_classes=2).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        preds, true = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                preds.extend(predicted.cpu().numpy())
                true.extend(labels.cpu().numpy())

        acc = accuracy_score(true, preds)

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Loss: {train_loss/len(train_loader):.4f} "
              f"Val Acc: {acc:.4f}")

    torch.save(model.state_dict(), "breast_model.pth")
    print("Model saved!")

if __name__ == "__main__":
    train()
