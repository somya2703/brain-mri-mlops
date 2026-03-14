import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch

from model import BrainCNN
from dataset import get_dataloaders

mlflow.set_experiment("brain_mri_classifier")

def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BrainCNN().to(device)

    loader = get_dataloaders("data/raw")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    with mlflow.start_run():

        mlflow.log_param("lr", 0.001)
        mlflow.log_param("epochs", 5)

        for epoch in range(5):

            total_loss = 0

            for images, labels in loader:

                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)

                loss = criterion(outputs, labels)

                loss.backward()

                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch} Loss {total_loss}")

            mlflow.log_metric("loss", total_loss)

        mlflow.pytorch.log_model(model, "model")

        torch.save(model.state_dict(), "models/model.pth")

if __name__ == "__main__":
    train()