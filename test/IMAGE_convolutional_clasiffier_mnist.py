import os
import sys
from os.path import dirname, join, abspath
sys.path.append(abspath(join(dirname(__file__), '..', 'src')))
######################################################

from mlexperiments.load_data.loader.downloadable.mnist_torch import LoadMnist
from mlexperiments.load_data.loader.downloadable.cifar10_torch import LoadCifar10

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

#dataset_to_use = "cifar10"
dataset_to_use = "mnist"
# dataset_to_use = "fashion"

if dataset_to_use == "cifar10":
    l_cifar10 = LoadCifar10()
    (Xtrain, Ytrain), (Xtest, Ytest) = l_cifar10.get_splited()
    Ytrain = Ytrain.reshape(len(Ytrain))
    Ytest = Ytest.reshape(len(Ytest))

if dataset_to_use == "mnist":
    l_mnist = LoadMnist()
    (Xtrain, Ytrain), (Xtest, Ytest) = l_mnist.get_splited()

n_classes = len(set(Ytrain))
print("number of classes are: ", n_classes)

n_train=len(Xtrain)  # 50000 datos de entrenamiento
n_test=len(Xtest)  # 10000 datos de test
print("n train: ", n_train)
print("n test: ", n_test)

# Modificando X
# ------------
# this shapes are for mnist
if dataset_to_use in ["mnist", "fashion"]:
    Xtrain = Xtrain.reshape(n_train, 1, 28, 28)  # (N, C, H, W) for PyTorch
    Xtest = Xtest.reshape(n_test, 1, 28, 28)
# / -------------

# ------------
# this shapes are for cifar10
if dataset_to_use == "cifar10":
    Xtrain = Xtrain.reshape(n_train, 3, 32, 32)  # (N, C, H, W) for PyTorch
    Xtest = Xtest.reshape(n_test, 3, 32, 32)
# / -------------
Xtrain = Xtrain.astype('float32')
Xtest = Xtest.astype('float32')
Xtrain /= 255
Xtest /= 255


# Convert to PyTorch tensors
Xtrain_t = torch.tensor(Xtrain)
Xtest_t = torch.tensor(Xtest)
Ytrain_t = torch.tensor(Ytrain, dtype=torch.long)
Ytest_t = torch.tensor(Ytest, dtype=torch.long)

train_dataset = TensorDataset(Xtrain_t, Ytrain_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# Define the CNN model
class ConvClassifier(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 12, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 18, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(18, 24, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(24 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


in_channels = 1 if dataset_to_use in ["mnist", "fashion"] else 3

print("creating the model...")
model = ConvClassifier(in_channels, n_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

print("start to train")
model.train()
for epoch in range(10):
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(batch_x)
        _, predicted = torch.max(outputs, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    print(f"Epoch {epoch+1}/10 - loss: {running_loss/total:.3f} - accuracy: {correct/total:.3f}")

model.eval()
with torch.no_grad():
    outputs = model(Xtest_t)
    test_loss = criterion(outputs, Ytest_t).item()
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == Ytest_t).sum().item()
    accuracy = correct / len(Ytest_t)
    print(f"\n Loss: {test_loss:.3f} \t Accuracy: {accuracy:.3f}")

# Save model
os.makedirs("data/created_models", exist_ok=True)
torch.save(model.state_dict(), os.path.join("data", "created_models", "mnist_convolve.pt"))
print("Saved model to disk")