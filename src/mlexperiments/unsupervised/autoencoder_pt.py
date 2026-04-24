from os.path import join, exists
from os import mkdir

import numpy as np
import torch
import torch.nn as nn


class _MLPAutoencoder(nn.Module):
    def __init__(self, n_input: int, n_hidden: int):
        super().__init__()
        self.encoder = nn.Linear(n_input, n_hidden)
        self.decoder = nn.Linear(n_hidden, n_input)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.sigmoid(self.encoder(x))
        out = self.sigmoid(self.decoder(h))
        return out, h


def MLP_reconstruct_data(X_train, X_test, n_hidden=2, epochs=10, name="reconstructed_model", batch_size=128, lr=1e-3):
    if len(X_train) < 1:
        raise ValueError("X has no length")
    n_input = len(X_train[0])

    if not exists(join("data/created_models", name)):
        mkdir(join("data/created_models", name))

    X_train_t = torch.tensor(np.array(X_train, dtype=np.float32))
    X_test_t = torch.tensor(np.array(X_test, dtype=np.float32))

    # Normalize to [0, 1] for sigmoid output
    x_min = X_train_t.min()
    x_max = X_train_t.max()
    if x_max > 1.0 or x_min < 0.0:
        X_train_t = (X_train_t - x_min) / (x_max - x_min + 1e-8)
        X_test_t = (X_test_t - x_min) / (x_max - x_min + 1e-8)

    model = _MLPAutoencoder(n_input, n_hidden)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(X_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Validation split: use last 20% of training data
    val_size = int(0.2 * len(X_train_t))
    train_size = len(X_train_t) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    model.train()
    for epoch in range(epochs):
        train_loss = 0.0
        for (batch_x,) in train_loader:
            optimizer.zero_grad()
            reconstructed, _ = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_x)
        train_loss /= train_size

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch_x,) in val_loader:
                reconstructed, _ = model(batch_x)
                loss = criterion(reconstructed, batch_x)
                val_loss += loss.item() * len(batch_x)
        val_loss /= val_size

        print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

    model.eval()
    with torch.no_grad():
        reconstructed, codings_val = model(X_test_t)

    return codings_val.numpy(), reconstructed.numpy()
