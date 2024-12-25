import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        # x = self.softmax(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(np.array(X), dtype=torch.float32)
        self.y = torch.tensor(np.argmax(y, axis=1), dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, X_train, y_train, epochs=3, batch_size=16, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_dataset = CustomDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            outputs = outputs.squeeze(1)
            #print(f"Shape of outputs: {outputs.shape}")  # Должно быть [batch_size, num_classes]
            #print(f"Shape of y_batch: {y_batch.shape}")  # Долж
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        print(f"Эпоха {epoch + 1}, Потери: {loss.item()}")

def evaluate_model(model, X_test, y_test):
    test_dataset = CustomDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            labels = y_batch.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    # print(f"all_labels: {all_labels[:5]}, shape: {np.array(all_labels).shape}")
    # print(f"all_preds: {all_preds[:5]}, shape: {np.array(all_preds).shape}")

    if len(np.array(all_labels).shape) > 1:
        all_labels = np.argmax(all_labels, axis=1)

    if len(np.array(all_preds).shape) > 1:
        all_preds = np.argmax(all_preds, axis=1)

    f1 = f1_score(all_labels, all_preds, average='weighted')
    return f1
