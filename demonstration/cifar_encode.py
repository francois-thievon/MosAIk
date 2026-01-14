import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

EPOCHS = 5

# --- 1. Chargement de MNIST ---
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

# 2. Définir un CNN simple
class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super(SimpleAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(True)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(256, 2048),
            nn.ReLU(True),
            nn.Unflatten(1, (32, 8, 8)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # [B, 16, 14, 14]
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),  # [B, 1, 28, 28]
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 3. Entraîner le modèle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Fonction pour évaluer la loss sur le jeu de test
def evaluate_test_loss():
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            test_loss += loss.item()
    return test_loss / len(test_loader)

# Entraînement simple
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader)}")
    res = evaluate_test_loss()
    print(f"Epoch [{epoch+1}/{EPOCHS}], Test Loss: {res}")

model.eval()
X_train, y_train = [], []
with torch.no_grad():
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        latents = model.encoder(inputs)
        X_train.append(latents.cpu().numpy())
        y_train.append(labels.cpu().numpy())
X_train = np.concatenate(X_train, axis=0)
y_train = np.concatenate(y_train, axis=0)

X_test, y_test = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        latents = model.encoder(inputs)
        X_test.append(latents.cpu().numpy())
        y_test.append(labels.cpu().numpy())
X_test = np.concatenate(X_test, axis=0)
y_test = np.concatenate(y_test, axis=0)

# Sauvegarde au format numpy
np.save('dataset/cifar/X_train_latent.npy', X_train)
np.save('dataset/cifar/y_train_latent.npy', y_train)
np.save('dataset/cifar/X_test_latent.npy', X_test)
np.save('dataset/cifar/y_test_latent.npy', y_test)