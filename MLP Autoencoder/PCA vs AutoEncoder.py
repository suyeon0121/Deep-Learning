import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

wine = load_wine()
X = wine.data
y = wine.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
dataset = TensorDataset(X_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim)
        )
        
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

model = AutoEncoder(input_dim=X.shape[1])

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 500
for epoch in range(epochs):
    total_loss = 0
    for (batch,) in loader:
        optimizer.zero_grad()
        recon = model(batch)
        loss = criterion(recon, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")
    
with torch.no_grad():
    reconstructed = model(X_tensor)
    reconstructed_error = torch.mean((X_tensor - reconstructed) ** 2, dim=1)

anomaly_scores = reconstructed_error.numpy()

pca = PCA(n_components=3)

X_pca = pca.fit_transform(X_scaled)

X_pca_recon = pca.inverse_transform(X_pca)

pca_error = np.mean((X_scaled - X_pca_recon) ** 2, axis=1)

print("PCA Reconstruction Error Mean:", pca_error.mean())
print("AutoEncoder Reconstruction Error Mean:", anomaly_scores.mean())

plt.hist(pca_error, bins=30, alpha=0.6, label="PCA")
plt.hist(anomaly_scores, bins=30, alpha=0.6, label="AutoEncoder")

plt.legend()
plt.title("Reconstruction Error Comparison")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.show()

idx = np.random.randint(0, len(X_scaled))

original = X_scaled[idx]
pca_recon_sample = X_pca_recon[idx]
ae_recon_sample = reconstructed[idx].numpy()

plt.plot(original, label="Original")
plt.plot(pca_recon_sample, label="PCA Reconstruction")
plt.plot(ae_recon_sample, label="AutoEncoder Reconstruction")
plt.legend()
plt.title("Reconstruction Comparison")
plt.show()