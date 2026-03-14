import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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

epochs = 100
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

# Error Distribution
plt.hist(anomaly_scores, bins=30)

plt.title("Reconstruction Error Distribution")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")

plt.show()

# Anomaly Threshold
threshold = anomaly_scores.mean() + 2 * anomaly_scores.std()
anomalies = anomaly_scores > threshold

print("Threshold:", threshold)
print("Number of anomalies:", anomalies.sum())

# PCA
with torch.no_grad():
    latent = model.encoder(X_tensor).numpy()

pca = PCA(n_components=2)
latent_pca = pca.fit_transform(latent)

plt.figure()

for i in np.unique(y):
    plt.scatter(
        latent_pca[y==i, 0],
        latent_pca[y==i, 1],
        label=f"class {i}"
    )

plt.legend()
plt.title("Latent Space Visualization(PCA)")
plt.show()

# t-SNE
tsne = TSNE(n_components=2, random_state=42)

latent_tsne = tsne.fit_transform(latent)
plt.figure()

for i in np.unique(y):
    plt.scatter(
        latent_tsne[y==i, 0],
        latent_tsne[y==i, 1],
        label=f"class {i}"
    )

plt.legend()
plt.title("Latent Space Visualization(t-SNE)")
plt.show()

# Reconstruction Example
idx = np.random.randint(0, len(X_scaled))

original = X_scaled[idx]
reconstructed_sample = reconstructed[idx].numpy()

plt.plot(original, label="Original")
plt.plot(reconstructed_sample, label="Reconstructed")

plt.legend()
plt.title("Original vs Reconstructed Sample")

plt.show()
