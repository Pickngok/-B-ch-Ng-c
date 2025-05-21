import torch
from simulate_data import generate_hmm_sequence
from model import InformerGaussian

def create_windows(data, window_size=10):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return torch.tensor(X).unsqueeze(-1), torch.tensor(y)

def gaussian_nll(mu, log_sigma, target):
    sigma = torch.exp(log_sigma)
    return torch.mean(0.5 * torch.log(2 * torch.pi * sigma**2) + ((target - mu)**2) / (2 * sigma**2))

states, observations = generate_hmm_sequence()
observations = observations.astype(float)
X, y = create_windows(observations, 10)

model = InformerGaussian()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(20):
    total_loss = 0.0
    for i in range(0, len(X), 8):
        xb, yb = X[i:i+8].float(), y[i:i+8]
        mu, log_sigma = model(xb)
        loss = gaussian_nll(mu, log_sigma, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")