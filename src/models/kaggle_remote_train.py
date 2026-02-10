import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import joblib # To save the scaler for local use

# 1. Load Data
df = pd.read_csv('/kaggle/input/aqi-hybrid-knowledge-base/fuzzified_delhi_data.csv')

# Feature selection: Raw Data + Fuzzy States + Rule Activations
features = [c for c in df.columns if any(x in c for x in ['PM2.5', 'Fuzzy', 'Rule'])]
target = 'PM2.5'

# 2. Scaling & Windowing
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features])

def create_sequences(data, window=14):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window, 0]) # Index 0 is raw PM2.5
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y)).view(-1, 1)

X, y = create_sequences(scaled_data)

# 3. Hybrid LSTM Architecture
class HybridLSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 128, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(128, 1)
        
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

# 4. Training Loop
device = torch.device('cuda')
model = HybridLSTM(X.shape[2]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(150):
    model.train()
    outputs = model(X.to(device))
    loss = criterion(outputs, y.to(device))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 25 == 0: print(f"Epoch {epoch+1} Loss: {loss.item():.6f}")

# 5. Save the artifacts
torch.save(model.state_dict(), 'hybrid_lstm_v1.pth')
joblib.dump(scaler, 'scaler.gz') 
print("Artifacts saved successfully!")
