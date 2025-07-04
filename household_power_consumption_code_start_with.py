# %%
import numpy as np
import pandas as pd

# %%
# load data
df = pd.read_csv('C:\\Users\\19647\\OneDrive\\桌面\\学习\\aiSummerCamp2025-master\\aiSummerCamp2025-master\\day3\\assignment\\data\\household_power_consumption\\household_power_consumption.txt', sep = ";")
df.head()

# %%
# check the data
df.info()

# %%
df['datetime'] = pd.to_datetime(df['Date'] + " " + df['Time'],format='%d/%m/%Y %H:%M:%S')
df.drop(['Date', 'Time'], axis = 1, inplace = True)
# handle missing values
df.dropna(inplace = True)

# %%
print("Start Date: ", df['datetime'].min())
print("End Date: ", df['datetime'].max())

# %%
# split training and test sets
# the prediction and test collections are separated over time
train, test = df.loc[df['datetime'] <= '2009-12-31'], df.loc[df['datetime'] > '2009-12-31']

# %%
# data normalization
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

scaler = MinMaxScaler()
feature_cols = [col for col in train.columns if col not in ['datetime', 'Global_active_power']]
train_features = scaler.fit_transform(train[feature_cols])
test_features = scaler.transform(test[feature_cols])
train_target = scaler.fit_transform(train[['Global_active_power']])
test_target = scaler.transform(test[['Global_active_power']])

# split X and y
def create_sequences(features, target, seq_length=24):
    xs, ys = [], []
    for i in range(len(features) - seq_length):
        xs.append(features[i:i+seq_length])
        ys.append(target[i+seq_length])
    return np.array(xs), np.array(ys)

seq_length = 24  # 1 day if hourly data
X_train, y_train = create_sequences(train_features, train_target, seq_length)
X_test, y_test = create_sequences(test_features, test_target, seq_length)

# create dataloaders
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 64
train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# build a LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

input_size = X_train.shape[2]
hidden_size = 64
num_layers = 2
output_size = 1
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 10
for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# evaluate the model on the test set
model.eval()
preds, trues = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        output = model(X_batch)
        preds.append(output.cpu().numpy())
        trues.append(y_batch.numpy())
preds = np.concatenate(preds).flatten()
trues = np.concatenate(trues).flatten()

# plotting the predictions against the ground truth
plt.figure(figsize=(12,6))
plt.plot(trues, label='True')
plt.plot(preds, label='Predicted')
plt.legend()
plt.title('LSTM Prediction vs Ground Truth')
plt.xlabel('Time Step')
plt.ylabel('Normalized Global Active Power')
plt.show()
