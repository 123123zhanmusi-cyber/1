import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
from model_lstm import ECGLSTM

# ========================
# GPU设置
# ========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# ========================
# 读取数据
# ========================

X = np.load("X.npy")
y = np.load("y.npy")

X = X.reshape(-1,300,1)

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# ========================
# Dataset
# ========================

class ECGDataset(Dataset):

    def __init__(self,X,y):

        self.X = torch.tensor(X,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):
        return self.X[idx],self.y[idx]

train_loader = DataLoader(
    ECGDataset(X_train,y_train),
    batch_size=64,
    shuffle=True
)

# ========================
# 模型
# ========================

model = ECGLSTM().to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 15

# ========================
# 训练
# ========================

for epoch in range(epochs):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for x,y in train_loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        output = model(x)

        loss = criterion(output,y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(output,1)

        correct += (preds == y).sum().item()

        total += y.size(0)

    acc = correct / total

    print(f"Epoch {epoch+1}/{epochs}")
    print("Loss:", total_loss)
    print("Accuracy:", acc)

# ========================
# 保存模型
# ========================

torch.save(model.state_dict(),"ecg_lstm_model.pth")

print("模型训练完成")