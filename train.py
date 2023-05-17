import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import sys

num_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 10

class NetflixDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.loc[index]
        runtime = sample['runtime']
        label = sample['type']

       
        return (runtime, label)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 64)  
        self.fc2 = nn.Linear(64, 2)  

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x.to(self.fc1.weight.dtype)))
        x = self.fc2(x)
        return x


train_dataset = NetflixDataset('train_data.csv')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
label_encoder = LabelEncoder()
labels = train_dataset.data['type'].tolist()
labels_encoded = label_encoder.fit_transform(labels)



for epoch in range(num_epochs):
    model.train()
    print(f"Epoch: {epoch+1}/{num_epochs}")
    for batch_idx, (runtimes, labels) in enumerate(train_loader):
        runtimes = runtimes.unsqueeze(1)  
        runtimes = runtimes.to(device)
        labels_tensor = torch.tensor(label_encoder.transform(labels)).to(device).long()

        #print(labels)

        optimizer.zero_grad()

        outputs = model(runtimes)
        loss = criterion(outputs, labels_tensor)

        loss.backward()
        optimizer.step()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch: {batch_idx+1}/{len(train_loader)} Loss: {loss.item():.4f}")
    
print("Training complete.")
torch.save(model.state_dict(), 'trained_model.pth')
