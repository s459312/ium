import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from train import NetflixDataset, Net
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

test_dataset = NetflixDataset('test_data.csv')
test_loader = DataLoader(test_dataset, batch_size=32)
model = Net()

model.load_state_dict(torch.load('trained_model.pth'))
model.eval()
predictions = []
true_values = []
correct = 0
total = 0
label_encoder = LabelEncoder()
labels = test_dataset.data['type'].tolist()
labels_encoded = label_encoder.fit_transform(labels)
with torch.no_grad():
    for runtimes, labels in test_loader:
        runtimes = runtimes.unsqueeze(1) 
        labels_tensor = torch.tensor(label_encoder.transform(labels)).long()
        outputs = model(runtimes)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.tolist())
        true_values.extend(labels_tensor.tolist())


with open('results.txt', 'w') as f:
    for pred, true in zip(predictions, true_values):
        pred_label = label_encoder.inverse_transform([pred])[0]
        true_label = label_encoder.inverse_transform([true])[0]
        f.write(f"Prediction: {pred_label}, True Value: {true_label}\n")
accuracy = accuracy_score(true_values, predictions)
precision = precision_score(true_values, predictions, average='micro')
recall = recall_score(true_values, predictions, average='micro')
f1 = f1_score(true_values, predictions, average='micro')
# with open('metrics_file.txt', 'a') as file:
#     file.write(f"\nAccuracy: {accuracy}\n")
#     file.write(f"Precision: {precision}\n")
#     file.write(f"Recall: {recall}\n")
#     file.write(f"F1-Score: {f1}\n")

with open('metrics_file.txt', 'a') as file:
    file.write(f"{accuracy}\t{precision}\t{recall}\t{f1}\n")
    
with open('metrics_file.txt', 'r') as file:
    lines = file.readlines()
    
accuracy_values = [float(line.split('\t')[0]) for line in lines]

rounded_accuracy_values = [round(accuracy, 4) for accuracy in accuracy_values]

x_values = list(range(1, len(rounded_accuracy_values) + 1))

plt.plot(x_values, rounded_accuracy_values, marker='o')
plt.xlabel('Build Number')
plt.ylabel('Accuracy')
plt.title('Accuracy Trend')
plt.grid(True)
plt.xticks(x_values)  
plt.savefig('plot.png') 

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
print('Results saved')