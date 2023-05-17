import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('titles.csv')

data = data[['type', 'runtime']]

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
test_data, val_data = train_test_split(test_data, test_size=0.5, random_state=42)

train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)
val_data.to_csv('val_data.csv', index=False)