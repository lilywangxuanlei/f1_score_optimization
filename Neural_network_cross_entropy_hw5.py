#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IPython.display import display, Javascript
display(Javascript('IPython.notebook.kernel.restart()'))


# In[1]:


import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score


# In[2]:


scaler = StandardScaler()
joblib.dump(scaler, "scaler.pkl")


# In[3]:


#Same as f-1 method
class ResidualNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ResidualNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        residual = out
        out = self.fc2(out)
        out += residual 
        out = self.relu(out)
        out = self.fc3(out)
        return out


# In[4]:


#Same as f-1 method
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import torch
import joblib
data = pd.read_csv("/Users/alvin/Desktop/HW5_data.csv")
scaler = StandardScaler()
joblib.dump(scaler, "scaler.pkl")
label_encoder = LabelEncoder()
data['F2_Encoded'] = label_encoder.fit_transform(data['F2'])
data['F4_Encoded'] = label_encoder.fit_transform(data['F4'])
data['F8_Encoded'] = label_encoder.fit_transform(data['F8'])
features = ['F0', 'F1', 'F3', 'F5', 'F6', 'F7', 'F2_Encoded', 'F4_Encoded', 'F8_Encoded']
X = scaler.fit_transform(data[features].astype(float).values)
y = data['Task1'].values
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
batch_size=64


# In[5]:


#Same as f-1 method
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
fold_metrics = []


# In[6]:


#Same as f-1 method
hidden_size = 32  
output_size = 1 


# In[7]:


from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np

fold_metrics = [] 
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}/{k}")
    X_train, X_test = X_tensor[train_index], X_tensor[test_index]
    y_train, y_test = y_tensor[train_index], y_tensor[test_index]
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = ResidualNN(input_size=X.shape[1], hidden_size=hidden_size, output_size=output_size)
    #cross-entropy loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()  
    for epoch in range(200): 
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch).view(-1)  # Reshape logits to (batch_size,)
            y_batch = y_batch.view(-1)  # Reshape target to match logits
            loss = criterion(logits, y_batch)  # Compute loss
            loss.backward()  
            optimizer.step() 
    model.eval()
    with torch.no_grad():
        logits_test = model(X_test).squeeze()
        predictions = (torch.sigmoid(logits_test) > 0.5).float()
        
        # Convert tensors to NumPy arrays
        y_true = y_test.numpy()
        y_pred = predictions.numpy()
        
        # Compute Macro F1-Score
        macro_f1 = f1_score(y_true, y_pred)
        print(f"Fold {fold + 1} Macro F1-Score: {macro_f1:.4f}")

        fold_metrics.append(macro_f1)

# Calculate average Macro F1-Score
average_f1_score = np.mean(fold_metrics)
print(f"Average Macro F1 Score (Neural Network with Cross-Entropy Loss): {average_f1_score:.4f}")


# In[8]:


y_test_series=pd.Series(y_test)
y_test_series.value_counts()


# In[9]:


y_pred_series=pd.Series(y_pred)
y_pred_series.value_counts()


# In[ ]:




