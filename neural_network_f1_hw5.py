#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#clear
from IPython.display import display, Javascript
display(Javascript('IPython.notebook.kernel.restart()'))


# In[1]:


import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score


# In[2]:


scaler = StandardScaler()
joblib.dump(scaler, "scaler.pkl")


# In[3]:


from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import torch
import joblib

# Load data
data = pd.read_csv("/Users/alvin/Desktop/HW5_data.csv")

# Initialize scaler
scaler = StandardScaler()
joblib.dump(scaler, "scaler.pkl")

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# change categorical data to numbers 
data['F2_Encoded'] = label_encoder.fit_transform(data['F2'])
data['F4_Encoded'] = label_encoder.fit_transform(data['F4'])
data['F8_Encoded'] = label_encoder.fit_transform(data['F8'])

#features 
features = ['F0', 'F1', 'F3', 'F5', 'F6', 'F7', 'F2_Encoded', 'F4_Encoded', 'F8_Encoded']

# data
X = scaler.fit_transform(data[features].astype(float).values)
# label
y = data['Task1'].values
# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

#batch size; 
#the number of training samples that are fed to the network at once 
#during each iteration of the training process
batch_size=64


# In[4]:


k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)


# In[5]:


hidden_size = 32  # Number of neurons in the hidden layer
output_size = 1  # Binary classification output


# In[6]:


class ResidualNN(torch.nn.Module):
    #3 layers
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
        out += residual  # Add residual connection
        out = self.relu(out)
        
        out = self.fc3(out)
        return out


# In[7]:


def f1_loss_approx(y_true, logits):
    p_k = torch.sigmoid(logits)  # Convert logits to probabilities
    TP = y_true * p_k  
    FP = (1 - y_true) * p_k  
    FN = y_true * (1 - p_k)  

    
    precision = TP.sum() / (TP.sum() + FP.sum() + 1e-8)
    recall = TP.sum() / (TP.sum() + FN.sum() + 1e-8)
    f1_score = 2 * precision * recall / (precision + recall + 1e-8)

    
    return -f1_score  # Negative F-1 for minimization


# In[8]:


fold_metrics = []  
for fold, (train_index, test_index) in enumerate(kf.split(X)):

    X_train, X_test = X_tensor[train_index], X_tensor[test_index]
    y_train, y_test = y_tensor[train_index], y_tensor[test_index]
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize Neural Network and Optimizer
    model = ResidualNN(input_size=X.shape[1], hidden_size=hidden_size, 
                       output_size=output_size)
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the Neural Network
    for epoch in range(200):  
        
        model.train()
        
        # Iterate over mini-batches
        for X_batch, y_batch in train_loader:  
            optimizer.zero_grad()
            
            logits = model(X_batch)  # Forward pass
            logits = logits.squeeze()  
            loss = f1_loss_approx(y_batch, logits)  # Use F-1 approximation as loss
            loss.backward()  #update parameters
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        
        #evaluate prediction
        logits_test = model(X_test).squeeze()
        predictions = (torch.sigmoid(logits_test) > 0.5).float()
        
        # Convert tensors to NumPy arrays
        y_true = y_test.numpy()
        y_pred = predictions.numpy()
        
        # Compute actual F1-Score
        f1 = f1_score(y_true, y_pred)
        print(f"Fold {fold + 1} F1-Score: {macro_f1:.4f}")

        fold_metrics.append(macro_f1)

average_f1_score = np.mean(fold_metrics)
print(f"Average F1 Score: {average_f1_score:.4f}")


# In[ ]:


y_test_series=pd.Series(y_test)
y_test_series.value_counts()


# In[15]:


y_pred_series=pd.Series(y_pred)
y_pred_series.value_counts()


# In[11]:


fold_metrics


# In[ ]:




