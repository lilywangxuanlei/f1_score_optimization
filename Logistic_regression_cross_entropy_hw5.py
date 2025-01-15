#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
import joblib
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader


# In[8]:


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


# In[9]:


#initialize weit and bias
w = torch.zeros(X_tensor.shape[1], requires_grad=True)  # equal weit
b = torch.tensor(0.0, requires_grad=True)  


# In[10]:


k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
fold_metrics = []


# In[11]:


optimizer = torch.optim.Adam([w, b], lr=0.001) #learning rate 0.001


# In[12]:


for fold, (train_index, test_index) in enumerate(kf.split(X)):
    
    # Split data into train and test sets
    X_train, X_test = X_tensor[train_index], X_tensor[test_index]
    
    y_train, y_test = y_tensor[train_index], y_tensor[test_index]

    #initialize min-batch
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # cross-entropy loss
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam([w, b], lr=0.001)

    # Training loop
    for epoch in range(200):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = X_batch @ w + b  # Compute logits
            loss = criterion(logits, y_batch)  
            loss.backward()
            optimizer.step()

    # Evaluation
    with torch.no_grad():
        logits = X_test @ w + b  # Compute raw logits
        p_k = torch.sigmoid(logits)  # Apply sigmoid for probabilities
        predictions = (p_k >= 0.5).float()  #threshold 0.5
        predictions_np = predictions.numpy()
        y_np = y_test.numpy()

        # Calculate TP, FP, FN
        TP = ((predictions_np == 1) & (y_np == 1)).sum()
        FP = ((predictions_np == 1) & (y_np == 0)).sum()
        FN = ((predictions_np == 0) & (y_np == 1)).sum()
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1_score = 2 * precision * recall / (precision + recall + 1e-8)

        # append F1 score
        fold_metrics.append(f1_score)
        print(f"Fold {fold + 1} F1 Score: {f1_score:.4f}")

# Final evaluation across folds
average_f1 = np.mean(fold_metrics)
print(f"Average F1 Score: {average_f1:.4f}")


# In[ ]:




