#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
from sklearn.metrics import f1_score


# In[11]:


data = pd.read_csv("/Users/alvin/Desktop/Student_performance_data.csv")
#load data
# less than 2 as 1
data['BinaryGradeClass'] = (data['GradeClass'] <=2).astype(float)

# features
features = ['Age','Gender','Ethnicity','ParentalEducation','StudyTimeWeekly','Absences',
            'Tutoring','ParentalSupport',
            'Extracurricular','Sports','Music','Volunteering','GPA']

scaler = StandardScaler()
joblib.dump(scaler, "scaler.pkl")
X = scaler.fit_transform(data[features].astype(float).values)
y = data['BinaryGradeClass'].values
rows, cols = X.shape
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)


# In[12]:


w = torch.zeros(X_tensor.shape[1], requires_grad=True)  # same weight for each feature
b = torch.tensor(0.0, requires_grad=True) #start as no bias


# In[13]:


k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
fold_metrics = []


# In[14]:


def train_model (X_train, y_train, w, b, optimizer, epochs):
    for epoch in range(rows):
        optimizer.zero_grad()
        logits = X_train @ w + b
        #use f1-loss-approx as loss
        loss = f1_loss_approx(y_train, logits)
        loss.backward()
        optimizer.step()


# In[15]:


def f1_loss_approx(y_train, logits):
    p_k = 1 / (1 + torch.exp(-logits))  # Sigmoid activation
    TP = y_train * p_k  
    FP = (1 - y_train) * p_k
    FN = y_train * (1 - p_k)
    
    precision = TP.sum() / (TP.sum() + FP.sum() + 1e-8)
    recall = TP.sum() / (TP.sum() + FN.sum() + 1e-8)
    
    f1_score_approx = 2 * precision * recall / (precision + recall + 1e-8)
    return -f1_score_approx


# In[16]:


def single_fold(X_test, y_test, w, b):
    logits_test = X_test @ w + b
    probabilities = torch.sigmoid(logits_test)  # Apply sigmoid for probabilities
    predictions = (probabilities > 0.5).float()  # Threshold probabilities
    y_true = y_test.numpy()
    y_pred = predictions.numpy()
    
    # Compute regular F1-Score
    f1 = f1_score(y_true, y_pred)
    return f1


# In[17]:


def cross_validation(X_tensor, y_tensor, k=5, lr=0.001):
    global w, b 
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_metrics = []
    
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}/{k}")
        
        # Split data into training and test sets
        X_train, X_test = X_tensor[train_index], X_tensor[test_index]
        y_train, y_test = y_tensor[train_index], y_tensor[test_index] 
        
        optimizer = torch.optim.Adam([w, b], lr=0.001)
        
        # Train the model
        train_model(X_train, y_train, w, b, optimizer, epochs=rows)
        
        # Evaluate on test set
        f1 = single_fold(X_test, y_test, w, b)
        print(f"Fold {fold + 1} F1-Score: {f1:.4f}")
        fold_metrics.append(f1)
    
    # Compute the average F1-Score
    average_f1_score = np.mean(fold_metrics)
    print(f"Average F1 Score (Neural Network with F1 Loss): {average_f1_score:.4f}")
    return


# In[18]:


if __name__ == "__main__":
    print("Running K-Fold with PyTorch Model:")
    average_f1_pytorch = cross_validation(X_tensor, y_tensor, k=5, lr=0.001)
    


# In[ ]:





# In[ ]:




