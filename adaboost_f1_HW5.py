#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IPython.display import display, Javascript
display(Javascript('IPython.notebook.kernel.restart()'))


# In[6]:


from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler


# In[7]:


scaler = StandardScaler()
joblib.dump(scaler, "scaler.pkl")


# In[1]:


class AdaBoost:
    def __init__(self, num_learner: int, num_cats: int):
        self.num_learner = num_learner
        self.num_cats = num_cats
        self.entry_weights = None
        self.learner_weights = []
        self.sorted_learners = []
        
    def f1_loss_approx(self, y, logits):
        p_k = 1 / (1 + np.exp(-logits))  # Sigmoid activation
        TP = np.sum(y * p_k)  
        FP = np.sum((1 - y) * p_k) 
        FN = np.sum(y * (1 - p_k))  

        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1_score_approx = 2 * precision * recall / (precision + recall + 1e-8)
        return -f1_score_approx  # Negative for minimization
    
    def train(self, X, y, learners):
        n = X.shape[0]
        self.entry_weights = np.full(n, 1 / n)
        self.learner_weights = []
        for learner in learners:
            learner.fit(X, y, sample_weight=self.entry_weights) #beginning weight
            logits = learner.predict_proba(X)[:,1] # Probabilities for positive class
        
            loss = self.f1_loss_approx(y, logits)
            
            alpha = -loss  # alpha=f1, the lower the loss, the higher the weight
            #f1 increase = predict corrent 
            
            self.learner_weights.append(alpha)
            
            # Update sample weights
            predictions = learner.predict(X)
            misclassified = (predictions != y).astype(float) #when predicted wrong
            
            self.entry_weights *= np.exp(np.clip(alpha * misclassified, -500, 500))
            self.entry_weights /= np.sum(self.entry_weights) #normalization
            
            self.sorted_learners.append(learner)
        self.learner_weights = np.array(self.learner_weights)
        self.learner_weights /= np.sum(self.learner_weights)
    
    def predict(self, X):
        pooled_predictions = np.zeros((X.shape[0], self.num_cats))
        for learner, weight in zip(self.sorted_learners, self.learner_weights):
            logits = learner.predict_proba(X)[:, 1]  # Use probabilities for positive class
            predictions = (logits > 0.5).astype(int)  # Binary threshold at 0.5
            for i, pred in enumerate(predictions):
                prediction = np.full(self.num_cats, -1 / (self.num_cats - 1))
                prediction[pred] = 1
                pooled_predictions[i] += prediction * weight
        return np.argmax(pooled_predictions, axis=1)


# In[9]:


data = pd.read_csv("/Users/alvin/Desktop/HW5_data.csv")

# Initialize scaler and save it
scaler = StandardScaler()
joblib.dump(scaler, "scaler.pkl")

# Target variable
y = data['Task1'].values

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode each categorical column ('F2', 'F4', 'F8') separately
data['F2_Encoded'] = label_encoder.fit_transform(data['F2'])
data['F4_Encoded'] = label_encoder.fit_transform(data['F4'])
data['F8_Encoded'] = label_encoder.fit_transform(data['F8'])

# Define features (replace original categorical columns with encoded ones)
features = ['F0', 'F1', 'F3', 'F5', 'F6', 'F7', 'F2_Encoded', 'F4_Encoded', 'F8_Encoded']

# Scale the feature matrix
X = scaler.fit_transform(data[features].astype(float).values)

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)


# In[10]:


num_learners=40
num_categories_task = 2
model_task= AdaBoost(num_learner=num_learners, num_cats=num_categories_task)
learners_task = [LogisticRegression(max_iter=100) for _ in range(num_learners)]
model_task.train(X, y, learners_task)


# In[11]:


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20)
f1_scores_total = []
for train_index, val_index in kf.split(X, y):
    # Split the data into training and validation sets
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Initialize AdaBoost model
    model_cv = AdaBoost(num_learner=num_learners, num_cats=num_categories_task)
    learners_cv = [LogisticRegression(max_iter=100) for _ in range(num_learners)]
    
    # Train on the current training fold
    model_cv.train(X_train_scaled, y_train, learners_cv)
    
    # Evaluate on the validation fold
    predictions_val = model_cv.predict(X_val_scaled)
    f1 = f1_score(y_val, predictions_val)
    f1_scores_total.append(f1)
    print(f'F1 score for fold: {f1:.4f}')
#     print("Actual y_val:", y_val)
#     print("Predicted y_val:", predictions_val)
# Report the average F1 score across folds
average_f1_score = np.mean(f1_scores_total)  # Ensure np.mean is not shadowed
print(f"Average F1 Score with Cross-Validation: {average_f1_score:.4f}")


# In[12]:


X.shape


# In[ ]:




