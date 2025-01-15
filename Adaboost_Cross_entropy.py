#!/usr/bin/env python
# coding: utf-8

# In[22]:


from IPython.display import display, Javascript
display(Javascript('IPython.notebook.kernel.restart()'))


# In[1]:


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


# In[2]:


scaler = StandardScaler()
joblib.dump(scaler, "scaler.pkl")


# In[3]:


class AdaBoost:
    def __init__(self, num_learner: int, num_cats: int):
        self.num_learner = num_learner
        self.num_cats = num_cats
        self.learner_weights = []
        self.sorted_learners = []

    def cross_entropy_loss_per_sample(self, y, logits):
        y_tensor = torch.tensor(y, dtype=torch.long)
        entry_weights_tensor = torch.tensor(self.entry_weights, dtype=torch.float32)

        p = logits
        probas = np.stack([1 - p, p], axis=1)
        logits_tensor = torch.tensor(probas, dtype=torch.float32)
        logits_tensor = torch.log(logits_tensor + 1e-9)

        criterion = nn.CrossEntropyLoss(reduction='none')

        loss = criterion(logits_tensor, y_tensor)  
        return loss.detach().numpy()  

    def train(self, X, y, learners):
        n = X.shape[0]
        self.entry_weights = np.full(n, 1.0 / n)
        self.learner_weights = []

        for i in range(self.num_learner): 
            # Initialize the learner with balanced class weights
            learner = LogisticRegression(max_iter=100, class_weight='balanced')

            # Train the learner on weighted data
            learner.fit(X, y, sample_weight=self.entry_weights)

            # Compute logits 
            logits = learner.predict_proba(X)[:, 1]

            # cross-entropy losses
            sample_losses = self.cross_entropy_loss_per_sample(y, logits)

            # Aggregate loss for alpha
            avg_loss = np.average(sample_losses, weights=self.entry_weights)

            # Define alpha from cross-entropy loss. loss decrease alpha increase
            alpha = -avg_loss

            self.learner_weights.append(alpha)

            # Update the sample weights using per-sample losses
            self.entry_weights *= np.exp(alpha * sample_losses)
            self.entry_weights /= np.sum(self.entry_weights)

            # Save the learner
            self.sorted_learners.append(learner)

            # Debug: Check if predictions vary
            preds = learner.predict(X)


    def predict(self, X):
        # Use all learners and their weights to produce final predictions
        learner_predictions = [learner.predict(X) for learner in self.sorted_learners]
        weighted_sum = np.zeros(X.shape[0])
        for w, preds in zip(self.learner_weights, learner_predictions):
            signed_preds = np.where(preds == 1, 1, -1)
            weighted_sum += w * signed_preds
        final_predictions = (weighted_sum > 0).astype(int)
        return final_predictions


# In[4]:


data = pd.read_csv("/Users/alvin/Desktop/Student_performance_data.csv")

data['BinaryGradeClass'] = (data['GradeClass'] <= 2).astype(float)

features = ['Age','Gender','Ethnicity','ParentalEducation',
            'StudyTimeWeekly', 'Absences','Tutoring','ParentalSupport',
            'Extracurricular','Sports', 'Music','GPA']
X_unnormalized = data[features].astype(float).values
X = scaler.fit_transform(X_unnormalized)
y = data['BinaryGradeClass'].values
y_tensor = torch.tensor(y, dtype=torch.long)


# In[5]:


num_learners=40
num_categories_task = 2
model_task= AdaBoost(num_learner=num_learners, num_cats=num_categories_task)
learners_task = [LogisticRegression(max_iter=50) for _ in range(num_learners)]
model_task.train(X, y, learners_task)


# In[6]:


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
#     print("Actual y_val:", y_val)

    # Evaluate on the validation fold
    predictions_val = model_cv.predict(X_val_scaled)
#     print("Predicted y_val:", predictions_val)
    TP = np.sum((y_val == 1) & (predictions_val == 1))  # True Positives
    FP = np.sum((y_val == 0) & (predictions_val == 1))  # False Positives
    FN = np.sum((y_val == 1) & (predictions_val == 0))  # False Negatives
    TN = np.sum((y_val == 0) & (predictions_val == 0))  # False Negatives
#     print(f'F1 score for fold: {f1:.4f}')
    precision = TP / (TP + FP) 
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Output results
    print(f"True Positives (TP): {TP}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")
    print(f"False Negatives (TN): {TN}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

# Report the average F1 score across folds


# In[ ]:




