#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import joblib
label_encoder = LabelEncoder()
# Load the data
data = pd.read_csv("/Users/alvin/Desktop/HW5_data.csv")
data['F2_Encoded'] = label_encoder.fit_transform(data['F2'])
data['F4_Encoded'] = label_encoder.fit_transform(data['F4'])
data['F8_Encoded'] = label_encoder.fit_transform(data['F8'])

features = ['F0', 'F1', 'F3', 'F5', 'F6', 'F7', 'F2_Encoded', 'F4_Encoded', 'F8_Encoded']
X_unnormalized = data[features].astype(float).values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X_unnormalized)

# Target values
y = data['Task1'].values
label_encoder = LabelEncoder()
label_encoder = LabelEncoder()


# CV
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Adaboost start
adaboost_model = AdaBoostClassifier(n_estimators=40, random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(adaboost_model, X_train, y_train, cv=cv, scoring="f1")

# results
print(f"F1 Score for Each Fold: {cv_scores}")
print(f"Mean CV F1 Score: {cv_scores.mean():.4f}")

# Train the model
adaboost_model.fit(X_train, y_train)

# Predict on the test set
y_pred = adaboost_model.predict(X_test)


# In[ ]:





# In[ ]:




