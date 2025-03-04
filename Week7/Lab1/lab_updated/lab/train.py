import numpy as np
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score
import joblib
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler

import os

os.makedirs("model", exist_ok=True)

# Load training data
train_list = ["20240501.pkl", "20240430.pkl"]
train = {}
train['input'] = []
train['labels'] = []
for i in train_list:
    with open("data/"+i,'rb') as f:
        tmp = pickle.load(f)
        train['input'].append(tmp['input'])
        train['labels'].append(tmp['labels'])

train['input'] = np.vstack(train['input'])
train['labels'] = np.hstack(train['labels'])

# 6 features : ['SNR', 'constellation', 'elevation', 'azimuth', 'residual', 'mean residual']

# Load test data
with open("data/20240507.pkl","rb") as f:
    test = pickle.load(f)


# Modification here to preprocess the significance
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train['input'])

# Same transformation for test data
test_scaled = scaler.transform(test['input'])

# Save the scalar
joblib.dump(scaler, "model/scaler.pkl")

# Train models with default parameters
lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier()
#svm = SVC()
gbdt = GradientBoostingClassifier()

models = {
    'logistic': lr,
    'random_forest': rf,
    'gbdt': gbdt
}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(train['input'], train['labels'])
    joblib.dump(model, f"model/{name}_model.pkl")