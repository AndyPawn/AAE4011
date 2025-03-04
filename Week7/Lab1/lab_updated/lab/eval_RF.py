import joblib
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load("bus_aug.model")
with open("data/20240525.pkl","rb") as f:
    test = pickle.load(f)

print(classification_report(test['labels'], model.predict(test['input'])))
conf_matrix = confusion_matrix(test['labels'], model.predict(test['input']))
print(conf_matrix)

importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
forest_importances = pd.Series(importances, index=['SNR', 'constellation', 'elevation', 'azimuth', 'residual', 'rms residual'])

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.show()