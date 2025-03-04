import numpy as np
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, classification_report,f1_score
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split


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

with open("data/20240503.pkl","rb") as f:
    test = pickle.load(f)


# train['input'] = train['input'][:,[0,2,3,4,5]]
# test['input'] = test['input'][:,[0,2,3,4,5]]

#X_train, X_test, y_train, y_test = train_test_split(train['input'], train['labels'], test_size=0.3, random_state=42)


param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
rf = RandomForestClassifier()
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=5, n_jobs=-1, verbose=2)

grid_search.fit(train['input'], train['labels'])
# 输出最佳参数
print(f"Best parameters: {grid_search.best_params_}")


# 1. 使用随机抽样缩小训练集
train_subset_size = 0.3  # 使用30%的数据进行网格搜索
X_train_subset, _, y_train_subset, _ = train_test_split(
    train['input'], train['labels'], train_size=train_subset_size, random_state=42
)

# # 2. 使用更小的参数网格，先进行粗略搜索，再细化
# param_grid_small = {
#     'n_estimators': [50, 100],  # 减少参数选项
#     'max_depth': [None, 20],
#     'min_samples_split': [2, 10],
#     'min_samples_leaf': [1, 2],
#     'bootstrap': [True]
# }

# # 3. 减少交叉验证折数
# rf_efficient = RandomForestClassifier()
# grid_search_efficient = GridSearchCV(
#     estimator=rf_efficient,
#     param_grid=param_grid_small,
#     cv=3,  # 减少交叉验证折数
#     n_jobs=-1,
#     verbose=2
# )

print(f"Best parameters: {grid_search.best_params_}")

# 使用最佳模型进行预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(test['input'])

# 输出分类报告
print(classification_report(test['labels'], y_pred))


# 输出分类报告
print("Classification Report:")
print(classification_report(train['labels'], best_model.predict(train['input'])))

# 输出分类报告
print("Classification Report:")
y_test = test['labels']
y_pred = best_model.predict(test['input'])
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)

print(conf_matrix)
joblib.dump(best_model,"bus_aug.model")