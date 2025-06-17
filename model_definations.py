#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install xgboost')


# In[2]:


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, StackingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib




# In[3]:


def get_svm_model():
    return (SVC(), {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto']  # Useful for 'rbf', 'poly', and 'sigmoid'
}),
def get_decision_tree_model():
    return (DecisionTreeClassifier(), {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}),


def get_bagging_model():
    (RandomForestClassifier(), {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}),



def get_random_forest_model():
    return (RandomForestClassifier(), {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}),

def get_ada_boost_model():
    return (AdaBoostClassifier(), {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1.0]
}),


def get_xgboost_model():
    return (XGBClassifier(), {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 6, 10],
    'subsample': [0.6, 0.8, 1.0]
}),

def get_neural_network_model():
    return (MLPClassifier(max_iter=500), {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
    'alpha': [0.0001, 0.001, 0.01],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'sgd']
}),

from sklearn.linear_model import LogisticRegression
def get_stacking_model():
    return (StackingClassifier(estimators=[
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('svc', SVC(kernel='rbf', probability=True))
], final_estimator=LogisticRegression()), {
    'final_estimator__C': [0.1, 1, 10],
    'final_estimator__penalty': ['l2'],
    'final_estimator__solver': ['lbfgs']
}),


# In[ ]:




