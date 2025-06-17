#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib




# In[2]:


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
# Remove the SMOTE import that's causing the error
# from imblearn.over_sampling import SMOTE

def preprocess_data(file_path, target_column, test_size=0.2, stratify=True):
    """
    Preprocess the data for machine learning.
    
    Parameters:
    file_path (str): Path to the CSV file
    target_column (str): Name of the target column
    test_size (float): Proportion of the dataset to include in the test split
    stratify (bool): Whether to use stratified sampling for train-test split
    
    Returns:
    X_train, X_test, y_train, y_test, preprocessor
    """
    # Load the data
    df = pd.read_csv(file_path)
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split the data
    if stratify and len(y.unique()) > 1 and all(y.value_counts() >= 2):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    else:
        # Use non-stratified split if stratify=False or if there are classes with fewer than 2 samples
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Encode target if it's categorical
    if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'  # Include any columns not explicitly transformed
    )
    
    # Split the data with stratification to handle class imbalance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # Fit and transform the training data
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Check for class imbalance
    unique_classes = np.unique(y_train)
    if len(unique_classes) > 1:  # Only check for classification problems
        class_counts = np.bincount(y_train)
        imbalance_ratio = np.max(class_counts) / np.min(class_counts)
        
        if imbalance_ratio > 5:
            print(f"Class imbalance detected (ratio: {imbalance_ratio:.2f}). Using class weights in models.")
            # We'll handle this in the model training instead of using SMOTE
    
    return X_train_transformed, X_test_transformed, y_train, y_test, preprocessor


# In[ ]:




