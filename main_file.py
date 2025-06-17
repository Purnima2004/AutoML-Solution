#!/usr/bin/env python
# coding: utf-8

# In[8]:


# In your main notebook
get_ipython().run_line_magic('run', 'preprocessing_hackathon.ipynb')
get_ipython().run_line_magic('run', 'model_definations.ipynb')


# In[9]:


from preprocessing_hackathon_new import preprocess_data


# In[3]:


from preprocessing_hackathon import preprocess_data
if __name__ == "__main__":
    data_path = input("Please enter the path to the dataset (CSV file): ")
    target_column = input("Please enter the name of the target column: ")

    # Input validation for test size
    while True:
        try:
            test_size = float(input("Please enter the test size (e.g., 0.2 for 20% test size): "))
            if 0 < test_size < 1:
                break
            else:
                print("Test size must be a float between 0 and 1.")
        except ValueError:
            print("Invalid input. Please enter a float between 0 and 1.")

    try:
        X_train, X_test, y_train, y_test, preprocessor = preprocess_data(data_path, target_column, test_size)
        print("Data preprocessing completed successfully.")
        
    except ValueError as e:
        print(e)
    except FileNotFoundError:
        print(f"File '{data_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")




# In[4]:


def select_ml_algorithm():
    """ user need to select a machine learning algorithm and update model_name. """
    algorithms = [
        'SVM',
        'Decision Tree',
        'Bagging',
        'Random Forest',
        'ADA Boost',
        'XG Boost',
        'Neural Network'
    ]
    
    # Display available algorithms
    print("Available machine learning algorithms:")
    for i, algo in enumerate(algorithms, start=1):
        print(f"{i}. {algo}")
    
    # Ask the user to select an algorithm
    while True:
        try:
            choice = int(input("Enter the number corresponding to the algorithm you want to use: "))
            if 1 <= choice <= len(algorithms):
                model_name = algorithms[choice - 1]
                print(f"Selected algorithm: {model_name}")
                return model_name
            else:
                print("Invalid choice. Please enter a number corresponding to the listed options.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

# Example usage:
model_name = select_ml_algorithm()
print(f"Model name updated to: {model_name}")


# In[5]:


models = {
    'SVM': (SVC(), {}),
    'Decision Tree': (DecisionTreeClassifier(), {}),
    'Bagging': (BaggingClassifier(), {}),
    'Random Forest': (RandomForestClassifier(), {}),
    'ADA Boost': (AdaBoostClassifier(), {}),
    'XG Boost': (XGBClassifier(), {}),
    'Neural Network': (MLPClassifier(max_iter=500), {}),
}


# In[10]:


from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

def get_user_input(model_name):
    """ Function to get user input for hyperparameters based on the model. """
    if model_name == 'SVM':
        C = list(map(float, input("Enter values for C (comma separated): ").split(',')))
        kernel = input("Enter values for kernel (comma separated, e.g., linear,rbf): ").split(',')
        gamma = input("Enter values for gamma (comma separated, e.g., scale,auto): ").split(',')
        return {'C': C, 'kernel': kernel, 'gamma': gamma}
    
    elif model_name == 'Decision Tree':
        criterion = input("Enter values for criterion (comma separated, e.g., gini,entropy): ").split(',')
        max_depth = list(map(int, input("Enter values for max_depth (comma separated, e.g., 10,20,30): ").split(',')))
        min_samples_split = list(map(int, input("Enter values for min_samples_split (comma separated, e.g., 2,5,10): ").split(',')))
        min_samples_leaf = list(map(int, input("Enter values for min_samples_leaf (comma separated, e.g., 1,2,4): ").split(',')))
        return {'criterion': criterion, 'max_depth': max_depth, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}
    
    elif model_name == 'Bagging':
        n_estimators = list(map(int, input("Enter values for n_estimators (comma separated, e.g., 10,50,100): ").split(',')))
        max_samples = list(map(float, input("Enter values for max_samples (comma separated, e.g., 0.5,0.7,1.0): ").split(',')))
        max_features = list(map(float, input("Enter values for max_features (comma separated, e.g., 0.5,0.7,1.0): ").split(',')))
        return {'n_estimators': n_estimators, 'max_samples': max_samples, 'max_features': max_features}
    
    elif model_name == 'Random Forest':
        n_estimators = list(map(int, input("Enter values for n_estimators (comma separated, e.g., 50,100,200): ").split(',')))
        max_depth = list(map(int, input("Enter values for max_depth (comma separated, e.g., 10,20,30): ").split(',')))
        min_samples_split = list(map(int, input("Enter values for min_samples_split (comma separated, e.g., 2,5,10): ").split(',')))
        min_samples_leaf = list(map(int, input("Enter values for min_samples_leaf (comma separated, e.g., 1,2,4): ").split(',')))
        max_features = input("Enter values for max_features (comma separated, e.g., auto,sqrt,log2): ").split(',')
        return {'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'max_features': max_features}
    
    elif model_name == 'ADA Boost':
        n_estimators = list(map(int, input("Enter values for n_estimators (comma separated, e.g., 50,100,200): ").split(',')))
        learning_rate = list(map(float, input("Enter values for learning_rate (comma separated, e.g., 0.01,0.1,1.0): ").split(',')))
        return {'n_estimators': n_estimators, 'learning_rate': learning_rate}
    
    elif model_name == 'XG Boost':
        n_estimators = list(map(int, input("Enter values for n_estimators (comma separated, e.g., 50,100,200): ").split(',')))
        learning_rate = list(map(float, input("Enter values for learning_rate (comma separated, e.g., 0.01,0.1,0.2): ").split(',')))
        max_depth = list(map(int, input("Enter values for max_depth (comma separated, e.g., 3,6,10): ").split(',')))
        subsample = list(map(float, input("Enter values for subsample (comma separated, e.g., 0.6,0.8,1.0): ").split(',')))
        return {'n_estimators': n_estimators, 'learning_rate': learning_rate, 'max_depth': max_depth, 'subsample': subsample}
    
    elif model_name == 'Neural Network':
        hidden_layer_sizes = input("Enter values for hidden_layer_sizes (comma separated tuples, e.g., (50,), (100,), (50,50)): ").split(',')
        hidden_layer_sizes = [tuple(map(int, layer.strip('()').split())) for layer in hidden_layer_sizes]
        alpha = list(map(float, input("Enter values for alpha (comma separated, e.g., 0.0001,0.001,0.01): ").split(',')))
        activation = input("Enter values for activation (comma separated, e.g., relu,tanh,logistic): ").split(',')
        solver = input("Enter values for solver (comma separated, e.g., adam,sgd): ").split(',')
        return {'hidden_layer_sizes': hidden_layer_sizes, 'alpha': alpha, 'activation': activation, 'solver': solver}
    
    else:
        raise ValueError("Model not recognized")


def update_models_with_user_input(model_name):
    """ Update the specific model dictionary with user input for hyperparameters. """
    print(f"\nUpdating hyperparameters for {model_name}:")
    param_grid = get_user_input(model_name)
    model, _ = models[model_name]
    updated_model = {model_name: (model, param_grid)}
    return updated_model



from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from sklearn.model_selection import RandomizedSearchCV

def train_and_evaluate(X_train, X_test, y_train, y_test):
    metrics = {}
    best_models = {}
    updated_models = update_models_with_user_input(model_name)
    
    for name, (model, param_grid) in updated_models.items():
        # Initialize RandomizedSearchCV with the model and parameters
        search = RandomizedSearchCV(model, param_grid, n_iter=min(10, len(param_grid)), cv=3, random_state=42, n_jobs=-1)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_models[name] = best_model
        y_pred = best_model.predict(X_test)
        metrics[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            
        }
    
    return metrics, best_models



# In[12]:


# Train and evaluate models
metrics, best_models = train_and_evaluate(X_train, X_test, y_train, y_test)

# Output the metrics
for model_name, metric in metrics.items():
    print(f"{model_name}: Accuracy={metric['accuracy']:.4f}")
    


# In[ ]:





# In[ ]:





# In[ ]:




