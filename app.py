import streamlit as st
import base64
import pandas as pd
import numpy as np
import joblib
import io
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import time

# Set page config first
st.set_page_config(
    page_title="AutoML Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'df' not in st.session_state:
    st.session_state.df = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'categorical_cols' not in st.session_state:
    st.session_state.categorical_cols = []
if 'numerical_cols' not in st.session_state:
    st.session_state.numerical_cols = []
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'evaluation_metrics' not in st.session_state:
    st.session_state.evaluation_metrics = None

# Custom CSS for the background and content
def set_background():
    st.markdown("""
    <style>
    /* Main app container */
    .stApp {
        background: linear-gradient(135deg, #f5f0ff 0%, #e3e9ff 50%, #d0e3ff 100%);
        background-attachment: fixed;
        min-height: 100vh;
    }
    
    /* Main content area */
    .main .block-container {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.5);
    }
    
    /* Sidebar */
    .stSidebar {
        background-color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Ensure content is above the background */
    .stApp > div:first-child > div:first-child > div:first-child > div:first-child {
        z-index: 1 !important;
    }
    
    
    /* Add glowing effect to elements */
    .glow {
        box-shadow: 0 0 20px rgba(100, 149, 255, 0.5);
        transition: all 0.3s ease-in-out;
    }
    
    .glow:hover {
        box-shadow: 0 0 30px rgba(100, 149, 255, 0.8);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(100, 149, 255, 0.5);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(100, 149, 255, 0.7);
    }
    </style>
    """, unsafe_allow_html=True)

# Set the background
set_background()

# Custom CSS for the main content
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Main app layout
def show_learn_more():
    st.markdown("""
    <style>
        .feature-card {
            background: white;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
        }
        .feature-icon {
            font-size: 2.5rem;
            margin-bottom: 1rem;
            color: #1e40af;
        }
    </style>
    
    <div style='text-align: center; margin-bottom: 3rem;'>
        <h1>Why Choose Our AutoML Solution?</h1>
        <p style='max-width: 800px; margin: 0 auto 2rem;'>
            Our platform makes machine learning accessible to everyone, regardless of technical expertise. 
            Here's what makes our solution stand out:
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features Grid
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='feature-card'>
            <div class='feature-icon'>🚀</div>
            <h3>No-Code Solution</h3>
            <p>Build and deploy machine learning models without writing a single line of code.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='feature-card'>
            <div class='feature-icon'>⚡</div>
            <h3>Lightning Fast</h3>
            <p>Automated model selection and hyperparameter tuning for optimal performance.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='feature-card'>
            <div class='feature-icon'>📊</div>
            <h3>Data Visualization</h3>
            <p>Interactive visualizations to understand your data and model performance.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='feature-card'>
            <div class='feature-icon'>🤖</div>
            <h3>Multiple Algorithms</h3>
            <p>Choose from various machine learning algorithms to find the best fit for your data.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='feature-card'>
            <div class='feature-icon'>🔍</div>
            <h3>Detailed Insights</h3>
            <p>Get comprehensive evaluation metrics and feature importance analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='feature-card'>
            <div class='feature-icon'>💾</div>
            <h3>Easy Export</h3>
            <p>Download your trained models and use them in your applications.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-top: 3rem;'>
        <h2>Ready to get started?</h2>
        <p>Upload your data and start building machine learning models in minutes!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Center the button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Get Started Now", type="primary", use_container_width=True, key=""):
            st.session_state.page = "📊 Data Upload"
            st.rerun()

def main():
    # Initialize page in session state if not exists
    if 'page' not in st.session_state:
        st.session_state.page = "🏠 Home"
    
    # Sidebar
    with st.sidebar:
        st.title("🤖 AutoML Pro")
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["🏠 Home", "📊 Data Upload", "🔧 Model Training", "📈 Results", "🔮 Predict", "📚 Learn More"],
            index=["🏠 Home", "📊 Data Upload", "🔧 Model Training", "📈 Results", "🔮 Predict", "📚 Learn More"].index(st.session_state.page) if st.session_state.page in ["🏠 Home", "📊 Data Upload", "🔧 Model Training", "📈 Results", "🔮 Predict", "📚 Learn More"] else 0
        )
        
        # Update page in session state when sidebar changes
        if page != st.session_state.page:
            st.session_state.page = page
            st.rerun()
            
        st.markdown("---")
    
    # Display the current page
    if st.session_state.page == "🏠 Home":
        show_home()
    elif st.session_state.page == "📊 Data Upload":
        show_data_upload()
    elif st.session_state.page == "🔧 Model Training":
        show_model_training()
    elif st.session_state.page == "📈 Results":
        show_results()
    elif st.session_state.page == "🔮 Predict":
        show_predict()
    elif st.session_state.page == "📚 Learn More":
        show_learn_more()
        

# Page functions
def show_home():
    # Add custom CSS for the title
    st.markdown("""
    <style>
        .main-title {
            font-size: 5rem;
            font-weight: 700;
            margin: 1rem 0;
            background: linear-gradient(45deg, #1e40af, #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            line-height: 1;
            letter-spacing: -1px;
        }
        .sub-title {
            font-size: 1.5rem;
            color: #555;
            max-width: 800px;
            margin: 0 auto;
        }
    </style>
    
    <div style='text-align: center; padding: 3rem 1rem;'>
        <div class='main-title'>Automated machine learning</div>
        <div class='main-title'>AutoML</div>
        <div class='sub-title'>AutoML Solution helps you build machine learning models without writing code.</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Button styling with 3D effect
    st.markdown("""
    <style>
        .stButton>button {
            border: none;
            border-radius: 8px;
            padding: 1rem 4rem;
            font-weight: 600;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.2s ease;
            position: relative;
            top: 5px;
            box-shadow: 0 4px 0 #1e3a8a, 0 6px 8px rgba(0,0,0,0.15);
            margin: -5rem 1rem;
            min-width: 140px;
        }
        .stButton>button:active {
            top: 2px;
            box-shadow: 0 1px 0 #1e3a8a, 0 2px 4px rgba(0,0,0,0.2);
        }
        /* Primary button */
        .stButton>button:first-child {
            background: #1e40af;
            color: white;
        }
        .stButton>button:first-child:hover {
            background: #1e3a8a;
            transform: translateY(-2px);
            box-shadow: 0 10px 0 #1e3a8a, 0 12px 16px rgba(0,0,0,0.15);
        }
        /* Secondary button */
        .stButton>button:not(:first-child) {
            background: white;
            color: #1e40af;
            border: 10px solid #1e40af;
        }
        .stButton>button:not(:first-child):hover {
            background: #f8fafc;
            transform: translateY(-2px);
            box-shadow: 0 16px 0 #1e3a8a, 0 18px 22px rgba(0,0,0,0.1);
        }
        .button-container {
            display: flex;
            justify-content: center;
            gap: -2rem;
            margin: -2rem 0;
            padding: 1rem 0;
            width: 160%;
        }
        .button-wrapper {
            display: flex;
            justify-content: center;
            width: 160%;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Create button container with centered layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Create a container for the buttons
        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        
        # Create columns for the buttons
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("Get Started", type="primary"):
                st.session_state.page = "📊 Data Upload"
                st.rerun()

                
        with col_btn2:
            if st.button("Learn More"):
                st.session_state.page = "📚 Learn More"
                st.rerun()

                
        st.markdown('</div>', unsafe_allow_html=True)

def preprocess_data(df, target_column):
    """Preprocess the data: handle missing values, encode categorical variables, etc."""
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Store column information in session state
    st.session_state.categorical_cols = categorical_cols
    st.session_state.numerical_cols = numerical_cols
    st.session_state.target_column = target_column
    
    # Create transformers
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Store preprocessor in session state
    st.session_state.preprocessor = preprocessor
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 2 else y
    )
    
    # Store data in session state
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    
    return X_train, X_test, y_train, y_test

def show_data_upload():
    st.header("📊 Data Upload & Preprocessing")
    st.markdown("Upload your dataset and configure the preprocessing steps.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            
            # Display dataset info
            st.success(f"File uploaded successfully! {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Show sample data
            with st.expander("View Sample Data", expanded=True):
                st.dataframe(df.head())
            
            # Data overview
            with st.expander("Data Overview"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Columns:**", ", ".join(df.columns.tolist()))
                    st.write("**Total Rows:**", df.shape[0])
                    st.write("**Total Columns:**", df.shape[1])
                
                with col2:
                    st.write("**Missing Values:**", df.isnull().sum().sum())
                    st.write("**Duplicate Rows:**", df.duplicated().sum())
            
            # Select target column
            target_column = st.selectbox(
                "Select the target column (what you want to predict)",
                options=df.columns.tolist(),
                index=len(df.columns)-1 if len(df.columns) > 0 else 0
            )
            
            if st.button("Preprocess Data", type="primary"):
                with st.spinner("Preprocessing data..."):
                    try:
                        # Preprocess the data
                        X_train, X_test, y_train, y_test = preprocess_data(df, target_column)
                        
                        # Show preprocessing summary
                        st.success("Data preprocessed successfully!")
                        st.write(f"Training set: {X_train.shape[0]} samples")
                        st.write(f"Test set: {X_test.shape[0]} samples")
                        
                        # Show class distribution
                        st.subheader("Class Distribution")
                        class_dist = pd.DataFrame({
                            'Count': y_train.value_counts(),
                            'Percentage': y_train.value_counts(normalize=True) * 100
                        })
                        st.dataframe(class_dist)
                        
                        # Visualize class distribution
                        fig = px.pie(
                            class_dist, 
                            values='Count', 
                            names=class_dist.index,
                            title='Class Distribution in Training Set',
                            hole=0.3
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Enable model training
                        st.session_state.data_preprocessed = True
                        
                    except Exception as e:
                        st.error(f"Error during preprocessing: {str(e)}")
                        st.session_state.data_preprocessed = False
                        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

def train_model(model_type, params):
    """Train the selected model with the given parameters"""
    try:
        # Get preprocessed data
        X_train = st.session_state.X_train
        y_train = st.session_state.y_train
        preprocessor = st.session_state.preprocessor
        
        # Create the model pipeline
        if model_type == "Random Forest":
            model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', None),
            min_samples_split=params.get('min_samples_split', 2),  # ADD THIS
            min_samples_leaf=params.get('min_samples_leaf', 1),    # ADD THIS
            random_state=42,
            n_jobs=-1
             ))
        ])
        
        
        elif model_type == "SVM":
           model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', SVC(
            C=params.get('C', 1.0),
            kernel=params.get('kernel', 'rbf'),
            gamma=params.get('gamma', 'scale'),  # ADD THIS
            probability=True,
            random_state=42
        ))
])
        elif model_type == "XGBoost":
            model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 3),
            learning_rate=params.get('learning_rate', 0.1),
            subsample=params.get('subsample', 1.0),  # ADD THIS
            random_state=42,
            n_jobs=-1
    ))
])
        elif model_type == "Neural Network":
            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', MLPClassifier(
                    hidden_layer_sizes=params.get('hidden_layer_sizes', (100,)),
                    activation=params.get('activation', 'relu'),
                    solver=params.get('solver', 'adam'),
                    alpha=params.get('alpha', 0.0001),
                    max_iter=params.get('max_iter', 200),
                    random_state=42
                ))
            ])
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Store the trained model
        st.session_state.model = model
        st.session_state.model_type = model_type
        
        # Evaluate the model
        evaluate_model(model)
        
        return model
    except Exception as e:
        st.error(f"Error during model training: {str(e)}")
        return None

def evaluate_model(model):
    """Evaluate the trained model and store metrics"""
    try:
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Store metrics
        st.session_state.evaluation_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        return True
    except Exception as e:
        st.error(f"Error during model evaluation: {str(e)}")
        return False

def show_model_training():
    st.header("🔧 Model Training")
    
    if 'data_preprocessed' not in st.session_state:
        st.warning("Please upload and preprocess your data in the Data Upload section first.")
        return
    
    # Model selection
    model_type = st.selectbox(
        "Select Model Type",
        ["Random Forest", "SVM", "XGBoost", "Neural Network"]
    )
    
    # Display model description
    model_descriptions = {
        "Random Forest": "An ensemble of decision trees, good for most classification tasks.",
        "SVM": "Powerful for high-dimensional spaces, works well with clear margin of separation.",
        "XGBoost": "Gradient boosting algorithm, often provides state-of-the-art results.",
        "Neural Network": "Powerful for complex patterns but requires more data and computation."
    }
    st.markdown(f"**Description:** {model_descriptions[model_type]}")
    
    # Model-specific hyperparameters
    st.markdown("### Hyperparameters")
    
    params = {}
    
    if model_type == "Random Forest":
        col1, col2 = st.columns(2)
        with col1:
            params['n_estimators'] = st.slider("Number of trees", 10, 500, 100, 10)
            params['max_depth'] = st.slider("Max depth", 1, 30, 10)
        with col2:
            params['min_samples_split'] = st.slider("Min samples split", 2, 20, 2)
            params['min_samples_leaf'] = st.slider("Min samples leaf", 1, 10, 1)
    
    elif model_type == "SVM":
        col1, col2 = st.columns(2)
        with col1:
            params['C'] = st.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.1)
            params['kernel'] = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
        with col2:
            params['gamma'] = st.selectbox("Gamma", ["scale", "auto"], index=0)
    
    elif model_type == "XGBoost":
        col1, col2 = st.columns(2)
        with col1:
            params['n_estimators'] = st.slider("Number of trees", 10, 500, 100, 10)
            params['max_depth'] = st.slider("Max depth", 1, 15, 3)
        with col2:
            params['learning_rate'] = st.slider("Learning rate", 0.01, 1.0, 0.1, 0.01)
            params['subsample'] = st.slider("Subsample ratio", 0.1, 1.0, 1.0, 0.05)
    
    elif model_type == "Neural Network":
        col1, col2 = st.columns(2)
        with col1:
            hidden_layers = st.slider("Number of hidden layers", 1, 5, 1)
            layer_sizes = []
            for i in range(hidden_layers):
                layer_size = st.slider(f"Neurons in layer {i+1}", 10, 200, 100, 10, key=f"nn_units_{i}")
                layer_sizes.append(layer_size)
            params['hidden_layer_sizes'] = tuple(layer_sizes)
        with col2:
            params['activation'] = st.selectbox("Activation function", ["relu", "tanh", "logistic"])
            params['solver'] = st.selectbox("Solver", ["adam", "sgd", "lbfgs"])
            params['alpha'] = st.slider("L2 penalty (alpha)", 0.0001, 0.1, 0.0001, 0.0001, format="%.4f")
    
    # Train button
    if st.button("Train Model", type="primary", use_container_width=True):
        with st.spinner("Training model. This may take a while..."):
            start_time = time.time()
            model = train_model(model_type, params)
            training_time = time.time() - start_time
            
            if model is not None:
                st.session_state.model_trained = True
                st.success(f"Model trained successfully in {training_time:.2f} seconds!")
                
                # Show evaluation metrics
                show_model_evaluation()
                
                # Enable download buttons
                show_model_download()
            else:
                st.error("Model training failed. Please check the error message above.")

def show_model_evaluation():
    """Display model evaluation metrics and visualizations"""
    if 'evaluation_metrics' not in st.session_state or st.session_state.evaluation_metrics is None:
        st.warning("No evaluation metrics available. Please train a model first.")
        return
    
    metrics = st.session_state.evaluation_metrics
    
    # Display metrics in columns
    st.markdown("### Model Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    with col2:
        st.metric("Precision", f"{metrics['precision']:.4f}")
    with col3:
        st.metric("Recall", f"{metrics['recall']:.4f}")
    with col4:
        st.metric("F1 Score", f"{metrics['f1']:.4f}")
    
    # Confusion Matrix
    st.markdown("### Confusion Matrix")
    
    # Create confusion matrix
    cm = confusion_matrix(metrics['y_true'], metrics['y_pred'])
    
    # Plot confusion matrix using Plotly
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=sorted(metrics['y_true'].unique()),
        y=sorted(metrics['y_true'].unique()),
        text_auto=True,
        aspect="auto",
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        coloraxis_colorbar=dict(title="Count")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature Importance (for tree-based models)
    if hasattr(st.session_state.model, 'named_steps') and \
       hasattr(st.session_state.model.named_steps['classifier'], 'feature_importances_'):
        try:
            st.markdown("### Feature Importance")
            
            # Get feature names after preprocessing
            preprocessor = st.session_state.model.named_steps['preprocessor']
            
            # Get feature names for numerical features
            numerical_features = st.session_state.numerical_cols
            
            # Get feature names for categorical features (after one-hot encoding)
            categorical_features = []
            if 'cat' in preprocessor.named_transformers_:
                ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
                if hasattr(ohe, 'get_feature_names_out'):
                    categorical_features = ohe.get_feature_names_out(st.session_state.categorical_cols).tolist()
            
            # Combine all feature names
            all_features = numerical_features + categorical_features
            
            # Get feature importances
            importances = st.session_state.model.named_steps['classifier'].feature_importances_
            
            # Create a DataFrame for visualization
            feature_importance = pd.DataFrame({
                'Feature': all_features[:len(importances)],
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(20)  # Show top 20 features
            
            # Plot feature importance
            fig = px.bar(
                feature_importance,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 20 Most Important Features',
                labels={'Importance': 'Importance Score', 'Feature': 'Feature Name'}
            )
            
            fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.warning(f"Could not compute feature importance: {str(e)}")

def show_model_download():
    """Display download buttons for the trained model and preprocessor"""
    st.markdown("### Download")
    
    # Create a buffer to save the model
    model_buffer = BytesIO()
    joblib.dump(st.session_state.model, model_buffer)
    
    # Create a download button for the model
    st.download_button(
        label="Download Trained Model",
        data=model_buffer.getvalue(),
        file_name=f"automl_{st.session_state.model_type.lower().replace(' ', '_')}.pkl",
        mime="application/octet-stream",
        help="Download the trained model for later use"
    )
    
    # Create a buffer to save the preprocessor
    if 'preprocessor' in st.session_state and st.session_state.preprocessor is not None:
        preprocessor_buffer = BytesIO()
        joblib.dump(st.session_state.preprocessor, preprocessor_buffer)
        
        # Create a download button for the preprocessor
        st.download_button(
            label="Download Data Preprocessor",
            data=preprocessor_buffer.getvalue(),
            file_name="data_preprocessor.pkl",
            mime="application/octet-stream",
            help="Download the data preprocessor for consistent preprocessing of new data"
        )

def show_results():
    st.header("📈 Model Evaluation")
    
    if 'model_trained' not in st.session_state or not st.session_state.model_trained:
        st.warning("Please train a model first in the Model Training section.")
        return
    
    # Show evaluation metrics and visualizations
    show_model_evaluation()
    
    # Show download options
    show_model_download()

def show_predict():
    st.header("🔮 Make Predictions")
    
    if 'model_trained' not in st.session_state or not st.session_state.model_trained:
        st.warning("Please train a model first in the Model Training section.")
        return
    
    # Get the trained model and preprocessor
    model = st.session_state.model
    
    # Get feature names (excluding target)
    feature_columns = [col for col in st.session_state.df.columns if col != st.session_state.target_column]
    
    st.markdown("### Input Features for Prediction")
    
    # Create input fields for each feature
    input_values = {}
    
    # Group features into columns for better layout
    num_cols = 3
    cols = st.columns(num_cols)
    
    for i, feature in enumerate(feature_columns):
        with cols[i % num_cols]:
            if st.session_state.df[feature].dtype in ['int64', 'float64']:
                # For numerical features
                min_val = float(st.session_state.df[feature].min())
                max_val = float(st.session_state.df[feature].max())
                default_val = float(st.session_state.df[feature].median())
                
                input_values[feature] = st.number_input(
                    label=feature,
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    step=(max_val - min_val) / 100 if max_val > min_val else 0.1,
                    help=f"Range: {min_val:.2f} to {max_val:.2f}"
                )
            else:
                # For categorical features
                unique_vals = st.session_state.df[feature].unique().tolist()
                input_values[feature] = st.selectbox(
                    label=feature,
                    options=unique_vals,
                    index=0 if len(unique_vals) > 0 else None
                )
    
    # Add a predict button
    if st.button("Make Prediction", type="primary", use_container_width=True):
        with st.spinner("Making prediction..."):
            try:
                # Create a DataFrame with the input values
                input_df = pd.DataFrame([input_values])
                
                # Make prediction
                prediction = model.predict(input_df)
                prediction_proba = model.predict_proba(input_df) if hasattr(model, 'predict_proba') else None
                
                # Display results
                st.markdown("### Prediction Result")
                
                # Show predicted class
                st.metric("Predicted Class", str(prediction[0]))
                
                # Show prediction probabilities if available
                if prediction_proba is not None:
                    st.markdown("#### Class Probabilities")
                    classes = model.classes_
                    proba_df = pd.DataFrame({
                        'Class': classes,
                        'Probability': prediction_proba[0]
                    }).sort_values('Probability', ascending=False)
                    
                    # Create a bar chart of class probabilities
                    fig = px.bar(
                        proba_df,
                        x='Class',
                        y='Probability',
                        color='Probability',
                        color_continuous_scale='Blues',
                        title='Prediction Probabilities'
                    )
                    
                    # Add probability labels
                    fig.update_traces(
                        texttemplate='%{y:.2%}',
                        textposition='outside',
                        hovertemplate='Class: %{x}<br>Probability: %{y:.2%}<extra></extra>'
                    )
                    
                    fig.update_layout(
                        xaxis_title="Class",
                        yaxis_title="Probability",
                        yaxis_tickformat='.0%',
                        showlegend=False,
                        uniformtext_minsize=8,
                        uniformtext_mode='hide'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    
    # Add an option to upload a CSV for batch prediction
    st.markdown("---")
    st.markdown("### Batch Prediction")
    st.markdown("Upload a CSV file with multiple rows to make predictions in bulk.")
    
    uploaded_file = st.file_uploader("Upload CSV for batch prediction", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            batch_df = pd.read_csv(uploaded_file)
            
            # Check if all required columns are present
            missing_cols = set(feature_columns) - set(batch_df.columns)
            if missing_cols:
                st.error(f"The uploaded file is missing the following required columns: {', '.join(missing_cols)}")
            else:
                # Make predictions
                with st.spinner("Making batch predictions..."):
                    try:
                        # Select only the required columns in the correct order
                        batch_input = batch_df[feature_columns]
                        
                        # Make predictions
                        batch_predictions = model.predict(batch_input)
                        
                        # Add predictions to the dataframe
                        result_df = batch_df.copy()
                        result_df['Prediction'] = batch_predictions
                        
                        # Add prediction probabilities if available
                        if hasattr(model, 'predict_proba'):
                            probas = model.predict_proba(batch_input)
                            for i, cls in enumerate(model.classes_):
                                result_df[f'P({cls})'] = probas[:, i]
                        
                        # Display the results
                        st.success(f"Successfully made predictions for {len(result_df)} rows!")
                        
                        # Show a sample of the results
                        st.dataframe(result_df.head())
                        
                        # Add a download button for the results
                        csv = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Predictions",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Error during batch prediction: {str(e)}")
                        
        except Exception as e:
            st.error(f"Error reading the uploaded file: {str(e)}")

if __name__ == "__main__":
    main()
