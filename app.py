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
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR

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

# Custom CSS 
def set_background():
    st.markdown("""
    <style>
    /* Light mode (default) */
    .stApp {
        background-image: radial-gradient(circle 248px at center, #16d9e3 0%, #30c7ec 47%, #46aef7 100%);
        background-attachment: fixed;
        min-height: 100vh;
        color: #1e40af;
    }
    /* Sidebar (light) */
    .stSidebar {
        background-color: rgba(255, 255, 255, 0.9) !important;
    }
    /* Dark mode */
    [data-theme="dark"] .stApp {
        background-image: radial-gradient(circle 248px at center, #16d9e3 0%, #30c7ec 47%, #46aef7 100%);
        background-attachment: fixed;
        min-height: 100vh;
        color: #fff !important;
    }
    [data-theme="dark"] .stSidebar {
        background-color: rgba(30, 36, 50, 0.95) !important;
    }
    [data-theme="dark"] .main-title, [data-theme="dark"] .feature-icon, [data-theme="dark"] .feature-card, [data-theme="dark"] .sub-title {
        color: #fff !important;
        background: none !important;
        -webkit-text-fill-color: #fff !important;
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
    [data-theme="dark"] ::-webkit-scrollbar-track {
        background: rgba(30, 36, 50, 0.2);
    }
    [data-theme="dark"] ::-webkit-scrollbar-thumb {
        background: rgba(100, 149, 255, 0.3);
    }
    [data-theme="dark"] ::-webkit-scrollbar-thumb:hover {
        background: rgba(100, 149, 255, 0.5);
    }
    </style>
    """, unsafe_allow_html=True)

set_background()


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
    if 'task_type' not in st.session_state:
        st.session_state.task_type = "Supervised Learning"
    
    # Sidebar
    with st.sidebar:
        st.title("🤖 AutoML Pro")
        st.markdown("---")
        # Task selector
        task_type = st.radio(
            "Select Task Type",
            ["Supervised Learning", "Unsupervised Learning"],
            index=["Supervised Learning", "Unsupervised Learning"].index(st.session_state.task_type)
        )
        if task_type != st.session_state.task_type:
            st.session_state.task_type = task_type
            st.rerun()
        # Navigation
        page = st.radio(
            "Navigation",
            ["🏠 Home", "📊 Data Upload", "🔧 Model Training", "📈 Results", "🔮 Predict", "📚 Learn More"],
            index=["🏠 Home", "📊 Data Upload", "🔧 Model Training", "📈 Results", "🔮 Predict", "📚 Learn More"].index(st.session_state.page) if st.session_state.page in ["🏠 Home", "📊 Data Upload", "🔧 Model Training", "📈 Results", "🔮 Predict", "📚 Learn More"] else 0
        )
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
        if st.session_state.task_type == "Supervised Learning":
           show_model_training()
        else:
            show_unsupervised_training()
    elif st.session_state.page == "📈 Results":
        if st.session_state.task_type == "Supervised Learning":
           show_results()
        else:
            show_unsupervised_results()
    elif st.session_state.page == "🔮 Predict":
        if st.session_state.task_type == "Supervised Learning":
           show_predict()
        else:
            st.info("Prediction is not available for unsupervised learning.")
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
    
    # --- PCA Option ---
    apply_pca = st.checkbox("Apply PCA for dimensionality reduction?")
    if apply_pca:
        max_components = min(500, X_train.shape[1]) if X_train.shape[1] > 2 else X_train.shape[1]
        n_components = st.slider("Number of PCA components", 2, max_components, min(50, max_components))
        pca = PCA(n_components=n_components)
        st.session_state.pca = pca
        st.session_state.pca_applied = True
        st.info(f"PCA will be applied: reduced to {n_components} components.")
    else:
        st.session_state.pca = None
        st.session_state.pca_applied = False
    
    return X_train, X_test, y_train, y_test

def show_data_upload():
    st.header("📊 Data Upload & Preprocessing")
    st.markdown("Upload your dataset and configure the preprocessing steps.")

    if st.session_state.task_type == "Supervised Learning":
        upload_mode = st.radio(
            "How do you want to provide your data?",
            ["Single CSV (auto split)", "Separate Train and Test CSVs"]
        )
        if upload_mode == "Single CSV (auto split)":
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.df = df
                    st.success(f"File uploaded successfully! {df.shape[0]} rows × {df.shape[1]} columns")
                    with st.expander("View Sample Data", expanded=True):
                        st.dataframe(df.head())
                    with st.expander("Data Overview"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Columns:**", ", ".join(df.columns.tolist()))
                            st.write("**Total Rows:**", df.shape[0])
                            st.write("**Total Columns:**", df.shape[1])
                        with col2:
                            st.write("**Missing Values:**", df.isnull().sum().sum())
                            st.write("**Duplicate Rows:**", df.duplicated().sum())
                    target_column = st.selectbox(
                        "Select the target column (what you want to predict)",
                        options=df.columns.tolist(),
                        index=len(df.columns)-1 if len(df.columns) > 0 else 0
                    )
                    # Handle missing values in target column
                    if df[target_column].isnull().sum() > 0:
                        st.warning(f"Target column contains {df[target_column].isnull().sum()} missing values.")
                        missing_option = st.radio(
                            "How do you want to handle missing values in the target column?",
                            ["Remove rows with missing target values", "Fill missing values with mean/median"]
                        )
                        if missing_option == "Remove rows with missing target values":
                            n_before = len(df)
                            df = df.dropna(subset=[target_column])
                            n_after = len(df)
                            st.info(f"Removed {n_before - n_after} rows with missing target values.")
                        else:
                            if pd.api.types.is_numeric_dtype(df[target_column]):
                                fill_method = st.selectbox("Fill missing values with:", ["mean", "median"])
                                if fill_method == "mean":
                                    fill_value = df[target_column].mean()
                                else:
                                    fill_value = df[target_column].median()
                                df[target_column] = df[target_column].fillna(fill_value)
                                st.info(f"Filled missing values in target column with {fill_method}: {fill_value}")
                            else:
                                st.error("Filling missing values is only supported for numeric target columns.")
                                return
                    if st.button("Preprocess Data", type="primary"):
                        with st.spinner("Preprocessing data..."):
                            try:
                                y = df[target_column]
                                value_counts = y.value_counts()
                                if value_counts.min() < 2:
                                    stratify = None
                                    st.warning("Some classes have only 1 sample. Stratified split is not possible.")
                                else:
                                    stratify = y
                                X = df.drop(columns=[target_column])
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=0.2, random_state=42, stratify=stratify
                                )
                                st.session_state.X_train = X_train
                                st.session_state.X_test = X_test
                                st.session_state.y_train = y_train
                                st.session_state.y_test = y_test
                                st.session_state.target_column = target_column
                                categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
                                numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
                                st.session_state.categorical_cols = categorical_cols
                                st.session_state.numerical_cols = numerical_cols
                                numerical_transformer = Pipeline(steps=[
                                    ('imputer', SimpleImputer(strategy='median')),
                                    ('scaler', StandardScaler())
                                ])
                                categorical_transformer = Pipeline(steps=[
                                    ('imputer', SimpleImputer(strategy='most_frequent')),
                                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                                ])
                                preprocessor = ColumnTransformer(
                                    transformers=[
                                        ('num', numerical_transformer, numerical_cols),
                                        ('cat', categorical_transformer, categorical_cols)
                                    ])
                                st.session_state.preprocessor = preprocessor
                                st.success("Data preprocessed successfully!")
                                st.session_state.data_preprocessed = True
                            except Exception as e:
                                st.error(f"Error during preprocessing: {str(e)}")
                                st.session_state.data_preprocessed = False
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
            # --- PCA Option (after preprocessing) ---
            if st.session_state.get('data_preprocessed', False):
                apply_pca = st.checkbox("Apply PCA for dimensionality reduction?")
                if apply_pca:
                    max_components = min(500, st.session_state.X_train.shape[1])
                    n_components = st.slider("Number of PCA components", 2, max_components, min(50, max_components))
                    pca = PCA(n_components=n_components)
                    st.session_state.pca = pca
                    st.session_state.pca_applied = True
                    st.info(f"PCA will be applied: reduced to {n_components} components.")
                else:
                    st.session_state.pca = None
                    st.session_state.pca_applied = False
        elif upload_mode == "Separate Train and Test CSVs":
            train_file = st.file_uploader("Upload your TRAIN CSV file", type="csv", key="train_csv")
            test_file = st.file_uploader("Upload your TEST CSV file", type="csv", key="test_csv")
            if train_file is not None and test_file is not None:
                try:
                    train_df = pd.read_csv(train_file)
                    test_df = pd.read_csv(test_file)
                    st.success(f"Train file: {train_df.shape[0]} rows × {train_df.shape[1]} columns")
                    st.success(f"Test file: {test_df.shape[0]} rows × {test_df.shape[1]} columns")
                    with st.expander("View Train Data", expanded=False):
                        st.dataframe(train_df.head())
                    with st.expander("View Test Data", expanded=False):
                        st.dataframe(test_df.head())
                    target_column_train = st.selectbox(
                        "Select the target column in TRAIN file",
                        options=train_df.columns.tolist(),
                        index=len(train_df.columns)-1 if len(train_df.columns) > 0 else 0,
                        key="target_train"
                    )
                    target_column_test = st.selectbox(
                        "Select the target column in TEST file",
                        options=test_df.columns.tolist(),
                        index=len(test_df.columns)-1 if len(test_df.columns) > 0 else 0,
                        key="target_test"
                    )
                    # Handle missing values in train target
                    if train_df[target_column_train].isnull().sum() > 0:
                        st.warning(f"Train target column contains {train_df[target_column_train].isnull().sum()} missing values.")
                        missing_option_train = st.radio(
                            "How do you want to handle missing values in the TRAIN target column?",
                            ["Remove rows with missing target values", "Fill missing values with mean/median"],
                            key="missing_option_train"
                        )
                        if missing_option_train == "Remove rows with missing target values":
                            n_before = len(train_df)
                            train_df = train_df.dropna(subset=[target_column_train])
                            n_after = len(train_df)
                            st.info(f"Removed {n_before - n_after} rows with missing target values in TRAIN.")
                        else:
                            if pd.api.types.is_numeric_dtype(train_df[target_column_train]):
                                fill_method = st.selectbox("Fill missing values in TRAIN with:", ["mean", "median"], key="fill_method_train")
                                if fill_method == "mean":
                                    fill_value = train_df[target_column_train].mean()
                                else:
                                    fill_value = train_df[target_column_train].median()
                                train_df[target_column_train] = train_df[target_column_train].fillna(fill_value)
                                st.info(f"Filled missing values in TRAIN target column with {fill_method}: {fill_value}")
                            else:
                                st.error("Filling missing values is only supported for numeric target columns in TRAIN.")
                                return
                    # Handle missing values in test target
                    if test_df[target_column_test].isnull().sum() > 0:
                        st.warning(f"Test target column contains {test_df[target_column_test].isnull().sum()} missing values.")
                        missing_option_test = st.radio(
                            "How do you want to handle missing values in the TEST target column?",
                            ["Remove rows with missing target values", "Fill missing values with mean/median"],
                            key="missing_option_test"
                        )
                        if missing_option_test == "Remove rows with missing target values":
                            n_before = len(test_df)
                            test_df = test_df.dropna(subset=[target_column_test])
                            n_after = len(test_df)
                            st.info(f"Removed {n_before - n_after} rows with missing target values in TEST.")
                        else:
                            if pd.api.types.is_numeric_dtype(test_df[target_column_test]):
                                fill_method = st.selectbox("Fill missing values in TEST with:", ["mean", "median"], key="fill_method_test")
                                if fill_method == "mean":
                                    fill_value = test_df[target_column_test].mean()
                                else:
                                    fill_value = test_df[target_column_test].median()
                                test_df[target_column_test] = test_df[target_column_test].fillna(fill_value)
                                st.info(f"Filled missing values in TEST target column with {fill_method}: {fill_value}")
                            else:
                                st.error("Filling missing values is only supported for numeric target columns in TEST.")
                                return
                    if st.button("Preprocess Data", type="primary"):
                        with st.spinner("Preprocessing data..."):
                            try:
                                X_train = train_df.drop(columns=[target_column_train])
                                y_train = train_df[target_column_train]
                                X_test = test_df.drop(columns=[target_column_test])
                                y_test = test_df[target_column_test]
                                st.session_state.X_train = X_train
                                st.session_state.X_test = X_test
                                st.session_state.y_train = y_train
                                st.session_state.y_test = y_test
                                st.session_state.target_column = target_column_train
                                categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
                                numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
                                st.session_state.categorical_cols = categorical_cols
                                st.session_state.numerical_cols = numerical_cols
                                numerical_transformer = Pipeline(steps=[
                                    ('imputer', SimpleImputer(strategy='median')),
                                    ('scaler', StandardScaler())
                                ])
                                categorical_transformer = Pipeline(steps=[
                                    ('imputer', SimpleImputer(strategy='most_frequent')),
                                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                                ])
                                preprocessor = ColumnTransformer(
                                    transformers=[
                                        ('num', numerical_transformer, numerical_cols),
                                        ('cat', categorical_transformer, categorical_cols)
                                    ])
                                st.session_state.preprocessor = preprocessor
                                st.success("Data preprocessed successfully!")
                                st.session_state.data_preprocessed = True
                            except Exception as e:
                                st.error(f"Error during preprocessing: {str(e)}")
                                st.session_state.data_preprocessed = False
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
            # --- PCA Option (after preprocessing) ---
            if st.session_state.get('data_preprocessed', False):
                apply_pca = st.checkbox("Apply PCA for dimensionality reduction?")
                if apply_pca:
                    max_components = min(500, st.session_state.X_train.shape[1])
                    n_components = st.slider("Number of PCA components", 2, max_components, min(50, max_components))
                    pca = PCA(n_components=n_components)
                    st.session_state.pca = pca
                    st.session_state.pca_applied = True
                    st.info(f"PCA will be applied: reduced to {n_components} components.")
                else:
                    st.session_state.pca = None
                    st.session_state.pca_applied = False
    else:
        # Unsupervised learning: keep original logic
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.success(f"File uploaded successfully! {df.shape[0]} rows × {df.shape[1]} columns")
                with st.expander("View Sample Data", expanded=True):
                    st.dataframe(df.head())
                st.session_state.data_preprocessed = True
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

def train_model(model_type, params):
    """Train the selected model with the given parameters"""
    try:
        X_train = st.session_state.X_train
        y_train = st.session_state.y_train
        preprocessor = st.session_state.preprocessor
        steps = [('preprocessor', preprocessor)]
        # Add PCA step if applied
        if st.session_state.get('pca_applied', False) and st.session_state.get('pca', None) is not None:
            steps.append(('pca', st.session_state.pca))
        # Add classifier/regressor step
        if model_type == "Random Forest":
            steps.append(('classifier', RandomForestClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', None),
                min_samples_split=params.get('min_samples_split', 2),
                min_samples_leaf=params.get('min_samples_leaf', 1),
            random_state=42,
            n_jobs=-1
            )))
        elif model_type == "Decision Tree":
            steps.append(('classifier', DecisionTreeClassifier(
                max_depth=params.get('max_depth', None),
                min_samples_split=params.get('min_samples_split', 2),
                min_samples_leaf=params.get('min_samples_leaf', 1),
                random_state=42
            )))
        elif model_type == "SVM":
            steps.append(('classifier', SVC(
            C=params.get('C', 1.0),
            kernel=params.get('kernel', 'rbf'),
                gamma=params.get('gamma', 'scale'),
            probability=True,
            random_state=42
            )))
        elif model_type == "XGBoost":
            steps.append(('classifier', XGBClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 3),
            learning_rate=params.get('learning_rate', 0.1),
                subsample=params.get('subsample', 1.0),
            random_state=42,
            n_jobs=-1
            )))
        elif model_type == "Neural Network":
            steps.append(('classifier', MLPClassifier(
                    hidden_layer_sizes=params.get('hidden_layer_sizes', (100,)),
                    activation=params.get('activation', 'relu'),
                    solver=params.get('solver', 'adam'),
                    alpha=params.get('alpha', 0.0001),
                    max_iter=params.get('max_iter', 200),
                    random_state=42
            )))
        elif model_type == "Logistic Regression":
            steps.append(('classifier', LogisticRegression(
                C=params.get('C', 1.0),
                penalty=params.get('penalty', 'l2'),
                solver=params.get('solver', 'lbfgs'),
                max_iter=params.get('max_iter', 100),
                random_state=42
            )))
        elif model_type == "Linear Regression":
            steps.append(('regressor', LinearRegression()))
        elif model_type == "Decision Tree Regressor":
            steps.append(('regressor', DecisionTreeRegressor(
                max_depth=params.get('max_depth', None),
                min_samples_split=params.get('min_samples_split', 2),
                min_samples_leaf=params.get('min_samples_leaf', 1),
                random_state=42
            )))
        elif model_type == "Random Forest Regressor":
            steps.append(('regressor', RandomForestRegressor(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', None),
                min_samples_split=params.get('min_samples_split', 2),
                min_samples_leaf=params.get('min_samples_leaf', 1),
                random_state=42,
                n_jobs=-1
            )))
        elif model_type == "SVR":
            steps.append(('regressor', SVR(
                C=params.get('C', 1.0),
                kernel=params.get('kernel', 'rbf'),
                gamma=params.get('gamma', 'scale')
            )))
        elif model_type == "XGBoost Regressor":
            steps.append(('regressor', XGBClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 3),
                learning_rate=params.get('learning_rate', 0.1),
                subsample=params.get('subsample', 1.0),
                random_state=42,
                n_jobs=-1
            )))
        elif model_type == "Neural Network Regressor":
            steps.append(('regressor', MLPClassifier(
                hidden_layer_sizes=params.get('hidden_layer_sizes', (100,)),
                activation=params.get('activation', 'relu'),
                solver=params.get('solver', 'adam'),
                alpha=params.get('alpha', 0.0001),
                max_iter=params.get('max_iter', 200),
                random_state=42
            )))
        else:
            raise ValueError(f"Model type '{model_type}' is not supported.")

        model = Pipeline(steps=steps)
        model.fit(X_train, y_train)
        st.session_state.model = model
        st.session_state.model_type = model_type
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
    supervised_type = st.selectbox("Problem Type", ["Classification", "Regression"])
    if supervised_type == "Classification":
        model_type = st.selectbox( 
        "Select Model Type",
            ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "XGBoost", "Neural Network"]
        )
    else:
        model_type = st.selectbox(
            "Select Model Type",
            ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor", "SVR", "XGBoost Regressor", "Neural Network Regressor"]
        )
    model_descriptions = {
        "Logistic Regression": "A linear model for binary or multiclass classification, interpretable and efficient.",
        "Decision Tree": "Simple to understand and visualize, but can easily overfit.",
        "Random Forest": "An ensemble of decision trees, good for most classification tasks.",
        "SVM": "Powerful for high-dimensional spaces, works well with clear margin of separation.",
        "XGBoost": "Gradient boosting algorithm, often provides state-of-the-art results.",
        "Neural Network": "Powerful for complex patterns but requires more data and computation.",
        "Linear Regression": "A basic regression technique for modeling linear relationships.",
        "Decision Tree Regressor": "Regression version of decision tree, can model non-linear relationships.",
        "Random Forest Regressor": "Ensemble of decision trees for regression, reduces overfitting.",
        "SVR": "Support Vector Regression, effective for high-dimensional regression tasks.",
        "XGBoost Regressor": "Gradient boosting for regression, often provides top performance.",
        "Neural Network Regressor": "Neural network for regression, captures complex patterns."
    }
    st.markdown(f"**Description:** {model_descriptions.get(model_type, 'No description available.')}")
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
    elif model_type == "Decision Tree":
        col1, col2 = st.columns(2)
        with col1:
            params['max_depth'] = st.slider("Max depth", 1, 30, 10)
            params['min_samples_split'] = st.slider("Min samples split", 2, 20, 2)
        with col2:
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
    elif model_type == "Logistic Regression":
        col1, col2 = st.columns(2)
        with col1:
            params['C'] = st.slider("Inverse regularization (C)", 0.01, 10.0, 1.0, 0.01)
            params['penalty'] = st.selectbox("Penalty", ["l2", "none"])
        with col2:
            params['solver'] = st.selectbox("Solver", ["lbfgs", "saga"])
            params['max_iter'] = st.slider("Max iterations", 50, 500, 100, 10)
    elif model_type == "Linear Regression":
        st.info("No hyperparameters to set for Linear Regression.")
    elif model_type == "Decision Tree Regressor":
        col1, col2 = st.columns(2)
        with col1:
            params['max_depth'] = st.slider("Max depth", 1, 30, 10)
            params['min_samples_split'] = st.slider("Min samples split", 2, 20, 2)
        with col2:
            params['min_samples_leaf'] = st.slider("Min samples leaf", 1, 10, 1)
    elif model_type == "Random Forest Regressor":
        col1, col2 = st.columns(2)
        with col1:
            params['n_estimators'] = st.slider("Number of trees", 10, 500, 100, 10)
            params['max_depth'] = st.slider("Max depth", 1, 30, 10)
        with col2:
            params['min_samples_split'] = st.slider("Min samples split", 2, 20, 2)
            params['min_samples_leaf'] = st.slider("Min samples leaf", 1, 10, 1)
    elif model_type == "SVR":
        col1, col2 = st.columns(2)
        with col1:
            params['C'] = st.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.1)
            params['kernel'] = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
        with col2:
            params['gamma'] = st.selectbox("Gamma", ["scale", "auto"], index=0)
    elif model_type == "XGBoost Regressor":
        col1, col2 = st.columns(2)
        with col1:
            params['n_estimators'] = st.slider("Number of trees", 10, 500, 100, 10)
            params['max_depth'] = st.slider("Max depth", 1, 15, 3)
        with col2:
            params['learning_rate'] = st.slider("Learning rate", 0.01, 1.0, 0.1, 0.01)
            params['subsample'] = st.slider("Subsample ratio", 0.1, 1.0, 1.0, 0.05)
    elif model_type == "Neural Network Regressor":
        col1, col2 = st.columns(2)
        with col1:
            hidden_layers = st.slider("Number of hidden layers", 1, 5, 1)
            layer_sizes = []
            for i in range(hidden_layers):
                layer_size = st.slider(f"Neurons in layer {i+1}", 10, 200, 100, 10, key=f"nn_units_reg_{i}")
                layer_sizes.append(layer_size)
            params['hidden_layer_sizes'] = tuple(layer_sizes)
        with col2:
            params['activation'] = st.selectbox("Activation function", ["relu", "tanh", "logistic"])
            params['solver'] = st.selectbox("Solver", ["adam", "sgd", "lbfgs"])
            params['alpha'] = st.slider("L2 penalty (alpha)", 0.0001, 0.1, 0.0001, 0.0001, format="%.4f")
    if st.button("Train Model", type="primary", use_container_width=True):
        with st.spinner("Training model. This may take a while..."):
            start_time = time.time()
            model = train_model(model_type, params)
            training_time = time.time() - start_time
            if model is not None:
                st.session_state.model_trained = True
                st.success(f"Model trained successfully in {training_time:.2f} seconds!")
                show_model_evaluation()
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
    st.markdown("### Feature Importance")
    model = st.session_state.model
    feature_importance_supported = False
    classifier = None
    # Try to get the classifier or regressor from the pipeline
    if hasattr(model, 'named_steps'):
        classifier = model.named_steps.get('classifier', None)
        if classifier is None:
            classifier = model.named_steps.get('regressor', None)
    # Check if the classifier supports feature_importances_
    if classifier is not None and hasattr(classifier, 'feature_importances_'):
        feature_importance_supported = True
    if feature_importance_supported:
        try:
            # Get the fitted preprocessor from the pipeline
            preprocessor = model.named_steps['preprocessor']
            # Get feature names for numerical features
            numerical_features = st.session_state.numerical_cols
            # Get feature names for categorical features (after one-hot encoding)
            categorical_features = []
            if 'cat' in preprocessor.named_transformers_:
                ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
                if hasattr(ohe, 'get_feature_names_out') and hasattr(ohe, 'categories_'):
                    # Only call get_feature_names_out if the encoder is fitted
                    categorical_features = ohe.get_feature_names_out(st.session_state.categorical_cols).tolist()
            # Combine all feature names
            all_features = numerical_features + categorical_features
            # Get feature importances
            importances = classifier.feature_importances_
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
    else:
        st.info("Feature importance is only available for tree-based models like Random Forest, Decision Tree, or XGBoost.")

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

# --- Unsupervised Learning ---
def show_unsupervised_training():
    st.header("🧩 Unsupervised Model Training")
    if st.session_state.df is None:
        st.warning("Please upload your data in the Data Upload section first.")
        return
    unsupervised_type = st.selectbox("Algorithm Type", ["Clustering", "Dimensionality Reduction", "Anomaly Detection"])
    if unsupervised_type == "Clustering":
        model_type = st.selectbox("Select Model", ["KMeans", "DBSCAN", "Agglomerative Clustering"])
        if model_type == "KMeans":
            n_clusters = st.slider("Number of clusters", 2, 10, 3)
            if st.button("Train Model"):
                model = KMeans(n_clusters=n_clusters, random_state=42)
                X = st.session_state.df.select_dtypes(include=[np.number])
                model.fit(X)
                st.session_state.model = model
                st.session_state.model_type = model_type
                st.session_state.unsup_labels = model.labels_
                st.success("KMeans clustering complete!")
        elif model_type == "DBSCAN":
            eps = st.slider("Epsilon (eps)", 0.1, 10.0, 0.5)
            min_samples = st.slider("Min samples", 2, 20, 5)
            if st.button("Train Model"):
                model = DBSCAN(eps=eps, min_samples=min_samples)
                X = st.session_state.df.select_dtypes(include=[np.number])
                model.fit(X)
                st.session_state.model = model
                st.session_state.model_type = model_type
                st.session_state.unsup_labels = model.labels_
                st.success("DBSCAN clustering complete!")
        elif model_type == "Agglomerative Clustering":
            n_clusters = st.slider("Number of clusters", 2, 10, 3)
            if st.button("Train Model"):
                model = AgglomerativeClustering(n_clusters=n_clusters)
                X = st.session_state.df.select_dtypes(include=[np.number])
                model.fit(X)
                st.session_state.model = model
                st.session_state.model_type = model_type
                st.session_state.unsup_labels = model.labels_
                st.success("Agglomerative clustering complete!")
    elif unsupervised_type == "Dimensionality Reduction":
        model_type = st.selectbox("Select Model", ["PCA"])
        n_components = st.slider("Number of components", 2, min(10, st.session_state.df.shape[1]), 2)
        if st.button("Apply Dimensionality Reduction"):
            model = PCA(n_components=n_components)
            X = st.session_state.df.select_dtypes(include=[np.number])
            X_reduced = model.fit_transform(X)
            st.session_state.model = model
            st.session_state.model_type = model_type
            st.session_state.X_reduced = X_reduced
            st.success("PCA complete!")
    elif unsupervised_type == "Anomaly Detection":
        model_type = st.selectbox("Select Model", ["Isolation Forest", "One-Class SVM"])
        if model_type == "Isolation Forest":
            contamination = st.slider("Contamination (outlier proportion)", 0.01, 0.5, 0.05)
            if st.button("Train Model"):
                model = IsolationForest(contamination=contamination, random_state=42)
                X = st.session_state.df.select_dtypes(include=[np.number])
                model.fit(X)
                st.session_state.model = model
                st.session_state.model_type = model_type
                st.session_state.unsup_labels = model.predict(X)
                st.success("Isolation Forest anomaly detection complete!")
        elif model_type == "One-Class SVM":
            nu = st.slider("Nu (anomaly proportion)", 0.01, 0.5, 0.05)
            if st.button("Train Model"):
                model = OneClassSVM(nu=nu)
                X = st.session_state.df.select_dtypes(include=[np.number])
                model.fit(X)
                st.session_state.model = model
                st.session_state.model_type = model_type
                st.session_state.unsup_labels = model.predict(X)
                st.success("One-Class SVM anomaly detection complete!")

def show_unsupervised_results():
    st.header("🧩 Unsupervised Results")
    if 'model' not in st.session_state or st.session_state.model is None:
        st.warning("Please train an unsupervised model first.")
        return
    model_type = st.session_state.model_type
    if model_type in ["KMeans", "DBSCAN", "Agglomerative Clustering"]:
        st.subheader("Cluster Assignments")
        st.write(pd.Series(st.session_state.unsup_labels, name="Cluster").value_counts())
        st.write("Cluster labels:", st.session_state.unsup_labels)
        # Optionally, show a 2D plot if possible
        X = st.session_state.df.select_dtypes(include=[np.number])
        if X.shape[1] >= 2:
            fig = px.scatter(X, x=X.columns[0], y=X.columns[1], color=st.session_state.unsup_labels.astype(str), title="Cluster Visualization")
            st.plotly_chart(fig, use_container_width=True)
    elif model_type == "PCA":
        st.subheader("PCA Components")
        st.write(st.session_state.X_reduced)
        fig = px.scatter(x=st.session_state.X_reduced[:,0], y=st.session_state.X_reduced[:,1], title="PCA 2D Projection")
        st.plotly_chart(fig, use_container_width=True)
    elif model_type in ["Isolation Forest", "One-Class SVM"]:
        st.subheader("Anomaly Detection Results")
        st.write(pd.Series(st.session_state.unsup_labels, name="Anomaly").value_counts())
        st.write("Anomaly labels:", st.session_state.unsup_labels)

if __name__ == "__main__":
    main()