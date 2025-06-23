# 🚀 AutoML Solution – Streamlit App

![MIT License](https://img.shields.io/badge/license-MIT-green)
![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-orange)

## 🔍 What is this?
**AutoML Solution** is a comprehensive, interactive web application built with Streamlit that empowers users to perform end-to-end machine learning workflows—**from data upload to model deployment**—with minimal coding. It supports both supervised and unsupervised learning, advanced preprocessing, dimensionality reduction, and robust evaluation, all through an intuitive UI.

Whether you're a student, data scientist, or business analyst, this app helps you:
- Rapidly prototype ML models
- Explore and visualize your data
- Compare algorithms and tune hyperparameters
- Download ready-to-use models for deployment

---

## 🧠 Features (In Depth)

### 📁 Flexible Data Upload
- **Single CSV Upload**: Upload one file and let the app automatically split it into training and testing sets.
- **Separate Train/Test Upload**: Upload distinct files for training and testing, ideal for competitions or real-world scenarios.
- **Target Column Selection**: Choose which column you want to predict for supervised tasks.
- **Data Preview & Stats**: Instantly see a sample of your data, column types, missing values, and duplicates.

### 🛠️ Smart Preprocessing
- **Missing Value Handling**: Remove rows with missing targets or fill them with mean/median (for numeric targets). All other missing values are imputed automatically.
- **Categorical Encoding**: One-hot encoding for categorical features, with robust handling of unseen categories.
- **Scaling**: Standardization of numeric features for optimal model performance.
- **Train/Test Split**: Stratified splitting for classification, with automatic fallback if classes are too rare.
- **Error Feedback**: Clear, actionable error messages for every step.

### ⚙️ Supervised Learning
- **Classification Models**: Decision Tree, Random Forest, SVM, XGBoost, Neural Network (MLP), Logistic Regression, Bagging, AdaBoost.
- **Regression Models**: Linear Regression, Decision Tree Regressor, Random Forest Regressor, SVR, XGBoost Regressor, Neural Network Regressor.
- **Hyperparameter Tuning**: Interactive sliders and selectors for all key parameters.
- **Pipeline Architecture**: All preprocessing, PCA, and modeling are handled in a single scikit-learn pipeline for reliability and reproducibility.
- **Class Imbalance Detection**: Warns and adapts if your data is imbalanced.

### 🧩 Unsupervised Learning
- **Clustering**: KMeans, DBSCAN, Agglomerative Clustering. Visualize clusters and their distribution.
- **Dimensionality Reduction**: PCA (Principal Component Analysis) with user-defined number of components. See 2D/3D projections of your data.
- **Anomaly Detection**: Isolation Forest and One-Class SVM for outlier and anomaly detection.
- **Results Visualization**: Interactive charts and tables for unsupervised results.

### 🔄 PCA as a Pipeline Step
- **Why PCA?**: Reduces dimensionality, speeds up training, and helps with memory issues on wide datasets.
- **How it works**: PCA is added as a step in the pipeline (after preprocessing), ensuring compatibility and preventing errors. You control the number of components.
- **No Data Loss**: Your original data remains intact; PCA is only applied during model training and inference.

### 📊 Evaluation & Results
- **Metrics**: Accuracy, F1, Recall, Precision, Confusion Matrix, Classification Report for classification; MSE, MAE, R2 for regression.
- **Batch & Single Prediction**: Upload new data for instant predictions, or enter values manually.
- **Downloadable Artifacts**: Trained models and preprocessors are saved as `.pkl` files for easy reuse.
- **Result Export**: Download predictions and results as CSV.

### 💾 Model Management
- **Save/Load**: Models and preprocessors are saved in the `models/` directory.
- **Versioning**: Keep multiple models for comparison and deployment.

### 🌗 Modern, Responsive UI
- **Light/Dark Theme**: Toggle for your preferred look.
- **Expandable Sections**: Keep the interface clean and focused.
- **Real-Time Feedback**: See progress, errors, and results instantly.

---

## 🌈 App Preview
![Watch the demo]([AutoML Pro demo (1).mov](https://youtu.be/4z0WebWXd_E))

---

## 🧑‍💻 How It Works: Step-by-Step Workflow

### **Supervised Learning (Classification/Regression)**
1. **Data Upload**: Choose single or separate CSVs. Select your target column.
2. **Preprocessing**: Handle missing values, encoding, and scaling automatically.
3. **(Optional) PCA**: Enable PCA and select the number of components for dimensionality reduction.
4. **Model Selection**: Pick your algorithm and set hyperparameters.
5. **Training**: Click to train. The app builds a pipeline and fits your model.
6. **Evaluation**: View metrics, confusion matrix, and download your model.
7. **Prediction**: Upload new data for batch or single predictions.

### **Unsupervised Learning (Clustering, PCA, Anomaly Detection)**
1. **Data Upload**: Upload your dataset (no target column needed).
2. **Algorithm Selection**: Choose clustering, dimensionality reduction, or anomaly detection.
3. **Configure Parameters**: Set number of clusters, PCA components, or anomaly detection settings.
4. **Run & Visualize**: See cluster assignments, PCA projections, or anomaly labels. Download results as needed.

---

## 📁 File Breakdown
```
├── app.py                    # Main Streamlit app
├── preprocessing_hackathon.py # Data preprocessing logic
├── model_definations.py      # ML models and hyperparameters
├── main_file.py              # Optional CLI/Notebook version
├── models/                   # Saved models (.pkl)
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

---

## 📦 Setup & Run Locally

**Clone This Repo**
```bash
git clone https://github.com/Purnima2004/AutoML-Solution.git
cd AutoML-Solution
```

**Install Requirements**
```bash
pip install -r requirements.txt
```

**Launch the App**
```bash
streamlit run app.py
```

---

## 🚀 Deployment

### **Local Machine**
- Follow the setup steps above. The app will open in your browser at `localhost:8501`.

### **Render (Recommended for Public Sharing)**
1. Push your code to a GitHub repository.
2. Go to [Render.com](https://render.com/) and create a new Web Service.
3. Connect your repo and set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Deploy and get a public URL to share your app!

### **Streamlit Community Cloud**
- Go to [streamlit.io/cloud](https://streamlit.io/cloud), connect your repo, and deploy for free.

---

## 🧩 Unsupervised Learning & PCA (Details)
- **Clustering**: Visualize and analyze clusters, download assignments, and explore cluster statistics.
- **PCA**: Reduce your data to 2D/3D for visualization or to speed up downstream models. Integrated as a pipeline step for best practices.
- **Anomaly Detection**: Identify outliers and anomalies in your data, with clear labeling and export options.

---

## 📌 Notes & Best Practices
- All preprocessing and PCA are handled in the pipeline for robust, error-free ML.
- UI provides clear feedback and error messages at every step.
- Handles imbalanced classes and rare categories gracefully.
- Supports both single and batch predictions.
- Clean, modern UI for a smooth user experience.
- Models and preprocessors are saved in `models/` as `.pkl` files for easy reuse.

---

## 💡 Example Use Cases
- **Students**: Quickly test and compare ML algorithms on class projects.
- **Data Scientists**: Prototype and benchmark models before production.
- **Business Analysts**: Explore data, run clustering, and generate insights without coding.
- **Hackathons**: Build and evaluate models in minutes, not hours.

---

## 🔁 Example Workflow
1. Upload your dataset (single or separate train/test)
2. Select the target column (for supervised tasks)
3. Handle missing values as prompted
4. Preprocess and (optionally) enable PCA
5. Select and configure your model
6. Train and evaluate
7. Download model or make predictions on new data
8. For unsupervised tasks, upload data and select algorithm

---

## 🙌 Contributing & Feedback
We welcome contributions, suggestions, and bug reports! Feel free to fork, star, or open an issue/PR. If you have ideas for new features or want to improve the UI, your input is appreciated.

---

## 📄 License
MIT License


