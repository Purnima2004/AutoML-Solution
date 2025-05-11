# AutoML Solution

## Overview
This application is an interactive AutoML solution built with [Streamlit](https://streamlit.io/) to automate the process of training, evaluating, and deploying machine learning models. It allows users to upload datasets, preprocess data, select and tune machine learning algorithms, train models, and make predictions, all through an intuitive web interface. The app supports a variety of popular classification algorithms and is designed for ease of use, even for those with limited coding experience.

## Features
- **Data Upload & Analysis**: Upload your CSV dataset and get instant data preview, statistics, and missing value analysis.
- **Preprocessing**: Automated preprocessing pipeline including scaling, encoding, and splitting into train/test sets.
- **Model Selection**: Choose from multiple ML algorithms:
  - SVM
  - Decision Tree
  - Bagging
  - Random Forest
  - ADA Boost
  - XG Boost
  - Neural Network (MLP)
- **Hyperparameter Tuning**: Interactive UI for specifying hyperparameter search spaces for each model.
- **Training & Evaluation**: Train models with randomized search, view accuracy, precision, recall, F1 score, and classification report.
- **Download Models**: Download trained model and preprocessor for future use.
- **Prediction**: Upload new data for batch predictions and download results as CSV.
- **Class Imbalance Handling**: Warns about imbalanced classes and applies balancing strategies where possible.
- **Modern UI**: Light/dark theme support and visually appealing controls.

## How It Works
1. **Data Upload**: Go to the "Data Upload" page, upload your CSV, and select the target column.
2. **Preprocessing**: The app preprocesses your data (imputation, encoding, scaling, splitting).
3. **Model Selection**: Choose an ML algorithm and set hyperparameters.
4. **Training**: Train your model with a single click. The app performs hyperparameter search and displays the best results.
5. **Results**: View evaluation metrics and download the trained model/preprocessor.
6. **Prediction**: Upload new data (without target column) and get predictions instantly.

## File Structure
- `app.py`: Main Streamlit application with all UI and workflow logic.
- `preprocessing_hackathon.py`: Data preprocessing functions.
- `model_definations.py`: Model definitions and hyperparameter grids.
- `main_file.py`: (CLI/Notebook) Example of pipeline usage.
- `models/`: Saved trained models and preprocessors.
- `requirements.txt`: Python dependencies.
- `README.md`: This file.

## Requirements
- Python 3.8+
- See `requirements.txt` for all required packages.

## Getting Started
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the app**:
   ```bash
   streamlit run app.py
   ```
3. **Open in your browser**: Follow the Streamlit link. You can now run the code locally on your server.

## Notes
- **Model Saving**: Trained models and preprocessors are saved in the `models/` directory as `.pkl` files.
- **Custom Hyperparameters**: Use the UI to specify ranges/values for hyperparameter search.
- **Imbalanced Data**: The app detects and warns about class imbalance, applying balancing techniques where possible.
- **Error Handling**: All steps include error messages for common issues (file not found, invalid input, etc).

## Example Workflow
1. Upload your dataset on the "Data Upload" page.
2. Select the target column and preprocess.
3. Go to "Model Selection", pick an algorithm, and set hyperparameters.
4. Proceed to "Training" and start model training.
5. After training, view results and download the model.
6. Use the "Results" page to upload new data and get predictions.


