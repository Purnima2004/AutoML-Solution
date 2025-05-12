# 🚀 AutoML Solution – Streamlit App

## 🔍 What is this?
Hey there!
This is my all-in-one AutoML web app built using Streamlit – designed to make training and evaluating machine learning models super easy. No heavy coding needed! Just upload your dataset, choose your model, tweak a few things, and boom – you’ve got results, metrics, and downloadable models. Perfect for students, beginners, or anyone who wants a quick ML pipeline.

## 🧠 What You Can Do with It

- 📁 **Upload CSVs** – Instantly see data previews, missing values, and stats
- 🛠️ **Preprocessing** – Auto handles encoding, scaling, and train-test split
- ⚙️ **Model Selection** – Choose from:
  - SVM
  - Decision Tree
  - Bagging
  - Random Forest
  - AdaBoost
  - XGBoost
  - Neural Network (MLP)
- 🎯 **Hyperparameter Tuning** – Adjust parameters interactively
- 📊 **Evaluation** – Accuracy, F1, recall, precision, classification report
- 💾 **Download Models** – Trained model & preprocessor in .pkl
- 🔮 **Prediction** – Upload new data and get predictions instantly
- ⚖️ **Class Imbalance Warnings** – Handles imbalanced classes
- 🌗 **Modern UI** – Light/Dark theme toggle + clean layout

## 🌈 App Preview
Here’s a look at the UI:

<!-- Replace this path with your actual image file location -->

## 🧑‍💻 How It Works (Quick Guide)
- Go to Data Upload, add your CSV, and choose the target column
- App handles preprocessing (imputation, encoding, scaling)
- Head to Model Selection, pick your algorithm + hyperparameters
- Click Train – App runs RandomizedSearchCV for tuning
- View your metrics and download the trained model
- Use the Prediction page to upload fresh data for predictions

## 📁 File Breakdown
```
├── app.py                    # Main Streamlit app
├── preprocessing_hackathon.py # Data preprocessing logic
├── model_definations.py     # ML models and hyperparameters
├── main_file.py             # Optional CLI/Notebook version
├── models/                  # Saved models (.pkl)
├── requirements.txt         # Dependencies
└── README.md                # This file
```

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
Now just follow the link Streamlit gives you and you're good to go!

## 📌 A Few Notes
- Models and preprocessors are saved in models/ as .pkl files
- Supports custom hyperparameter ranges via the UI
- Handles imbalanced class detection automatically
- UI is clean, responsive, and beginner-friendly

## 🔁 Full Example Flow
- Upload your dataset
- Pick the target column
- Preprocess with a click
- Select a model + set hyperparameters
- Train and evaluate
- Download model / make predictions on new data

Feel free to fork or star the repo if you like it – and contributions are always welcome. 😄
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

---

Feel free to fork or star the repo if you find it useful—contributions are always welcome! 😄
5. After training, view results and download the model.
6. Use the "Results" page to upload new data and get predictions.


