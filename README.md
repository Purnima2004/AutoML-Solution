

# 🚀 AutoML Solution – Streamlit App (v2.0)

> This project has been modernized with a `src` layout and the lightning-fast `uv` package manager!

## 🔍 What is this?
Hey there!
This is my all-in-one AutoML web app built using Streamlit – designed to make training and evaluating machine learning models super easy. No heavy coding needed! Just upload your dataset, choose your model, tweak a few things, and boom – you’ve got results, metrics, and downloadable models. Perfect for students, beginners, or anyone who wants a quick ML pipeline.

## 🧠 What You Can Do with It

- 📁 **Upload CSVs** – Instantly see data previews, missing values, and stats.
- 🛠️ **Preprocessing** – Auto handles encoding, scaling, and train-test split.
- ⚙️ **Model Selection** – Choose from:
  - SVM
  - Decision Tree
  - Bagging
  - Random Forest
  - AdaBoost
  - Gradient Boosting
  - XGBoost
  - Neural Network (MLP)
- 🎯 **Hyperparameter Tuning** – Adjust parameters interactively.
- 📊 **Evaluation** – Accuracy, F1, recall, precision, classification report.
- 💾 **Download Models** – Trained model & preprocessor in `.pkl` format.
- 🔮 **Prediction** – Upload new data and get predictions instantly.
- ⚖️ **Class Imbalance Warnings** – Automatically detects and handles imbalanced classes.
- 🌗 **Modern UI** – Clean, responsive layout.

## 🌈 App Preview
Here’s a look at the UI:



## 🏗️ Project Updates & New Structure

This project has been significantly refactored to use a modern Python stack, improving maintainability, performance, and reproducibility.

### Key Changes
- **Source Layout**: Code is now organized in a `src/` directory.
- **Dependency Management**: Migrated from `pip` and `requirements.txt` to **`uv`** and `pyproject.toml` for 10-100x faster dependency resolution.
- **Code Refactoring**: Legacy scripts have been removed and replaced with modular, class-based components.
- **Python Version**: Updated to support Python 3.13+.

### 📁 New File Breakdown

The project now follows a more standard, scalable structure:

```
├── src/
│   ├── core/                   # Core logic for data processing & model training
│   ├── ui/                     # Streamlit UI components
│   └── config/                 # Model definitions & hyperparameter grids
├── models/                     # Directory for saved models (.pkl)
├── pyproject.toml              # Project metadata & dependencies for uv
├── uv.lock                     # Auto-generated lock file for reproducible builds
├── run.py                      # Main application entry point
└── README.md                   # This file
```

### From Old to New

| Old File (`v1`)              | New Location (`v2`)            | Reason for Change                               |
| ---------------------------- | ------------------------------ | ----------------------------------------------- |
| `app.py`                     | `src/ui/streamlit_app.py`      | UI logic is now encapsulated in a class.        |
| `preprocessing_hackathon.py` | `src/core/data_processor.py`   | Modernized, robust data processing pipeline.    |
| `model_definations.py`       | `src/config/model_configs.py`  | Centralized model & hyperparameter management.  |
| `requirements.txt`           | `pyproject.toml`               | Using `uv` for modern dependency management.    |
| `main_file.py`               | (Removed)                      | Replaced by the streamlined `run.py` launcher.  |

## 📦 Setup & Run Locally with `uv`

Get the app running in under a minute with `uv`, the fast Python package installer & resolver.

**1. Install `uv`** (if you don't have it)
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**2. Clone This Repo**
```bash
git clone https://github.com/Purnima2004/AutoML-Solution.git
cd AutoML-Solution
```

**3. Install Dependencies**

`uv` will create a virtual environment and install all dependencies from the `uv.lock` file, ensuring a perfect, reproducible setup.

```bash
uv sync
```

**4. Launch the App**

Run the app using `uv run`, which automatically uses the project's virtual environment.

```bash
uv run streamlit run run.py
```
And that's it! The app should now be open in your browser.

## 📌 A Few Notes
- Models and preprocessors are saved in the `models/` directory as `.pkl` files.
- Supports custom hyperparameter ranges directly via the UI.
- The app automatically detects and provides warnings for class imbalance.
- The UI is clean, responsive, and designed to be beginner-friendly.

## 🔁 Full Example Flow
The user workflow remains as simple as ever:
1.  **Upload** your dataset.
2.  **Pick** the target column.
3.  **Preprocess** with a single click.
4.  **Select** a model and set hyperparameters.
5.  **Train** and evaluate your model.
6.  **Download** the best model or make predictions on new data.

Feel free to fork or star the repo if you like it – and contributions are always welcome. 😄
