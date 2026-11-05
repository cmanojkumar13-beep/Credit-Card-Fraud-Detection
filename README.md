# 💳 Credit Card Fraud Detection Using Machine Learning

## 📌 1. Project Overview

Credit Card Fraud Detection is a Machine Learning project that identifies fraudulent credit card transactions and distinguishes them from genuine transactions.

The objective is to build and compare multiple Machine Learning models to identify suspicious transactions and select the best-performing model.

## 🎯 2. Problem Statement

Credit card fraud is a major financial problem. Detecting fraudulent transactions accurately helps reduce financial losses and improve transaction security.

This project classifies transactions into:

- **0 – Genuine Transaction**
- **1 – Fraudulent Transaction**

## 🛠️ 3. Technologies Used

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, XGBoost
- **Model Saving:** Joblib
- **Web Application:** Streamlit
- **Development Environment:** VS Code, Jupyter Notebook
- **Version Control:** Git, GitHub

## 📂 4. Dataset Information

- **Dataset:** Credit Card Fraud Detection
- **Source:** Kaggle
- **File:** `creditcard.csv`
- **Target Variable:** `Class`

Dataset features include:

- Time
- V1 to V28
- Amount
- Class

The dataset is highly imbalanced, with fraudulent transactions representing a small percentage of total transactions.

**Note:** The dataset CSV files are maintained separately from the GitHub repository if they exceed GitHub's standard file size limit. Download the original dataset from Kaggle to reproduce the project.

## 🔄 5. Project Workflow

### Step 1: Data Collection
Collected the Credit Card Fraud Detection dataset from Kaggle.

### Step 2: Data Preprocessing
- Loaded the dataset using Pandas.
- Checked missing values.
- Checked duplicate records.
- Handled missing values where required.
- Analyzed and handled outliers.

### Step 3: Exploratory Data Analysis (EDA)
- Analyzed numerical features.
- Studied fraud and genuine transaction distributions.
- Visualized correlations.
- Identified patterns in fraudulent transactions.

### Step 4: Feature Engineering
- Selected relevant features.
- Handled outliers.
- Prepared features for Machine Learning.
- Separated independent and dependent variables.

### Step 5: Train-Test Split
Split the dataset into training and testing datasets.

### Step 6: Model Training
Implemented the following Machine Learning algorithms:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost

### Step 7: Hyperparameter Tuning
Used GridSearchCV to identify suitable model parameters and improve model performance.

### Step 8: Model Comparison
Compared different models using evaluation metrics and selected the best-performing model.

### Step 9: Model Evaluation
Evaluated the selected model using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC
- Precision-Recall AUC

### Step 10: Feature Importance
Analyzed feature importance to understand the contribution of input variables to fraud prediction.

### Step 11: Prediction System
Saved the best-performing model using Joblib and implemented a prediction system.

### Step 12: Web Application
Developed a Streamlit application to predict whether a credit card transaction is fraudulent or genuine.

## 🤖 6. Machine Learning Algorithms

| Algorithm | Purpose |
|---|---|
| Logistic Regression | Binary classification |
| Decision Tree | Rule-based classification |
| Random Forest | Ensemble learning |
| Gradient Boosting | Sequential ensemble learning |
| XGBoost | Optimized gradient boosting |

## 📊 7. Evaluation Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Overall correct predictions |
| Precision | Correct fraud predictions among predicted fraud |
| Recall | Fraud cases correctly identified |
| F1-Score | Balance between precision and recall |
| ROC-AUC | Classification performance across thresholds |
| PR-AUC | Performance on imbalanced datasets |

## 📁 8. Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── creditcard.csv
│   │
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── models/
│   └── best_fraud_model.pkl
│
├── reports/
│   └── model_comparison_results.csv
│
├── src/
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_comparison.py
│   ├── model_evaluation.py
│   ├── feature_importance.py
│   └── predict.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## ⚙️ 9. Installation and Setup

### Clone the Repository

```bash
git clone https://github.com/cmanojkumar13-beep/Credit-Card-Fraud-Detection.git
```

### Navigate to Project Folder

```bash
cd Credit-Card-Fraud-Detection
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Virtual Environment (Windows)

```powershell
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Dataset Setup:** Download `creditcard.csv` from Kaggle and place it inside `data/raw/`. Generate the processed CSV files by running the preprocessing and feature engineering scripts in the appropriate order.

## ▶️ 10. Run the Application

Run the Streamlit application:

```bash
streamlit run app/app.py
```

## 📈 11. Model Results

Model performance is compared using Precision, Recall, F1-Score, ROC-AUC, and PR-AUC.

| Evaluation Metric | Result |
|---|---|
| Best Model | To be updated |
| Precision | To be updated |
| Recall | To be updated |
| F1-Score | To be updated |
| ROC-AUC | To be updated |
| PR-AUC | To be updated |

*Replace the placeholders with actual model evaluation results.*

## 🚀 12. Future Enhancements

- Real-time fraud detection.
- Advanced anomaly detection.
- Improved handling of imbalanced datasets.
- Cloud deployment.
- Fraud transaction monitoring dashboard.
- API integration for real-time predictions.

## 🏁 13. Conclusion

This project demonstrates the application of Machine Learning algorithms for detecting fraudulent credit card transactions.

It covers the complete Machine Learning workflow, including data preprocessing, exploratory data analysis, feature engineering, model training, hyperparameter tuning, model comparison, evaluation, and deployment through a Streamlit application.

## 👨‍💻 14. Author

**Manoj Kumar**

Aspiring AI/ML Engineer | Python | SQL | Machine Learning | Data Science

- **GitHub:** https://github.com/cmanojkumar13-beep
- **LinkedIn:** https://www.linkedin.com/in/manoj-kumar-c1234/
