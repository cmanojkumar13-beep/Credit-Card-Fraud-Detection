from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load raw dataset and remove duplicates
DATA_PATH = BASE_DIR / "data" / "raw" / "creditcard.csv"

df = pd.read_csv(DATA_PATH)
df.drop_duplicates(inplace=True)

# Create report folder
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

from sklearn.model_selection import train_test_split


# Separate input and output
x = df.drop("Class", axis=1)
y = df["Class"]

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

from pathlib import Path

# Create processed folder
processed_dir = Path("data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

# Save training and testing data
x_train.to_csv(processed_dir / "X_train.csv", index=False)
x_test.to_csv(processed_dir / "X_test.csv", index=False)

y_train.to_csv(processed_dir / "y_train.csv", index=False)
y_test.to_csv(processed_dir / "y_test.csv", index=False)

print("Train and test datasets saved successfully!")

#------------------------
#Logistic Regression
#------------------------
print("\nLogistic Regression")
from sklearn.linear_model import LogisticRegression
model_l=LogisticRegression(max_iter=1000)
model_l.fit(x_train,y_train)
#prediction
train_pred=model_l.predict(x_train)
test_pred=model_l.predict(x_test)

#Evaluation Metrics 
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix

#accuracy_score
print("\ntrain accuracy_score:",accuracy_score(y_train,train_pred))
print("\ntest accuracy_score:",accuracy_score(y_test,test_pred))

#precision_score
print("\ntrain presision_score:",precision_score(y_train,train_pred))
print("\ntest pression_score:",precision_score(y_test,test_pred))

#recall_score
print("\ntrain recall_score:",recall_score(y_train,train_pred))
print("\ntest recall_score:",recall_score(y_test,test_pred))

#f1_Score
print("\ntrain f1_score:",f1_score(y_train,train_pred))
print("\ntest f1_score:",f1_score(y_test,test_pred))

#classification_report
print("\ntrain classification_report:",classification_report(y_train,train_pred))
print("\ntest classification_report:",classification_report(y_test,test_pred))

#confusion_matrix
print("\ntrain confusion_matrix:",confusion_matrix(y_train,train_pred))
print("\ntest confusion_matrix:",confusion_matrix(y_test,test_pred))

#-------------------
#Decision Tree
#-------------------
print("\nDecision Tree")

from sklearn.tree import DecisionTreeClassifier
model_d=DecisionTreeClassifier(random_state=42)
model_d.fit(x_train,y_train)

#predict
train_pred=model_d.predict(x_train)
test_pred=model_d.predict(x_test)

#hyperparameters
dt_model = DecisionTreeClassifier(
    random_state=42,
    class_weight="balanced"
)

param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [5, 10, 15, 20, None]
}

#Evaluation Metrics 
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix

#accuracy_score
print("\ntrain accuracy_score:",accuracy_score(y_train,train_pred))
print("\ntest accuracy_score:",accuracy_score(y_test,test_pred))

#precision_score
print("\ntrain presision_score:",precision_score(y_train,train_pred))
print("\ntest pression_score:",precision_score(y_test,test_pred))

#recall_score
print("\ntrain recall_score:",recall_score(y_train,train_pred))
print("\ntest recall_score:",recall_score(y_test,test_pred))

#f1_Score
print("\ntrain f1_score:",f1_score(y_train,train_pred))
print("\ntest f1_score:",f1_score(y_test,test_pred))

#classification_report
print("\ntrain classification_report:",classification_report(y_train,train_pred))
print("\ntest classification_report:",classification_report(y_test,test_pred))

#confusion_matrix
print("\ntrain confusion_matrix:",confusion_matrix(y_train,train_pred))
print("\ntest confusion_matrix:",confusion_matrix(y_test,test_pred))

#----------------------------
#Random Forest
#----------------------------
print("\nRandom Forest")
from sklearn.ensemble import RandomForestClassifier
model_r=RandomForestClassifier()
model_r.fit(x_train,y_train)

#predict
train_pred=model_r.predict(x_train)
test_pred=model_r.predict(x_test)

#hyperperameter

from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    "n_estimators": [10, 20, 30]
}

rf = RandomForestClassifier(
    random_state=42
)

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=10,
    cv=3,
    scoring="average_precision",
    random_state=42,
    n_jobs=-1,
    verbose=1
)

random_search.fit(x_train, y_train)

print("Best Parameters:", random_search.best_params_)
print("Best PR-AUC:", random_search.best_score_)

#Evaluation Metrics 
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix

#accuracy_score
print("\ntrain accuracy_score:",accuracy_score(y_train,train_pred))
print("\ntest accuracy_score:",accuracy_score(y_test,test_pred))

#precision_score
print("\ntrain presision_score:",precision_score(y_train,train_pred))
print("\ntest pression_score:",precision_score(y_test,test_pred))

#recall_score
print("\ntrain recall_score:",recall_score(y_train,train_pred))
print("\ntest recall_score:",recall_score(y_test,test_pred))

#f1_Score
print("\ntrain f1_score:",f1_score(y_train,train_pred))
print("\ntest f1_score:",f1_score(y_test,test_pred))

#classification_report
print("\ntrain classification_report:",classification_report(y_train,train_pred))
print("\ntest classification_report:",classification_report(y_test,test_pred))

#confusion_matrix
print("\ntrain confusion_matrix:",confusion_matrix(y_train,train_pred))
print("\ntest confusion_matrix:",confusion_matrix(y_test,test_pred))


#-------------------------
#GradientBoostingClassifier
#-------------------------
from sklearn.ensemble import GradientBoostingClassifier
model_g=GradientBoostingClassifier()
model_g.fit(x_train,y_train)
#prediction 
train_pred=model_g.predict(x_train)
test_pred=model_g.predict(x_test)

#hypertuning perameters
from sklearn.model_selection import GridSearchCV

# Define hyperparameters
param_grid = {
    "n_estimators": [50, 100, 150],
    "learning_rate": [0.01, 0.05, 0.1]}

# Create model
gb_model = GradientBoostingClassifier(random_state=42)

# GridSearchCV
grid_search = GridSearchCV(
    estimator=gb_model,
    param_grid=param_grid,
    cv=3,
    scoring="average_precision",
    n_jobs=-1,
    verbose=1
)

# Train and tune
grid_search.fit(x_train, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Best model
best_gb = grid_search.best_estimator_

# Prediction
best_gb_pred = best_gb.predict(x_test)

# Evaluation
print("\nClassification Report:\n")
print(classification_report(y_test, best_gb_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, best_gb_pred))


#Evaluation Metrics 
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix

#accuracy_score
print("\ntrain accuracy_score:",accuracy_score(y_train,train_pred))
print("\ntest accuracy_score:",accuracy_score(y_test,test_pred))

#precision_score
print("\ntrain presision_score:",precision_score(y_train,train_pred))
print("\ntest pression_score:",precision_score(y_test,test_pred))

#recall_score
print("\ntrain recall_score:",recall_score(y_train,train_pred))
print("\ntest recall_score:",recall_score(y_test,test_pred))

#f1_Score
print("\ntrain f1_score:",f1_score(y_train,train_pred))
print("\ntest f1_score:",f1_score(y_test,test_pred))

#classification_report
print("\ntrain classification_report:",classification_report(y_train,train_pred))
print("\ntest classification_report:",classification_report(y_test,test_pred))

#confusion_matrix
print("\ntrain confusion_matrix:",confusion_matrix(y_train,train_pred))
print("\ntest confusion_matrix:",confusion_matrix(y_test,test_pred))

#--------------------------------
#XGBoost
#-------------------------------
print("\nXGBoost")
from xgboost import XGBClassifier
model_x=XGBClassifier()
model_x.fit(x_train,y_train)
#prediction 
train_pred=model_x.predict(x_train)
test_pred=model_x.predict(x_test)

#hypertuning perameters
from sklearn.model_selection import GridSearchCV

# Define hyperparameters
param_grid = {
    "n_estimators": [10, 20],
    "learning_rate": [0.01, 0.1]}

# Create model
xgb_model = XGBClassifier(
    random_state=42)

# GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=3,
    scoring="average_precision",
    n_jobs=-1,
    verbose=1
)

# Train and tune
grid_search.fit(x_train, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Best model
best_xgb = grid_search.best_estimator_

# Prediction
best_xgb_pred = best_xgb.predict(x_test)

#Evaluation Metrics 
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix

#accuracy_score
print("\ntrain accuracy_score:",accuracy_score(y_train,train_pred))
print("\ntest accuracy_score:",accuracy_score(y_test,test_pred))

#precision_score
print("\ntrain presision_score:",precision_score(y_train,train_pred))
print("\ntest pression_score:",precision_score(y_test,test_pred))

#recall_score
print("\ntrain recall_score:",recall_score(y_train,train_pred))
print("\ntest recall_score:",recall_score(y_test,test_pred))

#f1_Score
print("\ntrain f1_score:",f1_score(y_train,train_pred))
print("\ntest f1_score:",f1_score(y_test,test_pred))

#classification_report
print("\ntrain classification_report:",classification_report(y_train,train_pred))
print("\ntest classification_report:",classification_report(y_test,test_pred))

#confusion_matrix
print("\ntrain confusion_matrix:",confusion_matrix(y_train,train_pred))
print("\ntest confusion_matrix:",confusion_matrix(y_test,test_pred))







