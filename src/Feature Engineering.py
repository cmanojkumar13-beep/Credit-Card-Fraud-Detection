from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load raw dataset and remove duplicates
DATA_PATH = BASE_DIR / "data" / "raw" / "creditcard.csv"

df = pd.read_csv(DATA_PATH)
df.drop_duplicates(inplace=True)

# Create report folder
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

#-------------------------------
#Handeling Null Values
#--------------------------------
print(df.isnull().sum())



# Select numeric columns
numeric_cols = df.select_dtypes(include="number").columns.tolist()

# Exclude Class because it is the target variable
numeric_cols.remove("Class")

n = len(numeric_cols)
cols = 3
rows = math.ceil(n / cols)

# Create subplots
fig, axes = plt.subplots(rows, cols,figsize=(18, rows * 3))
axes = axes.flatten()
# Plot boxplots
for i, col in enumerate(numeric_cols):

    sns.boxplot(data=df,x=col,ax=axes[i],color="steelblue")

    axes[i].set_title(f"{col} - Outlier Detection", fontsize=10)

# Remove empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Outlier Detection - All Numerical Features", fontsize=16)
plt.tight_layout()
plt.savefig(REPORT_DIR / "handling_nulvalues`.png")
plt.show()


#-------------------------------------
#Handling Outliers
#-----------------------------------

# Outlier detection using IQR
numeric_cols = df.select_dtypes(include="number").columns.drop("Class")

outlier_summary = []

for col in numeric_cols:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    count = (
        (df[col] < lower) |
        (df[col] > upper)
    ).sum()

    outlier_summary.append({
        "Feature": col,
        "Outliers": count
    })

# Display outlier summary
outlier_df = pd.DataFrame(outlier_summary)

print(outlier_df)

print("\nTotal rows preserved:", len(df))



#graph


# Select numerical columns
numeric_cols = df.select_dtypes(include="number").columns.tolist()

# Remove target column
numeric_cols.remove("Class")

# Create subplot layout
n = len(numeric_cols)
cols = 3
rows = math.ceil(n / cols)

fig, axes = plt.subplots(
    rows, cols,
    figsize=(18, rows * 3)
)

axes = axes.flatten()

# Draw boxplots
for i, col in enumerate(numeric_cols):

    sns.boxplot(
        data=df,
        x=col,
        ax=axes[i],
        color="steelblue",
        showfliers=True
    )

    axes[i].set_title(f"{col} - Outliers", fontsize=10)
    axes[i].set_xlabel(col)

# Remove empty plots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle(
    "Outlier Detection - All Numerical Features",
    fontsize=16
)

plt.tight_layout()
plt.savefig(REPORT_DIR / "handling_outliers.png")
plt.show()

#Feature Creation 

# Create transaction time in hours
df["Time_Hours"] = (df["Time"] / 3600) % 24

print(df[["Time", "Time_Hours"]].head())

correlation = df.corr()["Class"].drop("Class")

plt.figure(figsize=(12, 8))

correlation.sort_values().plot(kind="barh", color="steelblue")

plt.title("Feature Correlation with Fraud Class")
plt.xlabel("Correlation")
plt.tight_layout()
plt.savefig(REPORT_DIR / "correlation")
plt.show()

#--------------------------------------
#Train_Test_Split
#--------------------------------------
x=df.drop("Class", axis=1)
y=df["Class"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=40)
print(x_train.shape)
print(x_test.shape)

#----------------------------
#Scaling
#----------------------------
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# Select columns to scale
scale_columns = ["Amount", "Time"]

# Create copies
x_train = x_train.copy()
x_test = x_test.copy()

# Fit and transform training data
x_train[scale_columns] = sc.fit_transform(
    x_train[scale_columns]
)

# Transform testing data
x_test[scale_columns] = sc.transform(
    x_test[scale_columns]
)

#---------------------------
#Handle Imbalanced Data
#--------------------------

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

x_train, y_train = smote.fit_resample(
    x_train, y_train
)

print(y_train.value_counts())