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

#1.Fruad vs Normal Transactions
plt.figure(figsize=(7,5))
sns.countplot(data=df,x='Class')
plt.title("Fruad vs Narmal Transsaction")
plt.xlabel("Transaction Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Number of Transactions")

plt.savefig(REPORT_DIR / "fraud_distribution.png")
plt.show()

#2.Fraud Percentage
print("\nFraud Percentage:")
print(df['Class'].value_counts(normalize=True)*100)


#3.Transaction Amount Distribution
plt.figure(figsize=(7,5))

sns.histplot(data=df, x="Amount", bins=50)

plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Distribution")

plt.savefig(REPORT_DIR / "amount_distribution.png")
plt.show()

#4.Fraud vs Normal Transaction Amount
plt.figure(figsize=(8,5))

sns.boxplot(data=df,x="Class",y="Amount")

plt.title("Fraud vs Normal Transaction Ammount")
plt.xlabel("Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Amount")

plt.savefig(REPORT_DIR / "amount_by_class.png")

#5.Correlation Heatmap
plt.figure(figsize=(15,10))
correlation=df.corr()
sns.heatmap(correlation, cmap="coolwarm",annot=False)
plt.title("Feature Correlation Heatmap")

plt.savefig(REPORT_DIR / "correlation_heatmap.png")
plt.show()

print("\nEDA Completed Successfully!")