from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Get project folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset path
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "creditcard.csv")

print("Dataset Path:", DATA_PATH)

# Import dataset
df = pd.read_csv(DATA_PATH)

print("\nDisplay first 5 rows")
print(df.head())

print("\nDisplay last 5 rows")
print(df.tail())

print("\nDidplay null values")
print(df.isnull().sum())

print("\nDescribe all")
print(df.describe(include="all"))

print("\nDisplay Duplicated")
print(df.duplicated().sum())

print("\nRemove Duplicates")
print(df.drop_duplicates(inplace=True))

print("\n Display Shape")
print(df.shape)

print("\nDisplay Size")
print(df.size)