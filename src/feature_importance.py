
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance

# Load best model
model = joblib.load("models/best_fraud_model.pkl")

# Load test data
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

# Calculate feature importance using permutation
importance = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=5,
    random_state=42,
    scoring="average_precision",
    n_jobs=-1
)

# Create importance DataFrame
feature_importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": importance.importances_mean
})

# Sort features
feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance_df)

# Plot top 15 features
top_features = feature_importance_df.head(15)

plt.figure(figsize=(10, 6))
plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Permutation Importance")
plt.ylabel("Features")
plt.title("Top 15 Important Features")

plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig("models/feature_importance.png")
plt.show()

# Save results
feature_importance_df.to_csv(
    "models/feature_importance.csv",
    index=False
)

print("\nFeature Importance Analysis Completed!")
