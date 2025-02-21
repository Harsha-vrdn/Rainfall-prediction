import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

# Load dataset
file_path = "Rainfall.csv"  # Update path if needed
df = pd.read_csv(file_path)

# Reshape data: Convert wide format to long format
df_melted = df.melt(
    id_vars=["SUBDIVISION", "YEAR"],
    value_vars=[
        "JAN",
        "FEB",
        "MAR",
        "APR",
        "MAY",
        "JUN",
        "JUL",
        "AUG",
        "SEP",
        "OCT",
        "NOV",
        "DEC",
    ],
    var_name="MONTH",
    value_name="RAINFALL",
)

# Convert month names to numbers
df_melted["MONTH"] = pd.to_datetime(df_melted["MONTH"], format="%b").dt.month

# Handle missing values
df_melted["RAINFALL"].fillna(
    df_melted.groupby(["SUBDIVISION", "MONTH"])["RAINFALL"].transform("mean"),
    inplace=True,
)

# Encode subdivision names
label_encoder = LabelEncoder()
df_melted["SUBDIVISION"] = label_encoder.fit_transform(df_melted["SUBDIVISION"])

# Define features and target
X = df_melted[["SUBDIVISION", "YEAR", "MONTH"]]
y = df_melted["RAINFALL"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")

# Ensure model directory exists
model_dir = "prediction/models"
os.makedirs(model_dir, exist_ok=True)

# Save the trained model and label encoder
joblib.dump(model, os.path.join(model_dir, "rainfall_model.pkl"))
joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder.pkl"))

print("Model and label encoder saved successfully!")
