import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

# Load dataset
file_path = "Rainfall.csv"
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


# Prediction function
def predict_rainfall(subdivision, year, month):
    sub_encoded = label_encoder.transform([subdivision])[0]
    month_num = pd.to_datetime(month, format="%b").month
    pred = model.predict(np.array([[sub_encoded, year, month_num]]))
    return pred[0]


# Example usage
subdivision_input = "ANDAMAN & NICOBAR ISLANDS"
year_input = 2025
month_input = "JAN"
predicted_rainfall = predict_rainfall(subdivision_input, year_input, month_input)
print(
    f"Predicted Rainfall for {subdivision_input} in {month_input} {year_input}: {predicted_rainfall:.2f} mm"
)
