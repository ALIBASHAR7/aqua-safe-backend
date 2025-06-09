import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib
from sklearn.preprocessing import StandardScaler
import os

# Get the current file path and adjust it to the data directory
current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, '..', 'data', 'water_quality.csv')
model_path = os.path.join(current_dir, 'water_quality_model.pkl')

# Load dataset
try:
    data = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"Error: Dataset not found at {data_path}. Please ensure 'water_quality.csv' is in the 'Backend/data/' directory.")
    exit()



# Select features and target variable
# Correct column names based on your app.py's required_fields
X = data[['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
          'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']]
y = data['Potability']

# Handle missing values by dropping rows (simple approach, consider imputation for production)
X = X.dropna()
y = y[X.index] # Align y with cleaned X

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------- Scale the Data -------------------
# It's important to scale if your model was trained on scaled data.
# If your original train_model.py did not include StandardScaler, you might want to adjust.
# For consistency, I'm adding it as it's a common preprocessing step for ML models.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ------------------- Random Forest Model -------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train) # Train on scaled data
rf_predictions = rf_model.predict(X_test_scaled) # Predict on scaled data

rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_f1 = f1_score(y_test, rf_predictions)

# Save Random Forest model
# Also save the scaler if you plan to use it for new predictions
joblib.dump(rf_model, model_path)
# joblib.dump(scaler, os.path.join(model_save_dir, 'scaler.pkl')) # You might want to save scaler too

# ------------------- Print Evaluation Results -------------------
print("Model Evaluation Results:")
print(f"Random Forest - Accuracy: {rf_accuracy * 100:.2f}%, F1 Score: {rf_f1:.2f}")
print(f"Model saved to: {model_path}")
