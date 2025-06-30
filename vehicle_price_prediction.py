import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# Load the dataset
data = pd.read_csv('dataset.csv')

# Data Cleaning
# Remove rows with missing price values
data = data.dropna(subset=['price'])

# Convert price to numeric, handling non-numeric values
data['price'] = pd.to_numeric(data['price'], errors='coerce')
data = data.dropna(subset=['price'])

# Feature Engineering
# Encode categorical variables
le_make = LabelEncoder()
le_model = LabelEncoder()
le_transmission = LabelEncoder()
le_fuel = LabelEncoder()
le_body = LabelEncoder()
le_drivetrain = LabelEncoder()

data['make_encoded'] = le_make.fit_transform(data['make'])
data['model_encoded'] = le_model.fit_transform(data['model'])
data['transmission_encoded'] = le_transmission.fit_transform(data['transmission'])
data['fuel_encoded'] = le_fuel.fit_transform(data['fuel'])
data['body_encoded'] = le_body.fit_transform(data['body'])
data['drivetrain_encoded'] = le_drivetrain.fit_transform(data['drivetrain'])

# Select features for prediction
features = ['year', 'make_encoded', 'model_encoded', 'cylinders', 'mileage', 
            'transmission_encoded', 'fuel_encoded', 'body_encoded', 'drivetrain_encoded']
X = data[features]
y = data['price']

# Handle missing values in numerical columns
X = X.fillna(X.mean())

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')

# Feature Importance
feature_importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance in Price Prediction')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Example prediction for a new vehicle
new_vehicle = pd.DataFrame({
    'year': [2024],
    'make_encoded': [le_make.transform(['Jeep'])[0]],
    'model_encoded': [le_model.transform(['Wagoneer'])[0]],
    'cylinders': [6],
    'mileage': [10],
    'transmission_encoded': [le_transmission.transform(['8-Speed Automatic'])[0]],
    'fuel_encoded': [le_fuel.transform(['Gasoline'])[0]],
    'body_encoded': [le_body.transform(['SUV'])[0]],
    'drivetrain_encoded': [le_drivetrain.transform(['Four-wheel Drive'])[0]]
})
new_vehicle_scaled = scaler.transform(new_vehicle)
predicted_price = model.predict(new_vehicle_scaled)
print(f'Predicted Price for new vehicle: ${predicted_price[0]:,.2f}')

# Save the model and preprocessing objects
joblib.dump(model, 'vehicle_price_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le_make, 'le_make.pkl')
joblib.dump(le_model, 'le_model.pkl')
joblib.dump(le_transmission, 'le_transmission.pkl')
joblib.dump(le_fuel, 'le_fuel.pkl')
joblib.dump(le_body, 'le_body.pkl')
joblib.dump(le_drivetrain, 'le_drivetrain.pkl')

print("Model and preprocessing objects saved successfully.")