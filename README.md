Vehicle Price Prediction Project
Overview
This repository contains a Jupyter Notebook (vehicle_price_prediction.ipynb) that implements a machine learning model to predict vehicle prices based on various features such as make, model, year, mileage, and more. The solution uses a RandomForestRegressor from scikit-learn and includes data preprocessing, feature engineering, model training, evaluation, and visualization.
Features

Data cleaning and handling of missing values.
Encoding of categorical variables (e.g., make, model, transmission).
Feature scaling using StandardScaler.
Training and evaluation of a RandomForestRegressor model.
Visualization of feature importance.
Prediction capability for new vehicle data.
Model and preprocessing objects saved for reuse.

Requirements

Python 3.9 or higher
Libraries and versions:
pandas >= 1.5.0
numpy >= 1.23.0
scikit-learn >= 1.2.0
matplotlib >= 3.6.0
joblib >= 1.2.0


Dataset: dataset.csv (place it in the same directory as the notebook)

Install dependencies using:
pip install pandas>=1.5.0 numpy>=1.23.0 scikit-learn>=1.2.0 matplotlib>=3.6.0 joblib>=1.2.0

Usage

Clone the repository:git clone https://github.com/vishavjit-singh/vehicle-price-prediction.git


Navigate to the project directory:cd vehicle-price-prediction


Ensure dataset.csv is in the directory.
Open and run vehicle_price_prediction.ipynb in Jupyter Notebook or any compatible environment (Jupyter Notebook >= 6.4.0 recommended).
The notebook will train the model, display results, and save the model/preprocessing objects.

Files

vehicle_price_prediction.ipynb: The main Jupyter Notebook with the prediction model (tested with Jupyter Notebook 6.4.0).
vehicle_price_model.pkl: Saved RandomForestRegressor model.
scaler.pkl and le_*.pkl: Saved preprocessing objects (scaler and label encoders).
dataset.csv: Input dataset (to be provided by the user).

Results

The model evaluates performance using Mean Squared Error (MSE) and R² Score.
Feature importance is visualized to show which factors most influence price predictions.
Example prediction for a new vehicle is included in the notebook output.

Contributing
Feel free to fork this repository, make improvements, and submit pull requests. Issues and suggestions are welcome!
License
This project is licensed under the MIT License - see the LICENSE file for details (add a LICENSE file if desired).
Acknowledgments

Built using scikit-learn, pandas, and matplotlib.
Inspired by vehicle price prediction challenges.

Version Information

Project Version: 1.0.0 (Initial release)
Last Updated: July 01, 2025
