# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
data = pd.read_csv('car_data.csv')

# Exploratory Data Analysis (EDA)
print(data.info())
print(data.describe())
print(data.head())

# Data Preprocessing
# Handling missing values (simple imputation example)
data.fillna(data.mean(), inplace=True)

# Feature and target separation
X = data.drop('price', axis=1)
y = data['price']

# Column transformer for preprocessing
numeric_features = ['year', 'mileage', 'horsepower']
categorical_features = ['brand', 'model', 'transmission', 'fuel', 'color']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Define the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print('MAE:', mean_absolute_error(y_test, y_pred))
print('MSE:', mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))

# Save the model
import joblib
joblib.dump(model, 'car_price_model.pkl')
