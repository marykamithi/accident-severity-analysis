import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib

#  Prepare a Sample Dataset
data = {
    'accident_severity': np.random.randint(1, 5, 100),
    'speed': np.random.randint(20, 100, 100),
    'weather_condition': np.random.choice(['clear', 'rainy', 'foggy', 'snowy'], 100),
    'road_condition': np.random.choice(['dry', 'wet', 'icy'], 100),
    'vehicle_condition': np.random.choice(['good', 'fair', 'poor'], 100),
    'time_of_day': np.random.choice(['morning', 'afternoon', 'evening', 'night'], 100)
}

df = pd.DataFrame(data)
print("Sample DataFrame:")
print(df.head())

# Specify the Dependent and Independent Variables
X = df.drop('accident_severity', axis=1)
y = df['accident_severity']

# Preprocess the Data and Create the Linear Regression Model
categorical_features = ['weather_condition', 'road_condition', 'vehicle_condition', 'time_of_day']
numeric_features = ['speed']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict using the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'\nMean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Save the Model for Future Use
joblib.dump(model, 'road_accident_severity_model.pkl')
print("\nModel saved as 'road_accident_severity_model.pkl'")

# Provide an Example Prediction
hypothetical_data = pd.DataFrame({
    'speed': [70],
    'weather_condition': ['rainy'],
    'road_condition': ['wet'],
    'vehicle_condition': ['fair'],
    'time_of_day': ['evening']
})

predicted_severity = model.predict(hypothetical_data)
print(f'\nPredicted Accident Severity for hypothetical data: {predicted_severity[0]}')


