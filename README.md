# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the Dataset
2.Split the Data
3.Normalize the Data
4.Train the Model Using SGD
5.Predict and Evaluate

## Program:
DEVELOPED BY : SRI HARI KRISHNA D T 
REGISTER NO  : 212224240160
```
import pandas as pd
from google.colab import files

# Upload dataset
uploaded = files.upload()
file_name = list(uploaded.keys())[0]
df = pd.read_csv(file_name)

# Display first few rows
df.head()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Selecting independent variables (features)
X = df[["CYLINDERS", "ENGINESIZE", "FUELCONSUMPTION_COMB"]]

# Selecting dependent variable (target)
y = df["CO2EMISSIONS"]

# Splitting dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing features (important for SGD)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Initialize and train the SGD model
sgd_regressor = SGDRegressor(max_iter=1000, tol=1e-3, learning_rate='optimal', random_state=42)
sgd_regressor.fit(X_train_scaled, y_train)

# Predict on test data
y_pred = sgd_regressor.predict(X_test_scaled)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)
print("Model Coefficients:", sgd_regressor.coef_)
print("Model Intercept:", sgd_regressor.intercept_)
```

## Output:
![image](https://github.com/user-attachments/assets/67c75e51-b692-4466-a793-50acdc18678e)



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
