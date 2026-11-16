# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load dataset, encode target (Placed = 1, Not Placed = 0), select features, and normalize using StandardScaler.

2.Initialize parameters (θ), define sigmoid and cost function, and add bias term to input.

3.Apply gradient descent: compute predictions, calculate gradient, and update θ iteratively.

4.Predict outputs using learned θ, compute accuracy, and plot cost vs iterations.  

## Program:
```
*/

Developed by: MEGANATHAN R
Register Number: 212224230156


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Create Dataset
data = {
    "R&D Spend": [165349, 162598, 153441, 144372, 142107,
                  131877, 134615, 130298, 120542, 123334],
    "Profit": [192261, 191792, 191050, 182902, 166187,
               156991, 156122, 155752, 152211, 149760]
}

df = pd.DataFrame(data)

# Save Dataset
df.to_csv("50_Startups.csv", index=False)

# Linear Regression using Gradient Descent
def linear_regression(X1, y, learning_rate=0.1, num_iters=1000):

    # Add Bias Column
    X = np.c_[np.ones((len(X1), 1)), X1]

    # Initialize Theta
    theta = np.zeros((X.shape[1], 1))

    # Gradient Descent
    for i in range(num_iters):
        predictions = X.dot(theta)
        errors = predictions - y
        gradients = (1 / len(X1)) * X.T.dot(errors)
        theta = theta - learning_rate * gradients

    return theta

# Load Dataset
data = pd.read_csv("50_Startups.csv")

# Select Input and Output
X1 = data[["R&D Spend"]].values
y = data[["Profit"]].values

# Standardize Input
scaler = StandardScaler()
X1_scaled = scaler.fit_transform(X1)

# Train Model
theta = linear_regression(X1_scaled, y)

# Display Parameters
print("Theta:")
print(theta)

# Predict Profit
X = np.c_[np.ones((len(X1_scaled), 1)), X1_scaled]
predictions = X.dot(theta)

# Actual vs Predicted
results = pd.DataFrame({
    "Actual Profit": y.flatten(),
    "Predicted Profit": predictions.flatten()
})

print("\nActual vs Predicted Profit:")
print(results)

```

## Output:
<img width="393" height="268" alt="image" src="https://github.com/user-attachments/assets/f643af3e-49c3-4f71-8545-42dd786c7992" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
