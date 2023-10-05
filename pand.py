import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


data = pd.read_csv('Exercise1.csv')
#Load data


x_train = data['Data']
y_train = data['Validation']
x_test = data['Test']
#Extract the data and set them as training and testing values

def Model1(x):
    return x + 0.5

def Model2(x):
    return 0.8 * x**2 + 0.5
#Using the equations given in the exercise, generate two models

plt.scatter(x_train, y_train, label='Validation Data')
plt.plot(x_train, Model1(x_train), label='Model1', color='red')
plt.plot(x_train, Model2(x_train), label='Model2', color='green')
plt.xlabel('Data')
plt.ylabel('Validation')
plt.legend()
plt.title('Data and Models')
plt.show()
#Using the matplotlib library plot the points and create the graphs

y_pred_model1_val = Model1(x_train)
y_pred_model2_val = Model2(x_train)

rmse_model1_val = np.sqrt(mean_squared_error(y_train, y_pred_model1_val))
r2_model1_val = r2_score(y_train, y_pred_model1_val)
rmse_model2_val = np.sqrt(mean_squared_error(y_train, y_pred_model2_val))
r2_model2_val = r2_score(y_train, y_pred_model2_val)
#This quantifies Root Mean Squared Error and $R^2$


print("Model 1 (Validation Data):")
print("RMSE:", rmse_model1_val)
print("R^2:", r2_model1_val)
print("Model 2 (Validation Data):")
print("RMSE:", rmse_model2_val)
print("R^2:", r2_model2_val)
#Display all data
