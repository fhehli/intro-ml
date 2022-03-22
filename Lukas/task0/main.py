import pandas as pd
import numpy as np
from numpy.linalg import inv

#import the csv files
train_df = pd.read_csv("train.csv")
train_df = train_df.set_index('Id')
test_df = pd.read_csv("test.csv")
test_df = test_df.set_index('Id')

#Constructing the model via stationary point condition for linear regression

x_train_df = train_df.iloc[:, 1:]
y_train_df = train_df.iloc[:, 0]

x = x_train_df.to_numpy()
y = y_train_df.to_numpy()
x_transpose = np.transpose(x)

#Using the linear regression formula to calculate the weights

weights = np.matmul(inv(np.matmul(x_transpose, x)), np.matmul(x_transpose, y))

print(weights)
#Predict with the weights


test_df['y'] = test_df.mean(0)

export_df = test_df[['y']]

export_df.to_csv('submission4.csv')
