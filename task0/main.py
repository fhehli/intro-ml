import pandas as pd
import numpy as np

# import the csv file
test_df = pd.read_csv("./task0/test.csv")
test_df = test_df.set_index('Id')

# Predict with the weights
test = test_df.to_numpy()
y_pred = np.mean(test, axis=1)

# export
test_df['y'] = y_hat
export_df = test_df[['y']]
export_df.to_csv('submission4.csv')
