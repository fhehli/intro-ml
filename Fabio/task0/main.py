import pandas as pd

# import the csv file
df = pd.read_csv("./task0/test.csv", index_col=0)

# Predict with the weights
y_pred = df.mean(axis=1)

# export
y_pred.to_csv("submission.csv")
