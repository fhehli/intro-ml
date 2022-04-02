import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from statistics import mean


# read data
df = pd.read_csv("train.csv")

#randomize order

df = df.sample(frac=1)

#split the dataframe into equal parts

df_split = np.array_split(df, 10)

lambdas = [0.1, 1, 10, 100, 200]

rmses = []

for lambdai in lambdas:
    lambda_rmse = []
    for i in range(10):
        frames = []
        for j in range(10):
            if j!=i:
                frames.append(df_split[j])
        train_df = pd.concat(frames)
        test_df = df_split[i]
        X_train, y_train = train_df.loc[:, "x1":], train_df["y"]
        X_test, y_test = test_df.loc[:, "x1":], test_df["y"]
        model = Ridge(alpha = lambdai)
        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_hat, squared = False)
        lambda_rmse.append(rmse)
    rmses.append(mean(lambda_rmse))

print(rmses)
submission = pd.DataFrame(rmses)
submission.to_csv('submission2.csv', header=False, index=False)
        
        
            
