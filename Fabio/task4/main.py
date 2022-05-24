import lightgbm
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

X_pretrain = pd.read_csv("Fabio/task4/data/pretrain_features.csv").to_numpy()[:, 2:]
X_train = pd.read_csv("Fabio/task4/data/train_features.csv").to_numpy()[:, 2:]
X_test = pd.read_csv("Fabio/task4/data/test_features.csv").to_numpy()[:, 2:]
y_pretrain = (
    pd.read_csv("Fabio/task4/data/pretrain_labels.csv").to_numpy()[:, 1].reshape(-1, 1)
)
y_train = (
    pd.read_csv("Fabio/task4/data/train_labels.csv").to_numpy()[:, 1].reshape(-1, 1)
)

model = lightgbm.LGBMRegressor(n_estimators=1000)
model.fit(X_pretrain, y_pretrain)

energy_train = model.predict(X_train).reshape(-1, 1)
reg = LinearRegression().fit(energy_train, y_train)

energy_test = model.predict(X_test).reshape(-1, 1)
pred = reg.predict(energy_test)

df_pred = pd.DataFrame(pred)
df_pred.index += 50_100
df_pred.to_csv(
    "gbm.csv",
    index=True,
    index_label="Id",
    header=["y"],
)
