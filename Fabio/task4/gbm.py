import numpy as np
import lightgbm
import pandas as pd

df_train = pd.read_csv("data/train_features.csv")
df_labels = pd.read_csv("data/train_labels.csv")

df_train = df_train.drop(["Id", "smiles"], axis=1)
df_labels = df_labels.drop("Id", axis=1)

model = lightgbm.LGBMRegressor()
model.fit(df_train, df_labels)

df_test = pd.read_csv("data/test_features.csv")
df_test = df_test.drop(["Id", "smiles"], axis=1)
ids = df_test["Id"].to_numpy()

preds = model.predict(df_test)
out = np.vstack((ids, preds))

pd.DataFrame(out.T).to_csv("gbm.csv", index=False, header=["Id", "y"])
