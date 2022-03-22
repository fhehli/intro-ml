import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor

# import data
df = pd.read_csv("Lukas/task1b/train.csv", index_col=0)
data = df.to_numpy()
X, y = data[:, 1:], data[:, 0]
Phi = np.concatenate(
    (X, X ** 2, np.exp(X), np.cos(X), np.ones((X.shape[0], 1))), axis=1
)

# fit model
clf = SGDRegressor(max_iter=10000, tol=1e-4)
clf.fit(Phi, y)

# export
weights = clf.coef_
pd.DataFrame(weights).to_csv("Lukas/task1b/sgd2.csv", header=False, index=False)
