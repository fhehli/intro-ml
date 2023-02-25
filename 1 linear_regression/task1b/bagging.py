import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate

# import data
df = pd.read_csv("train.csv", index_col=0)
data = df.to_numpy()
X, y = data[:, 1:], data[:, 0]
Phi = np.concatenate(
    (X, X**2, np.exp(X), np.cos(X), np.ones((X.shape[0], 1))), axis=1
)

# fit model
K = 8
cv = cross_validate(
    Ridge(alpha=11.0, fit_intercept=False), Phi, y, cv=K, return_estimator=True
)

bagging_weights = np.zeros(Phi.shape[1])
for estimator in cv["estimator"]:
    bagging_weights += estimator.coef_ / K

# export
pd.DataFrame(bagging_weights).to_csv("bagging.csv", header=False, index=False)
