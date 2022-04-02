import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# import data
df = pd.read_csv("task1b/train.csv", index_col=0)
data = df.to_numpy()
X, y = data[:, 1:], data[:, 0]
Phi = np.concatenate(
    (X, X ** 2, np.exp(X), np.cos(X), np.ones((X.shape[0], 1))), axis=1
)

# fit model
clf = Ridge(alpha=22.0, fit_intercept=False)
clf.fit(Phi, y)

# export
weights = clf.coef_
pd.DataFrame(weights).to_csv("task1b/ridge.csv", header=False, index=False)
