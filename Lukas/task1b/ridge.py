import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

df = pd.read_csv("train.csv", index_col=0)
data = df.to_numpy()
X, y = data[:, 1:], data[:, 0]
Phi = np.concatenate(
    (X, X ** 2, np.exp(X), np.cos(X), np.ones((X.shape[0], 1))), axis=1)

clf = Ridge(alpha=350, fit_intercept=False)
clf.fit(Phi, y)

# export
weights = clf.coef_
pd.DataFrame(weights).to_csv("ridge_alpha350.csv", header=False, index=False)
