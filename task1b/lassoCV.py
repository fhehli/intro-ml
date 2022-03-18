import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV

df = pd.read_csv("task1b/train.csv", index_col=0)
data = df.to_numpy()
X, y = data[:, 1:], data[:, 0]
Phi = np.concatenate(
    (X, X ** 2, np.exp(X), np.cos(X), np.ones((X.shape[0], 1))), axis=1
)

clf = LassoCV(cv=5, max_iter=10000)
clf.fit(Phi, y)

weights = clf.coef_
pd.DataFrame(weights).to_csv("task1b/lassoCV.csv", header=False, index=False)
