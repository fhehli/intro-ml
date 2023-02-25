import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor

# import data
df = pd.read_csv("train.csv", index_col=0)
data = df.to_numpy()
X, y = data[:, 1:], data[:, 0]
Phi = np.concatenate(
    (X, X**2, np.exp(X), np.cos(X), np.ones((X.shape[0], 1))), axis=1
)

# fit models
models = [
    Ridge(alpha=11.0, fit_intercept=True),
    Lasso(alpha=0.034697, fit_intercept=False, max_iter=10000),
    ElasticNet(alpha=0.01, max_iter=10000, fit_intercept=False),
    HuberRegressor(alpha=20.0, max_iter=1000, fit_intercept=False),
]
for model in models:
    model.fit(Phi, y)
weights = {model: model.coef_ for model in models}

# export weights
for model, weight in weights.items():
    pd.DataFrame(weight).to_csv(
        f"{str(model).split('(')[0]}.csv", header=False, index=False
    )
