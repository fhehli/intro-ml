import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge

df = pd.read_csv("train.csv")
X, y = df.loc[:, "x1":], df["y"]

# cross validate
lambdas = [0.1, 1.0, 10.0, 100.0, 200.0]
scores = []
for lmbda in lambdas:
    fold_scores = cross_val_score(
        Ridge(alpha=lmbda, fit_intercept=False),
        X,
        y,
        cv=10,
        scoring="neg_root_mean_squared_error",
    )
    scores.append(-fold_scores.mean())

submission = pd.DataFrame(scores)
submission.to_csv("submission.csv", header=False, index=False)
