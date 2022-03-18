import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge

# read data
df = pd.read_csv("./task1a/train.csv")
X, y = df.loc[:, "x1":], df["y"]

# cross validate
lambdas = [0.1, 1.0, 10.0, 100.0, 200.0]
scores = []
for lmbda in lambdas:
    fold_scores = cross_val_score(Ridge(alpha=lmbda, solver="svd"), X, y, cv=10)
    scores.append(-fold_scores.mean())

# export
submission = pd.DataFrame(scores)
submission.to_csv("./submission.csv", header=False, index=False)
