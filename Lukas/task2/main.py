import pandas as pd
import numpy as np

from sklearn.linear_model import SGDRegressor

df_features = pd.read_csv("train_features.csv")
df_test_features = pd.read_csv("test_features.csv")
df_labels = pd.read_csv("train_labels.csv")

averages = df_features.mean(axis=0)

#probably a mistake, but decided to remove all the sparsity in the dataset (it would be great to at least include weighted average)
df_features = df_features.groupby(['pid']).mean()

df_features.fillna(value=averages, inplace = True)

df_classification, df_sepsis, df_regression = df_labels.loc[:, "LABEL_BaseExcess":"LABEL_EtCO2"], df_labels["LABEL_Sepsis"], df_labels.loc[:, "LABEL_RRate":]
#subtask1

# TODO: predict whether a test will be ordered by the clinician (Binary classification).
# LABEL_BaseExcess, LABEL_Fibrinogen, LABEL_AST, LABEL_Alkalinephos, LABEL_Bilirubin_total, LABEL_Lactate, LABEL_TroponinI, LABEL_SaO2, LABEL_Bilirubin_direct, LABEL_EtCO2

#subtask2 bi

# TODO: sepsis prediction (binary classification)

#subtask3

#TODO: mean value of vital signs (regression)

sgd = SGDRegressor()
