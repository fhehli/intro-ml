import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier


def get_features(df):
    df.drop("Time", axis=1, inplace=True)
    df_mean = df.groupby("pid").mean().add_suffix("_mean")
    df_var = df.groupby("pid").var().add_suffix("_var")
    df_min = df.groupby("pid").min().add_suffix("_min")
    df_max = df.groupby("pid").max().add_suffix("_max")

    return pd.concat([df_mean, df_var, df_min, df_max], axis=1)


# get data
df_train = pd.read_csv("Fabio/task2/train_features.csv")
df_training_labels = pd.read_csv("Fabio/task2/train_labels.csv")
df_test = pd.read_csv("Fabio/task2/test_features.csv")

# get features
df_train_features = get_features(df_train)
df_test_features = get_features(df_test)

# impute missing values
imputer = SimpleImputer()
imputer.fit(df_train_features)
X_train = imputer.transform(df_train_features)
X_test = imputer.transform(df_test_features)

# center and descale
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# prepare df for predictions
df_pred = pd.DataFrame(columns=df_training_labels.columns)
df_pred["pid"] = df_test_features.index

## subtask 1 -----------------------------------------------------------
TESTS = [
    "LABEL_BaseExcess",
    "LABEL_Fibrinogen",
    "LABEL_AST",
    "LABEL_Alkalinephos",
    "LABEL_Bilirubin_total",
    "LABEL_Lactate",
    "LABEL_TroponinI",
    "LABEL_SaO2",
    "LABEL_Bilirubin_direct",
    "LABEL_EtCO2",
]

params = {
    "loss": "binary_crossentropy",
    "l2_regularization": 0.01,
    "early_stopping": True,
    "scoring": "roc_auc",
}

HGB = HistGradientBoostingClassifier(**params)

for target in TESTS:
    HGB.fit(X_train, df_training_labels[target])
    df_pred[target] = HGB.predict_proba(X_test)[:, 1]


## subtask 2 -----------------------------------------------------------
HGB.fit(X_train, df_training_labels["LABEL_Sepsis"])
df_pred["LABEL_Sepsis"] = HGB.predict_proba(X_test)[:, 1]


## subtask 3 -----------------------------------------------------------
# TODO


# export
# TODO

