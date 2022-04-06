import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor


VITALS = ["LABEL_RRate", "LABEL_ABPm", "LABEL_SpO2", "LABEL_Heartrate"]
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


def get_features(df):
    df.drop("Time", axis=1, inplace=True)
    df_mean = df.groupby("pid", sort=False).mean().add_suffix("_mean")
    df_var = df.groupby("pid", sort=False).var().add_suffix("_var")
    df_min = df.groupby("pid", sort=False).min().add_suffix("_min")
    df_max = df.groupby("pid", sort=False).max().add_suffix("_max")
    df_n_measurements = (
        df.groupby("pid", sort=False)
        .apply(lambda x: 12 - x.isna().sum())
        .add_suffix("_n_meas")
        .drop("pid_n_meas", axis=1)
    )  # Number of measurements
    df_age = df_mean.loc[:, "Age_mean"]

    return pd.concat(
        [df_age, df_mean, df_var, df_min, df_max, df_n_measurements], axis=1
    )


# get data
df_train = pd.read_csv("Fabio/task2/train_features.csv")
df_training_labels = pd.read_csv("Fabio/task2/train_labels.csv")
df_test = pd.read_csv("Fabio/task2/test_features.csv")

# get features
print("Getting features...")
df_train_features = get_features(df_train)
df_test_features = get_features(df_test)
print("Features loaded.")

print("Imputing and transforming...")
# impute missing values
imputer = SimpleImputer(strategy="median")
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

print("Finished data handling.")


# subtask 1&2 -----------------------------------------------------------
print("Fitting models for subtask 1&2...")
params = {
    "loss": "binary_crossentropy",
    "learning_rate": 0.1,
    "l2_regularization": 0.0,
    "early_stopping": True,
    "scoring": "roc_auc",
    "max_depth": None,
}

hgbc = HistGradientBoostingClassifier(**params)

for target in TESTS + ["LABEL_Sepsis"]:
    hgbc.fit(X_train, df_training_labels[target])  # fit
    df_pred[target] = hgbc.predict_proba(X_test)[:, 1]  # predict

print("Finished fitting models for subtask 1&2.")

# subtask 3 -------------------------------------------------------------
print("Fitting models for subtask 3...")
params = {
    "loss": "squared_error",
    "learning_rate": 0.1,
    "l2_regularization": 0,
    "early_stopping": True,
    "scoring": "r2",
    "max_depth": None,
}

hgbr = HistGradientBoostingRegressor(**params)

for target in VITALS:
    hgbr.fit(X_train, df_training_labels[target])  # fit
    df_pred[target] = hgbr.predict(X_test)  # predict

print("Finished fitting models for subtask 3.")
# -----------------------------------------------------------------------

# export
df_pred.to_csv("prediction.zip", index=False, float_format="%.3f", compression="zip")
