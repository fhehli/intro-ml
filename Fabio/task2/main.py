import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor


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
    "LABEL_Sepsis",
]
VITALS = [
    "LABEL_RRate",
    "LABEL_ABPm",
    "LABEL_SpO2",
    "LABEL_Heartrate",
]

shared_params_tests = {
    "loss": "binary_crossentropy",
    "early_stopping": True,
    "scoring": "roc_auc",
}

shared_params_labels = {
    "loss": "squared_error",
    "early_stopping": True,
    "scoring": "r2",
}

# optimal parameters found by grid search
individual_params_tests = {
    "LABEL_BaseExcess": {
        "l2_regularization": 0.1,
        "learning_rate": 0.1,
        "max_depth": 5,
    },
    "LABEL_Fibrinogen": {
        "l2_regularization": 0.1,
        "learning_rate": 0.2,
        "max_depth": 3,
    },
    "LABEL_AST": {"l2_regularization": 0, "learning_rate": 0.05, "max_depth": None},
    "LABEL_Alkalinephos": {
        "l2_regularization": 0.1,
        "learning_rate": 0.05,
        "max_depth": None,
    },
    "LABEL_Bilirubin_total": {
        "l2_regularization": 0.1,
        "learning_rate": 0.05,
        "max_depth": None,
    },
    "LABEL_Lactate": {"l2_regularization": 0.01, "learning_rate": 0.1, "max_depth": 3},
    "LABEL_TroponinI": {"l2_regularization": 0.1, "learning_rate": 0.1, "max_depth": 5},
    "LABEL_SaO2": {"l2_regularization": 0.01, "learning_rate": 0.1, "max_depth": 5},
    "LABEL_Bilirubin_direct": {
        "l2_regularization": 0.1,
        "learning_rate": 0.05,
        "max_depth": 5,
    },
    "LABEL_EtCO2": {"l2_regularization": 0.1, "learning_rate": 0.2, "max_depth": 3},
    "LABEL_Sepsis": {"l2_regularization": 0.01, "learning_rate": 0.1, "max_depth": 3},
}

individual_params_labels = {
    "LABEL_RRate": {"l2_regularization": 0, "learning_rate": 0.05, "max_depth": 5},
    "LABEL_ABPm": {"l2_regularization": 0, "learning_rate": 0.05, "max_depth": None},
    "LABEL_SpO2": {"l2_regularization": 0.01, "learning_rate": 0.1, "max_depth": 5},
    "LABEL_Heartrate": {"l2_regularization": 0, "learning_rate": 0.05, "max_depth": 5},
}


def get_features(df):
    df.drop("Time", axis=1, inplace=True)

    df_age = df.groupby("pid", sort=False).first()["Age"]

    # Simple statistics
    df_mean = df.groupby("pid", sort=False).mean().add_suffix("_mean")
    df_var = df.groupby("pid", sort=False).var().add_suffix("_var")
    df_min = df.groupby("pid", sort=False).min().add_suffix("_min")
    df_max = df.groupby("pid", sort=False).max().add_suffix("_max")

    # Number of measurements
    df_n_measurements = df.groupby("pid", sort=False).count().add_suffix("_n_meas")

    # Most recent measurement
    df_last_measurement = (
        df.groupby("pid", sort=False)
        .agg(
            lambda x: x[x.last_valid_index()]
            if x.last_valid_index() is not None
            else np.nan
        )
        .add_suffix("_last_measurement")
    )

    # Difference between first and last measurement
    df_diff = (
        df.groupby("pid", sort=False)
        .agg(
            lambda x: x[x.last_valid_index()] - x[x.first_valid_index()]
            if x.last_valid_index() != x.first_valid_index()
            else np.nan
        )
        .add_suffix("_diff")
    )

    # Difference between first and last measurement divided by time difference
    df_diff_by_time = (
        df.groupby("pid", sort=False)
        .agg(
            lambda x: (x[x.last_valid_index()] - x[x.first_valid_index()])
            / (x.last_valid_index() - x.first_valid_index())
            if x.last_valid_index() != x.first_valid_index()
            else np.nan
        )
        .add_suffix("_diff_by_time")
    )

    return pd.concat(
        [
            df_age,
            df_mean,
            df_var,
            df_min,
            df_max,
            df_n_measurements,
            df_last_measurement,
            df_diff,
            df_diff_by_time,
        ],
        axis=1,
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

for target in TESTS:
    hgbc = HistGradientBoostingClassifier(
        **shared_params_tests, **individual_params_tests[target]
    )
    hgbc.fit(X_train, df_training_labels[target])  # fit
    df_pred[target] = hgbc.predict_proba(X_test)[:, 1]  # predict

print("Finished fitting models for subtask 1&2.")

# subtask 3 -------------------------------------------------------------
print("Fitting models for subtask 3...")

for target in VITALS:
    hgbr = HistGradientBoostingRegressor(
        **shared_params_labels, **individual_params_labels[target]
    )
    hgbr.fit(X_train, df_training_labels[target])  # fit
    df_pred[target] = hgbr.predict(X_test)  # predict

print("Finished fitting models for subtask 3.")

# -----------------------------------------------------------------------

# export
df_pred.to_csv(
    "Fabio/task2/prediction.zip", index=False, float_format="%.3f", compression="zip"
)
