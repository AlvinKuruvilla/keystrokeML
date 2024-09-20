import os
import ast
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneOut,
    StratifiedKFold,
)
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier


df = pd.read_csv(os.path.join(os.getcwd(), "features_data.csv"))


# Deserialization of columns
def deserialize_column(df, column_name):
    df[column_name] = df[column_name].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    return df


# Flattening the columns with lists into separate columns
def flatten_column(df, column_name):
    df[
        [
            f"{column_name}_mean",
            f"{column_name}_median",
            f"{column_name}_mode",
            f"{column_name}_q1",
            f"{column_name}_q3",
        ]
    ] = pd.DataFrame(df[column_name].tolist(), index=df.index)
    df = df.drop(columns=[column_name])
    return df


def run_xgboost_model(X_train, X_test, y_train, y_test):
    # Encode the labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    # Use Leave-One-Out Cross-Validation
    # TODO: This is a sort of hack because apparently there is one class (user_id) with only one sample
    loo = LeaveOneOut()

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "reg_lambda": [1, 3, 5],
    }

    # Perform grid search with LOO
    grid_search_xgb = GridSearchCV(
        XGBClassifier(), param_grid, scoring="accuracy", cv=loo, n_jobs=-1, verbose=1
    )
    grid_search_xgb.fit(X_train, y_train_encoded)

    best_xgb = grid_search_xgb.best_estimator_

    # Predict on the test set
    y_pred_xgb = best_xgb.predict(X_test)

    # Decode the predictions
    y_pred_xgb_decoded = label_encoder.inverse_transform(y_pred_xgb)

    # Calculate and print accuracy
    accuracy_xgb = accuracy_score(y_test, y_pred_xgb_decoded)
    print(f"XGBoost Accuracy: {accuracy_xgb}")


def run_random_forest_model(X_train, X_test, y_train, y_test):
    # Encode the labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    stratified_kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    # Add more hyperparameters to tune
    param_grid = {
        "n_estimators": [100, 200, 500],
        "max_depth": [10, 20, 30, None],  # Add deeper trees and unlimited depth
        "min_samples_split": [2, 10, 20],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [
            "auto",
            "sqrt",
            "log2",
        ],  # Vary the number of features used for splitting
        "bootstrap": [True, False],  # Try both bootstrap sampling methods
    }

    # Perform GridSearchCV for hyperparameter tuning
    grid_search_rf = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        scoring="accuracy",
        cv=stratified_kfold,
        n_jobs=-1,  # Use all processors for faster training
        verbose=1,  # Show the process
    )
    grid_search_rf.fit(X_train, y_train_encoded)

    # Get the best estimator
    best_rf = grid_search_rf.best_estimator_

    # Predict using the test set
    y_pred_rf = best_rf.predict(X_test)

    # Decode the predictions back to original labels
    y_pred_rf_decoded = label_encoder.inverse_transform(y_pred_rf)

    # Compute accuracy using the decoded predictions
    accuracy_rf = accuracy_score(y_test, y_pred_rf_decoded)
    print(f"Random Forest Accuracy: {accuracy_rf}")


def run_catboost_model(X_train, X_test, y_train, y_test):
    catboost_model = CatBoostClassifier(verbose=0, random_seed=42)
    stratified_kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    # Define the hyperparameter grid
    param_grid = {
        "iterations": [300, 500, 700],
        "depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05],
        "l2_leaf_reg": [1, 3, 5],
        "border_count": [32, 64, 128],
        "boosting_type": ["Ordered", "Plain"],
        "od_type": ["IncToDec", "Iter"],
        "od_wait": [20, 40],  # Early stopping patience
    }

    # Initialize GridSearchCV with CatBoost model and hyperparameter grid
    grid_search = GridSearchCV(
        estimator=catboost_model,
        param_grid=param_grid,
        scoring="accuracy",
        cv=stratified_kfold,
        verbose=1,
        n_jobs=-1,
    )

    # Fit the grid search
    grid_search.fit(X_train, y_train)

    # Get the best parameters and results
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"Best Parameters: {best_params}")
    print(f"Best Cross-Validation Score: {best_score}")

    # Train the model with the best parameters
    best_catboost_model = CatBoostClassifier(**best_params, random_seed=42)
    best_catboost_model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = best_catboost_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    # Output results
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{classification_rep}")


# Columns to deserialize
columns_to_deserialize = [
    "er",
    "in",
    "ti",
    "on",
    "es",
    "te",
    "en",
    "is",
    "ic",
    "ri",
    "ra",
    "at",
    "al",
    "an",
    "re",
]

# Apply deserialization and flattening to each relevant column
for col in columns_to_deserialize:
    df = deserialize_column(df, col)
    df = flatten_column(df, col)

# Converting 'user_id' and 'platform_id' to numeric values
df["user_id"] = df["user_id"].apply(
    lambda x: int(ast.literal_eval(x)[0]) if isinstance(x, str) else x
)
df["platform_id"] = df["platform_id"].apply(
    lambda x: int(ast.literal_eval(x)[0]) if isinstance(x, str) else x
)
X_train = df[df["platform_id"].isin([1, 2])]
y_train = df[df["platform_id"].isin([1, 2])]["user_id"]
X_test = df[df["platform_id"] == 3]
y_test = df[df["platform_id"] == 3]["user_id"]

print(X_train)
print(y_train)
print(X_test)
print(y_test)
# Dropping 'user_id' and 'platform_id' from the features
X = df.drop(columns=["user_id", "platform_id"])
y = df["user_id"]  # 'user_id' is the target

# Splitting data into training and test sets
# NOTE: This is kind of a temporary split, to test things. The way I am imaging we should actually do
#       this is we train on all of one platform's data and then test on another platform's.
#       We can just remove any users that only have one platform's data from consideration.
#       Before implementing it here though, I will probably attempt it in our old paper's codebase and see how it does.
X_train = df[df["platform_id"].isin([1, 2])].drop(columns=["user_id", "platform_id"])
y_train = df[df["platform_id"].isin([1, 2])]["user_id"]

# Test Data: Users from platform 3
X_test = df[df["platform_id"] == 3].drop(columns=["user_id", "platform_id"])
y_test = df[df["platform_id"] == 3]["user_id"]
# run_random_forest_model(X_train, X_test, y_train, y_test)
