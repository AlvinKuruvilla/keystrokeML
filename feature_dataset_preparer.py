import os
import ast
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier, Pool
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneOut,
    StratifiedKFold,
)
from sklearn.metrics import accuracy_score, classification_report, top_k_accuracy_score
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
import numpy as np
import bob.measure

from core.extended_minmax import ExtendedMinMaxScalar
from core.feature_table import (
    CKP_SOURCE,
    create_full_user_and_platform_table,
    table_to_cleaned_df,
)
import json
# This is the csv saving logic for the features, we shouldn't have to always run this unless we change this with the users
# or the features
# We will also need to rerun this if we ever change the source because currently the columns are fixed

# source = CKP_SOURCE.ALPHA_WORDS
# rows = create_full_user_and_platform_table(source)
# cleaned = table_to_cleaned_df(rows, source)
# cleaned.to_csv("alpha_features_data.csv")

df = pd.read_csv(os.path.join(os.getcwd(), "alpha_features_data.csv"))
with open(os.path.join(os.getcwd(), "classifier_config.json"), "r") as f:
    config = json.load(f)


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
            f"{column_name}_min",
            f"{column_name}_max",
            f"{column_name}_mean",
            f"{column_name}_median",
            f"{column_name}_mode",
        ]
    ] = pd.DataFrame(df[column_name].tolist(), index=df.index)
    df = df.drop(columns=[column_name])
    return df


# TODO: Eventually I want to replace LOO with StratifiedKFold but not now
def run_xgboost_model(X_train, X_test, y_train, y_test, max_k=5):
    # Scale the features
    scaler = ExtendedMinMaxScalar()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Encode the labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Use Leave-One-Out Cross-Validation
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
    grid_search_xgb.fit(X_train_scaled, y_train_encoded)

    best_xgb = grid_search_xgb.best_estimator_

    # Predict on the test set
    y_pred_xgb = best_xgb.predict(X_test_scaled)

    # Decode the predictions
    y_pred_xgb_decoded = label_encoder.inverse_transform(y_pred_xgb)

    # Calculate and print accuracy
    accuracy_xgb = accuracy_score(y_test, y_pred_xgb_decoded)
    print(f"XGBoost Accuracy: {accuracy_xgb}")

    # Get the probabilities for top-k accuracy
    y_pred_proba_xgb = best_xgb.predict_proba(X_test_scaled)

    # Compute classification report
    classification_rep = classification_report(y_test, y_pred_xgb_decoded)
    print(f"Classification Report:\n{classification_rep}")

    # Top-k accuracy loop
    for k in range(1, max_k + 1):
        top_k_acc_xgb = top_k_accuracy_score(y_test_encoded, y_pred_proba_xgb, k=k)
        print(f"XGBoost Top-{k} Accuracy: {top_k_acc_xgb}")

    # Prepare the structure for computing recognition rate using bob.measure
    rr_scores = []
    for i, true_label in enumerate(y_test_encoded):
        # Get positive (true class score) and negative (all other class scores)
        pos_score = y_pred_proba_xgb[i, true_label]  # The score for the correct class
        neg_scores = np.delete(
            y_pred_proba_xgb[i], true_label
        )  # The scores for all other classes

        # Append as (negative scores, positive score) tuple for each probe
        rr_scores.append((neg_scores, [pos_score]))

    # Calculate recognition rate at rank 1
    recognition_rate = bob.measure.recognition_rate(rr_scores, rank=1)
    print(f"Recognition Rate at rank 1: {recognition_rate:.2f}")

    # Plot feature importance
    if config["draw_feature_importance_graph"]:
        plt.figure(figsize=(10, 8))
        plot_importance(
            best_xgb, importance_type="weight", max_num_features=10
        )  # Plot top 10 important features
        plt.title("XGBoost Feature Importance")
        plt.savefig("XGBoost Feature Importance.png")


def run_random_forest_model(X_train, X_test, y_train, y_test, max_k=5):
    # Scale the features
    scaler = ExtendedMinMaxScalar()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Encode the labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    stratified_kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    # Define the hyperparameter grid
    param_grid = {
        "n_estimators": [100, 200, 500],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 10, 20],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["auto", "sqrt", "log2"],
        "bootstrap": [True, False],
    }

    # Perform GridSearchCV for hyperparameter tuning
    grid_search_rf = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        scoring="accuracy",
        cv=stratified_kfold,
        n_jobs=-1,
        verbose=1,
    )
    grid_search_rf.fit(X_train_scaled, y_train_encoded)

    # Get the best estimator
    best_rf = grid_search_rf.best_estimator_

    # Predict using the test set
    y_pred_rf = best_rf.predict(X_test_scaled)

    # Decode the predictions back to original labels
    y_pred_rf_decoded = label_encoder.inverse_transform(y_pred_rf)

    # Compute accuracy using the decoded predictions
    accuracy_rf = accuracy_score(y_test, y_pred_rf_decoded)
    print(f"Random Forest Accuracy: {accuracy_rf}")

    # Get the probabilities for top-k accuracy
    y_pred_proba_rf = best_rf.predict_proba(X_test_scaled)

    # Compute classification report
    classification_rep = classification_report(y_test, y_pred_rf_decoded)
    print(f"Classification Report:\n{classification_rep}")

    # Top-k accuracy loop
    for k in range(1, max_k + 1):
        top_k_acc_rf = top_k_accuracy_score(y_test_encoded, y_pred_proba_rf, k=k)
        print(f"Random Forest Top-{k} Accuracy: {top_k_acc_rf}")

    # Prepare the structure for computing recognition rate using bob.measure
    rr_scores = []
    for i, true_label in enumerate(y_test_encoded):
        # Get positive (true class score) and negative (all other class scores)
        pos_score = y_pred_proba_rf[i, true_label]  # The score for the correct class
        neg_scores = np.delete(
            y_pred_proba_rf[i], true_label
        )  # The scores for all other classes

        # Append as (negative scores, positive score) tuple for each probe
        rr_scores.append((neg_scores, [pos_score]))

    # Calculate recognition rate at rank 1
    recognition_rate = bob.measure.recognition_rate(rr_scores, rank=1)
    print(f"Recognition Rate at rank 1: {recognition_rate:.2f}")

    # Plot feature importance
    if config["draw_feature_importance_graph"]:
        # Plot feature importance
        plt.figure(figsize=(10, 8))
        feature_importances = best_rf.feature_importances_

        # Assuming X_train is a DataFrame, use its columns as feature names
        feature_names = X_train.columns

        indices = np.argsort(feature_importances)[-10:]
        plt.barh(range(len(indices)), feature_importances[indices], align="center")
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Feature Importance")
        plt.title("Random Forest Feature Importance")
        plt.savefig("Random Forest Feature Importance.png")


def run_catboost_model(X_train, X_test, y_train, y_test, max_k=5):
    # Scale the features
    scaler = ExtendedMinMaxScalar()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Encode labels to ensure proper indices
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    catboost_model = CatBoostClassifier(verbose=0, random_seed=42)
    stratified_kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    # Define the hyperparameter grid
    param_grid = {
        "iterations": [100, 200, 300],
        "depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "l2_leaf_reg": [1, 3, 5],
        "border_count": [32, 64, 128],
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
    grid_search.fit(X_train_scaled, y_train_encoded)

    # Get the best parameters and results
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"Best Parameters: {best_params}")
    print(f"Best Cross-Validation Score: {best_score}")

    # Train the model with the best parameters
    best_catboost_model = CatBoostClassifier(**best_params, random_seed=42)
    best_catboost_model.fit(X_train_scaled, y_train_encoded)

    # Predict and evaluate
    y_pred = best_catboost_model.predict(X_test_scaled)
    y_pred_proba = best_catboost_model.predict_proba(X_test_scaled)

    # Evaluate the model
    classification_rep = classification_report(y_test_encoded, y_pred)
    print(f"Classification Report:\n{classification_rep}")

    # Top-k accuracy
    for k in range(1, max_k + 1):
        top_k_acc = top_k_accuracy_score(y_test_encoded, y_pred_proba, k=k)
        print(f"Catboost Top-{k} Accuracy: {top_k_acc}")

    # Prepare the structure for computing recognition rate using bob.measure
    rr_scores = []
    for i, true_label in enumerate(y_test_encoded):  # Use encoded labels
        # Get positive (true class score) and negative (all other class scores)
        pos_score = y_pred_proba[i, true_label]  # The score for the correct class
        neg_scores = np.delete(
            y_pred_proba[i], true_label
        )  # The scores for all other classes

        # Append as (negative scores, positive score) tuple for each probe
        rr_scores.append((neg_scores, [pos_score]))

        # Calculate recognition rate at rank 1
        for k in range(1, max_k + 1):
            recognition_rate = bob.measure.recognition_rate(rr_scores, rank=k)
            print(f"Catboost Top-{k} Recognition Rate: {recognition_rate}")

    # Feature importance
    if config["draw_feature_importance_graph"]:
        feature_importances = best_catboost_model.get_feature_importance(
            Pool(X_train_scaled, label=y_train_encoded)
        )
        feature_names = X_train.columns  # Assuming X_train is a pandas DataFrame

        # Plot feature importance
        plt.figure(figsize=(10, 8))
        sorted_indices = feature_importances.argsort()[-10:]  # Get top 10 features
        plt.barh(
            range(len(sorted_indices)),
            feature_importances[sorted_indices],
            align="center",
        )
        plt.yticks(
            range(len(sorted_indices)), [feature_names[i] for i in sorted_indices]
        )
        plt.xlabel("Feature Importance")
        plt.title("CatBoost Feature Importance")
        plt.savefig("Catboost Feature Importance.png")


# Columns to deserialize
# TODO: We need a more generic way to deserialize columns
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
df.to_csv("cleaned_features_data.csv")

X_train = df[df["platform_id"].isin([1, 2])].drop(columns=["user_id", "platform_id"])
y_train = df[df["platform_id"].isin([1, 2])]["user_id"]

# Test Data: Users from platform 3
X_test = df[df["platform_id"] == 3].drop(columns=["user_id", "platform_id"])
y_test = df[df["platform_id"] == 3]["user_id"]
run_catboost_model(X_train, X_test, y_train, y_test, 5)
