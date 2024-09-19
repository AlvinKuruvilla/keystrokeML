import os
import ast
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report


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


def run_catboost_model(X_train, X_test, y_train, y_test):
    catboost_model = CatBoostClassifier(verbose=0, random_seed=42)

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
        cv=3,
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


def run_svm_model(X_train, X_test, y_train, y_test):
    gbm_model = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42
    )
    gbm_model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred_gbm = gbm_model.predict(X_test)
    accuracy_gbm = accuracy_score(y_test, y_pred_gbm)
    print(f"Gradient Boosting Accuracy: {accuracy_gbm}")


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

# Dropping 'user_id' and 'platform_id' from the features
X = df.drop(columns=["user_id", "platform_id"])
y = df["user_id"]  # 'user_id' is the target

# Splitting data into training and test sets
# NOTE: This is kind of a temporary split, to test things. The way I am imaging we should actually do
#       this is we train on all of one platform's data and then test on another platform's.
#       We can just remove any users that only have one platform's data from consideration.
#       Before implementing it here though, I will probably attempt it in our old paper's codebase and see how it does.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# run_catboost_model(X_train, X_test, y_train, y_test)
run_svm_model(X_train, X_test, y_train, y_test)
