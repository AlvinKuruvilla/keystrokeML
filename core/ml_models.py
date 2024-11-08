import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
from core.deft import flatten_list
from core.feature_table import (
    CKP_SOURCE,
    create_full_user_and_platform_table,
    only_user_id,
    table_to_cleaned_df,
)
from core.utils import all_ids


def prepare_data(source):
    rows = create_full_user_and_platform_table(source)
    cleaned = table_to_cleaned_df(rows, source)

    user_ids = []
    features = []

    for x in all_ids():
        user_data = only_user_id(cleaned, x)
        # print(user_data)
        # input("Got user data")
        num_rows = len(user_data)

        for i in range(num_rows):  # Safely iterate over available rows, up to 3
            if num_rows == 1:
                continue
            row_data = user_data.iloc[i].to_dict()
            row_data.pop("user_id", None)
            row_data.pop("platform_id", None)
            feature_vector = flatten_list(list(row_data.values()))
            features.append(feature_vector)
            user_ids.append(x)
    print(features)
    input()
    features = np.array(features)
    labels = np.array(user_ids)

    return features, labels


def train_and_evaluate_svm_with_kfold(features, labels, n_splits=3, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    accuracies = []

    for train_index, test_index in skf.split(features, labels):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # Step 2: Create a pipeline that first scales the data then applies SVM
        svm_pipeline = make_pipeline(
            StandardScaler(), SVC(kernel="rbf", probability=True, C=1.0)
        )

        # Step 3: Train the model on the training set
        svm_pipeline.fit(X_train, y_train)

        # Step 4: Test the model on the testing set
        y_pred = svm_pipeline.predict(X_test)

        # Step 5: Evaluate the model's performance
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    mean_accuracy = np.mean(accuracies)
    print(f"SVM Mean Cross-Validation Accuracy: {mean_accuracy:.4f}")

    return mean_accuracy


def train_and_evaluate_svm_with_hyperparameter_tuning(
    features, labels, n_splits=3, random_state=42
):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    svm = SVC(probability=True)
    param_grid = {
        "C": [0.1, 1, 10, 100],
        "gamma": [1, 0.1, 0.01, 0.001],
        "kernel": ["rbf", "linear"],
    }

    grid = GridSearchCV(svm, param_grid, refit=True, cv=skf, verbose=2)
    grid.fit(features, labels)

    print(f"Best parameters found: {grid.best_params_}")
    best_svm = grid.best_estimator_

    # Evaluate the best model on the entire dataset
    y_pred = best_svm.predict(features)
    accuracy = accuracy_score(labels, y_pred)
    print(f"SVM Accuracy with best parameters: {accuracy:.4f}")

    return best_svm, accuracy


def train_hist_gb(features, labels):
    hist_gb_model = HistGradientBoostingClassifier()
    kfold = KFold(
        n_splits=min(3, len(np.unique(labels)))
    )  # Set n_splits to at most the number of unique labels
    scores = cross_val_score(
        hist_gb_model, features, labels, cv=kfold, scoring="accuracy"
    )
    print(
        f"Histogram Gradient Boosting Cross-Validation Accuracy: {np.mean(scores):.4f}"
    )

    hist_gb_model.fit(features, labels)
    return hist_gb_model


def evaluate_model(model, scaler, features, labels):
    features_scaled = scaler.transform(features) if scaler else features
    predictions = model.predict(features_scaled)
    accuracy = accuracy_score(labels, predictions)
    print(f"Model Accuracy: {accuracy:.4f}")
    print(classification_report(labels, predictions))


def random_forest_test():
    source = CKP_SOURCE.ALPHA_WORDS
    rows = create_full_user_and_platform_table(source)
    cleaned = table_to_cleaned_df(rows, source)
    print(cleaned)
    input()
    cleaned.to_csv("features_data.csv")
    user_ids = all_ids()
    facebook_features = []
    y_train = []
    y_test = []

    max_length = 0  # Track the maximum feature length for padding

    for x in user_ids:
        user_data = only_user_id(cleaned, x)
        num_rows = len(user_data)
        print("Number of platforms are: ", num_rows)

        if (
            num_rows >= 2
        ):  # Ensure there are at least two platforms (Facebook and Instagram)
            facebook_data = user_data.iloc[0].to_dict()
            facebook_data.pop("user_id", None)
            facebook_data.pop("platform_id", None)
            facebook_feature_vector = flatten_list(list(facebook_data.values()))
            max_length = max(max_length, len(facebook_feature_vector))
            facebook_features.append(facebook_feature_vector)
            y_train.append(x)

    X = facebook_features
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_train, test_size=0.2, random_state=42
    )
    print(X_train)
    input()
    print(X_test)
    input()
    # Initialize the KNN model
    knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")

    # Train the KNN model
    knn.fit(X_train, y_train)

    # Predict on the test set
    y_pred = knn.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, zero_division=0)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)


def catboost_test():
    source = CKP_SOURCE.ALPHA_WORDS
    rows = create_full_user_and_platform_table(source)
    cleaned = table_to_cleaned_df(rows, source)
    user_ids = all_ids()
    facebook_features = []
    y_train = []
    y_test = []

    max_length = 0  # Track the maximum feature length for padding

    for x in user_ids:
        user_data = only_user_id(cleaned, x)
        num_rows = len(user_data)
        print("Number of platforms are: ", num_rows, "for user:", str(x))

        if (
            num_rows >= 2
        ):  # Ensure there are at least two platforms (Facebook and Instagram)
            facebook_data = user_data.iloc[0].to_dict()
            facebook_data.pop("user_id", None)
            facebook_data.pop("platform_id", None)
            facebook_feature_vector = flatten_list(list(facebook_data.values()))
            max_length = max(max_length, len(facebook_feature_vector))
            facebook_features.append(facebook_feature_vector)
            y_train.append(x)

    X = facebook_features
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_train, test_size=0.2, random_state=42
    )

    # Initialize the CatBoost model
    catboost_model = CatBoostClassifier(
        iterations=1000, learning_rate=0.1, depth=6, verbose=0
    )

    # Train the CatBoost model
    catboost_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = catboost_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, zero_division=0)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)
