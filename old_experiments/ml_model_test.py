from core.ml_models import (
    catboost_test,
    evaluate_model,
    prepare_data,
    random_forest_test,
    train_and_evaluate_svm_with_hyperparameter_tuning,
    train_hist_gb,
)


# source = CKP_SOURCE.NORVIG
# features, labels = prepare_data(source)

# print(features)
# print(labels)
# Train SVM
# print("SVM Model Testing:")
# train_and_evaluate_svm_with_hyperparameter_tuning(features, labels)
# Train Histogram Gradient Boosting
# hist_gb_model = train_hist_gb(features, labels)

# print("Histogram Gradient Boosting Model Evaluation:")
# evaluate_model(hist_gb_model, None, features, labels)
# random_forest_test()
random_forest_test()
