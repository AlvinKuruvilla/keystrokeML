import os
import tensorflow as tf
import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16
from PIL import Image

from core.extractor import (
    clean_directories,
    create_test_images_folder,
    create_train_images_folder,
    validate_folder_split,
    vgg_feature_extract,
)


def vgg_model():
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # print(model.summary())
    return model


def build_dense_classifier(input_shape, num_classes):
    model = Sequential()
    model.add(Dense(512, activation="relu", input_shape=(input_shape,)))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def dbg_print_image_dimensions(dir):
    files = os.listdir(dir)
    for file in files:
        print(Image.open(os.path.join(os.getcwd(), dir, file)).size)


def run_model():
    np.random.seed(0)
    tf.random.set_seed(0)
    # Load VGG model for feature extraction
    model = vgg_model()

    # Extract features from organized directories
    train_features, train_labels = vgg_feature_extract("train_images/", model)
    test_features, test_labels = vgg_feature_extract("test_images/", model)
    print(train_labels)
    # Split features into training and testing data is now not necessary as they are already split
    # Build and train the classifier
    num_classes = train_labels.shape[1]  # Correctly capture the number of classes
    classifier = build_dense_classifier(
        train_features.shape[1], num_classes
    )  # Set the correct input shape and num_classes
    classifier.fit(train_features, train_labels, epochs=10, validation_split=0.1)

    # Evaluate the classifier
    test_loss, test_acc = classifier.evaluate(test_features, test_labels)
    print(f"Test Accuracy: {test_acc:.3f}")


if __name__ == "__main__":
    clean_directories()
    create_train_images_folder()
    create_test_images_folder()
    validate_folder_split()
    run_model()
    # train_directory = os.path.join(os.getcwd(), "test_images")
    # dbg_print_image_dimensions(train_directory)
