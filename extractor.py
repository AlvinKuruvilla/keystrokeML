import os
import shutil
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical


def clean_directories():
    train_directory = os.path.join(os.getcwd(), "train_images")
    test_directory = os.path.join(os.getcwd(), "test_images")
    if not os.path.exists(train_directory):
        print("No train directory found - skipping clean")
        return
    if not os.path.exists(test_directory):
        print("No test directory found - skipping clean")
        return
    for directory_path in [train_directory, test_directory]:
        try:
            files = os.listdir(directory_path)
            for file in files:
                file_path = os.path.join(directory_path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        except OSError:
            print("Error occurred while deleting files.")
    print("All files deleted successfully.")


def create_train_images_folder():
    train_directory = os.path.join(os.getcwd(), "train_images")
    if not os.path.exists(train_directory):
        os.makedirs(train_directory)
    for filename in os.listdir(os.path.join(os.getcwd(), "result_images")):
        # filename has the extension so move back 4 more spaces first
        if int(filename[-5]) in [1, 2, 3]:
            print("here")
            src_path = os.path.join(
                os.path.join(os.getcwd(), "result_images"), filename
            )
            dest_path = os.path.join(train_directory, filename)
            # Check if the file already exists in the destination directory
            if not os.path.exists(dest_path):
                # Move the file
                shutil.copy(src_path, dest_path)
                print(f'Moved "{filename}" to {train_directory}')
            else:
                print(f'Skipping "{filename}", already exists in {train_directory}')


def create_test_images_folder():
    test_directory = os.path.join(os.getcwd(), "test_images")
    if not os.path.exists(test_directory):
        os.makedirs(test_directory)
    for filename in os.listdir(os.path.join(os.getcwd(), "result_images")):
        # filename has the extension so move back 4 more spaces first
        if int(filename[-5]) in [4, 5, 6]:
            src_path = os.path.join(
                os.path.join(os.getcwd(), "result_images"), filename
            )
            dest_path = os.path.join(test_directory, filename)
            # Check if the file already exists in the destination directory
            if not os.path.exists(dest_path):
                # Move the file
                shutil.copy(src_path, dest_path)
                print(f'Moved "{filename}" to {test_directory}')
            else:
                print(f'Skipping "{filename}", already exists in {test_directory}')


def validate_folder_split():
    train_directory = os.path.join(os.getcwd(), "train_images")
    test_directory = os.path.join(os.getcwd(), "test_images")
    orginial_directory = os.path.join(os.getcwd(), "result_images")
    original_count = 0
    train_count = 0
    test_count = 0
    for _ in os.listdir(orginial_directory):
        original_count += 1
    for _ in os.listdir(train_directory):
        train_count += 1
    for _ in os.listdir(test_directory):
        test_count += 1
    assert train_count + test_count == original_count


def vgg_feature_extract(directory, model):
    features = []
    labels = []
    for filename in os.listdir(directory):
        # Extract label from filename assuming format 'userID_platform_sessionID.png, where the label is the user_id'
        label = filename.split("_")[0]
        labels.append(label)

        # Prepare image for feature extraction
        image_path = os.path.join(directory, filename)
        img = load_img(image_path, target_size=(224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        # Extract features
        feature = model.predict(img, verbose=0)
        features.append(feature.flatten())

    # Convert labels to one-hot encoding
    unique_labels = list(set(labels))
    label_indices = [unique_labels.index(label) for label in labels]
    labels_one_hot = to_categorical(label_indices)

    return np.array(features), labels_one_hot


clean_directories()
create_train_images_folder()
create_test_images_folder()
validate_folder_split()
