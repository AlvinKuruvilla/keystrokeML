import os
import numpy as np
from PIL import Image
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, LeakyReLU, Dropout
from sklearn.model_selection import train_test_split


# Ex: https://www.datacamp.com/tutorial/convolutional-neural-networks-python
def resize_images(input_dir, size: int):
    """
    Resizes all images in the specified directory to size x size pixels and returns them as a numpy array.

    Parameters:
    - input_dir: str, the path to the directory containing image files.
    - size: int, the size to change the image too.

    Returns:
    - numpy.ndarray: An array of shape (num_images, 128, 128, 3) containing the resized images.
    """
    images = []

    # Iterate through all files in the directory
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        with Image.open(file_path) as img:
            # Check if the image is 'RGB'
            if img.mode != "RGB":
                img = img.convert("RGB")
            # Resize the image
            img = img.resize((size, size), Image.Resampling.LANCZOS)

            # Convert image to numpy array and normalize the pixel values
            img_array = np.array(img) / 255.0
            images.append(img_array)

    # Convert list of images to a numpy array
    return np.stack(images)


def change_type_and_reshape(train_X, test_X, dim_size):
    train_X = X_train.reshape(-1, dim_size, dim_size, 3)
    test_X = X_test.reshape(-1, dim_size, dim_size, 3)

    train_X = train_X.astype("float32")

    test_X = test_X.astype("float32")
    return train_X, test_X


def get_labels(input_dir):
    lables = []
    for filename in os.listdir(input_dir):
        lables.append(filename.split("_")[0])
    return to_categorical(lables)


def build_cnn(num_classes, axis_dim: int):
    fashion_model = Sequential()
    fashion_model.add(
        Conv2D(
            32,
            kernel_size=(3, 3),
            activation="linear",
            input_shape=(axis_dim, axis_dim, 3),
            padding="same",
        )
    )
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D((2, 2), padding="same"))
    fashion_model.add(Dropout(0.25))
    fashion_model.add(Conv2D(64, (3, 3), activation="linear", padding="same"))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    fashion_model.add(Dropout(0.25))
    fashion_model.add(Conv2D(128, (3, 3), activation="linear", padding="same"))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    fashion_model.add(Dropout(0.4))
    fashion_model.add(Flatten())
    fashion_model.add(Dense(128, activation="linear"))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(Dropout(0.3))
    fashion_model.add(Dense(num_classes, activation="softmax"))

    fashion_model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(),
        metrics=["accuracy"],
    )
    return fashion_model


IMAGE_DIMENSION = 64
print(os.path.join(os.getcwd(), "train_images"))
X_train = resize_images(os.path.join(os.getcwd(), "train_images"), IMAGE_DIMENSION)
X_test = resize_images(os.path.join(os.getcwd(), "test_images"), IMAGE_DIMENSION)
X_train, X_test = change_type_and_reshape(X_train, X_test, IMAGE_DIMENSION)
train_Y_one_hot = get_labels(os.path.join(os.getcwd(), "train_images"))
test_Y_one_hot = get_labels(os.path.join(os.getcwd(), "test_images"))

train_X, valid_X, train_label, valid_label = train_test_split(
    X_train, train_Y_one_hot, test_size=0.2, random_state=13
)


print(train_X.shape, valid_X.shape, train_label.shape, valid_label.shape)
batch_size = 64
epochs = 20
num_classes = get_labels(os.path.join(os.getcwd(), "train_images")).shape[1]
print(num_classes)
print("Number of classes from labels:", train_Y_one_hot.shape[1])

fashion_model = build_cnn(num_classes, IMAGE_DIMENSION)
fashion_train = fashion_model.fit(
    train_X,
    train_label,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(valid_X, valid_label),
)
test_eval = fashion_model.evaluate(X_test, test_Y_one_hot, verbose=1)
print("Test loss:", test_eval[0])
print("Test accuracy:", test_eval[1])
