import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from core.sample_models.typenet import create_typenet_model, prepare_lstm_data
from core.synthetic.simple import extract_features


# TODO: The SEQUENCE_LENGTH of 70 was specifically for the synthetic data. Figure out what the SEQUENCE_LENGTH
#       should be for our actual dataset
SEQUENCE_LENGTH = 70
if __name__ == "__main__":
    df = pd.read_csv(
        os.path.join(os.getcwd(), "dataset", "synthetic_keystroke_data.csv"),
    )
    num_users = df["user_id"].nunique()
    print(f"Number of unique users: {num_users}")

    # Define input shape (sequence_length, num_features)
    input_shape = (SEQUENCE_LENGTH, 5)  # 5 features: keycode, HL, IL, PL, RL
    num_classes = num_users  # Set the number of classes based on unique users
    model = create_typenet_model(input_shape, num_classes)
    model.summary()
    scaler = MinMaxScaler()
    features = scaler.fit_transform(extract_features(df))
    X, y = prepare_lstm_data(df, features, SEQUENCE_LENGTH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Compile and train the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Train the model
    history = model.fit(
        X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test)
    )
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    plt.plot(history.history["accuracy"], label="train accuracy")
    plt.plot(history.history["val_accuracy"], label="val accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
