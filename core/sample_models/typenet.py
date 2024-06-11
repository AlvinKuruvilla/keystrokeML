import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
    Masking,
)


def create_typenet_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Masking(mask_value=0.0)(inputs)
    x = LSTM(128, return_sequences=True, dropout=0.2)(x)
    x = BatchNormalization()(x)
    x = LSTM(128, dropout=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="tanh")(
        x
    )  # Change 'tanh' to 'softmax' for classification
    model = Model(inputs, outputs)
    return model


# Prepare data for LSTM
def prepare_lstm_data(df, features, sequence_length):
    X, y = [], []
    for user_id, group in df.groupby("user_id"):
        sequences = [
            features[i : i + sequence_length]
            for i in range(0, len(group), sequence_length)
        ]
        X.extend(sequences)
        y.extend([user_id] * len(sequences))
    return np.array(X), np.array(y)
