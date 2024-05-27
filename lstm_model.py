import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utils import read_compact_format
from keras.models import Model
from keras.layers import Dense, LSTM, Input, BatchNormalization, Activation, Subtract
from sklearn.metrics import roc_curve, accuracy_score, f1_score


# Create pairs of sequences
def create_pairs(sequences, labels):
    pairs = []
    targets = []
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            pairs.append([sequences[i], sequences[j]])
            if labels[i] == labels[j]:
                targets.append(1)
            else:
                targets.append(0)
    return np.array(pairs), np.array(targets)


# Define the LSTM network
def create_base_network(input_shape):
    input = Input(shape=input_shape)
    x = LSTM(128, return_sequences=True, dropout=0.5, recurrent_dropout=0.2)(input)
    x = BatchNormalization()(x)
    x = Activation("tanh")(x)
    x = LSTM(128, return_sequences=False, dropout=0.5, recurrent_dropout=0.2)(x)
    x = BatchNormalization()(x)
    x = Activation("tanh")(x)
    x = Dense(128)(x)
    return Model(input, x)


data = read_compact_format()
# Convert press and release times from nanoseconds to milliseconds
data["press_time"] = data["press_time"] / 1e6
data["release_time"] = data["release_time"] / 1e6

# Calculate duration of each key press
data["duration"] = data["release_time"] - data["press_time"]
count = 5
sequences = []
labels = []

for i in range(len(data) - count):
    seq = data.iloc[i : i + count]
    label = data.iloc[i + count]["duration"]
    sequences.append(seq[["duration"]].values)
    labels.append(label)

sequences = np.array(sequences)
labels = np.array(labels)
# Normalize the data
scaler = MinMaxScaler()
sequences = scaler.fit_transform(sequences.reshape(-1, sequences.shape[-1])).reshape(
    sequences.shape
)
labels = scaler.fit_transform(labels.reshape(-1, 1)).reshape(-1)
pairs, targets = create_pairs(sequences, labels)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    pairs, targets, test_size=0.2, random_state=42
)
input_shape = (count, 1)
base_network = create_base_network(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

processed_a = base_network(input_a)
processed_b = base_network(input_b)

# Compute the absolute difference between the outputs
distance = Subtract()([processed_a, processed_b])
distance = Dense(1, activation="sigmoid")(distance)

model = Model([input_a, input_b], distance)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
history = model.fit(
    [X_train[:, 0], X_train[:, 1]],
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=([X_test[:, 0], X_test[:, 1]], y_test),
)

# Evaluate the model
y_pred = model.predict([X_test[:, 0], X_test[:, 1]])

# Compute ROC curve and EER
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
eer_threshold = thresholds[np.nanargmin(np.abs(fpr - (1 - tpr)))]

# Convert predictions to binary based on the EER threshold
y_pred_binary = (y_pred > eer_threshold).astype(int)

# Calculate accuracy and F1 score
accuracy = accuracy_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)

print(f"EER Threshold: {eer_threshold}")
print(f"Test Accuracy: {accuracy}")
print(f"Test F1 Score: {f1}")
