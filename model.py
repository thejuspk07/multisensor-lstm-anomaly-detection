from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Input
from tensorflow.keras.optimizers import Adam


def build_lstm_autoencoder(seq_len, n_features):
    model = Sequential()
    model.add(Input(shape=(seq_len, n_features)))
    model.add(LSTM(32, activation="tanh"))
    model.add(RepeatVector(seq_len))
    model.add(LSTM(32, activation="tanh", return_sequences=True))
    model.add(TimeDistributed(Dense(n_features)))

    model.compile(
        optimizer=Adam(0.001),
        loss="mse"
    )
    return model
