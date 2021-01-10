from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM
from tensorflow.keras.optimizers import Adam


def build_model_mlp_classfication(hp):
    model = Sequential()
    for i in range(hp.Int('num_layers', min_value=2, max_value=3, step=1)):
        model.add(Dense(units=hp.Choice("num_units" + str(i), values=[8, 16, 32, 64]), activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=hp.Choice("lr", values=[1e-2, 1e-3, 1e-4])))
    return model

def build_model_cnn_classification(hp):
    model = Sequential()
    for i in range(2):
        model.add(Conv1D(filters=hp.Choice("nof_filters", values=[6, 12, 24, 48]),
                         kernel_size=hp.Int("kernel_size", min_value=3, max_value=9, step=3),
                         activation='relu'))
    model.add(Flatten())
    for j in range(1):
        model.add(Dense(units=hp.Choice("num_units" + str(j), values=[8, 16, 32, 64]), activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=hp.Choice("lr", values=[1e-3, 1e-4])))
    return model



# def build_model_lstm_classification(hp):
#     model = Sequential()
#     model.add(LSTM(64, activation=hp.Choice('activation', values=['relu', 'tanh', 'sigmoid']), return_sequences=True))
#     model.add(LSTM(64, activation=hp.Choice('activation', values=['relu', 'tanh', 'sigmoid'])))
#     model.add(Dense(3, activation=hp.Choice('activation_end', values=['softmax', 'sigmoid'])))
#     model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=hp.Choice("lr", values=[1e-2, 1e-3, 1e-4])))
#     return model

# def build_model_lstm_classification(hp):
#     model = Sequential()
#     model.add(LSTM(50, activation='relu'))
#     model.add(Dense(3, activation='softmax'))
#     model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=1e-3))
#     return model