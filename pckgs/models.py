from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dense, LSTM, Input, Reshape, Conv1D, concatenate, Dropout, Flatten, Conv2D, \
    MaxPooling2D, MaxPooling1D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import Model


def get_model_both_sent():
    model = Sequential()
    model.add(LSTM(16, activation='relu', input_shape=(20, 2)))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=1e-3))
    print(model.summary())
    return model
    # plot_model(model, "my_first_model.png")


def get_model_both_sent_multi():
    inputs = Input(shape=(20, 2))
    layer1 = LSTM(16, activation='relu')(inputs)
    classification = Dense(3, activation='softmax', name='classification')(layer1)
    regression = Dense(1, activation='tanh', name='regression')(layer1)
    model = Model(inputs=inputs, outputs=[classification, regression])
    losses = {'classification': 'categorical_crossentropy', 'regression': 'mape'}
    metrics = {'classification': 'accuracy', 'regression': 'mape'}
    weights = {'classification': 0.5, 'regression': 0.02}  #####!!!!!!!!!
    model.compile(loss=losses, metrics=metrics, loss_weights=weights, optimizer=Adam(learning_rate=1e-3))
    print(model.summary())
    return model


def get_model_price():
    model = Sequential()
    model.add(Reshape((20, 1), input_shape=(20,)))
    model.add(LSTM(16, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=1e-3))
    print(model.summary())
    return model
