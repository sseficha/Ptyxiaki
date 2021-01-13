from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dense, LSTM, Input, Reshape, Conv1D, concatenate, Dropout, Flatten, Conv2D, \
    MaxPooling2D, MaxPooling1D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l1, l2



def get_model_both_sent():
    model = Sequential()
    model.add(LSTM(16, activation='relu', input_shape=(20, 2)))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=1e-3))
    # print(model.summary())
    return model
    # plot_model(model, "my_first_model.png")

def get_model_price():
    model = Sequential()
    model.add(Reshape((20, 1), input_shape=(20,)))
    model.add(LSTM(16, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=1e-3))
    # print(model.summary())
    return model


def get_model_both_emb():
    model1 = Sequential()
    model1.add(Reshape((20, 1), input_shape=(20,)))
    model1.add(LSTM(16, activation='relu'))

    model2 = Sequential()
    model2.add(Conv1D(filters=8, kernel_size=1, activation='relu', input_shape=(20, 768)))
    model2.add(LSTM(16, activation='relu', kernel_regularizer=l2(1e-2), recurrent_regularizer=l2(1e-2)))

    merged = concatenate([model1.output, model2.output])
    # merged = (Dense(16, activation='relu'))(merged)
    merged = (Dropout(0.4))(merged)
    merged = (Dense(3, activation='softmax'))(merged)
    model_merged = Model(inputs=[model1.input, model2.input], outputs=merged)
    model_merged.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=1e-3))
    # dot_img_file = '/tmp/model_both_emb.png'
    # plot_model(model_merged, to_file=dot_img_file, show_shapes=True)
    # print(model_merged.summary())
    return model_merged











# def get_model_both_sent_multi():
#     inputs = Input(shape=(20, 2))
#     layer1 = LSTM(16, activation='relu')(inputs)
#     classification = Dense(3, activation='softmax', name='classification')(layer1)
#     regression = Dense(1, activation='tanh', name='regression')(layer1)
#     model = Model(inputs=inputs, outputs=[classification, regression])
#     losses = {'classification': 'categorical_crossentropy', 'regression': 'mape'}
#     metrics = {'classification': 'accuracy', 'regression': 'mape'}
#     weights = {'classification': 0.5, 'regression': 0.02}  #####!!!!!!!!!
#     model.compile(loss=losses, metrics=metrics, loss_weights=weights, optimizer=Adam(learning_rate=1e-3))
#     print(model.summary())
#     return model


