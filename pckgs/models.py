from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dense, LSTM, Input, Reshape, Conv1D, concatenate, Dropout, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import load_model
from pckgs.helper import PnlCallback


def get_model_lstm(neuron, lr):
    model = Sequential()
    model.add(LSTM(neuron, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss=CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'], optimizer=Adam(learning_rate=lr))
    return model


def get_model_mlp(layer,neuron,lr,dropout):
    model = Sequential()
    for i in range(layer):
        model.add(Dense(neuron, activation='relu'))
        model.add(Dropout(dropout))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss=CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'], optimizer=Adam(learning_rate=lr))
    return model


def get_model_cnn():
    model = Sequential()
    model.add(Reshape((20, 1), input_shape=(20,)))
    model.add(Conv1D(filters=8, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=8, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss=CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'], optimizer=Adam(learning_rate=1e-3))
    return model

# def tune_model_mlp(hp):
#     model = Sequential()
#     model.add(Dense(units=hp.Choice("num_units1", values=[4,8,16,32,64,128,256]), activation='relu'))
#     model.add(Dropout(rate=hp.Choice("num_units_drop1", values=[0.2,0.4,0.6,0.8])))
#     model.add(Dense(units=hp.Choice("num_units2", values=[4,8,16,32,64,128,256]), activation='relu'))
#     model.add(Dropout(rate=hp.Choice("num_units_drop2", values=[0.2,0.4,0.6,0.8])))
#     model.add(Dense(3, activation='softmax'))
#     model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=1e-3))#hp.Choice("lr", values=[1e-2,5e-3,1e-3,5e-4,1e-4])))
#     return model



# def tune_model_lstm(hp):
#     model = Sequential()
#     model.add(Reshape((20, 1), input_shape=(20,)))
#     model.add(LSTM(units=hp.Choice("num_units1", values=[4,8,16,32,64]), activation='relu', return_state=True))
#     model.add(LSTM(units=hp.Choice("num_units2", values=[4,8,16,32,64]), activation='relu'))
#     model.add(Dense(3, activation='softmax'))
#     model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=1e-3))#hp.Choice("lr", values=[1e-2,5e-3,1e-3,5e-4,1e-4])))
#     return model

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


import tensorflow.keras.backend as K
from pckgs.helper import ModelSave

def train_model(model, data, train_candle, test_candle, pathname_model, epochs=200, verbose=0):
    f_lr = K.eval(model.optimizer.lr) * 0.1
    s_lr = K.eval(model.optimizer.lr)
    def scheduler(epoch, lr):
        return lr + ((f_lr - s_lr) / epochs)

    (x_train, x_test, y_train, y_test) = data

    pnl = PnlCallback(x_test, test_candle, x_train, train_candle, pathname_model)
    sched = LearningRateScheduler(scheduler)

    history = model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=32,
                        epochs=epochs, verbose=verbose, callbacks=[pnl, sched])

    return history, pnl.stats_test, pnl.stats_train, pnl.pnls_test, pnl.pnls_train










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


