import tensorflow as tf
import time
import os
import json

from tensorflow.keras.layers import LSTM, TimeDistributed, Dense, Flatten, BatchNormalization, Dropout, Input
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.metrics import MeanAbsolutePercentageError, MeanSquaredError, MeanSquaredLogarithmicError
from tensorflow.keras.callbacks import History, EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras import Model

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# function api


def lstm_model(history, future, num_features, model_id):

    # model input
    inputs = Input(shape=(history, num_features))

    # model layers
    x = LSTM(256, return_sequences=True)(inputs)
    x = LSTM(128, return_sequences=True)(x)
    x = LSTM(64, return_sequences=True)(x)
    x = TimeDistributed(Dense(64, activation="relu"))(x)
    x = Flatten()(x)
    x = Dense(64)(x)
    x = Dense(32)(x)

    # model output
    outputs = Dense(future)(x)

    # group layers into model object
    model = Model(inputs=inputs, outputs=outputs, name=model_id)

    return model


def train(train_dev_test, _model_func, _name, _dir_name, BATCH_SIZE=128, EPOCHS=100):

    output_folder = f"../Models/{_dir_name}/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    timestamp = int(time.time())

    history, future, n_features = train_dev_test.in_out

    # get Datasets
    X_Y_train = train_dev_test.train_ds(batch_size=BATCH_SIZE)
    X_Y_dev = train_dev_test.dev_ds(batch_size=BATCH_SIZE)
    X_Y_test = train_dev_test.test_ds(batch_size=BATCH_SIZE)

    # create model
    model = _model_func(history, future, n_features, f"{_name}_{timestamp}")

    # compile parameters
    optimizer = Adam(learning_rate=1e-3)
    loss = 'mean_absolute_percentage_error'#MeanAbsoluteError()
    metrics = [MeanAbsoluteError(), MeanSquaredError(),MeanSquaredLogarithmicError()]

    # compile model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    #
    out_name = output_folder + f"/{_name}_{timestamp}/"
    # fit parametersls
    callbacks = [
        History(),
        EarlyStopping("val_loss", min_delta=1e-2, patience=5),
        ModelCheckpoint(filepath=out_name +
                        "model_{epoch}_{val_loss:.4f},", save_best_only=True),
        TensorBoard(log_dir=f"../Tensorboard_Logs/{_name}_{timestamp}")
    ]

    history = model.fit(X_Y_train, epochs=EPOCHS,
                        validation_data=X_Y_dev, callbacks=callbacks, verbose=1)
    result = model.evaluate(X_Y_test)

    history = history.history

    loss_compare = {
        "train loss": history['loss'][-1],
        "dev loss": history['val_loss'][-1],
        "test loss": result[0]
    }

    dict_keys = []
    for key in history.keys():
        if 'val' not in key:
            dict_keys.append(f"test_{key}")


    test_results = dict(zip(dict_keys, result))

    loss_dict = {**history,**test_results}
    with open(f"{output_folder}model_loss.json", 'w') as fp:
        json.dump(loss_dict, fp)


    return model, loss_compare
