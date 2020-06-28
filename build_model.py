import tensorflow as tf
import time

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],enable=True)

def lstm_model(history,future,n_feat_cols):
    pass

def train(train_dev_test,train_run_name,base_dir):
    pass