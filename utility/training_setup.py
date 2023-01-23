from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sys import path
from os import getcwd
path.append("/".join(getcwd().split("\\")[::-1][1:][::-1])+"/ML Project")
from config import VALIDATION_SPLIT, TRAIN_DATA_PATH, TEST_DATA_PATH, \
                IMAGE_SIZE, BATCH_SIZE, DECAY_RATE, KAGGLE_DATASET, ABS_DATA_PATH
from kaggle import api

def download_data() -> None:
    global ABS_DATA_PATH
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=ABS_DATA_PATH)

# preparing generators for preprocessing and loading the data
def get_data() -> tuple:
    global VALIDATION_SPLIT
    global TRAIN_DATA_PATH
    global TEST_DATA_PATH
    global IMAGE_SIZE
    global BATCH_SIZE
    train_preproc = ImageDataGenerator(rescale=1/255,
                                                horizontal_flip=True,
                                                vertical_flip=True,
                                                rotation_range=90,
                                                brightness_range=[0, 1],
                                                validation_split=VALIDATION_SPLIT)
    test_preproc = ImageDataGenerator(rescale=1/255)
    train_data = train_preproc.flow_from_directory(
        TRAIN_DATA_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        subset="training",
        shuffle=True
    )
    val_data = train_preproc.flow_from_directory(
        TRAIN_DATA_PATH,
        subset="validation",
        shuffle=True,
        batch_size=BATCH_SIZE,
        target_size=IMAGE_SIZE
    )
    test_data = test_preproc.flow_from_directory(
        TEST_DATA_PATH,
        target_size=IMAGE_SIZE
    )
    return train_data, test_data, val_data

# a call back to terminate training when reaching certain accuracy
class StopTraining(Callback):
    global LOG_FREQ
    def __init__(self, threshold):
        self.THRESHOLD = threshold
        self.i = 0
    def on_epoch_end(self, epoch, logs={}):
        try:
            if(logs.get('accuracy') >= self.THRESHOLD):  
                    self.model.stop_training = True
        except TypeError: print("Accuracy threshold has been reached.")
        self.i+=1

# learning rate decay function
def alpha_decay(epoch, alpha):
    global DECAY_RATE
    if alpha == 0.0: return 0.0001
    decay_step = 1
    if epoch % decay_step == 0 and epoch:
        return alpha * DECAY_RATE**(epoch // decay_step)
    return alpha
