from config import LEARNING_RATE, OPTIMIZER, INPUT_SHAPE, TARGET_ACCURACY, \
    STEPS, EPOCHS, VAL_STEPS, LOSS, METRICS, LOG_FILE
from utility.training_setup import StopTraining, alpha_decay, get_data, download_data
from utility.metrics import get_confusion_matrices, get_performance_graphs
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.python.platform import tf_logging
from models.vision_transformer import ViTransformer
from utility.logger import logging, LOG_CONFIG
from models.inceptionv3 import InceptionV3
from models.convitrans import ConViTrans
from models.inresnet import InResNet
from models.resnet50 import ResNet50
from PIL import ImageFile
import tensorflow as tf

# configuring environment
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
LOG_CONFIG['handlers']['file']['filename'] = LOG_FILE
logging.config.dictConfig(LOG_CONFIG)
tf_logging._logging.config.dictConfig(LOG_CONFIG)
tf_logging._logger = logging.getLogger(__name__)
log = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True
log.info("Environment setup complete.")

# acuqiring data
try:
    train_data, test_data, val_data = get_data()
    log.info("Data acquisition complete.")
except FileNotFoundError:
    log.info("Data not found, attempting to download.")
    download_data()
    log.info("Data download complete.")
    train_data, test_data, val_data = get_data()
    log.info("Data acquisition complete.")

# compiling the models
tf.keras.backend.clear_session()
ip = tf.keras.layers.Input(INPUT_SHAPE)
models = [ConViTrans(ip), InResNet(ip),
          ViTransformer(ip), ResNet50(ip), InceptionV3(ip)]
OPTIMIZER = tf.keras.optimizers.RMSprop(LEARNING_RATE)
for model in models:
    model.compile(optimizer=OPTIMIZER,
                loss=LOSS,
                metrics=METRICS)
    model.summary(print_fn = log.debug)
log.info("Models compilation successfully completed.")

# training the models
callbacks = [StopTraining(TARGET_ACCURACY),
            LearningRateScheduler(alpha_decay, verbose=1)]
model_histories = []
for model in models:
    log.info(f"Model training of {model.name} beginning.")
    # normal training of the model:
    history_1 = model.fit(train_data,
                          steps_per_epoch=STEPS,
                          epochs=EPOCHS,
                          callbacks=callbacks[0:1],
                          verbose=1,
                          validation_data=val_data,
                          validation_steps=VAL_STEPS)
    # adaptive second training of the model:
    log.info(f"Adaptive odel training of {model.name} beginning.")
    history_2 = model.fit(train_data,
                      steps_per_epoch=STEPS,
                      epochs=EPOCHS//5,
                      callbacks=callbacks,
                      validation_data=val_data,
                      validation_steps=VAL_STEPS)
    model_histories.append([history_1, history_2])

# getting results of the test data, performance curves and confusion matrices
for model in models:
    log.info(f"Test Accuracy & Loss of {model.name}: {model.evaluate(test_data)}")
figs, axes = get_performance_graphs(model_histories)
displays = get_confusion_matrices(models, test_data)
