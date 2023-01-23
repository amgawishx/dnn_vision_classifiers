import tensorflow as tf
from sys import path
from os import getcwd
path.append("/".join(getcwd().split("\\")[::-1][1:][::-1])+"/dnn_vision_classifiers")
from config import NO_CLASSES, DROPOUT_RATE, INPUT_SHAPE

def ResNet50(ip):
    global INPUT_SHAPE
    global NO_CLASSES
    global DROPOUT_RATE
    x = tf.keras.applications.resnet50.ResNet50(
        weights=None,
        input_shape=INPUT_SHAPE,
        include_top=False,
        pooling="max"
    )(ip)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    x = tf.keras.layers.Dense(NO_CLASSES, activation='softmax')(x)
    return tf.keras.Model(ip, x)
