import tensorflow as tf
from sys import path
from os import getcwd
path.append("/".join(getcwd().split("\\")[::-1][1:][::-1])+"/dnn_vision_classifiers")
from config import NO_CLASSES, NO_LAYERS, DROPOUT_RATE

# Define an inception layer
def InceptionLayer(x: tf.Tensor):
    a = tf.keras.layers.Conv2D(64, (1,1))(x)
    b = tf.keras.layers.Conv2D(64, (1,1))(x)
    c = tf.keras.layers.Conv2D(64, (1,1))(x)
    d = tf.keras.layers.MaxPool2D(pool_size=(2,2),padding='same')(x)
    c = tf.keras.layers.MaxPool2D(pool_size=(2,2),padding='same')(c)
    a = tf.keras.layers.Conv2D(64, (5,5),
                               padding='same', strides=(2,2))(a)
    b = tf.keras.layers.Conv2D(64, (3,3),
                              padding='same',strides=(2,2))(b)
    d = tf.keras.layers.Conv2D(64, (1,1))(d)
    op = tf.concat([a,b,c,d], -1)
    return op

# Stem of InResNet
def NetStem(x: tf.Tensor):
    x = tf.keras.layers.Conv2D(16, (3,3))(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2),padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32, (3,3))(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2),padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x

# My very own -probably not working- net.
def InResNet(ip):
    global NO_LAYERS
    global NO_CLASSES
    global DROPOUT_RATE
    x = NetStem(ip)
    for i in range(NO_LAYERS):
        x = InceptionLayer(x)
        x = InceptionLayer(x)
        x = tf.keras.layers.Activation('relu')(x)+x
    x = InceptionLayer(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    op = tf.keras.layers.Dense(NO_CLASSES, activation='softmax')(x)
    return tf.keras.Model(ip, op, name = "InResNet")
