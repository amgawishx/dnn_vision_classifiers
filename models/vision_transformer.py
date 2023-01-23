from sys import path
from os import getcwd
path.append("/".join(getcwd().split("\\")[::-1][1:][::-1])+"/ML Project")
from config import VIT_CLASS_MLP, VIT_CUT_SIZES, VIT_ENCODER_MLP, \
    VIT_NO_PATCHES, VIT_NO_HEADS, NO_CLASSES, NO_LAYERS, DROPOUT_RATE, \
    VIT_LATENT_DIM
import tensorflow as tf


class CuttingLayer(tf.keras.layers.Layer):
    global VIT_CUT_SIZES

    def __init__(self, sizes=None, padding='VALID'):
        super(CuttingLayer, self).__init__()
        self.sizes = sizes if sizes else VIT_CUT_SIZES
        self.padding = padding

    def call(self, ip):
        patches = tf.image.extract_patches(ip, sizes=self.sizes,
                                           strides=self.sizes, padding=self.padding,
                                           rates=[1, 1, 1, 1])
        return tf.reshape(patches, [tf.shape(ip)[0], -1, patches.shape[-1]])


class PositionEncoder(tf.keras.layers.Layer):
    global VIT_LATENT_DIM
    global VIT_NO_PATCHES

    def __init__(self, latent_dim=None, no_patches=None):
        latent_dim = latent_dim if latent_dim else VIT_LATENT_DIM
        no_patches = no_patches if no_patches else VIT_NO_PATCHES
        super(PositionEncoder, self).__init__()
        self.no_patches = no_patches
        self.linear_embedding = tf.keras.layers.Dense(latent_dim)
        self.embedding_pos = tf.keras.layers.Embedding(no_patches, latent_dim)

    def call(self, ip):
        positions = tf.range(start=0, limit=self.no_patches, delta=1)
        return self.linear_embedding(ip)+self.embedding_pos(positions)


def MLP(ip, layers, dropout=None):
    global DROPOUT_RATE
    dropout = dropout if dropout else DROPOUT_RATE
    x = ip
    for layer_units in layers:
        x = tf.keras.layers.Dense(layer_units, activation=tf.nn.gelu)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    return x


def TransformerEncoder(x, no_heads=None, latent_dim=None,
                       encoder_mlp=None):
    no_heads = no_heads if no_heads else VIT_NO_HEADS
    encoder_mlp = encoder_mlp if encoder_mlp else VIT_ENCODER_MLP
    latent_dim = latent_dim if latent_dim else VIT_LATENT_DIM
    x_norm = tf.keras.layers.LayerNormalization()(x)
    y = tf.keras.layers.MultiHeadAttention(num_heads=no_heads,
                                           key_dim=latent_dim)(x_norm, x_norm)
    x = x+y
    x_norm = tf.keras.layers.LayerNormalization()(x)
    y = MLP(x_norm, encoder_mlp)
    return x+y


def ViTransformer(ip, ip_tensor=None, cutting_sizes=None, latent_dim=None, no_patches=None,
                  dropout=None, no_heads=None, encoder_mlp=None, name="ViTransformer"):
    global VIT_CLASS_MLP
    global NO_LAYERS
    global NO_CLASSES
    ip_tensor = ip_tensor if ip_tensor else ip
    x = CuttingLayer(sizes=cutting_sizes)(ip)
    x = PositionEncoder(latent_dim=latent_dim, no_patches=no_patches)(x)
    for i in range(NO_LAYERS):
        x = TransformerEncoder(x, no_heads=no_heads,
                               latent_dim=latent_dim, encoder_mlp=encoder_mlp)
    x = tf.keras.layers.Flatten()(x)
    x = MLP(x, VIT_CLASS_MLP, dropout=dropout)
    op = tf.keras.layers.Dense(NO_CLASSES, activation='softmax')(x)
    return tf.keras.Model(ip_tensor, op, name=name)
