import tensorflow as tf
from .vision_transformer import ViTransformer
from .inresnet import NetStem

def ConViTrans(ip):
    x = NetStem(ip)
    return ViTransformer(x, ip_tensor=ip, latent_dim=18, encoder_mlp=[75, 18],
                      cutting_sizes=[1,6,6,1], no_patches = 9, name="ConViTrans")