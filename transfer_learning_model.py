import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    Flatten,
    Dense,
    Dropout,
    MaxPooling2D,
    BatchNormalization,
)
from tensorflow.keras.models import Sequential, load_model

from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B1
from tensorflow.keras import layers

IMG_SIZE=224
IMAGE_SHAPE = (IMG_SIZE,IMG_SIZE,3) 

def build_model():
    pretrained_model = EfficientNetV2B1(include_top=False,
                                      input_shape=IMAGE_SHAPE,
                                      weights='imagenet',
                                      pooling='avg')
    
    # freeze pretrained weight
    pretrained_model.trainable = False
    
    # rebuild top
    inputs = layers.Input(shape=IMAGE_SHAPE)
    x = BatchNormalization()(pretrained_model.output)
    x = Flatten()(x)
    # x = layers.GlobalAveragePooling2D(name='avg_pool')(pretrained_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    x = Dropout(0.2, name='top_dropout')(x)
    outputs = Dense(1, activation='sigmoid', name='pred')(x)

    model = tf.keras.Model(pretrained_model.input, outputs, name='EfficientNet')
    
    
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=0.001,
        decay_steps=1200,
        decay_rate=0.35,
    )
    
    model.compile(
        optimizer= tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(name="roc_auc")]
    )
    
    return model