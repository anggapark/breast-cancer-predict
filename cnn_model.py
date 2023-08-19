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


def create_model():
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(256, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ]
    )

    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=0.001,
        decay_steps=1200,
        decay_rate=0.35,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="roc_auc")],
    )

    return model


if __name__ == "__main__":
    model = create_model()
    model.summary()
