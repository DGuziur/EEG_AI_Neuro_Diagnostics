import tensorflow as tf
from tensorflow.keras import layers, models, Input
import numpy as np

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, maxlen, dim):
        super().__init__()
        pos = tf.range(maxlen, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(dim, dtype=tf.float32)[tf.newaxis, :]
        angle_rates = 1.0 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(dim, tf.float32))
        angle_rads = pos * angle_rates
        sin_part = tf.math.sin(angle_rads[:, 0::2])
        cos_part = tf.math.cos(angle_rads[:, 1::2])
        angle_encoded = tf.concat([sin_part, cos_part], axis=-1)
        self.pos_encoding = angle_encoded[tf.newaxis, ...]

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]

def build_cnn_transformer_model(input_shape, num_labels,
                                head_size=64, num_heads=4, ff_dim=128, num_transformer_blocks=2):
    inputs = Input(shape=input_shape)  # (1000, 19)

    # 🧠 Wstępna ekstrakcja cech czasowych przez CNN
    x = layers.Conv1D(64, kernel_size=7, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    # 🎯 Positional encoding + Transformer
    x = PositionalEncoding(maxlen=input_shape[0], dim=128)(x)

    for _ in range(num_transformer_blocks):
        attn_input = layers.LayerNormalization(epsilon=1e-6)(x)
        attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(attn_input, attn_input)
        x = layers.Add()([x, attn_output])

        ffn_input = layers.LayerNormalization(epsilon=1e-6)(x)
        ffn_output = layers.Dense(ff_dim, activation='relu')(ffn_input)
        ffn_output = layers.Dense(128)(ffn_output)
        x = layers.Add()([x, ffn_output])

    # 📦 Agregacja i klasyfikacja
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_labels, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    return model

X_test = np.load("X_test.npy")
Y_test = np.load("Y_test.npy")
X_train = np.load("X_train.npy")
Y_train = np.load("Y_train.npy")

X_train = np.transpose(X_train, (0, 2, 1))
X_test = np.transpose(X_test, (0, 2, 1))

model = build_cnn_transformer_model(
    input_shape=X_train.shape[1:],  # (1000, 19)
    num_labels=Y_train.shape[1],
    head_size=64,
    num_heads=4,
    ff_dim=128,
    num_transformer_blocks=2
)

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    X_train, Y_train,
    validation_data=(X_test, Y_test),
    epochs=10,
    batch_size=64
)

model.save_weights("cnn_transformer_hybrid.weights.h5")