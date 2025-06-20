import numpy as np

# Zakładamy że masz dane z testgene
X_test = np.load("X_test.npy")
Y_test = np.load("Y_test.npy")
X_train = np.load("X_train.npy")
Y_train = np.load("Y_train.npy")

# Transpozycja na (samples, time, channels) – wymagane przez Transformer
X_train = np.transpose(X_train, (0, 2, 1))  # (samples, 1000, 19)
X_test = np.transpose(X_test, (0, 2, 1))    # (samples, 1000, 19)

import tensorflow as tf
from tensorflow.keras import layers, models, Layer

class PositionalEncoding(Layer):
    def __init__(self, maxlen, dim):
        super().__init__()
        pos = tf.range(maxlen, dtype=tf.float32)[:, tf.newaxis]  # (maxlen, 1)
        i = tf.range(dim, dtype=tf.float32)[tf.newaxis, :]       # (1, dim)
        
        angle_rates = 1.0 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(dim, tf.float32))
        angle_rads = pos * angle_rates  # (maxlen, dim)
        
        # Zastosuj sin i cos do odpowiednich kanałów
        sin_part = tf.math.sin(angle_rads[:, 0::2])
        cos_part = tf.math.cos(angle_rads[:, 1::2])
        
        # Przeplot sin i cos (interleave)
        even_dims = tf.shape(sin_part)[1]
        odd_dims = tf.shape(cos_part)[1]
        min_dims = tf.minimum(even_dims, odd_dims)
        
        angle_encoded = tf.reshape(tf.stack(
            [sin_part[:, :min_dims], cos_part[:, :min_dims]],
            axis=-1
        ), (maxlen, -1))

        # Jeśli liczba wymiarów jest nieparzysta – dopełnij
        if dim % 2 == 1:
            angle_encoded = tf.concat([angle_encoded, sin_part[:, -1:]], axis=-1)

        self.pos_encoding = angle_encoded[tf.newaxis, ...]  # (1, maxlen, dim)

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]


def build_transformer_model(input_shape, num_labels, head_size=64, num_heads=4, ff_dim=128, num_transformer_blocks=2):
    inputs = tf.keras.Input(shape=input_shape)

    x = PositionalEncoding(maxlen=1000, dim=19)(inputs)
    for _ in range(num_transformer_blocks):
        # Normalization + Multi-head self-attention
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        x1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(x1, x1)
        x = layers.Add()([x, x1])

        # Feed-forward block
        x2 = layers.LayerNormalization(epsilon=1e-6)(x)
        x2 = layers.Dense(ff_dim, activation='relu')(x2)
        x2 = layers.Dense(input_shape[-1])(x2)
        x = layers.Add()([x, x2])

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_labels, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)
    return model

input_shape = X_train.shape[1:]  # (1000, 19)
num_labels = Y_train.shape[1]

model = build_transformer_model(input_shape, num_labels)

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# Trening
history = model.fit(
    X_train, Y_train,
    validation_data=(X_test, Y_test),
    epochs=10,
    batch_size=64
)

# Zapis modelu
model.save_weights("transformer_model_PosEnc_v2.weights.h5")


from sklearn.metrics import classification_report, roc_auc_score, multilabel_confusion_matrix

Y_pred = model.predict(X_test)
Y_pred_bin = (Y_pred > 0.5).astype(int)

print("=== Transformer classification report ===")
print(classification_report(Y_test, Y_pred_bin, target_names=[
    'AD', 'FTD', 'CN', 'HasDementia', 'Decline', 'SevereDecline', 'Age>65'
]))

print("\n=== AUROC per label ===")
for i, label in enumerate(['AD', 'FTD', 'CN', 'HasDementia', 'Decline', 'SevereDecline', 'Age>65']):
    try:
        auc = roc_auc_score(Y_test[:, i], Y_pred[:, i])
        print(f"{label}: AUC = {auc:.3f}")
    except ValueError:
        print(f"{label}: brak AUROC")

