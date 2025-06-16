from testgene import TestDataGenerator
import numpy as np

# Zakładamy że masz dane z testgene
X_train, X_test, Y_train, Y_test = TestDataGenerator.generateTestData()

# Transpozycja na (samples, time, channels) – wymagane przez Transformer
X_train = np.transpose(X_train, (0, 2, 1))  # (samples, 1000, 19)
X_test = np.transpose(X_test, (0, 2, 1))    # (samples, 1000, 19)

import tensorflow as tf
from tensorflow.keras import layers, models

def build_transformer_model(input_shape, num_labels, head_size=32, num_heads=2, ff_dim=64, num_transformer_blocks=1):
    inputs = tf.keras.Input(shape=input_shape)

    x = inputs
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
model.save_weights("transformer_model.weights.h5")


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

