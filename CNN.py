import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from testgene import TestDataGenerator
from sklearn.metrics import classification_report, roc_auc_score, multilabel_confusion_matrix
import matplotlib.pyplot as plt

# ⛔ Wyłączenie oneDNN (wymagane dla channels_first na CPU)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 📥 Dane EEG
X_test = np.load("X_test.npy")
Y_test = np.load("Y_test.npy")
X_train = np.load("X_train.npy")
Y_train = np.load("Y_train.npy")

# 🔍 Parametry
n_channels = X_train.shape[1]
n_samples = X_train.shape[2]
n_labels = Y_train.shape[1]

# 🧱 Model CNN
model = models.Sequential([
    layers.Input(shape=(n_channels, n_samples)),
    
    layers.Conv1D(32, kernel_size=7, activation='relu', padding='same', data_format='channels_first'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2, data_format='channels_first'),
    
    layers.Conv1D(64, kernel_size=5, activation='relu', padding='same', data_format='channels_first'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2, data_format='channels_first'),

    layers.Conv1D(128, kernel_size=3, activation='relu', padding='same', data_format='channels_first'),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling1D(data_format='channels_first'),

    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(n_labels, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# 🏋️‍♂️ Trening
history = model.fit(
    X_train, Y_train,
    validation_data=(X_test, Y_test),
    epochs=10,
    batch_size=64
)

# 💾 Zapis wag
model.save_weights("cnn_model.weights.h5")

# 📊 Ewaluacja
label_names = ['AD', 'FTD', 'CN', 'HasDementia', 'Decline', 'SevereDecline', 'Age>65']

# Predykcja
Y_pred = model.predict(X_test)
Y_pred_bin = (Y_pred > 0.5).astype(int)

# F1 / precision / recall
print("=== Classification report ===")
print(classification_report(Y_test, Y_pred_bin, target_names=label_names))

# AUROC per label
print("\n=== AUROC per label ===")
for i, label in enumerate(label_names):
    try:
        auc = roc_auc_score(Y_test[:, i], Y_pred[:, i])
        print(f"{label}: AUROC = {auc:.3f}")
    except ValueError:
        print(f"{label}: AUROC = brak (etykieta jednostronna)")

# Confusion matrix per label
print("\n=== Confusion Matrix (TP/FP/FN/TN) ===")
cm = multilabel_confusion_matrix(Y_test, Y_pred_bin)
for i, label in enumerate(label_names):
    tn, fp, fn, tp = cm[i].ravel()
    print(f"{label}: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

# 📈 Wykres loss
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title("Loss per epoch")
plt.xlabel("Epoch")
plt.ylabel("Binary Crossentropy Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
