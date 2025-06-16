import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, multilabel_confusion_matrix
from tensorflow.keras.models import load_model

# === ZAŁADUJ DANE ===
# Zakładam, że masz zapisane testowe dane w plikach .npy
X_test = np.load("X_test.npy")
Y_test = np.load("Y_test.npy")

# === ZAŁADUJ MODEL ===
# Jeśli zapisałeś cały model:
model = load_model("cnn_model_full.keras")

# Jeśli tylko wagi i masz kod architektury w innym pliku, wczytaj go ręcznie + model.load_weights(...)

# === PREDYKCJA ===
Y_pred = model.predict(X_test)
Y_pred_bin = (Y_pred > 0.5).astype(int)

# === NAZWY ETYKIET ===
label_names = ['AD', 'FTD', 'CN', 'HasDementia', 'Decline', 'SevereDecline', 'Age>65']

# === KLASYFIKACJA (F1, precision, recall) ===
print("=== Klasyfikacja per etykieta ===")
print(classification_report(Y_test, Y_pred_bin, target_names=label_names))

# === AUROC ===
print("\n=== AUROC per etykieta ===")
for i, label in enumerate(label_names):
    try:
        auc = roc_auc_score(Y_test[:, i], Y_pred[:, i])
        print(f"{label}: AUC = {auc:.3f}")
    except ValueError:
        print(f"{label}: AUC = brak (etykieta jednostronna)")

# === CONFUSION MATRIX ===
print("\n=== Confusion matrix (multi-label) ===")
cm = multilabel_confusion_matrix(Y_test, Y_pred_bin)

for i, label in enumerate(label_names):
    tn, fp, fn, tp = cm[i].ravel()
    print(f"{label}: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
