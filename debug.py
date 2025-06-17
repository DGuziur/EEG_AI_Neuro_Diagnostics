import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from CNN_model import build_model  # zakładamy, że masz funkcję build_model
import matplotlib.pyplot as plt

# === 1. Wczytaj dane ===
X_test = np.load("X_test.npy")  # shape: (N, 19, 1000)
Y_test = np.load("Y_test.npy")

# Wczytaj model i wagi
n_channels, n_samples = X_test.shape[1], X_test.shape[2]
n_labels = Y_test.shape[1]
labels = ['AD', 'FTD', 'CN', 'HasDementia', 'Decline', 'SevereDecline', 'Age>65']

model = build_model(n_channels, n_samples, n_labels)
model.load_weights("cnn_model.weights.h5")

# === 2. Rozkład etykiet ===
print("\n=== Rozkład etykiet (TEST) ===")
pd.DataFrame(Y_test, columns=labels).sum().plot(kind='bar', title="Liczba próbek na etykietę")
plt.ylabel("Liczba próbek")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# === 3. Statystyki danych wejściowych ===
print("\n=== Statystyki sygnału (X_test) ===")
print("Min:", np.min(X_test))
print("Max:", np.max(X_test))
print("Średnia:", np.mean(X_test))
print("Stddev:", np.std(X_test))

# === 4. Porównanie pred vs true ===
print("\n=== Przykładowe predykcje (raw sigmoid) ===")
for i in range(3):
    x = X_test[i][np.newaxis, :, :]  # (1, 19, 1000)
    y_true = Y_test[i]
    y_pred = model.predict(x, verbose=0)[0]
    
    print(f"\nPróbka {i}")
    for j, label in enumerate(labels):
        print(f"{label:15s} | True: {int(y_true[j])} | Pred: {y_pred[j]:.4f}")

# === 5. Histogram prawdopodobieństw ===
all_preds = model.predict(X_test, verbose=1)
plt.figure(figsize=(10, 5))
for i in range(n_labels):
    plt.hist(all_preds[:, i], bins=50, alpha=0.5, label=labels[i])
plt.title("Rozkład wyjść sigmoid (dla każdej etykiety)")
plt.xlabel("Prawdopodobieństwo")
plt.ylabel("Liczba próbek")
plt.legend()
plt.tight_layout()
plt.show()