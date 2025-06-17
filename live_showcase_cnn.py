import mne
import numpy as np
import matplotlib.pyplot as plt
import time
from CNN_model import build_model

# === Ścieżka do EEG (.set)
file_path = "dataset/derivatives/sub-077/eeg/sub-077_task-eyesclosed_eeg.set"

# === Wczytaj dane EEG
raw = mne.io.read_raw_eeglab(file_path, preload=True)
raw.pick('eeg')
data, times = raw.get_data(return_times=True)  # data shape: (n_channels, n_samples)
sfreq = raw.info['sfreq']  # np. 500 Hz
ch_names = raw.ch_names
n_channels = data.shape[0]
n_labels = 7
labels = ['AD', 'FTD', 'CN', 'HasDementia', 'Decline', 'SevereDecline', 'Age>65']

# === Parametry modelu
n_samples = 1000  # 2 sekundy przy 500Hz
model = build_model(n_channels, n_samples, n_labels)
model.load_weights("cnn_model.weights.h5")

# === Parametry przesuwającego się okna
window_sec = 2.0
step_sec = 0.1
window_samples = int(window_sec * sfreq)
step_samples = int(step_sec * sfreq)

# === Wykres setup
offset = 150  # µV
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
lines = [ax1.plot([], [], label=ch)[0] for ch in ch_names]
bar_container = ax2.bar(labels, [0] * n_labels, color='gray')

ax1.set_xlim(0, window_sec)
ax1.set_ylim(-offset, offset * n_channels)
ax1.set_title("EEG – Symulacja badania 2s (co 0.1s)")
ax1.set_xlabel("Czas (s)")
ax1.legend(loc='upper right')

ax2.set_ylim(0, 1)
ax2.set_ylabel("Prawdopodobieństwo")
ax2.set_title("Predykcja CNN")

plt.ion()
plt.tight_layout()
plt.show()

# === Symulacja przesuwającego się okna
for start in range(0, data.shape[1] - window_samples, step_samples):
    end = start + window_samples
    t = times[start:end]

    # Przygotuj dane dla modelu CNN
    window = data[:, start:end]             # shape: (19, 1000)
    model_input = window[np.newaxis, :, :]  # shape: (1, 19, 1000)

    # === Predykcja
    y_pred = model.predict(model_input, verbose=0)[0]

    # === Rysuj EEG
    for i in range(n_channels):
        signal = window[i] * 1e6 + i * offset
        lines[i].set_data(t - t[0], signal)

    # === Rysuj predykcje
    for i, bar in enumerate(bar_container):
        bar.set_height(y_pred[i])
        # Interpolacja koloru: czerwony (0) → żółty (0.5) → zielony (1)
        prob = y_pred[i]
        if prob < 0.5:
            r = 1.0
            g = 2 * prob  # 0 → 0, 0.5 → 1
        else:
            r = 2 * (1 - prob)  # 0.5 → 1, 1 → 0
            g = 1.0
        b = 0.0
        bar.set_color((r, g, b))

    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(step_sec)

plt.ioff()
plt.show()
