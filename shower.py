import mne
import matplotlib.pyplot as plt
import numpy as np
import time

# Wczytaj dane EEG
file_path = "dataset/derivatives/sub-test-01/sub-087_task-eyesclosed_eeg.set"
raw = mne.io.read_raw_eeglab(file_path, preload=True)
raw.pick('eeg')

# Parametry danych
data, times = raw.get_data(return_times=True)
sfreq = raw.info['sfreq']  # np. 500 Hz

# Parametry okna
window_sec = 2.0           # długość okna (sekundy)
step_sec = 0.1             # przesunięcie okna (sekundy)
window_samples = int(window_sec * sfreq)
step_samples = int(step_sec * sfreq)

n_channels = data.shape[0]
offset = 150               # µV – przesunięcie kanałów w pionie

# Przygotuj wykres
fig, ax = plt.subplots(figsize=(12, 6))
lines = [ax.plot([], [], label=raw.ch_names[i])[0] for i in range(n_channels)]

ax.set_xlim(0, window_sec)
ax.set_ylim(-offset, offset * n_channels)
ax.set_xlabel("Czas (s)")
ax.set_title("EEG – Symulacja badania w czasie rzeczywistym (okno 2s, krok 0.1s)")
ax.legend(loc='upper right')
plt.ion()
plt.show()

# Symulacja przesuwającego się okna
for start in range(0, data.shape[1] - window_samples, step_samples):
    end = start + window_samples
    t = times[start:end]

    for i in range(n_channels):
        signal = data[i, start:end] * 1e6 + i * offset  # µV + przesunięcie
        lines[i].set_data(t - t[0], signal)

    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(step_sec)  # Czekaj tyle co przesunięcie

plt.ioff()
plt.show()
