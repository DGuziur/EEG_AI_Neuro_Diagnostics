import os
import numpy as np
import mne
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tensorflow.keras import layers, models, Input, Layer

eeg_file_path = "dataset\derivatives\sub-test-01\sub-087_task-eyesclosed_eeg.set"
model_weights_path = "cnn_transformer_hybrid.weights.h5"
labels = ['AD', 'FTD', 'CN', 'HasDementia', 'Decline', 'SevereDecline', 'Age>65']

class PositionalEncoding(Layer):
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
    inputs = Input(shape=input_shape)
    x = layers.Conv1D(64, kernel_size=7, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = PositionalEncoding(maxlen=input_shape[0], dim=128)(x)

    for _ in range(num_transformer_blocks):
        attn_input = layers.LayerNormalization(epsilon=1e-6)(x)
        attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(attn_input, attn_input)
        x = layers.Add()([x, attn_output])

        ffn_input = layers.LayerNormalization(epsilon=1e-6)(x)
        ffn_output = layers.Dense(ff_dim, activation='relu')(ffn_input)
        ffn_output = layers.Dense(128)(ffn_output)
        x = layers.Add()([x, ffn_output])

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_labels, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    return model

def smooth_predictions(probs, window_size=5):
    smoothed = np.copy(probs)
    for i in range(probs.shape[1]):
        smoothed[:, i] = np.convolve(probs[:, i], np.ones(window_size) / window_size, mode='same')
    return smoothed

def run_demo(eeg_path):
    print("Ładowanie modelu...")
    input_shape = (1000, 19)
    model = build_cnn_transformer_model(input_shape=input_shape, num_labels=len(labels))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.load_weights(model_weights_path)

    print("Wczytywanie EEG...")
    raw = mne.io.read_raw_eeglab(eeg_path, preload=True)
    data = raw.get_data()
    data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)

    window_size = 1000
    step_size = 500

    segments = []
    for start in range(0, data.shape[1] - window_size, step_size):
        segment = data[:, start:start + window_size].T
        segments.append(segment)
    segments = np.array(segments)
    print(f"Liczba segmentów: {segments.shape[0]}")

    print("Symulacja predykcji w czasie rzeczywistym...")
    Y_pred = model.predict(segments)
    Y_pred_smoothed = smooth_predictions(Y_pred, window_size=100)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels, Y_pred_smoothed[0], color='skyblue')
    ax.set_xlim(0, 1)
    ax.set_title("Live predykcja modelu hybrydowego EEG")

    def update(frame):
        for i, b in enumerate(bars):
            b.set_width(Y_pred_smoothed[frame, i])
            b.set_color('orange' if Y_pred_smoothed[frame, i] > 0.5 else 'lightgray')
        ax.set_xlabel(f"Klatka {frame + 1} / {len(Y_pred_smoothed)}")
        return bars

    ani = animation.FuncAnimation(fig, update, frames=len(Y_pred_smoothed), interval=100, repeat=False)
    plt.show()

if __name__ == '__main__':
    run_demo(eeg_file_path)
