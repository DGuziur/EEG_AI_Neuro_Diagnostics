import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import mne
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load data
raw = mne.io.read_raw_eeglab('dataset/derivatives/sub-001/eeg/sub-001_task-eyesclosed_eeg.set', preload=True)

# 2. Create fixed-length epochs (no events needed)
epochs = mne.make_fixed_length_epochs(raw, duration=2.0, overlap=1.0, preload=True)
print(f"Created {len(epochs)} epochs")  # Should show e.g., "Created 100 epochs"

# 3. Get data (X) - we'll create dummy labels (y) since this is resting-state
X = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
n_epochs, n_channels, n_times = X.shape

# Create dummy labels (replace with real labels if available)
y = np.zeros(n_epochs)  # Single class (resting-state)

# 4. Normalize data per channel
scaler = StandardScaler()
X = scaler.fit_transform(X.reshape(n_epochs, -1)).reshape(n_epochs, n_channels, n_times)

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Build model (using n_times for dynamic input length)
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(n_channels, n_times)),
    tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary output (dummy task)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 7. Train (with dummy labels)
history = model.fit(X_train, y_train, epochs=10, batch_size=32,
                    validation_data=(X_test, y_test))

# 8. Plot training history
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Test')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

model.save('eeg_model.keras')