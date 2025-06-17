import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(n_channels, n_samples, n_labels): 
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
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        optimizer='adam',
        metrics=['accuracy']
    )
    return model
