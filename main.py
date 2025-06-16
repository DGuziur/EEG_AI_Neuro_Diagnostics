import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.python.keras.layers import Input, Conv2D, LayerNormalization, MultiHeadAttention
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import AdamW
import numpy as np

class EEGSpecAugment(tf.keras.layers.Layer):
    """Augmentacja specyficzna dla EEG"""
    def __init__(self, freq_mask_param=10, time_mask_param=50):
        super().__init__()
        self.freq_mask = freq_mask_param
        self.time_mask = time_mask_param
    
    def call(self, inputs):
        if tf.random.uniform(()) > 0.5:
            return inputs
        
        # Maskowanie częstotliwości
        freq_mask = tf.random.uniform((1, self.freq_mask, 1, 1), maxval=inputs.shape[2])
        inputs = tf.where(
            (tf.range(inputs.shape[2])[None,None,:,None] >= freq_mask) & 
            (tf.range(inputs.shape[2])[None,None,:,None] < freq_mask + self.freq_mask),
            tf.zeros_like(inputs), inputs)
        
        # Maskowanie czasu
        time_mask = tf.random.uniform((1, 1, self.time_mask, 1), maxval=inputs.shape[3])
        inputs = tf.where(
            (tf.range(inputs.shape[3])[None,None,None,:] >= time_mask) & 
            (tf.range(inputs.shape[3])[None,None,None,:] < time_mask + self.time_mask),
            tf.zeros_like(inputs), inputs)
        return inputs

class ChannelAttention(tf.keras.layers.Layer):
    """Mechanizm uwagi między kanałami EEG"""
    def build(self, input_shape):
        self.gap = tf.keras.layers.GlobalAvgPool1D()
        self.dense1 = tf.keras.layers.Dense(input_shape[-1]//8, activation='relu')
        self.dense2 = tf.keras.layers.Dense(input_shape[-1], activation='sigmoid')
    
    def call(self, inputs):
        x = self.gap(inputs)
        x = self.dense1(x)
        weights = self.dense2(x)
        return inputs * weights[:,None,:]

def create_hybrid_model(input_shape=(19, 1000, 1), num_classes=5):
    inputs = Input(input_shape)
    
    # Augmentacja
    x = EEGSpecAugment()(inputs)
    
    # Część CNN
    x = Conv2D(32, (1, 25), activation='elu', padding='same')(x)  # Filtrowanie przestrzenne
    x = Conv2D(64, (3, 1), activation='elu', padding='valid')(x)  # Filtrowanie czasowe
    x = tf.keras.layers.MaxPool2D((1, 4))(x)
    
    # Transformacja do sekwencji
    seq_len = x.shape[2]
    x = tf.reshape(x, (-1, x.shape[1], seq_len * x.shape[3]))
    
    # Normalizacja
    x = LayerNormalization()(x)
    
    # Multi-head attention
    attn = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = tf.keras.layers.Concatenate()([x, attn])
    
    # Uwaga między kanałami
    x = ChannelAttention()(x)
    
    # Klasyfikacja
    x = tf.keras.layers.GlobalAvgPool1D()(x)
    x = tf.keras.layers.Dense(128, activation='gelu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    
    # Custom loss z wagami klas
    def weighted_loss(y_true, y_pred):
        weights = tf.where(y_true == 1, 2.0, 1.0)  # Większa kara za FP
        return tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(y_true, y_pred) * weights)
    
    model.compile(
        optimizer=AdamW(learning_rate=3e-4, weight_decay=1e-4),
        loss=weighted_loss,
        metrics=[
            tf.keras.metrics.AUC(name='auroc'),
            tf.keras.metrics.PrecisionAtRecall(0.9, name='par90')
        ])
    return model

# Przykład użycia
model = create_hybrid_model()
model.summary()

# Callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True),
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
]

# Trening (przykładowe dane)
X_train = np.random.randn(100, 19, 1000, 1)  # 100 próbek, 19 kanałów, 1000 punktów czasowych
y_train = np.random.randint(0, 2, (100, 5))   # 5 klas wieloetykietowych

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    callbacks=callbacks
)

from sklearn.metrics import multilabel_confusion_matrix
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    # Predykcja
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)
    
    # Macierze pomyłek per klasa
    cm = multilabel_confusion_matrix(y_test, y_pred)
    
    # Wizualizacja
    fig, axes = plt.subplots(1, len(cm), figsize=(20, 5))
    for i, (matrix, ax) in enumerate(zip(cm, axes)):
        ax.matshow(matrix, cmap='Blues')
        ax.set_title(f'Class {i}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    plt.show()
    
    # Raport metryk
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred, target_names=[f'Class_{i}' for i in range(y_test.shape[1])]))

# Przykładowe wywołanie
evaluate_model(model, X_train[:20], y_train[:20])  # Na prawdziwych danych podmienić