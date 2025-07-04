import os
import numpy as np
import pandas as pd
import mne
from sklearn.model_selection import train_test_split


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class TestDataGenerator:
    def generateTestData():
        print('Generuje dane testowe')
        data_dir = 'dataset/derivatives'
        labels_path = 'multi_labels.csv'

        sampling_rate = 500 
        window_sec = 2       
        step_sec = 1         
        window_size = sampling_rate * window_sec
        step_size = sampling_rate * step_sec

        labels_df = pd.read_csv(labels_path)
        labels_df.set_index('participant_id', inplace=True)

        X = []
        Y = []

        for sub in os.listdir(data_dir):
            sub_path = os.path.join(data_dir, sub, 'eeg')
            if not os.path.isdir(sub_path):
                continue

            eeg_file = [f for f in os.listdir(sub_path) if f.endswith('.set')]
            if not eeg_file:
                continue

            file_path = os.path.join(sub_path, eeg_file[0])
            raw = mne.io.read_raw_eeglab(file_path, preload=True)

            data = raw.get_data()
            data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)

            # Segmentuj dane na okna czasowe
            total_samples = data.shape[1]
            for start in range(0, total_samples - window_size, step_size):
                segment = data[:, start:start+window_size]  # shape: (channels, samples)
                X.append(segment)

                participant_id = sub
                if participant_id not in labels_df.index:
                    continue
                Y.append(labels_df.loc[participant_id].values)

        # Konwersja do NumPy
        X = np.array(X)
        Y = np.array(Y).astype(int)

        print(f'X shape: {X.shape}, Y shape: {Y.shape}')


        # Podział na train/test
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        print('Zapisuje test/train data')
        # np.save("X_test.npy", X_test)
        # np.save("Y_test.npy", Y_test)
        # np.save("X_train.npy", X_train)
        # np.save("Y_train.npy", Y_train)
        print("Dane gotowe do treningu!")

        return X_train, X_test, Y_train, Y_test

if __name__ == '__main__':
    TestDataGenerator.generateTestData()
