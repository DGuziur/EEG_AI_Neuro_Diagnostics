import os
import numpy as np
import pandas as pd
import mne
from CVerClean import MedicalMultilabelClassifier  # zakładamy, że Twój kod to klasa w tym pliku

def load_eeg_multilabel(data_dir):
    subjects = []
    labels = []
    features = []

    for subject_dir in sorted(os.listdir(data_dir)):
        if subject_dir.startswith("sub-"):
            subject_path = os.path.join(data_dir, subject_dir, f"{subject_dir}_eeg.set")
            if not os.path.exists(subject_path):
                continue

            try:
                raw = mne.io.read_raw_eeglab(subject_path, preload=True, verbose=False)
                raw.filter(0.5, 45, verbose=False)

                psds, freqs = mne.time_frequency.psd_welch(raw, fmin=0.5, fmax=45, verbose=False)
                psds_mean = psds.mean(axis=1)  # średnia moc dla każdej elektrody
                features.append(psds_mean)

                # Sztuczne metadane — w prawdziwym przypadku wczytaj z pliku CSV z MMSE i wiekiem
                idx = int(subject_dir[-2:])
                if idx <= 36:
                    group = "AD"
                    mmse = 17.5
                    age = 66.4
                elif idx <= 59:
                    group = "FTD"
                    mmse = 22.2
                    age = 63.6
                else:
                    group = "CN"
                    mmse = 30
                    age = 67.9

                multilabel = [group]
                if mmse < 24:
                    multilabel.append("MMSE_low")
                if age > 65:
                    multilabel.append("age_above_65")

                labels.append(multilabel)
                subjects.append(subject_dir)

            except Exception as e:
                print(f"Błąd wczytywania {subject_path}: {e}")

    X = pd.DataFrame(features)
    y = labels
    return X, y

# 2. Uruchomienie klasyfikatora
def run_eeg_multilabel_classification(data_dir):
    X, y = load_eeg_multilabel(data_dir)
    print(f"Wczytano {len(X)} próbek z {X.shape[1]} cechami.")

    clf = MedicalMultilabelClassifier()
    clf.fit(
        X, y,
        numerical_columns=X.columns.tolist(),
        balance_method='smote',
        feature_selection_method='rfe',
        tune_params=False
    )

    clf.visualize_results()
    clf.analyze_label_correlations(clf.mlb.transform(y))

# 3. Start
if __name__ == "__main__":
    data_dir = "./dataset/derivatives"  # <-- zmień na ścieżkę do datasetu
    run_eeg_multilabel_classification(data_dir)
