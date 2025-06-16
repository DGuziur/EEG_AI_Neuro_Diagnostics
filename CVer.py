#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Klasyfikacja wieloetykietowa zbiorów medycznych
====================================================
Implementacja kompleksowego systemu klasyfikacji wieloetykietowej
dla danych medycznych wraz z analizą wydajności różnych metod.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            hamming_loss, jaccard_score, classification_report)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, csr_matrix
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import time
import itertools
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.inspection import permutation_importance
from scipy.stats import rankdata
import shap

# Ignorowanie ostrzeżeń dla czytelności
warnings.filterwarnings('ignore')

class MedicalMultilabelClassifier:
    """
    Klasa implementująca klasyfikację wieloetykietową dla danych medycznych.
    """
    
    def __init__(self, random_state=42):
        """
        Inicjalizacja klasyfikatora.
        
        Parametry:
        ----------
        random_state : int, domyślnie 42
            Ziarno generatora liczb losowych dla powtarzalności wyników.
        """
        self.random_state = random_state
        self.mlb = MultiLabelBinarizer()
        self.models = {}
        self.feature_names = None
        self.best_model_name = None
        self.best_model = None
        self.feature_importances = {}
        
    def preprocess_data(self, X, text_columns=None, categorical_columns=None, numerical_columns=None):
        """
        Przetwarzanie danych wejściowych.
        
        Parametry:
        ----------
        X : pandas.DataFrame
            Dane wejściowe.
        text_columns : list, opcjonalnie
            Lista nazw kolumn zawierających dane tekstowe.
        categorical_columns : list, opcjonalnie
            Lista nazw kolumn zawierających dane kategoryczne.
        numerical_columns : list, opcjonalnie
            Lista nazw kolumn zawierających dane numeryczne.
            
        Zwraca:
        -------
        X_processed : numpy.ndarray
            Przetworzone dane wejściowe.
        """
        processed_parts = []
        self.feature_names = []
        
        # Przetwarzanie danych numerycznych
        if numerical_columns:
            X_num = X[numerical_columns].fillna(X[numerical_columns].mean())
            scaler = StandardScaler()
            X_num_scaled = scaler.fit_transform(X_num)
            processed_parts.append(X_num_scaled)
            self.feature_names.extend(numerical_columns)
        
        # Przetwarzanie danych kategorycznych
        if categorical_columns:
            X_cat = pd.get_dummies(X[categorical_columns], drop_first=True)
            processed_parts.append(X_cat.values)
            self.feature_names.extend(X_cat.columns.tolist())
        
        # Przetwarzanie danych tekstowych
        if text_columns:
            for col in text_columns:
                try:
                    nltk.data.find('corpora/stopwords')
                except LookupError:
                    nltk.download('stopwords', quiet=True)
                    nltk.download('punkt', quiet=True)
                
                # Przetwarzanie tekstu z użyciem TF-IDF
                tfidf = TfidfVectorizer(max_features=100, stop_words=stopwords.words('english'))
                X_text = tfidf.fit_transform(X[col].fillna('').astype(str))
                processed_parts.append(X_text)
                self.feature_names.extend([f"{col}_{i}" for i in range(X_text.shape[1])])
        
        # Połączenie wszystkich przetworzonych części
        if len(processed_parts) == 1:
            X_processed = processed_parts[0]
        else:
            # Konwersja wszystkich części do formatu sparse matrix dla efektywności
            for i in range(len(processed_parts)):
                if not isinstance(processed_parts[i], csr_matrix):
                    processed_parts[i] = csr_matrix(processed_parts[i])
            X_processed = hstack(processed_parts).toarray()
        
        return X_processed
    
    def handle_imbalance(self, X, y, method='smote', sampling_strategy='auto'):
        """
        Obsługa problemu niezbalansowania klas.
        
        Parametry:
        ----------
        X : numpy.ndarray
            Dane wejściowe.
        y : numpy.ndarray
            Etykiety (w postaci binarnej).
        method : str, domyślnie 'smote'
            Metoda obsługi niezbalansowania klas ('smote', 'undersampling', 'hybrid').
        sampling_strategy : dict lub str, domyślnie 'auto'
            Strategia próbkowania.
            
        Zwraca:
        -------
        X_resampled : numpy.ndarray
            Zbalansowane dane wejściowe.
        y_resampled : numpy.ndarray
            Zbalansowane etykiety.
        """
        # Analiza rozkładu klas
        class_counts = np.sum(y, axis=0)
        print(f"Rozkład klas przed balansowaniem: {class_counts}")
        
        # Obliczamy, które klasy są niezbalansowane
        mean_samples = np.mean(class_counts)
        minority_classes = np.where(class_counts < mean_samples * 0.5)[0]
        
        if len(minority_classes) == 0:
            print("Dane są względnie zbalansowane, pomijam balansowanie.")
            return X, y
        
        print(f"Wykryto {len(minority_classes)} klas mniejszościowych.")
        
        X_resampled, y_resampled = X.copy(), y.copy()
        
        if method == 'smote':
            # Zastosowanie SMOTE dla każdej klasy mniejszościowej
            for i in minority_classes:
                # Używamy tylko tych cech, które są istotne dla tej klasy
                smote = SMOTE(sampling_strategy={i: int(mean_samples)}, random_state=self.random_state)
                try:
                    X_temp, y_temp = smote.fit_resample(X, y[:, i])
                    # Aktualizujemy tylko próbki klasy mniejszościowej
                    new_samples_mask = (y_temp == 1) & (y[:, i] == 0)
                    if np.any(new_samples_mask):
                        X_resampled = np.vstack([X_resampled, X_temp[new_samples_mask]])
                        new_labels = np.zeros((np.sum(new_samples_mask), y.shape[1]))
                        new_labels[:, i] = 1
                        y_resampled = np.vstack([y_resampled, new_labels])
                except ValueError as e:
                    print(f"Błąd podczas stosowania SMOTE dla klasy {i}: {e}")
                    print("Pomijam tę klasę.")
        
        elif method == 'undersampling':
            # Undersampling klas większościowych
            majority_classes = np.where(class_counts > mean_samples * 1.5)[0]
            for i in majority_classes:
                rus = RandomUnderSampler(sampling_strategy={i: int(mean_samples)}, random_state=self.random_state)
                try:
                    X_resampled, y_resampled = rus.fit_resample(X_resampled, y_resampled[:, i])
                except ValueError as e:
                    print(f"Błąd podczas stosowania undersamplingu dla klasy {i}: {e}")
        
        elif method == 'hybrid':
            # Najpierw undersampling, potem oversampling
            X_resampled, y_resampled = self.handle_imbalance(X, y, method='undersampling')
            X_resampled, y_resampled = self.handle_imbalance(X_resampled, y_resampled, method='smote')
        
        print(f"Rozkład klas po balansowaniu: {np.sum(y_resampled, axis=0)}")
        return X_resampled, y_resampled
    
    def feature_selection(self, X, y, method='rfe', k=None):
        """
        Selekcja cech.
        
        Parametry:
        ----------
        X : numpy.ndarray
            Dane wejściowe.
        y : numpy.ndarray
            Etykiety (w postaci binarnej).
        method : str, domyślnie 'rfe'
            Metoda selekcji cech ('selectk', 'rfe', 'model_based').
        k : int, opcjonalnie
            Liczba cech do wybrania.
            
        Zwraca:
        -------
        X_selected : numpy.ndarray
            Dane wejściowe z wybranymi cechami.
        selected_features : list
            Lista nazw wybranych cech.
        """
        if k is None:
            # Automatyczne określenie liczby cech na podstawie wielkości zbioru
            k = max(int(X.shape[1] * 0.3), 10)  # Co najmniej 10 cech lub 30% oryginalnych
        
        k = min(k, X.shape[1])  # Nie więcej niż liczba dostępnych cech
        
        selected_features_indices = None
        
        if method == 'selectk':
            # Wybór k najlepszych cech na podstawie testu ANOVA F
            selector = SelectKBest(f_classif, k=k)
            X_selected = selector.fit_transform(X, y.sum(axis=1))  # Uproszczenie do problemu jednoetykietowego
            selected_features_indices = np.where(selector.get_support())[0]
        
        elif method == 'rfe':
            # Rekurencyjna eliminacja cech
            base_model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            rfe = RFE(estimator=base_model, n_features_to_select=k, step=0.1)
            X_selected = rfe.fit_transform(X, y.sum(axis=1))
            selected_features_indices = np.where(rfe.support_)[0]
        
        elif method == 'model_based':
            # Selekcja cech na podstawie ważności z modelu
            model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            model.fit(X, y.sum(axis=1))
            
            # Wybór k najważniejszych cech
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:k]
            selected_features_indices = indices
            X_selected = X[:, indices]
        
        else:
            raise ValueError(f"Nieznana metoda selekcji cech: {method}")
        
        # Zachowanie nazw wybranych cech
        if self.feature_names and len(self.feature_names) == X.shape[1]:
            selected_features = [self.feature_names[i] for i in selected_features_indices]
        else:
            selected_features = [f"feature_{i}" for i in selected_features_indices]
        
        print(f"Wybrano {len(selected_features)} cech metodą {method}.")
        return X_selected, selected_features
    
    def build_models(self):
        """
        Budowa różnych modeli klasyfikacji wieloetykietowej.
        
        Zwraca:
        -------
        dict
            Słownik zawierający zbudowane modele.
        """
        models = {
            'logistic_regression': OneVsRestClassifier(
                LogisticRegression(max_iter=1000, random_state=self.random_state)
            ),
            'svm': OneVsRestClassifier(
                SVC(probability=True, random_state=self.random_state)
            ),
            'random_forest': OneVsRestClassifier(
                RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            ),
            'xgboost': OneVsRestClassifier(
                xgb.XGBClassifier(random_state=self.random_state)
            ),
            'mlp': OneVsRestClassifier(
                MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=self.random_state)
            )
        }
        
        return models
    
    def train_neural_network(self, X_train, y_train, X_val, y_val):
        """
        Trenowanie głębokiej sieci neuronowej dla klasyfikacji wieloetykietowej.
        
        Parametry:
        ----------
        X_train : numpy.ndarray
            Dane treningowe.
        y_train : numpy.ndarray
            Etykiety treningowe.
        X_val : numpy.ndarray
            Dane walidacyjne.
        y_val : numpy.ndarray
            Etykiety walidacyjne.
            
        Zwraca:
        -------
        model : tensorflow.keras.models.Sequential
            Wytrenowany model sieci neuronowej.
        history : tensorflow.keras.callbacks.History
            Historia treningu.
        """
        input_dim = X_train.shape[1]
        output_dim = y_train.shape[1]
        
        # Budowa modelu
        model = Sequential()
        model.add(Dense(256, input_dim=input_dim, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(output_dim, activation='sigmoid'))
        
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        # Wczesne zatrzymanie, aby uniknąć przeuczenia
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Trenowanie modelu
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=0
        )
        
        return model, history
    
    def tune_hyperparameters(self, X, y, model_name, param_grid):
        """
        Dostrajanie hiperparametrów modelu.
        
        Parametry:
        ----------
        X : numpy.ndarray
            Dane wejściowe.
        y : numpy.ndarray
            Etykiety.
        model_name : str
            Nazwa modelu do dostrojenia.
        param_grid : dict
            Siatka parametrów do przeszukania.
            
        Zwraca:
        -------
        best_model : sklearn.base.BaseEstimator
            Najlepszy model.
        best_params : dict
            Najlepsze parametry.
        """
        models = self.build_models()
        model = models[model_name]
        
        # Używamy zmodyfikowanego F1 jako metryki dla problemu wieloetykietowego
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        
        # Uproszczenie problemu wieloetykietowego dla potrzeb CV
        y_simplified = np.argmax(y, axis=1) if y.shape[1] > 1 else y
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='f1_macro',
            cv=cv,
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X, y)
        
        print(f"Najlepsze parametry dla {model_name}: {grid_search.best_params_}")
        print(f"Najlepszy wynik: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def fit(self, X, y, text_columns=None, categorical_columns=None, numerical_columns=None,
            balance_method=None, feature_selection_method=None, tune_params=False):
        """
        Trenowanie klasyfikatora wieloetykietowego.
        
        Parametry:
        ----------
        X : pandas.DataFrame
            Dane wejściowe.
        y : list of lists lub numpy.ndarray
            Etykiety wieloetykietowe.
        text_columns : list, opcjonalnie
            Lista nazw kolumn zawierających dane tekstowe.
        categorical_columns : list, opcjonalnie
            Lista nazw kolumn zawierających dane kategoryczne.
        numerical_columns : list, opcjonalnie
            Lista nazw kolumn zawierających dane numeryczne.
        balance_method : str, opcjonalnie
            Metoda zrównoważenia klas ('smote', 'undersampling', 'hybrid').
        feature_selection_method : str, opcjonalnie
            Metoda selekcji cech ('selectk', 'rfe', 'model_based').
        tune_params : bool, domyślnie False
            Czy dostroić hiperparametry.
            
        Zwraca:
        -------
        self : MedicalMultilabelClassifier
            Instancja klasyfikatora.
        """
        start_time = time.time()
        
        # Konwersja etykiet do formatu binarnego
        if isinstance(y, list) or (isinstance(y, np.ndarray) and y.ndim == 1):
            y_bin = self.mlb.fit_transform(y)
        else:
            y_bin = y
            
        print(f"Zbiór danych: {X.shape[0]} próbek, {X.shape[1]} cech, {y_bin.shape[1]} etykiet")
        
        # Podział na zbiory treningowy i testowy
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_bin, test_size=0.2, random_state=self.random_state, stratify=y_bin.sum(axis=1) > 0
        )
        
        # Podział zbioru treningowego na treningowy i walidacyjny
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=self.random_state, stratify=y_train.sum(axis=1) > 0
        )
        
        print(f"Zbiór treningowy: {X_train.shape[0]} próbek")
        print(f"Zbiór walidacyjny: {X_val.shape[0]} próbek")
        print(f"Zbiór testowy: {X_test.shape[0]} próbek")
        
        # Przetwarzanie danych
        X_train_processed = self.preprocess_data(X_train, text_columns, categorical_columns, numerical_columns)
        X_val_processed = self.preprocess_data(X_val, text_columns, categorical_columns, numerical_columns)
        X_test_processed = self.preprocess_data(X_test, text_columns, categorical_columns, numerical_columns)
        
        print(f"Po przetworzeniu: {X_train_processed.shape[1]} cech")
        
        # Opcjonalne zrównoważenie klas
        if balance_method:
            X_train_processed, y_train = self.handle_imbalance(X_train_processed, y_train, method=balance_method)
        
        # Opcjonalna selekcja cech
        if feature_selection_method:
            X_train_processed, selected_features = self.feature_selection(
                X_train_processed, y_train, method=feature_selection_method
            )
            
            # Zastosowanie tej samej transformacji do zbiorów walidacyjnego i testowego
            if feature_selection_method == 'selectk':
                selector = SelectKBest(f_classif, k=len(selected_features))
                selector.fit(X_train_processed, y_train.sum(axis=1))
                X_val_processed = selector.transform(X_val_processed)
                X_test_processed = selector.transform(X_test_processed)
            elif feature_selection_method == 'rfe' or feature_selection_method == 'model_based':
                # Uproszczone podejście: wybieramy te same kolumny
                selected_indices = [i for i, feat in enumerate(self.feature_names) if feat in selected_features]
                X_val_processed = X_val_processed[:, selected_indices]
                X_test_processed = X_test_processed[:, selected_indices]
            
            self.feature_names = selected_features
        
        # Budowa modeli
        models = self.build_models()
        self.models = {}
        
        # Trenowanie modeli
        for name, model in models.items():
            print(f"\nTrenowanie modelu: {name}")
            
            if tune_params and name in ['random_forest', 'xgboost', 'svm']:
                # Dostrajanie hiperparametrów
                if name == 'random_forest':
                    param_grid = {
                        'estimator__n_estimators': [50, 100],
                        'estimator__max_depth': [None, 10, 20]
                    }
                elif name == 'xgboost':
                    param_grid = {
                        'estimator__n_estimators': [50, 100],
                        'estimator__max_depth': [3, 5, 7]
                    }
                elif name == 'svm':
                    param_grid = {
                        'estimator__C': [0.1, 1, 10],
                        'estimator__kernel': ['linear', 'rbf']
                    }
                
                model, best_params = self.tune_hyperparameters(X_train_processed, y_train, name, param_grid)
                print(f"Dostrojone parametry: {best_params}")
            
            model.fit(X_train_processed, y_train)
            self.models[name] = model
            
            # Ocena modelu na zbiorze walidacyjnym
            y_val_pred = model.predict(X_val_processed)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            val_f1_micro = f1_score(y_val, y_val_pred, average='micro')
            val_f1_macro = f1_score(y_val, y_val_pred, average='macro')
            val_hamming = hamming_loss(y_val, y_val_pred)
            val_jaccard = jaccard_score(y_val, y_val_pred, average='samples')
            
            print(f"Dokładność walidacyjna: {val_accuracy:.4f}")
            print(f"F1 mikro: {val_f1_micro:.4f}, F1 makro: {val_f1_macro:.4f}")
            print(f"Strata Hamminga: {val_hamming:.4f}")
            print(f"Podobieństwo Jaccarda: {val_jaccard:.4f}")
            
            # Obliczenie ważności cech
            if name in ['random_forest', 'xgboost']:
                if hasattr(model, 'feature_importances_'):
                    feature_importances = model.feature_importances_
                else:
                    # Dla OneVsRestClassifier bierzemy średnią z wszystkich klasyfikatorów
                    feature_importances = np.mean([clf.feature_importances_ for clf in model.estimators_], axis=0)
                
                feature_importance_dict = {}
                for i, importance in enumerate(feature_importances):
                    feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
                    feature_importance_dict[feature_name] = importance
                
                self.feature_importances[name] = feature_importance_dict
        
        # Dodanie głębokiej sieci neuronowej
        print("\nTrenowanie głębokiej sieci neuronowej")
        nn_model, history = self.train_neural_network(X_train_processed, y_train, X_val_processed, y_val)
        
        # Ocena sieci neuronowej
        y_val_pred_nn = (nn_model.predict(X_val_processed) > 0.5).astype(int)
        val_accuracy_nn = accuracy_score(y_val, y_val_pred_nn)
        val_f1_micro_nn = f1_score(y_val, y_val_pred_nn, average='micro')
        val_f1_macro_nn = f1_score(y_val, y_val_pred_nn, average='macro')
        val_hamming_nn = hamming_loss(y_val, y_val_pred_nn)
        val_jaccard_nn = jaccard_score(y_val, y_val_pred_nn, average='samples')
        
        print(f"NN - Dokładność walidacyjna: {val_accuracy_nn:.4f}")
        print(f"NN - F1 mikro: {val_f1_micro_nn:.4f}, F1 makro: {val_f1_macro_nn:.4f}")
        print(f"NN - Strata Hamminga: {val_hamming_nn:.4f}")
        print(f"NN - Podobieństwo Jaccarda: {val_jaccard_nn:.4f}")
        
        self.models['neural_network'] = nn_model
        
        # Wybór najlepszego modelu na podstawie F1 makro (dobrze radzi sobie z niezbalansowaniem)
        best_model_name = None
        best_f1_macro = -1
        
        for name in self.models:
            if name == 'neural_network':
                y_val_pred = (self.models[name].predict(X_val_processed) > 0.5).astype(int)
            else:
                y_val_pred = self.models[name].predict(X_val_processed)
            
            f1_macro = f1_score(y_val, y_val_pred, average='macro')
            
            if f1_macro > best_f1_macro:
                best_f1_macro = f1_macro
                best_model_name = name
        
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        
        print(f"\nNajlepszy model: {best_model_name} z F1 makro: {best_f1_macro:.4f}")
        
        # Ostateczna ocena na zbiorze testowym
        self.evaluate(X_test_processed, y_test)
        
        end_time = time.time()
        print(f"\nCałkowity czas wykonania: {end_time - start_time:.2f} sekund")
        
        return self
    
    def predict(self, X, text_columns=None, categorical_columns=None, numerical_columns=None):
        """
        Predykcja etykiet dla nowych danych.
        
        Parametry:
        ----------
        X : pandas.DataFrame
            Dane wejściowe.
        text_columns : list, opcjonalnie
            Lista nazw kolumn zawierających dane tekstowe.
        categorical_columns : list, opcjonalnie
            Lista nazw kolumn zawierających dane kategoryczne.
        numerical_columns : list, opcjonalnie
            Lista nazw kolumn zawierających dane numeryczne.
            
        Zwraca:
        -------
        y_pred : numpy.ndarray
            Przewidziane etykiety (w oryginalnym formacie).
        y_pred_proba : numpy.ndarray
            Prawdopodobieństwa przynależności do klas.
        """
        if not self.best_model:
            raise ValueError("Model nie został jeszcze wytrenowany. Najpierw wywołaj metodę fit().")
        
        # Przetwarzanie danych wejściowych
        X_processed = self.preprocess_data(X, text_columns, categorical_columns, numerical_columns)
        
        # Predykcja
        if self.best_model_name == 'neural_network':
            y_pred_proba = self.best_model.predict(X_processed)
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = self.best_model.predict(X_processed)
            # Nie wszystkie modele zwracają prawdopodobieństwa
            try:
                y_pred_proba = self.best_model.predict_proba(X_processed)
            except:
                y_pred_proba = y_pred
        
        # Konwersja z powrotem do oryginalnego formatu etykiet
        y_pred_original = self.mlb.inverse_transform(y_pred)
        
        return y_pred_original, y_pred_proba
    
    def evaluate(self, X_test, y_test):
        """
        Ewaluacja modelu na zbiorze testowym.
        
        Parametry:
        ----------
        X_test : numpy.ndarray
            Testowe dane wejściowe.
        y_test : numpy.ndarray
            Testowe etykiety.
            
        Zwraca:
        -------
        dict
            Słownik zawierający metryki wydajności.
        """
        metrics = {}
        
        for name, model in self.models.items():
            print(f"\nEwaluacja modelu: {name}")
            
            if name == 'neural_network':
                y_pred_proba = model.predict(X_test)
                y_pred = (y_pred_proba > 0.5).astype(int)
            else:
                y_pred = model.predict(X_test)
            
            # Obliczenie metryk
            accuracy = accuracy_score(y_test, y_pred)
            precision_micro = precision_score(y_test, y_pred, average='micro', zero_division=0)
            precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall_micro = recall_score(y_test, y_pred, average='micro', zero_division=0)
            recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1_micro = f1_score(y_test, y_pred, average='micro')
            f1_macro = f1_score(y_test, y_pred, average='macro')
            hamming = hamming_loss(y_test, y_pred)
            jaccard = jaccard_score(y_test, y_pred, average='samples')
            
            # Zapisanie metryk
            metrics[name] = {
                'accuracy': accuracy,
                'precision_micro': precision_micro,
                'precision_macro': precision_macro,
                'recall_micro': recall_micro,
                'recall_macro': recall_macro,
                'f1_micro': f1_micro,
                'f1_macro': f1_macro,
                'hamming_loss': hamming,
                'jaccard_score': jaccard
            }
            
            # Wyświetlenie metryk
            print(f"Dokładność: {accuracy:.4f}")
            print(f"Precyzja mikro: {precision_micro:.4f}, Precyzja makro: {precision_macro:.4f}")
            print(f"Czułość mikro: {recall_micro:.4f}, Czułość makro: {recall_macro:.4f}")
            print(f"F1 mikro: {f1_micro:.4f}, F1 makro: {f1_macro:.4f}")
            print(f"Strata Hamminga: {hamming:.4f}")
            print(f"Podobieństwo Jaccarda: {jaccard:.4f}")
            
            # Wyświetlenie dokładnego raportu dla każdej klasy
            print("\nRaport klasyfikacji per klasa:")
            for i in range(y_test.shape[1]):
                class_precision = precision_score(y_test[:, i], y_pred[:, i], zero_division=0)
                class_recall = recall_score(y_test[:, i], y_pred[:, i], zero_division=0)
                class_f1 = f1_score(y_test[:, i], y_pred[:, i], zero_division=0)
                class_support = np.sum(y_test[:, i])
                
                class_name = f"Klasa {i}"
                if hasattr(self.mlb, 'classes_'):
                    class_name = self.mlb.classes_[i]
                
                print(f"{class_name}: Precyzja={class_precision:.4f}, Czułość={class_recall:.4f}, "
                      f"F1={class_f1:.4f}, Wsparcie={class_support}")
        
        return metrics
    
    def analyze_misclassifications(self, X_test, y_test, text_columns=None, categorical_columns=None, numerical_columns=None):
        """
        Analiza błędnych klasyfikacji.
        
        Parametry:
        ----------
        X_test : pandas.DataFrame
            Testowe dane wejściowe.
        y_test : numpy.ndarray
            Testowe etykiety.
        text_columns : list, opcjonalnie
            Lista nazw kolumn zawierających dane tekstowe.
        categorical_columns : list, opcjonalnie
            Lista nazw kolumn zawierających dane kategoryczne.
        numerical_columns : list, opcjonalnie
            Lista nazw kolumn zawierających dane numeryczne.
            
        Zwraca:
        -------
        dict
            Słownik zawierający analizę błędnych klasyfikacji.
        """
        X_test_processed = self.preprocess_data(X_test, text_columns, categorical_columns, numerical_columns)
        
        if self.best_model_name == 'neural_network':
            y_pred_proba = self.best_model.predict(X_test_processed)
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = self.best_model.predict(X_test_processed)
        
        # Identyfikacja błędnie sklasyfikowanych próbek
        incorrect_mask = (y_pred != y_test).any(axis=1)
        incorrect_indices = np.where(incorrect_mask)[0]
        
        print(f"Liczba błędnie sklasyfikowanych próbek: {len(incorrect_indices)} z {len(y_test)}")
        
        # Analiza błędów per klasa
        error_analysis = {}
        
        for i in range(y_test.shape[1]):
            # Prawdziwie pozytywne, fałszywie pozytywne, fałszywie negatywne, prawdziwie negatywne
            tp = np.sum((y_test[:, i] == 1) & (y_pred[:, i] == 1))
            fp = np.sum((y_test[:, i] == 0) & (y_pred[:, i] == 1))
            fn = np.sum((y_test[:, i] == 1) & (y_pred[:, i] == 0))
            tn = np.sum((y_test[:, i] == 0) & (y_pred[:, i] == 0))
            
            class_name = f"Klasa {i}"
            if hasattr(self.mlb, 'classes_'):
                class_name = self.mlb.classes_[i]
                
            error_analysis[class_name] = {
                'true_positives': int(tp),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_negatives': int(tn),
                'fp_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
                'fn_rate': float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
            }
            
            print(f"\n{class_name}:")
            print(f"  Prawdziwie pozytywne: {tp}")
            print(f"  Fałszywie pozytywne: {fp} (wskaźnik: {error_analysis[class_name]['fp_rate']:.4f})")
            print(f"  Fałszywie negatywne: {fn} (wskaźnik: {error_analysis[class_name]['fn_rate']:.4f})")
            print(f"  Prawdziwie negatywne: {tn}")
        
        # Analiza współwystępowania błędów między klasami
        error_correlation = np.zeros((y_test.shape[1], y_test.shape[1]))
        
        for i in range(y_test.shape[1]):
            for j in range(y_test.shape[1]):
                if i != j:
                    # Liczba przypadków, gdzie obie klasy są błędnie sklasyfikowane
                    both_errors = np.sum(
                        ((y_test[:, i] != y_pred[:, i]) & (y_test[:, j] != y_pred[:, j]))
                    )
                    # Liczba przypadków, gdzie klasa i jest błędnie sklasyfikowana
                    i_errors = np.sum(y_test[:, i] != y_pred[:, i])
                    
                    error_correlation[i, j] = both_errors / i_errors if i_errors > 0 else 0
        
        # Zidentyfikowanie najbardziej skorelowanych par błędów
        top_correlated_errors = []
        for i in range(y_test.shape[1]):
            for j in range(i+1, y_test.shape[1]):
                corr = error_correlation[i, j]
                if corr > 0.3:  # Próg korelacji
                    class_i = f"Klasa {i}"
                    class_j = f"Klasa {j}"
                    if hasattr(self.mlb, 'classes_'):
                        class_i = self.mlb.classes_[i]
                        class_j = self.mlb.classes_[j]
                    
                    top_correlated_errors.append((class_i, class_j, corr))
        
        top_correlated_errors.sort(key=lambda x: x[2], reverse=True)
        
        print("\nNajbardziej skorelowane błędy między klasami:")
        for class_i, class_j, corr in top_correlated_errors[:5]:
            print(f"  {class_i} i {class_j}: {corr:.4f}")
        
        return {
            'error_analysis': error_analysis,
            'error_correlation': error_correlation,
            'top_correlated_errors': top_correlated_errors
        }
    
    def visualize_results(self, metrics=None, feature_importances=None):
        """
        Wizualizacja wyników modelu.
        
        Parametry:
        ----------
        metrics : dict, opcjonalnie
            Słownik zawierający metryki do wizualizacji.
        feature_importances : dict, opcjonalnie
            Słownik zawierający ważności cech do wizualizacji.
        """
        plt.figure(figsize=(15, 10))
        
        # Porównanie metryk między modelami
        if metrics:
            # Ustawienie szerokości słupków i pozycji
            models = list(metrics.keys())
            metrics_to_plot = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
            x = np.arange(len(metrics_to_plot))
            width = 0.8 / len(models)
            
            plt.subplot(2, 2, 1)
            for i, model_name in enumerate(models):
                values = [metrics[model_name][metric] for metric in metrics_to_plot]
                plt.bar(x + i * width, values, width, label=model_name)
            
            plt.xlabel('Metryka')
            plt.ylabel('Wartość')
            plt.title('Porównanie metryk dla różnych modeli')
            plt.xticks(x + width * (len(models) - 1) / 2, metrics_to_plot)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Strata Hamminga
            plt.subplot(2, 2, 2)
            hamming_losses = [metrics[model]['hamming_loss'] for model in models]
            plt.bar(models, hamming_losses, color='orange')
            plt.xlabel('Model')
            plt.ylabel('Strata Hamminga')
            plt.title('Porównanie straty Hamminga')
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.7)
        
        # Wizualizacja ważności cech
        if feature_importances and self.feature_names:
            plt.subplot(2, 2, 3)
            
            for model_name, importances in feature_importances.items():
                # Wybór top 10 najważniejszych cech
                sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
                features = [x[0] for x in sorted_importances]
                importance_values = [x[1] for x in sorted_importances]
                
                plt.figure(figsize=(12, 6))
                plt.barh(features, importance_values, color='skyblue')
                plt.xlabel('Ważność')
                plt.ylabel('Cecha')
                plt.title(f'Top 10 najważniejszych cech - {model_name}')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.show()
        
        # Macierz korelacji błędów (jeśli dostępna)
        if hasattr(self, 'error_correlation') and self.error_correlation is not None:
            plt.subplot(2, 2, 4)
            plt.imshow(self.error_correlation, cmap='viridis')
            plt.colorbar(label='Korelacja błędów')
            plt.title('Macierz korelacji błędów między klasami')
            plt.tight_layout()
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """
        Zapisanie modelu do pliku.
        
        Parametry:
        ----------
        filepath : str
            Ścieżka do pliku.
        """
        model_data = {
            'models': self.models,
            'mlb': self.mlb,
            'feature_names': self.feature_names,
            'best_model_name': self.best_model_name,
            'random_state': self.random_state,
            'feature_importances': self.feature_importances
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model zapisany w: {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """
        Wczytanie modelu z pliku.
        
        Parametry:
        ----------
        filepath : str
            Ścieżka do pliku.
            
        Zwraca:
        -------
        MedicalMultilabelClassifier
            Wczytany model.
        """
        model_data = joblib.load(filepath)
        
        classifier = cls(random_state=model_data['random_state'])
        classifier.models = model_data['models']
        classifier.mlb = model_data['mlb']
        classifier.feature_names = model_data['feature_names']
        classifier.best_model_name = model_data['best_model_name']
        classifier.best_model = classifier.models[classifier.best_model_name]
        classifier.feature_importances = model_data['feature_importances']
        
        print(f"Model wczytany z: {filepath}")
        return classifier
    
    def explain_predictions(self, X, index=0, text_columns=None, categorical_columns=None, numerical_columns=None):
        """
        Wyjaśnienie predykcji modelu dla wybranej próbki.
        
        Parametry:
        ----------
        X : pandas.DataFrame
            Dane wejściowe.
        index : int, domyślnie 0
            Indeks próbki do wyjaśnienia.
        text_columns : list, opcjonalnie
            Lista nazw kolumn zawierających dane tekstowe.
        categorical_columns : list, opcjonalnie
            Lista nazw kolumn zawierających dane kategoryczne.
        numerical_columns : list, opcjonalnie
            Lista nazw kolumn zawierających dane numeryczne.
            
        Zwraca:
        -------
        dict
            Słownik zawierający wyjaśnienie predykcji.
        """
        # Przetwarzanie danych wejściowych
        X_processed = self.preprocess_data(X, text_columns, categorical_columns, numerical_columns)
        
        # Wybór pojedynczej próbki
        X_single = X.iloc[[index]]
        X_processed_single = X_processed[[index]]
        
        # Predykcja
        if self.best_model_name == 'neural_network':
            y_pred_proba = self.best_model.predict(X_processed_single)
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = self.best_model.predict(X_processed_single)
            try:
                y_pred_proba = self.best_model.predict_proba(X_processed_single)
            except:
                y_pred_proba = y_pred
        
        # Konwersja z powrotem do oryginalnego formatu etykiet
        y_pred_original = self.mlb.inverse_transform(y_pred)[0]
        
        print(f"Przewidziane etykiety: {y_pred_original}")
        
        explanation = {
            'prediction': y_pred_original,
            'probabilities': {}
        }
        
        # Dla każdej klasy pokazujemy prawdopodobieństwo i najważniejsze cechy
        for i, prob in enumerate(y_pred_proba[0]):
            class_name = f"Klasa {i}"
            if hasattr(self.mlb, 'classes_'):
                class_name = self.mlb.classes_[i]
            
            explanation['probabilities'][class_name] = float(prob)
            print(f"{class_name}: Prawdopodobieństwo = {prob:.4f}")
        
        # Wyjaśnienie ważności cech dla tej predykcji
        if self.best_model_name in ['random_forest', 'xgboost']:
            # Używamy permutacyjnej ważności cech dla pojedynczej próbki
            print("\nNajważniejsze cechy dla tej predykcji:")
            
            if len(self.feature_names) > 10:
                # Wybór top 10 najważniejszych cech
                if hasattr(self.best_model, 'feature_importances_'):
                    feature_importances = self.best_model.feature_importances_
                else:
                    feature_importances = np.mean([clf.feature_importances_ for clf in self.best_model.estimators_], axis=0)
                
                # Sortowanie i wybór top 10
                indices = np.argsort(feature_importances)[::-1][:10]
                top_features = [(self.feature_names[i], feature_importances[i]) for i in indices]
                
                for feature, importance in top_features:
                    value = X_single[feature].values[0] if feature in X_single.columns else "N/A"
                    print(f"  {feature}: Ważność = {importance:.4f}, Wartość = {value}")
                    explanation[feature] = {'importance': float(importance), 'value': value}
        
        return explanation
    
    def get_decision_boundaries(self, X, y, feature1, feature2, class_idx=0):
        """
        Wizualizacja granic decyzyjnych modelu dla dwóch wybranych cech.
        
        Parametry:
        ----------
        X : numpy.ndarray
            Dane wejściowe.
        y : numpy.ndarray
            Etykiety.
        feature1 : int
            Indeks pierwszej cechy.
        feature2 : int
            Indeks drugiej cechy.
        class_idx : int, domyślnie 0
            Indeks klasy do wizualizacji.
            
        Zwraca:
        -------
        matplotlib.figure.Figure
            Obiekt figury.
        """
        # Wybieramy tylko dwie określone cechy
        X_subset = X[:, [feature1, feature2]]
        
        # Etykiety dla wybranej klasy
        y_class = y[:, class_idx]
        
        # Utworzenie siatki punktów
        h = 0.02  # krok siatki
        x_min, x_max = X_subset[:, 0].min() - 1, X_subset[:, 0].max() + 1
        y_min, y_max = X_subset[:, 1].min() - 1, X_subset[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        # Tworzenie figury
        plt.figure(figsize=(10, 8))
        
        # Dla każdego modelu rysujemy granice decyzyjne
        for i, (name, model) in enumerate(self.models.items()):
            plt.subplot(2, 3, i + 1)
            
            # Przygotowanie danych dla modelu
            if name == 'neural_network':
                # Dla sieci neuronowej musimy stworzyć kompletny zestaw cech
                # Używamy średnich wartości dla pozostałych cech
                X_mean = np.mean(X, axis=0)
                grid = np.zeros((xx.ravel().shape[0], X.shape[1]))
                for j in range(X.shape[1]):
                    grid[:, j] = X_mean[j]
                grid[:, feature1] = xx.ravel()
                grid[:, feature2] = yy.ravel()
                
                # Predykcja
                Z = model.predict(grid)[:, class_idx]
                Z = Z.reshape(xx.shape)
            else:
                # Dla innych modeli możemy używać ich specyficznych metod
                if hasattr(model, 'estimators_'):
                    # Dla OneVsRestClassifier wybieramy klasyfikator dla danej klasy
                    clf = model.estimators_[class_idx]
                    
                    # Przygotowanie danych dla wybranych cech
                    grid = np.c_[xx.ravel(), yy.ravel()]
                    
                    # Predykcja
                    if hasattr(clf, 'predict_proba'):
                        Z = clf.predict_proba(grid)[:, 1]
                    else:
                        Z = clf.decision_function(grid)
                    Z = Z.reshape(xx.shape)
                else:
                    # Dla innych modeli tworzymy pełny zestaw cech
                    X_mean = np.mean(X, axis=0)
                    grid = np.zeros((xx.ravel().shape[0], X.shape[1]))
                    for j in range(X.shape[1]):
                        grid[:, j] = X_mean[j]
                    grid[:, feature1] = xx.ravel()
                    grid[:, feature2] = yy.ravel()
                    
                    # Predykcja
                    if hasattr(model, 'predict_proba'):
                        Z = model.predict_proba(grid)[:, class_idx]
                    else:
                        Z = model.decision_function(grid)[:, class_idx]
                    Z = Z.reshape(xx.shape)
            
            # Rysowanie konturów
            plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
            
            # Rysowanie punktów treningowych
            plt.scatter(X_subset[y_class == 0, 0], X_subset[y_class == 0, 1], 
                        c='blue', marker='o', label='Klasa 0')
            plt.scatter(X_subset[y_class == 1, 0], X_subset[y_class == 1, 1], 
                        c='red', marker='x', label='Klasa 1')
            
            plt.title(f"Granice decyzyjne - {name}")
            plt.xlabel(f"Cecha {feature1}")
            plt.ylabel(f"Cecha {feature2}")
            
            if i == 0:
                plt.legend()
        
        plt.tight_layout()
        return plt.gcf()
    
    def analyze_label_correlations(self, y):
        """
        Analiza korelacji między etykietami.
        
        Parametry:
        ----------
        y : numpy.ndarray
            Etykiety (w postaci binarnej).
            
        Zwraca:
        -------
        numpy.ndarray
            Macierz korelacji między etykietami.
        """
        # Obliczenie korelacji Phi (odpowiednik korelacji Pearsona dla zmiennych binarnych)
        n_labels = y.shape[1]
        corr_matrix = np.zeros((n_labels, n_labels))
        
        for i in range(n_labels):
            for j in range(n_labels):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    # Współczynnik korelacji phi
                    n11 = np.sum((y[:, i] == 1) & (y[:, j] == 1))
                    n10 = np.sum((y[:, i] == 1) & (y[:, j] == 0))
                    n01 = np.sum((y[:, i] == 0) & (y[:, j] == 1))
                    n00 = np.sum((y[:, i] == 0) & (y[:, j] == 0))
                    
                    # Unikanie dzielenia przez zero
                    n1_ = n11 + n10
                    n0_ = n01 + n00
                    n_1 = n11 + n01
                    n_0 = n10 + n00
                    
                    if n1_ * n0_ * n_1 * n_0 == 0:
                        corr_matrix[i, j] = 0
                    else:
                        corr_matrix[i, j] = (n11 * n00 - n10 * n01) / np.sqrt(n1_ * n0_ * n_1 * n_0)
        
        # Wizualizacja macierzy korelacji
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Dodanie nazw etykiet, jeśli są dostępne
        if hasattr(self.mlb, 'classes_'):
            class_names = self.mlb.classes_
            plt.xticks(np.arange(n_labels) + 0.5, class_names, rotation=90)
            plt.yticks(np.arange(n_labels) + 0.5, class_names)
        
        plt.title('Macierz korelacji etykiet')
        plt.tight_layout()
        plt.show()
        
        # Identyfikacja silnie skorelowanych par etykiet
        print("\nSilnie skorelowane pary etykiet:")
        strong_correlations = []
        
        for i in range(n_labels):
            for j in range(i+1, n_labels):
                corr = corr_matrix[i, j]
                if abs(corr) > 0.5:  # Próg korelacji
                    label_i = f"Etykieta {i}"
                    label_j = f"Etykieta {j}"
                    if hasattr(self.mlb, 'classes_'):
                        label_i = self.mlb.classes_[i]
                        label_j = self.mlb.classes_[j]
                    
                    strong_correlations.append((label_i, label_j, corr))
        
        strong_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        
        for label_i, label_j, corr in strong_correlations:
            corr_type = "pozytywna" if corr > 0 else "negatywna"
            print(f"  {label_i} i {label_j}: korelacja {corr_type} = {corr:.4f}")
        
        return corr_matrix

# Przykładowe użycie
def generate_synthetic_medical_data(n_samples=1000, n_features=20, n_labels=5, random_state=42):
    """
    Generowanie syntetycznych danych medycznych.
    
    Parametry:
    ----------
    n_samples : int, domyślnie 1000
        Liczba próbek.
    n_features : int, domyślnie 20
        Liczba cech.
    n_labels : int, domyślnie 5
        Liczba etykiet.
    random_state : int, domyślnie 42
        Ziarno generatora liczb losowych.
        
    Zwraca:
    -------
    X : pandas.DataFrame
        Syntetyczne dane wejściowe.
    y : list of lists
        Syntetyczne etykiety.
    """
    np.random.seed(random_state)
    
    # Generowanie cech
    X = np.random.randn(n_samples, n_features)
    
    # Generowanie etykiet
    y_bin = np.zeros((n_samples, n_labels))
    
    # Dodanie zależności między cechami a etykietami
    for i in range(n_labels):
        # Każda etykieta zależy od kilku cech
        relevant_features = np.random.choice(n_features, size=3, replace=False)