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

warnings.filterwarnings('ignore')

class MedicalMultilabelClassifier:
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.mlb = MultiLabelBinarizer()
        self.models = {}
        self.feature_names = None
        self.best_model_name = None
        self.best_model = None
        self.feature_importances = {}
        
    def preprocess_data(self, X, text_columns=None, categorical_columns=None, numerical_columns=None):
        processed_parts = []
        self.feature_names = []
        
        if numerical_columns:
            X_num = X[numerical_columns].fillna(X[numerical_columns].mean())
            scaler = StandardScaler()
            X_num_scaled = scaler.fit_transform(X_num)
            processed_parts.append(X_num_scaled)
            self.feature_names.extend(numerical_columns)
        
        if categorical_columns:
            X_cat = pd.get_dummies(X[categorical_columns], drop_first=True)
            processed_parts.append(X_cat.values)
            self.feature_names.extend(X_cat.columns.tolist())
        
        if text_columns:
            for col in text_columns:
                try:
                    nltk.data.find('corpora/stopwords')
                except LookupError:
                    nltk.download('stopwords', quiet=True)
                    nltk.download('punkt', quiet=True)
                
                tfidf = TfidfVectorizer(max_features=100, stop_words=stopwords.words('english'))
                X_text = tfidf.fit_transform(X[col].fillna('').astype(str))
                processed_parts.append(X_text)
                self.feature_names.extend([f"{col}_{i}" for i in range(X_text.shape[1])])
        
        if len(processed_parts) == 1:
            X_processed = processed_parts[0]
        else:
            for i in range(len(processed_parts)):
                if not isinstance(processed_parts[i], csr_matrix):
                    processed_parts[i] = csr_matrix(processed_parts[i])
            X_processed = hstack(processed_parts).toarray()
        
        return X_processed
    
    def handle_imbalance(self, X, y, method='smote', sampling_strategy='auto'):
        class_counts = np.sum(y, axis=0)
        print(f"Rozkład klas przed balansowaniem: {class_counts}")
        
        mean_samples = np.mean(class_counts)
        minority_classes = np.where(class_counts < mean_samples * 0.5)[0]
        
        if len(minority_classes) == 0:
            print("Dane są względnie zbalansowane, pomijam balansowanie.")
            return X, y
        
        print(f"Wykryto {len(minority_classes)} klas mniejszościowych.")
        
        X_resampled, y_resampled = X.copy(), y.copy()
        
        if method == 'smote':
            for i in minority_classes:
                smote = SMOTE(sampling_strategy={i: int(mean_samples)}, random_state=self.random_state)
                try:
                    X_temp, y_temp = smote.fit_resample(X, y[:, i])
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
            majority_classes = np.where(class_counts > mean_samples * 1.5)[0]
            for i in majority_classes:
                rus = RandomUnderSampler(sampling_strategy={i: int(mean_samples)}, random_state=self.random_state)
                try:
                    X_resampled, y_resampled = rus.fit_resample(X_resampled, y_resampled[:, i])
                except ValueError as e:
                    print(f"Błąd podczas stosowania undersamplingu dla klasy {i}: {e}")
        
        elif method == 'hybrid':
            X_resampled, y_resampled = self.handle_imbalance(X, y, method='undersampling')
            X_resampled, y_resampled = self.handle_imbalance(X_resampled, y_resampled, method='smote')
        
        print(f"Rozkład klas po balansowaniu: {np.sum(y_resampled, axis=0)}")
        return X_resampled, y_resampled
    
    def feature_selection(self, X, y, method='rfe', k=None):
        if k is None:
            k = max(int(X.shape[1] * 0.3), 10)
        
        k = min(k, X.shape[1])
        
        selected_features_indices = None
        
        if method == 'selectk':
            selector = SelectKBest(f_classif, k=k)
            X_selected = selector.fit_transform(X, y.sum(axis=1))
            selected_features_indices = np.where(selector.get_support())[0]
        
        elif method == 'rfe':
            base_model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            rfe = RFE(estimator=base_model, n_features_to_select=k, step=0.1)
            X_selected = rfe.fit_transform(X, y.sum(axis=1))
            selected_features_indices = np.where(rfe.support_)[0]
        
        elif method == 'model_based':
            model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            model.fit(X, y.sum(axis=1))
            
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:k]
            selected_features_indices = indices
            X_selected = X[:, indices]
        
        else:
            raise ValueError(f"Nieznana metoda selekcji cech: {method}")
        
        if self.feature_names and len(self.feature_names) == X.shape[1]:
            selected_features = [self.feature_names[i] for i in selected_features_indices]
        else:
            selected_features = [f"feature_{i}" for i in selected_features_indices]
        
        print(f"Wybrano {len(selected_features)} cech metodą {method}.")
        return X_selected, selected_features
    
    def build_models(self):
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
        input_dim = X_train.shape[1]
        output_dim = y_train.shape[1]
        
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
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
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
        models = self.build_models()
        model = models[model_name]
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        
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

        start_time = time.time()
        
        if isinstance(y, list) or (isinstance(y, np.ndarray) and y.ndim == 1):
            y_bin = self.mlb.fit_transform(y)
        else:
            y_bin = y
            
        print(f"Zbiór danych: {X.shape[0]} próbek, {X.shape[1]} cech, {y_bin.shape[1]} etykiet")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_bin, test_size=0.2, random_state=self.random_state, stratify=y_bin.sum(axis=1) > 0
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=self.random_state, stratify=y_train.sum(axis=1) > 0
        )
        
        print(f"Zbiór treningowy: {X_train.shape[0]} próbek")
        print(f"Zbiór walidacyjny: {X_val.shape[0]} próbek")
        print(f"Zbiór testowy: {X_test.shape[0]} próbek")
        
        X_train_processed = self.preprocess_data(X_train, text_columns, categorical_columns, numerical_columns)
        X_val_processed = self.preprocess_data(X_val, text_columns, categorical_columns, numerical_columns)
        X_test_processed = self.preprocess_data(X_test, text_columns, categorical_columns, numerical_columns)
        
        print(f"Po przetworzeniu: {X_train_processed.shape[1]} cech")
        
        if balance_method:
            X_train_processed, y_train = self.handle_imbalance(X_train_processed, y_train, method=balance_method)
        
        if feature_selection_method:
            X_train_processed, selected_features = self.feature_selection(
                X_train_processed, y_train, method=feature_selection_method
            )
            
            if feature_selection_method == 'selectk':
                selector = SelectKBest(f_classif, k=len(selected_features))
                selector.fit(X_train_processed, y_train.sum(axis=1))
                X_val_processed = selector.transform(X_val_processed)
                X_test_processed = selector.transform(X_test_processed)
            elif feature_selection_method == 'rfe' or feature_selection_method == 'model_based':
                selected_indices = [i for i, feat in enumerate(self.feature_names) if feat in selected_features]
                X_val_processed = X_val_processed[:, selected_indices]
                X_test_processed = X_test_processed[:, selected_indices]
            
            self.feature_names = selected_features
        
        models = self.build_models()
        self.models = {}
        
        for name, model in models.items():
            print(f"\nTrenowanie modelu: {name}")
            
            if tune_params and name in ['random_forest', 'xgboost', 'svm']:
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
            
            if name in ['random_forest', 'xgboost']:
                if hasattr(model, 'feature_importances_'):
                    feature_importances = model.feature_importances_
                else:
                    feature_importances = np.mean([clf.feature_importances_ for clf in model.estimators_], axis=0)
                
                feature_importance_dict = {}
                for i, importance in enumerate(feature_importances):
                    feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
                    feature_importance_dict[feature_name] = importance
                
                self.feature_importances[name] = feature_importance_dict
        
        print("\nTrenowanie głębokiej sieci neuronowej")
        nn_model, history = self.train_neural_network(X_train_processed, y_train, X_val_processed, y_val)
        
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
        
        self.evaluate(X_test_processed, y_test)
        
        end_time = time.time()
        print(f"\nCałkowity czas wykonania: {end_time - start_time:.2f} sekund")
        
        return self
    
    def predict(self, X, text_columns=None, categorical_columns=None, numerical_columns=None):
        if not self.best_model:
            raise ValueError("Model nie został jeszcze wytrenowany. Najpierw wywołaj metodę fit().")
        
        X_processed = self.preprocess_data(X, text_columns, categorical_columns, numerical_columns)
        
        if self.best_model_name == 'neural_network':
            y_pred_proba = self.best_model.predict(X_processed)
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = self.best_model.predict(X_processed)
            try:
                y_pred_proba = self.best_model.predict_proba(X_processed)
            except:
                y_pred_proba = y_pred
        
        y_pred_original = self.mlb.inverse_transform(y_pred)
        
        return y_pred_original, y_pred_proba
    
    def evaluate(self, X_test, y_test):
        metrics = {}
        
        for name, model in self.models.items():
            print(f"\nEwaluacja modelu: {name}")
            
            if name == 'neural_network':
                y_pred_proba = model.predict(X_test)
                y_pred = (y_pred_proba > 0.5).astype(int)
            else:
                y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision_micro = precision_score(y_test, y_pred, average='micro', zero_division=0)
            precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall_micro = recall_score(y_test, y_pred, average='micro', zero_division=0)
            recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1_micro = f1_score(y_test, y_pred, average='micro')
            f1_macro = f1_score(y_test, y_pred, average='macro')
            hamming = hamming_loss(y_test, y_pred)
            jaccard = jaccard_score(y_test, y_pred, average='samples')
            
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
            
            print(f"Dokładność: {accuracy:.4f}")
            print(f"Precyzja mikro: {precision_micro:.4f}, Precyzja makro: {precision_macro:.4f}")
            print(f"Czułość mikro: {recall_micro:.4f}, Czułość makro: {recall_macro:.4f}")
            print(f"F1 mikro: {f1_micro:.4f}, F1 makro: {f1_macro:.4f}")
            print(f"Strata Hamminga: {hamming:.4f}")
            print(f"Podobieństwo Jaccarda: {jaccard:.4f}")
            
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
        X_test_processed = self.preprocess_data(X_test, text_columns, categorical_columns, numerical_columns)
        
        if self.best_model_name == 'neural_network':
            y_pred_proba = self.best_model.predict(X_test_processed)
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = self.best_model.predict(X_test_processed)
        
        incorrect_mask = (y_pred != y_test).any(axis=1)
        incorrect_indices = np.where(incorrect_mask)[0]
        
        print(f"Liczba błędnie sklasyfikowanych próbek: {len(incorrect_indices)} z {len(y_test)}")
        
        error_analysis = {}
        
        for i in range(y_test.shape[1]):
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
        
        error_correlation = np.zeros((y_test.shape[1], y_test.shape[1]))
        
        for i in range(y_test.shape[1]):
            for j in range(y_test.shape[1]):
                if i != j:
                    both_errors = np.sum(
                        ((y_test[:, i] != y_pred[:, i]) & (y_test[:, j] != y_pred[:, j]))
                    )
                    i_errors = np.sum(y_test[:, i] != y_pred[:, i])
                    
                    error_correlation[i, j] = both_errors / i_errors if i_errors > 0 else 0
        
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
        plt.figure(figsize=(15, 10))
        
        if metrics:
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
            
            plt.subplot(2, 2, 2)
            hamming_losses = [metrics[model]['hamming_loss'] for model in models]
            plt.bar(models, hamming_losses, color='orange')
            plt.xlabel('Model')
            plt.ylabel('Strata Hamminga')
            plt.title('Porównanie straty Hamminga')
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.7)
        
        if feature_importances and self.feature_names:
            plt.subplot(2, 2, 3)
            
            for model_name, importances in feature_importances.items():
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
        
        if hasattr(self, 'error_correlation') and self.error_correlation is not None:
            plt.subplot(2, 2, 4)
            plt.imshow(self.error_correlation, cmap='viridis')
            plt.colorbar(label='Korelacja błędów')
            plt.title('Macierz korelacji błędów między klasami')
            plt.tight_layout()
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
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
        X_processed = self.preprocess_data(X, text_columns, categorical_columns, numerical_columns)
        
        X_single = X.iloc[[index]]
        X_processed_single = X_processed[[index]]
        
        if self.best_model_name == 'neural_network':
            y_pred_proba = self.best_model.predict(X_processed_single)
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = self.best_model.predict(X_processed_single)
            try:
                y_pred_proba = self.best_model.predict_proba(X_processed_single)
            except:
                y_pred_proba = y_pred
        
        y_pred_original = self.mlb.inverse_transform(y_pred)[0]
        
        print(f"Przewidziane etykiety: {y_pred_original}")
        
        explanation = {
            'prediction': y_pred_original,
            'probabilities': {}
        }
        
        for i, prob in enumerate(y_pred_proba[0]):
            class_name = f"Klasa {i}"
            if hasattr(self.mlb, 'classes_'):
                class_name = self.mlb.classes_[i]
            
            explanation['probabilities'][class_name] = float(prob)
            print(f"{class_name}: Prawdopodobieństwo = {prob:.4f}")
        
        if self.best_model_name in ['random_forest', 'xgboost']:
            print("\nNajważniejsze cechy dla tej predykcji:")
            
            if len(self.feature_names) > 10:
                if hasattr(self.best_model, 'feature_importances_'):
                    feature_importances = self.best_model.feature_importances_
                else:
                    feature_importances = np.mean([clf.feature_importances_ for clf in self.best_model.estimators_], axis=0)
                
                indices = np.argsort(feature_importances)[::-1][:10]
                top_features = [(self.feature_names[i], feature_importances[i]) for i in indices]
                
                for feature, importance in top_features:
                    value = X_single[feature].values[0] if feature in X_single.columns else "N/A"
                    print(f"  {feature}: Ważność = {importance:.4f}, Wartość = {value}")
                    explanation[feature] = {'importance': float(importance), 'value': value}
        
        return explanation
    
    def get_decision_boundaries(self, X, y, feature1, feature2, class_idx=0):
        X_subset = X[:, [feature1, feature2]]
        
        y_class = y[:, class_idx]
        
        h = 0.02
        x_min, x_max = X_subset[:, 0].min() - 1, X_subset[:, 0].max() + 1
        y_min, y_max = X_subset[:, 1].min() - 1, X_subset[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        plt.figure(figsize=(10, 8))
        
        for i, (name, model) in enumerate(self.models.items()):
            plt.subplot(2, 3, i + 1)
            
            if name == 'neural_network':
                X_mean = np.mean(X, axis=0)
                grid = np.zeros((xx.ravel().shape[0], X.shape[1]))
                for j in range(X.shape[1]):
                    grid[:, j] = X_mean[j]
                grid[:, feature1] = xx.ravel()
                grid[:, feature2] = yy.ravel()
                
                Z = model.predict(grid)[:, class_idx]
                Z = Z.reshape(xx.shape)
            else:
                if hasattr(model, 'estimators_'):
                    clf = model.estimators_[class_idx]
                    
                    grid = np.c_[xx.ravel(), yy.ravel()]
                    
                    if hasattr(clf, 'predict_proba'):
                        Z = clf.predict_proba(grid)[:, 1]
                    else:
                        Z = clf.decision_function(grid)
                    Z = Z.reshape(xx.shape)
                else:
                    X_mean = np.mean(X, axis=0)
                    grid = np.zeros((xx.ravel().shape[0], X.shape[1]))
                    for j in range(X.shape[1]):
                        grid[:, j] = X_mean[j]
                    grid[:, feature1] = xx.ravel()
                    grid[:, feature2] = yy.ravel()
                    
                    if hasattr(model, 'predict_proba'):
                        Z = model.predict_proba(grid)[:, class_idx]
                    else:
                        Z = model.decision_function(grid)[:, class_idx]
                    Z = Z.reshape(xx.shape)
            
            plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
            
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
        n_labels = y.shape[1]
        corr_matrix = np.zeros((n_labels, n_labels))
        
        for i in range(n_labels):
            for j in range(n_labels):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    n11 = np.sum((y[:, i] == 1) & (y[:, j] == 1))
                    n10 = np.sum((y[:, i] == 1) & (y[:, j] == 0))
                    n01 = np.sum((y[:, i] == 0) & (y[:, j] == 1))
                    n00 = np.sum((y[:, i] == 0) & (y[:, j] == 0))
                    
                    n1_ = n11 + n10
                    n0_ = n01 + n00
                    n_1 = n11 + n01
                    n_0 = n10 + n00
                    
                    if n1_ * n0_ * n_1 * n_0 == 0:
                        corr_matrix[i, j] = 0
                    else:
                        corr_matrix[i, j] = (n11 * n00 - n10 * n01) / np.sqrt(n1_ * n0_ * n_1 * n_0)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        
        if hasattr(self.mlb, 'classes_'):
            class_names = self.mlb.classes_
            plt.xticks(np.arange(n_labels) + 0.5, class_names, rotation=90)
            plt.yticks(np.arange(n_labels) + 0.5, class_names)
        
        plt.title('Macierz korelacji etykiet')
        plt.tight_layout()
        plt.show()
        
        print("\nSilnie skorelowane pary etykiet:")
        strong_correlations = []
        
        for i in range(n_labels):
            for j in range(i+1, n_labels):
                corr = corr_matrix[i, j]
                if abs(corr) > 0.5:
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

def generate_synthetic_medical_data(n_samples=1000, n_features=20, n_labels=5, random_state=42):
    """
    Generowanie syntetycznych danych medycznych dla klasyfikacji wieloetykietowej
    
    Args:
        n_samples: liczba próbek
        n_features: liczba cech
        n_labels: liczba etykiet
        random_state: seed dla powtarzalności
    
    Returns:
        X: macierz cech
        y: macierz etykiet binarnych
        feature_names: nazwy cech
        label_names: nazwy etykiet
    """
    np.random.seed(random_state)
    
    # Generowanie podstawowych cech medycznych
    X = np.random.randn(n_samples, n_features)
    y_bin = np.zeros((n_samples, n_labels))
    
    # Symulacja zależności między cechami a etykietami
    for i in range(n_labels):
        # Każda etykieta zależy od różnych podzbiorów cech
        relevant_features = np.random.choice(n_features, size=3, replace=False)
        
        # Liniowa kombinacja wybranych cech z szumem
        linear_combination = np.sum(X[:, relevant_features], axis=1)
        
        # Dodanie nieliniowych zależności
        nonlinear_term = np.sin(X[:, relevant_features[0]]) * X[:, relevant_features[1]]
        
        # Próg decyzyjny z szumem
        threshold = np.random.normal(0, 0.5)
        noise = np.random.normal(0, 0.3, n_samples)
        
        decision_values = linear_combination + nonlinear_term + noise
        y_bin[:, i] = (decision_values > threshold).astype(int)
    
    # Wprowadzenie korelacji między etykietami (współwystępowanie chorób)
    correlation_pairs = [(0, 1), (2, 3), (1, 4)]  # Przykładowe pary skorelowanych etykiet
    
    for label1, label2 in correlation_pairs:
        # Jeśli pierwsza etykieta jest pozytywna, zwiększ prawdopodobieństwo drugiej
        positive_mask = y_bin[:, label1] == 1
        correlation_strength = 0.7
        
        # Dla próbek z pozytywną pierwszą etykietą
        n_positive = np.sum(positive_mask)
        if n_positive > 0:
            # Zwiększ prawdopodobieństwo drugiej etykiety
            flip_indices = np.random.choice(
                np.where(positive_mask)[0], 
                size=int(n_positive * correlation_strength), 
                replace=False
            )
            y_bin[flip_indices, label2] = 1
    
    # Zapewnienie, że każda próbka ma przynajmniej jedną etykietę
    empty_samples = np.sum(y_bin, axis=1) == 0
    if np.any(empty_samples):
        for idx in np.where(empty_samples)[0]:
            random_label = np.random.randint(0, n_labels)
            y_bin[idx, random_label] = 1
    
    # Generowanie realistycznych nazw cech medycznych
    feature_names = [
        'age', 'bmi', 'blood_pressure_systolic', 'blood_pressure_diastolic',
        'heart_rate', 'temperature', 'glucose_level', 'cholesterol_total',
        'cholesterol_hdl', 'cholesterol_ldl', 'hemoglobin', 'white_blood_cells',
        'platelets', 'creatinine', 'sodium', 'potassium', 'protein_total',
        'albumin', 'bilirubin', 'ast_enzyme'
    ][:n_features]
    
    # Generowanie nazw etykiet (chorób)
    label_names = [
        'diabetes', 'hypertension', 'cardiovascular_disease', 
        'kidney_disease', 'liver_disease'
    ][:n_labels]
    
    # Dodanie jednostek i normalizacja niektórych cech do realistycznych zakresów
    for i, feature in enumerate(feature_names):
        if feature == 'age':
            X[:, i] = np.abs(X[:, i]) * 20 + 40  # Wiek 40-100
        elif feature == 'bmi':
            X[:, i] = np.abs(X[:, i]) * 10 + 20  # BMI 20-40
        elif 'blood_pressure' in feature:
            if 'systolic' in feature:
                X[:, i] = np.abs(X[:, i]) * 40 + 100  # 100-180 mmHg
            else:
                X[:, i] = np.abs(X[:, i]) * 30 + 60   # 60-120 mmHg
        elif feature == 'heart_rate':
            X[:, i] = np.abs(X[:, i]) * 40 + 60   # 60-140 bpm
        elif feature == 'temperature':
            X[:, i] = X[:, i] * 2 + 37  # Około 37°C
        elif feature == 'glucose_level':
            X[:, i] = np.abs(X[:, i]) * 100 + 70  # 70-250 mg/dL
        else:
            # Normalizacja do dodatnich wartości dla pozostałych biomarkerów
            X[:, i] = np.abs(X[:, i]) * 50 + 10
    
    # Stworzenie DataFrame dla lepszej czytelności
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Dodanie kategorycznych cech medycznych
    X_df['gender'] = np.random.choice(['M', 'F'], size=n_samples)
    X_df['smoking_status'] = np.random.choice(['never', 'former', 'current'], size=n_samples)
    X_df['family_history'] = np.random.choice(['yes', 'no'], size=n_samples, p=[0.3, 0.7])
    
    # Dodanie tekstowych opisów objawów
    symptoms = [
        'chest pain, shortness of breath',
        'fatigue, dizziness',
        'nausea, vomiting',
        'headache, blurred vision',
        'frequent urination, excessive thirst',
        'no symptoms reported',
        'joint pain, swelling',
        'fever, chills'
    ]
    
    X_df['symptoms'] = np.random.choice(symptoms, size=n_samples)
    
    print(f"Wygenerowano {n_samples} próbek z {X_df.shape[1]} cechami i {n_labels} etykietami")
    print(f"Rozkład etykiet: {np.sum(y_bin, axis=0)}")
    print(f"Średnia liczba etykiet na próbkę: {np.mean(np.sum(y_bin, axis=1)):.2f}")
    
    return X_df, y_bin, feature_names + ['gender', 'smoking_status', 'family_history', 'symptoms'], label_names


# Przykład użycia i testowania klasyfikatora
if __name__ == "__main__":
    # Generowanie danych syntetycznych
    X, y, feature_names, label_names = generate_synthetic_medical_data(
        n_samples=2000, n_features=15, n_labels=5, random_state=42
    )
    
    # Inicjalizacja klasyfikatora
    classifier = MedicalMultilabelClassifier(random_state=42)
    
    # Definicja typów kolumn
    numerical_columns = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
    categorical_columns = ['gender', 'smoking_status', 'family_history']
    text_columns = ['symptoms']
    
    print("\nRozpoczynanie trenowania modelu...")
    
    # Trenowanie modelu z różnymi opcjami
    classifier.fit(
        X, y,
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        text_columns=text_columns,
        balance_method='smote',  # Balansowanie klas
        feature_selection_method='rfe',  # Selekcja cech
        tune_params=True  # Dostrajanie hiperparametrów
    )
    
    # Analiza korelacji etykiet
    print("\n" + "="*50)
    print("ANALIZA KORELACJI ETYKIET")
    print("="*50)
    classifier.analyze_label_correlations(y)
    
    # Analiza błędnych klasyfikacji
    print("\n" + "="*50)
    print("ANALIZA BŁĘDNYCH KLASYFIKACJI")
    print("="*50)
    X_test_sample = X.sample(200, random_state=42)
    y_test_sample = y[X_test_sample.index]
    
    error_analysis = classifier.analyze_misclassifications(
        X_test_sample, y_test_sample,
        text_columns=text_columns,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns
    )
    
    # Wyjaśnienie predykcji dla przykładowej próbki
    print("\n" + "="*50)
    print("WYJAŚNIENIE PREDYKCJI")
    print("="*50)
    explanation = classifier.explain_predictions(
        X.head(5), index=0,
        text_columns=text_columns,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns
    )
    
    # Predykcja na nowych danych
    print("\n" + "="*50)
    print("TESTOWANIE PREDYKCJI")
    print("="*50)
    new_data = X.sample(10, random_state=123)
    predictions, probabilities = classifier.predict(
        new_data,
        text_columns=text_columns,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns
    )
    
    print("Przykładowe predykcje:")
    for i, (pred, prob) in enumerate(zip(predictions[:3], probabilities[:3])):
        print(f"Próbka {i+1}: {pred}")
        if hasattr(prob, 'shape') and len(prob.shape) > 1:
            print(f"  Prawdopodobieństwa: {prob[0]}")
        else:
            print(f"  Prawdopodobieństwa: {prob}")
    
    # Zapisanie modelu
    classifier.save_model('medical_multilabel_model.pkl')
    
    print("\n" + "="*50)
    print("ZAKOŃCZONO POMYŚLNIE")
    print("="*50)
    print(f"Najlepszy model: {classifier.best_model_name}")
    print("Model został zapisany jako 'medical_multilabel_model.pkl'")
    
    # Demonstracja ładowania modelu
    print("\nTestowanie ładowania modelu...")
    loaded_classifier = MedicalMultilabelClassifier.load_model('medical_multilabel_model.pkl')
    
    # Test predykcji na załadowanym modelu
    test_predictions, _ = loaded_classifier.predict(
        X.head(3),
        text_columns=text_columns,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns
    )
    print(f"Predykcje z załadowanego modelu: {test_predictions}")