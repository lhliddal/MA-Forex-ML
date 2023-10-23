#code 1

# Importieren von Bibliotheken und Modulen
import os
import re
import sys
import time
import matplotlib
import sklearn
import joblib
import xgboost
import glob
import datetime
import configparser
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor


#Code Name
print(f"code 1")
print("\n")  # Einen Zeilenumbruch einfügen

# Python und Paketversionen anzeigen
print('Python version:{}'.format(sys.version))
print('Numpy version:{}'.format(np.__version__))
print('Pandas version:{}'.format(pd.__version__))
print('MatlpotLib version:{}'.format(matplotlib.__version__))
print('Seaborn version:{}'.format(sns.__version__))
print('Sci-Kit Learn version:{}'.format(sklearn.__version__))
print('XGBoost version:{}'.format(xgboost.__version__))
print("\n")  # Einen Zeilenumbruch einfügen


# Einstellen von Konfigurationen und Parametern
matplotlib.use('Qt5Agg')
sns.set_style("whitegrid")


#Konfigurationsdatei laden
def load_config():
    config = configparser.ConfigParser()
    # Pfad zum Verzeichnis, in dem sich dieses Skript befindet
    current_dir = os.path.dirname(__file__)
    # Relativer Pfad zur config.ini-Datei
    config_path = os.path.join(current_dir, '..', '05config', 'config.ini')
    config.read(config_path)
    return config


#Alle Inputs des Users (Graphiken, Modelle und Anzahl Ausführungen)
def get_user_inputs():
    show_graphs_input = input("Möchten Sie die Grafiken anzeigen? (j/n): ").strip().lower() == "j"
    save_all_input = input("Möchten Sie alle Modelle speichern? (j/n): ").strip().lower() == "j"
    num_runs = int(input("Wie oft möchten Sie den Code ausführen? "))
    return show_graphs_input, save_all_input, num_runs


#Graphiken speichern
def get_next_run_number(base_path):
    """
    Ermittelt die nächste Run-Nummer für einen neuen Ordner.
    """
    # Überprüft, ob der Basisordner existiert
    if not os.path.exists(base_path):
        return 1.0
    
    # Durchsucht den Ordner nach bestehenden Run-Nummern
    existing_folders = os.listdir(base_path)
    run_numbers = []
    for folder in existing_folders:
        match = re.search(r'Run_([\d.]+)_', folder)
        if match:
            run_numbers.append(float(match.group(1)))
    
    # Bestimmt die nächste Run-Nummer
    if run_numbers:
        return round(max(run_numbers) + 0.1, 1)
    else:
        return 1.0

def save_graph(base_path, run_number, file_name, graph_number):
    """
    Speichert den Graphen in einem speziellen Ordner.
    """
    # Erstellt den Ordnerpfad und den Ordnernamen
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    new_folder_name = f"Run_{run_number}_{current_date}"
    new_folder_path = os.path.join(base_path, new_folder_name)
    
    # Erstellt den Ordner, falls nicht vorhanden
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    
    # Speichert den Graphen im Ordner
    complete_file_name = f"{file_name}_{graph_number}.png"
    plt.savefig(os.path.join(new_folder_path, complete_file_name))


def load_and_filter_data(file_path, start_date, end_date, kontroll_output_path):
    """
    Diese Funktion lädt die Forex-Daten, konvertiert das Datumsformat, filtert die Daten und generiert Features.
    """
    
    # Forex-Daten werden aus der angegebenen Datei geladen
    eu = pd.read_csv(file_path, parse_dates=True, skipinitialspace=True)
   
    # Konvertiere die 'date'-Spalte zum Datumsformat (falls noch nicht im Datumsformat)
    eu['date'] = pd.to_datetime(eu['date'], format='%Y.%m.%d')

    # Setze die 'date'-Spalte als Index
    eu.set_index('date', inplace=True)

    # Füge eine neue Spalte für den Wochentag hinzu
    eu['weekday'] = eu.index.weekday

    # Filtern der Forex-Daten entsprechend des Start- und Enddatums
    eu_filtered = eu[(eu.index >= pd.Timestamp(start_date)) & (eu.index <= pd.Timestamp(end_date))]
    
    # Generieren des vollständigen Pfads für die Kontrolldatei nach der Datenfilterung
    kontroll_output_full_path = os.path.join(kontroll_output_path, 'data_pregf_code1.csv')
    
    # Speichern des DataFrame in einer CSV-Datei für die Kontrolle
    eu_filtered.to_csv(kontroll_output_full_path, index=True)
    
    # Generieren von Features für die gefilterten Daten
    data = generate_features(eu_filtered)

    return data


def visual1(eu, start_date, end_date, show_graphs, base_path, graph_number, run_number):
    """
    Diese Funktion erstellt Zeitreihenplots der USD/CHF-Wechselkurse und markiert das Startdatum.
    """
    # Plots werden erstellt
    date_range = pd.date_range(start=start_date, end=end_date, freq='YS')

    plt.figure(figsize=(15, 6))
    plt.plot(eu.index, eu['close'], label="Close Price")
    plt.title('USD vs CHF')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()

    # Achsen werden eingestellt
    plt.xlim(left=pd.Timestamp(start_date), right=pd.Timestamp(end_date))
    plt.xticks(rotation=45)
    plt.xticks(date_range)

    # Layout wird optimiert und zugleich entschieden ob der Graph gezeigt wird oder nicht.
    plt.tight_layout()
    
    graph_name = "USD.CHF_1995-2023"

    # Speichert den Graphen
    save_graph(base_path, run_number, graph_name, graph_number)
    
    if show_graphs:
        plt.show()
    else:
        plt.close('all')

    
def generate_features(df):
    """
    Diese Funktion generiert Features für Währungspaar-Daten, einschließlich selbst gewählter Indikatoren.
    """
    df_new = pd.DataFrame()
    
    # One-Hot-Encoding der 'weekday'-Spalte, um Wochenenden von Wochentagen zu unterscheiden
    weekday_onehot = pd.get_dummies(df['weekday'], prefix='weekday')
    df_new = pd.concat([df_new, weekday_onehot], axis=1)
    
    # 5 Original-Features
    df_new['open'] = df['open']
    df_new['open_1'] = df['open'].shift(1)
    df_new['close_1'] = df['close'].shift(1)
    df_new['high_1'] = df['high'].shift(1)
    df_new['low_1'] = df['low'].shift(1)
    
    # Durchschnittspreis
    df_new['avg_price_5'] = df['close'].rolling(window=5).mean().shift(1)
    df_new['avg_price_30'] = df['close'].rolling(window=21).mean().shift(1)
    df_new['avg_price_90'] = df['close'].rolling(window=63).mean().shift(1)
    df_new['avg_price_365'] = df['close'].rolling(window=252).mean().shift(1)
    
    # Verhältnisse der Durchschnittspreise
    df_new['ratio_avg_price_5_30'] = df_new['avg_price_5'] / df_new['avg_price_30']
    df_new['ratio_avg_price_5_90'] = df_new['avg_price_5'] / df_new['avg_price_90']
    df_new['ratio_avg_price_5_365'] = df_new['avg_price_5'] / df_new['avg_price_365']
    df_new['ratio_avg_price_30_90'] = df_new['avg_price_30'] / df_new['avg_price_90']
    df_new['ratio_avg_price_30_365'] = df_new['avg_price_30'] / df_new['avg_price_365']
    df_new['ratio_avg_price_90_365'] = df_new['avg_price_90'] / df_new['avg_price_365']                                            
    
    # Standardabweichung der Preise
    df_new['std_price_5'] = df['close'].rolling(window=5).std().shift(1)
    df_new['std_price_30'] = df['close'].rolling(window=21).std().shift(1)
    df_new['std_price_90'] = df['close'].rolling(window=63).std().shift(1)                                               
    df_new['std_price_365'] = df['close'].rolling(window=252).std().shift(1)
    
    # Verhältnisse der Standardabweichungen der Preise
    df_new['ratio_std_price_5_30'] = df_new['std_price_5'] / df_new['std_price_30']
    df_new['ratio_std_price_5_90'] = df_new['std_price_5'] / df_new['std_price_90']
    df_new['ratio_std_price_5_365'] = df_new['std_price_5'] / df_new['std_price_365']
    df_new['ratio_std_price_30_90'] = df_new['std_price_30'] / df_new['std_price_90'] 
    df_new['ratio_std_price_30_365'] = df_new['std_price_30'] / df_new['std_price_365']                                               
    df_new['ratio_std_price_90_365'] = df_new['std_price_90'] / df_new['std_price_365']                                                
    
    # Rendite
    df_new['return_1'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1)).shift(1)
    df_new['return_5'] = ((df['close'] - df['close'].shift(5)) / df['close'].shift(5)).shift(1)
    df_new['return_30'] = ((df['close'] - df['close'].shift(21)) / df['close'].shift(21)).shift(1)
    df_new['return_90'] = ((df['close'] - df['close'].shift(63)) / df['close'].shift(63)).shift(1)                                                
    df_new['return_365'] = ((df['close'] - df['close'].shift(252)) / df['close'].shift(252)).shift(1)
    
    # Durchschnittliche Rendite
    df_new['moving_avg_5'] = df_new['return_1'].rolling(window=5).mean()
    df_new['moving_avg_30'] = df_new['return_1'].rolling(window=21).mean()
    df_new['moving_avg_90'] = df_new['return_1'].rolling(window=63).mean()
    df_new['moving_avg_365'] = df_new['return_1'].rolling(window=252).mean()
    
    # Neue Indikatoren
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df_new['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9, adjust=False).mean()
    df_new['macd'] = macd
    df_new['macd_signal'] = signal
    df_new['macd_hist'] = macd - signal
    
    # Bollinger-Bänder
    sma = df['close'].rolling(window=20).mean()
    rolling_std = df['close'].rolling(window=20).std()
    df_new['bollinger_upper'] = sma + (rolling_std * 2)
    df_new['bollinger_middle'] = sma
    df_new['bollinger_lower'] = sma - (rolling_std * 2)
    
    # Stochastischer Oszillator
    low_min = df['low'].rolling(window=14).min()
    high_max = df['high'].rolling(window=14).max()
    df_new['k_stochastic'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
    df_new['d_stochastic'] = df_new['k_stochastic'].rolling(window=3).mean()
    
    # Durchschnittliche True Range (ATR)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close']).abs()
    low_close = (df['low'] - df['close']).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df_new['atr'] = true_range.rolling(window=14).mean()
    
    # Commodity Channel Index (CCI)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    moving_avg = typical_price.rolling(window=20).mean()
    mean_deviation = typical_price.rolling(window=20).apply(lambda x: abs(x - x.mean()).mean())
    df_new['cci'] = (typical_price - moving_avg) / (0.015 * mean_deviation)
    
    # Momentum
    df_new['momentum'] = df['close'] - df['close'].shift(4)
    
    # Rate of Change (ROC)
    df_new['roc'] = df['close'].pct_change(periods=4) * 100
    
    # Williams %R
    low_min = df['low'].rolling(window=14).min()
    high_max = df['high'].rolling(window=14).max()
    df_new['williams_r'] = ((high_max - df['close']) / (high_max - low_min)) * -100
    
    # Price Rate of Change (PROC)
    df_new['proc'] = (df['close'].pct_change(periods=9)) * 10
    
    # TRIX
    ex1 = df['close'].ewm(span=18, adjust=False).mean()
    ex2 = ex1.ewm(span=18, adjust=False).mean()
    ex3 = ex2.ewm(span=18, adjust=False).mean()
    df_new['trix'] = ex3.pct_change(periods=1) * 100
    
    # Percentage Price Oscillator (PPO)
    short_ema = df['close'].ewm(span=12, adjust=False).mean()
    long_ema = df['close'].ewm(span=26, adjust=False).mean()
    df_new['ppo'] = ((short_ema - long_ema) / long_ema) * 100
    
    # Das Ziel
    df_new['close'] = df['close']
    df_new = df_new.dropna(axis=0)
    
    return df_new


def visual2(data, show_graphs, base_path, graph_number, run_number):
    """
    Erstellt eine Heatmap zur Visualisierung der Korrelation zwischen den Features.
    Filtert die 'weekday'-Features heraus und zeigt keine Anmerkungen an.
    """
    # Filtere alle Spalten heraus, die nicht mit 'weekday' beginnen
    filtered_data = data[[col for col in data.columns if not col.startswith('weekday')]]

    # Größe des Heatmap-Plots festlegen
    plt.figure(figsize=(19, 16))

    # Einstellungen für die Darstellung der Heatmap
    sns.set(font_scale=0.7)
    ax = sns.heatmap(
        filtered_data.corr(),  
        annot=True, 
        cmap='coolwarm',
        center=0,
        fmt=".1f",  
        annot_kws={"size": 8},  
        linewidths=.5
    )

    # Titel und Achsenbeschriftungen
    plt.title("Korrelations-Heatmap der Features (ohne 'weekday'-Features)", fontsize=14)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )

    # Layout wird optimiert
    plt.tight_layout()
    
    # Graphenname und Nummer für diesen Code
    graph_name = "Heatmap_Korrelation"

    # Speichert den Graphen
    save_graph(base_path, run_number, graph_name, graph_number)
    
    # Entscheiden, ob die Heatmap angezeigt wird oder nicht.
    if show_graphs:
        plt.show()
    plt.close('all')


#=============================================================================================================================================================================================================#

def split_data(data, start_train, end_train, start_test, end_test):
    """
    Teilt die Daten in Trainings- und Testsets auf.
    """
    # Aufteilen der Daten in Trainings- und Testsets
    data_train = data.loc[start_train:end_train]
    data_test = data.loc[start_test:end_test]

    # Feature und Zielvariablen für das Training setzen
    X_train = data_train.drop('close', axis='columns')
    y_train = data_train.close

    # Feature und Zielvariablen für das Testen setzen
    X_test = data_test.drop('close', axis='columns')
    y_test = data_test.close

    # Informationen über die Form der Daten ausgeben
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    return X_train, y_train, X_test, y_test


def scale_features(X_train, X_test, scaler_path):
    """
    Skaliert die Features für Trainings- und Testdatensätze und speichert den Scaler.
    """
    # Skalieren der Daten
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_scaled_train = scaler.transform(X_train)
    X_scaled_test = scaler.transform(X_test)
    
    # Scaler speichern, wenn er nicht existiert
    if not os.path.exists(scaler_path):
        joblib.dump(scaler, scaler_path)
        
    return X_scaled_train, X_scaled_test


#=============================================================================================================================================================================================================#

#Linear Regression
def train_evaluate_linear_model(X_train, y_train, X_test, y_test, graph_path, graph_number, run_number, show_graphs=True):
    """
    Trainiert ein lineares Modell (Linear Regression) mit Ridge-Regularisierung und evaluiert es.
    """
    print('\nLinear Regression -->\n')
    
    # Definieren des Hyperparameter-Raums für Ridge-Regularisierung
    param_grid = {
        'alpha': [0.1, 1.0, 10.0]  # Regularisierungsstärken (vereinfacht für schnellere Ausführung)
    }
    
    ridge = Ridge()
    grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print("Beste gefundene Parameter: ", grid_search.best_params_)
    
    model = grid_search.best_estimator_
    predictions = model.predict(X_test)
    
    # Evaluieren des Modells und Erstellen von Plots
    evaluate_and_plot(predictions, y_test, 'Linear Regression with Ridge', show_graphs, graph_path, graph_number, run_number)
    
    return model, predictions


#SGD-Regressor
def train_evaluate_sgd_model(X_train, y_train, X_test, y_test, graph_path, graph_number, run_number, show_graphs=True):
    """
    Trainiert ein SGD-Regressor-Modell und evaluiert es.
    """
    print('\nSGD Regressor Model Training -->\n')
    
    # Definieren des Hyperparameter-Raums für die Grid-Suche
    param_grid = {
        'max_iter': [500, 1000],  # Maximale Iterationen (reduziert für schnelleren Durchlauf)
        'penalty': ['l1', 'l2'], 
        'alpha': [1e-5, 1e-4]    
    }
    
    # Initialisieren des SGD-Regressors
    sgd = SGDRegressor()
    
    # Durchführen der Grid-Suche für die besten Hyperparameter
    grid_search = GridSearchCV(sgd, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print("Beste gefundene Parameter: ", grid_search.best_params_)
    
    # Erstellen des Modells mit den besten Hyperparametern
    model = grid_search.best_estimator_
    
    # Vorhersagen mit dem Modell auf den Testdaten
    predictions = model.predict(X_test)
    
    # Evaluieren des Modells und Erstellen von Plots
    evaluate_and_plot(predictions, y_test, 'SGD Regressor', show_graphs, graph_path, graph_number, run_number)
    
    return model, predictions


#XGBoost-Regressor
def train_evaluate_xgb_model(X_train, y_train, X_test, y_test, graph_path, graph_number, run_number, show_graphs=True):
    """
    Trainiert ein XGBoost-Regressor-Modell und evaluiert es.
    """
    print('\nXGBoost Regressor Model Training -->\n')
       
    # Definieren des Hyperparameter-Raums für die Grid-Suche
    xgb_param_grid = {
        'n_estimators': [50, 100],  # Anzahl der Bäume (reduziert für schnelleren Durchlauf)
        'max_depth': [3, 5]
    }
    
    # Initialisieren des XGBoost-Regressors
    xgb = XGBRegressor()
    
    # Zeitbasierte Cross-Validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Durchführen der Grid-Suche für die besten Hyperparameter
    grid_search = GridSearchCV(estimator=xgb, param_grid=xgb_param_grid, scoring='neg_mean_squared_error', cv=tscv, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train) 
       
    print("Beste gefundene Parameter: ", grid_search.best_params_)
    
    # Erstellen des Modells mit den besten Hyperparametern
    model = grid_search.best_estimator_
    
    # Vorhersagen mit dem Modell auf den Testdaten
    predictions = model.predict(X_test)
    
    # Evaluieren des Modells und Erstellen von Plots
    evaluate_and_plot(predictions, y_test, 'XGBoost Regressor', show_graphs, graph_path, graph_number, run_number)
    
    return model, predictions


#Bagging-Regressor
def train_evaluate_bgr_model(X_train, y_train, X_test, y_test, estimator, graph_path, graph_number, run_number, show_graphs=True):
    """
    Trainiert ein Bagging-Regressor-Modell und evaluiert es.
    """
    print('\nBagging Regressor Model Training -->\n')
    
    # Definieren des Hyperparameter-Raums für die Grid-Suche
    param_grid = {
        'n_estimators': [50, 100], 
        'max_samples': [0.7, 1.0],  
        'bootstrap': [True, False]  
    }
    
    # Initialisieren des Bagging-Regressors mit dem angegebenen Schätzer
    bagging = BaggingRegressor(estimator=estimator)
    
    # Durchführen der Grid-Suche für die besten Hyperparameter
    grid_search = GridSearchCV(bagging, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print("Beste gefundene Parameter: ", grid_search.best_params_)
    
    # Erstellen des Modells mit den besten Hyperparametern
    model = grid_search.best_estimator_
    
    # Vorhersagen mit dem Modell auf den Testdaten
    predictions = model.predict(X_test)
    
    # Evaluieren des Modells und Erstellen von Plots
    evaluate_and_plot(predictions, y_test, 'Bagging Regressor', show_graphs, graph_path, graph_number, run_number)
    
    return model, predictions


# Random Forest Regressor
def train_evaluate_rf_model(X_train, y_train, X_test, y_test, graph_path, graph_number, run_number, show_graphs=True):
    """
    Trainiert ein Random Forest-Regressor-Modell und evaluiert es.
    """
    print('\nRandom Forest Regressor Model Training -->\n')
    
    # Definieren des Hyperparameter-Raums für die Grid-Suche
    param_grid = {
        'n_estimators': [10],  # Reduzierte Anzahl von Bäumen
        'max_depth': [10],     # Begrenzte maximale Tiefe
        'bootstrap': [True]    # Bootstrapping aktiviert
    }
    
    # Initialisieren des Random Forest-Regressors
    rf = RandomForestRegressor()
    
    # Durchführen der Grid-Suche für die besten Hyperparameter
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print("Beste gefundene Parameter: ", grid_search.best_params_)
    
    # Erstellen des Modells mit den besten Hyperparametern
    model = grid_search.best_estimator_
    
    # Vorhersagen mit dem Modell auf den Testdaten
    predictions = model.predict(X_test)
       
    # Evaluieren des Modells und Erstellen von Plots
    evaluate_and_plot(predictions, y_test, 'Random Forest Regressor', show_graphs, graph_path, graph_number, run_number)
    
    return model, predictions


def evaluate_and_plot(predictions, y_test, model_name, show_graphs, graph_path, graph_number, run_number):
    """
    Evaluieren des Modells und Erstellen eines Vergleichsgrafen.
    """
    print('RMSE: {:.3f}'.format(mean_squared_error(y_test, predictions)**0.5))
    print('MAE: {:.3f}'.format(mean_absolute_error(y_test, predictions)))
    print('R^2: {:.3f}'.format(r2_score(y_test, predictions)))
    
    dates = y_test.index.values
    
    # Erstellen des Vergleichsgrafen
    plt.figure(figsize=(18,9))
    plt.plot(dates, y_test, label='Truth')
    plt.plot(dates, predictions, label=model_name)
    plt.legend(fontsize='xx-large')
    plt.title(f'USD vs CHF: Prediction vs Truth - {model_name}', fontsize=24)
    plt.xlabel('Date', fontsize=20)
    plt.ylabel('Price', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # Speichert den Graphen
    save_graph(graph_path, run_number, model_name, graph_number)
    
    if show_graphs:
        plt.show()
    else:
        plt.close('all')


#=============================================================================================================================================================================================================#


def save_all_models(models, all_pkl_path):
    """
    Speichert alle Modelle in .pkl-Dateiformaten und erhöht die Versionsnummer,
    wenn eine Datei mit demselben Namen bereits existiert.
    """
    # Liste aller .pkl-Dateien im Ordner "all_pkl_files" (Aktualisierter Ordnername)
    existing_files = glob.glob(os.path.join(all_pkl_path, '*.pkl'))
    
    for model_name, model in models.items():
        # Extrahieren der Versionsnummern und Ermitteln der höchsten vorhandenen Version
        versions = [float(file.split(os.sep)[-1].split("_")[-1].replace(".pkl", "")) 
                    for file in existing_files 
                    if file.startswith(os.path.join(all_pkl_path, model_name.replace(" ", "_").lower())) and file.endswith('.pkl')]
    
        # Bestimmen der neuen Versionsnummer
        if versions:
            latest_version = max(versions)
            new_version = latest_version + 0.01
        else:
            new_version = 1.00
    
        # Generieren des neuen Dateinamens mit dem Ordnerpfad
        model_filename = os.path.join(all_pkl_path, f'{model_name.replace(" ", "_").lower()}_{new_version:.2f}.pkl')
    
        # Speichern des Modells
        joblib.dump(model, model_filename)
        print(f"\nModell {model_name} wurde als {model_filename} gespeichert.\n")


def evaluate_select_and_plot_models(predictions, y_test, show_graphs=True):
    """
    Evaluieren der Modelle, Auswahl des besten Modells basierend auf dem MAE und Visualisierung der Leistung.
    """
    # Berechnung des MAE für jedes Modell
    mae_scores = {name: mean_absolute_error(y_test, pred) for name, pred in predictions.items()}

    # Bestimmen des Modells mit dem niedrigsten MAE
    best_model_name = min(mae_scores, key=mae_scores.get)
    print(f"Best performing model: {best_model_name} with MAE: {mae_scores[best_model_name]:.4f}")

    # Visualisierung des MAE jedes Modells
    plt.bar(mae_scores.keys(), mae_scores.values(), color='skyblue')
    plt.ylabel('Mean Absolute Error')
    plt.title('Model Performance Comparison based on MAE')
    plt.axhline(y=mae_scores[best_model_name], color='r', linestyle='-')
    plt.text(0, mae_scores[best_model_name] + 0.0002, f'Best MAE: {mae_scores[best_model_name]:.4f}', color='r')
    plt.xticks(rotation=45, ha='right')

    if show_graphs:
        plt.show()
    else:
        plt.close('all')
    
    return best_model_name


def save_best_model(best_model_name, models, best_pkl_path):
    """
    Speichert das beste Modell in einem .pkl-Dateiformat und erhöht die Versionsnummer,
    wenn eine Datei mit demselben Namen bereits existiert.
    """
    # Liste aller .pkl-Dateien im Ordner "pkl_files"
    existing_files = glob.glob(os.path.join(best_pkl_path, "*.pkl"))
    
    # Extrahieren der Versionsnummern und Ermitteln der höchsten vorhandenen Version
    versions = [float(file.split(os.sep)[-1].split("_")[-1].replace(".pkl", "")) 
                for file in existing_files 
                if file.startswith(os.path.join(best_pkl_path, best_model_name.replace(" ", "_").lower())) and file.endswith('.pkl')]

    # Bestimmen der neuen Versionsnummer
    if versions:
        latest_version = max(versions)
        new_version = latest_version + 0.01
    else:
        new_version = 1.00

    # Generieren des neuen Dateinamens mit dem Ordnerpfad
    model_filename = os.path.join(best_pkl_path, f'{best_model_name.replace(" ", "_").lower()}_{new_version:.2f}.pkl')

    # Speichern des Modells
    joblib.dump(models[best_model_name], model_filename)
    print(f"\nModell {best_model_name} wurde als {model_filename} gespeichert.\n")


def visualize_and_evaluate_model(loaded_model, X_test, y_test, best_model_name, show_graphs=True):
    """
    Visualisiert die Vorhersagen des geladenen Modells und bewertet seine Leistung.
    """
    
    # Die Vorhersagen werden visualisiert
    predictions = loaded_model.predict(X_test)
    
    plt.figure(figsize=(15,7))
    plt.plot(y_test.index, y_test, 'r', label='Tatsächlicher Preis')
    plt.plot(y_test.index, predictions, 'b', label='Vorhergesagter Preis')
    plt.title(f"{best_model_name} Modell Vorhersagen")
    plt.xlabel('Datum')
    plt.ylabel('Preis')
    plt.legend()
    if show_graphs:
        plt.show()
    else:
        plt.close('all')

    # Regressionlinie wird dargestellt
    plt.scatter(y_test, predictions, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=3, label='Regressionslinie')
    plt.title(f"Regressionslinie für {best_model_name}")
    plt.legend()
    if show_graphs:
        plt.show()
    else:
        plt.close('all')

    # Modellleistungs-Bewertungsmetriken
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"MSE (Mittlerer quadratischer Fehler): {mse:.4f}")
    print(f"RMSE (Quadratwurzel des mittleren quadratischen Fehlers): {rmse:.4f}")
    print(f"MAE (Mittlerer absoluter Fehler): {mae:.4f}")
    print(f"R^2 (Bestimmtheitsmaß): {r2:.4f}")

#=============================================================================================================================================================================================================#
#=============================================================================================================================================================================================================#

def main(show_graphs, save_all):
    """
    Diese Funktion koordiniert den Ablauf des Modellspeicher-Prozesses für das USD/CHF Währungspaar.
    """
    
    # Konfigurationsdatei wird geladen
    config = load_config()
    
    # Dateipfade aus der Konfigurationsdatei werden abgerufen
    file_path = os.path.abspath(config['Paths']['file_path'])
    scaler_path = os.path.abspath(config['Paths']['scaler_path'])
    all_pkl_path = os.path.abspath(config['Paths']['all_pkl_path'])
    best_pkl_path = os.path.abspath(config['Paths']['best_pkl_path'])
    kontroll_output_path = os.path.abspath(config['Paths']['kontroll_output_path'])
    graph_path = os.path.abspath(config['Paths']['graphiken_code1'])    
    
    # Überprüfen, ob die angegebenen Dateipfade existieren
    if not os.path.exists(file_path):
        print(f"Fehler: Die Datei {file_path} existiert nicht.")
        return   
    
    # Festlegen der Zeitspanne für die Datenanalyse
    start_date = '1995-01-03'
    end_date = '2023-08-01'
    
    # Laden und Filtern der Forex-Daten
    data = load_and_filter_data(file_path, start_date, end_date, kontroll_output_path)
    
    # Ermittelt die nächste Run-Nummer einmal für diese Ausführung des Codes
    run_number = get_next_run_number(graph_path)
    graph_number = 1
    
    # Initiierung der ersten Datenvisualisierung
    visual1(data, start_date, end_date, show_graphs, graph_path, graph_number, run_number)
    graph_number += 1  # Inkrementieren der Graphennummer

    # Datenvisualisierung mit dem aktualisierten DataFrame
    visual2(data, show_graphs, graph_path, graph_number, run_number)
    graph_number += 1  # Inkrementieren der Graphennummer
    
    # Festlegen der Zeiträume für Trainings- und Testdatensätze
    start_train = datetime.datetime(1995, 3, 1, 0, 0)
    end_train = datetime.datetime(2021, 12, 31, 0, 0)
    start_test = datetime.datetime(2022, 1, 1, 0, 0)
    end_test = datetime.datetime(2023, 8, 1, 0, 0)

    # Aufteilen der Forex-Daten in Trainings- und Testdaten          
    X_train, y_train, X_test, y_test = split_data(data, start_train, end_train, start_test, end_test)
    
    # Skalierung der Forex-Daten
    X_scaled_train, X_scaled_test = scale_features(X_train, X_test, scaler_path)
   
    # Training und Evaluation der Modelle mit den skalierten Daten
    lin_model, predictions_lin = train_evaluate_linear_model(X_scaled_train, y_train, X_scaled_test, y_test, graph_path, graph_number, run_number, show_graphs)   
    graph_number += 1
    sgd_model, predictions_sgd = train_evaluate_sgd_model(X_scaled_train, y_train, X_scaled_test, y_test, graph_path, graph_number, run_number, show_graphs)    
    graph_number += 1
    xgb_model, predictions_xgb = train_evaluate_xgb_model(X_scaled_train, y_train, X_scaled_test, y_test, graph_path, graph_number, run_number, show_graphs)   
    graph_number += 1
    bgr_model, predictions_bgr = train_evaluate_bgr_model(X_scaled_train, y_train, X_scaled_test, y_test, lin_model, graph_path, graph_number, run_number, show_graphs)   
    graph_number += 1
    rf_model, predictions_rf = train_evaluate_rf_model(X_scaled_train, y_train, X_scaled_test, y_test, graph_path, graph_number, run_number, show_graphs)

    
    # Erstellung eines Dictionaries mit allen Modellen
    models = {
        'Lineare Regression': lin_model,
        'SGD Regressor': sgd_model,
        'XGBoost Regressor': xgb_model,
        'Bagging Regressor': bgr_model,
        'Random Forest Regressor': rf_model
    }

    #Auswahl des besten Modells
    predictions = {
        'Lineare Regression': predictions_lin,
        'SGD Regressor': predictions_sgd,
        'XGBoost Regressor': predictions_xgb,
        'Bagging Regressor': predictions_bgr,
        'Random Forest Regressor': predictions_rf
    }
    
    #Speichern aller Modelle, wenn vom User gewünscht
    if save_all:
        save_all_models(models, all_pkl_path)
    
    #Auswahl und Speicherung des besten Modells
    best_model_name = evaluate_select_and_plot_models(predictions, y_test, show_graphs)
        
    #Speichern des besten Modells
    save_best_model(best_model_name, models, best_pkl_path)
    
    # Das beste Modell wird für weitere Analysen verwendet
    best_model = models[best_model_name]
    
    # Das beste Modell wird geladen, visualisiert und bewertet
    visualize_and_evaluate_model(best_model, X_scaled_test, y_test, best_model_name, show_graphs)
    
    #Alle offenen Grafiken werden geschlossen, um Konflikte mit dem System zu vermeiden
    plt.close('all')

#=============================================================================================================================================================================================================#
#=============================================================================================================================================================================================================#
    
#Main funktion wird bei starten des Programs ausgeführt
if __name__ == "__main__":
    # Benutzereingaben abrufen: show_graphs_input (Grafiken anzeigen), save_all_input (Daten speichern), num_runs (Anzahl der Durchläufe)
    show_graphs_input, save_all_input, num_runs = get_user_inputs()

    # Zeitmessung starten
    start_time = time.time()

    # Schleife für die Anzahl der Durchläufe
    for i in range(num_runs):
        print(f"Durchlauf {i + 1} von {num_runs}")
        main(show_graphs_input, save_all_input)

    # Zeitmessung beenden und die verstrichene Zeit ausgeben
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_minutes, elapsed_time_seconds = divmod(elapsed_time, 60)
    print(f"Die Ausführung dauerte {elapsed_time_minutes:.0f} Minuten und {elapsed_time_seconds:.2f} Sekunden.")