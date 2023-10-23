#code 2

# Importieren von Bibliotheken und Modulen
import os
import re
import sys
import joblib
import matplotlib
import configparser
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
from pandas.tseries.offsets import BDay

# Code Name
print("code 2")
print("\n")  # Einen Zeilenumbruch einfügen

# Python und Paketversionen anzeigen
print('Python version:{}'.format(sys.version))
print('Pandas version:{}'.format(pd.__version__))
print('Matplotlib version:{}'.format(matplotlib.__version__))
print('Seaborn version:{}'.format(sns.__version__))
print("\n")  # Einen Zeilenumbruch einfügen


def load_config():
    """
    Lädt die Konfigurationsdatei und gibt sie als ConfigParser-Objekt zurück.
    """
    config = configparser.ConfigParser()
    # Pfad zum Verzeichnis, in dem sich dieses Skript befindet
    current_dir = os.path.dirname(__file__)
    # Relativer Pfad zur config.ini-Datei
    config_path = os.path.join(current_dir, '..', '05config', 'config.ini')
    config.read(config_path)
    return config

#Vorbereitung (usern inputs und quellen)
#==========================================================================================================================================================================================#

def get_user_preferences():
    """
    Diese Funktion ruft die Benutzereinstellungen ab, einschließlich der Anzahl der Tage für die Vorhersage,
    der Entscheidung zur Anzeige von Graphen, der Modellauswahl und des Modellverzeichnisses.
    """
    AZ_TAGE_PRED = get_user_input_for_days()
    print("\n")  # Einen Zeilenumbruch einfügen
    show_graphs = get_user_input_for_visualization()
    print("\n")  # Einen Zeilenumbruch einfügen
    model_choice = get_user_model_choice()
    print("\n")  # Einen Zeilenumbruch einfügen
    model_directory = get_directory_from_choice(model_choice)
    selected_model = select_model_from_directory(model_directory)
    print("\n" * 2)  # Zwei Zeilenumbrüche einfügen
    return AZ_TAGE_PRED, show_graphs, model_directory, selected_model

def get_user_input_for_days():
    """
    Diese Funktion fragt den Benutzer nach der Anzahl der Tage für die Vorhersage und gibt diese zurück.
    Wenn die Eingabe ungültig ist, wird der Standardwert von 3 Tagen verwendet.
    """
    try:
        num_days = int(input("Geben Sie die Anzahl der Tage für die Vorhersage ein: "))
        if num_days <= 0:
            print("Ungültige Eingabe. Verwende den Standardwert von 3 Tagen.")
            return 3
        return num_days
    except ValueError:
        print("Ungültige Eingabe. Verwende den Standardwert von 3 Tagen.")
        return 3

def get_user_input_for_visualization():
    """
    Diese Funktion fragt den Benutzer, ob Graphen angezeigt werden sollen, und gibt True zurück, wenn die Antwort 'j' ist.
    Andernfalls gibt es False zurück.
    """
    choice = input("Möchten Sie die Graphen anzeigen? (Vorhersagen werden unabhängig der Antworten angezeigt) (j/n): ")
    return choice.lower() == 'j'

def get_user_model_choice():
    """
    Diese Funktion fragt den Benutzer nach der Modellauswahl ('best_pkl_files' oder 'all_pkl_files') und gibt die Auswahl zurück.
    """
    choice = input("Möchten Sie aus 'best_pkl_files' (b) oder 'all_pkl_files' (a) wählen? (Eingabe: b/a): ")
    return choice.lower()

def get_directory_from_choice(choice):
    """
    Diese Funktion gibt das Modellverzeichnis basierend auf der Benutzerwahl ('b' oder 'a') aus der Konfigurationsdatei zurück.
    Wenn die Auswahl ungültig ist, wird 'best_pkl_files' als Standard verwendet.
    """
    config = load_config()
    if choice == 'b':
        return config.get('Paths', 'best_pkl_path')
    elif choice == 'a':
        return config.get('Paths', 'all_pkl_path')
    else:
        print("Ungültige Auswahl. Verwende 'best_pkl_files' als Standard.")
        return config.get('Paths', 'best_pkl_path')

def select_model_from_directory(directory_path):
    """
    Diese Funktion lässt den Benutzer ein Modell aus dem angegebenen Verzeichnis auswählen und gibt den Dateinamen zurück.
    """
    files = [f for f in os.listdir(directory_path) if f.endswith('.pkl')]
    for idx, file in enumerate(files, 1):
        print(f"{idx}. {file}")
    choice = int(input("Wählen Sie die Nummer des Modells (XGBoost empfohlen): "))
    return files[choice - 1]

#==========================================================================================================================================================================================#
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
    current_date = datetime.now().strftime("%Y-%m-%d")
    new_folder_name = f"Run_{run_number}_{current_date}"
    new_folder_path = os.path.join(base_path, new_folder_name)
    
    # Erstellt den Ordner, falls nicht vorhanden
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    
    # Speichert den Graphen im Ordner
    complete_file_name = f"{file_name}_{graph_number}.png"
    plt.savefig(os.path.join(new_folder_path, complete_file_name))


def initialize(model_directory, selected_model_name):
    """Initialisiert Pfade und lädt Modell, Scaler und die Datenquelle."""
    config = load_config()

    # Änderung hier: Pfad zu IBKR-Daten
    data_directory = config.get('Paths', 'ibrk_path')  # Änderung von 'yahoo_path' zu 'ibrk_path'
    scaler_directory = config.get('Paths', 'scaler_path')

    model_path = os.path.join(model_directory, selected_model_name)
    scaler_path = os.path.join(scaler_directory)

    ibkrfile = get_latest_file(data_directory)  # Änderung: Variable von 'yahoofile' zu 'ibkrfile'
    model = load_model(model_path)
    scaler = load_scaler(scaler_path)

    if model is None or scaler is None:
        print("Fehler beim Laden von Modell oder Scaler.")
        sys.exit(1)

    return model, scaler, ibkrfile 

def get_latest_file(directory):
    """Ermittelt die neueste CSV-Datei im angegebenen Verzeichnis."""
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith("usdchf_") and f.endswith(".csv")]
    return max(files, key=os.path.getctime)

def load_model(model_path):
    """Lädt das ML-Modell aus den angegebenen Pfaden."""
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print("Datei nicht gefunden.")
        return None
    return model

def load_scaler(scaler_path):
    """Lädt den Scaler aus den angegebenen Pfaden."""
    try:
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        print("Datei nicht gefunden.")
        return None
    return scaler

#==========================================================================================================================================================================================#

#Die Features sind die Gleichen wie in code 1
def generate_features(df):
    """ Generate features for a stock/index/currency/commodity based on historical price and performance
    Args:
        df (dataframe with columns "open", "close", "high", "low")
    Returns:
        dataframe, data set with new features
    """
    df_new = pd.DataFrame()
    
    # One-Hot-Encoding der 'weekday'-Spalte, um wochenenden von wochentagen zu unterscheiden
    weekday_onehot = pd.get_dummies(df['weekday'], prefix='weekday')
    df_new = pd.concat([df_new, weekday_onehot], axis=1)
    
    # 5 original features
    df_new['open'] = df['open']
    df_new['open_1'] = df['open'].shift(1)
    df_new['close_1'] = df['close'].shift(1)
    df_new['high_1'] = df['high'].shift(1)
    df_new['low_1'] = df['low'].shift(1)
    
    # average price
    df_new['avg_price_5'] = df['close'].rolling(window=5).mean().shift(1)
    df_new['avg_price_30'] = df['close'].rolling(window=21).mean().shift(1)
    df_new['avg_price_90'] = df['close'].rolling(window=63).mean().shift(1)
    df_new['avg_price_365'] = df['close'].rolling(window=252).mean().shift(1)
    
    # average price ratio
    df_new['ratio_avg_price_5_30'] = df_new['avg_price_5'] / df_new['avg_price_30']
    df_new['ratio_avg_price_5_90'] = df_new['avg_price_5'] / df_new['avg_price_90']
    df_new['ratio_avg_price_5_365'] = df_new['avg_price_5'] / df_new['avg_price_365']
    df_new['ratio_avg_price_30_90'] = df_new['avg_price_30'] / df_new['avg_price_90']
    df_new['ratio_avg_price_30_365'] = df_new['avg_price_30'] / df_new['avg_price_365']
    df_new['ratio_avg_price_90_365'] = df_new['avg_price_90'] / df_new['avg_price_365']                                            
    
    # standard deviation of prices
    df_new['std_price_5'] = df['close'].rolling(window=5).std().shift(1)
    df_new['std_price_30'] = df['close'].rolling(window=21).std().shift(1)
    df_new['std_price_90'] = df['close'].rolling(window=63).std().shift(1)                                               
    df_new['std_price_365'] = df['close'].rolling(window=252).std().shift(1)
    
    # standard deviation ratio of prices 
    df_new['ratio_std_price_5_30'] = df_new['std_price_5'] / df_new['std_price_30']
    df_new['ratio_std_price_5_90'] = df_new['std_price_5'] / df_new['std_price_90']
    df_new['ratio_std_price_5_365'] = df_new['std_price_5'] / df_new['std_price_365']
    df_new['ratio_std_price_30_90'] = df_new['std_price_30'] / df_new['std_price_90'] 
    df_new['ratio_std_price_30_365'] = df_new['std_price_30'] / df_new['std_price_365']                                               
    df_new['ratio_std_price_90_365'] = df_new['std_price_90'] / df_new['std_price_365']                                                
    
    # return
    df_new['return_1'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1)).shift(1)
    df_new['return_5'] = ((df['close'] - df['close'].shift(5)) / df['close'].shift(5)).shift(1)
    df_new['return_30'] = ((df['close'] - df['close'].shift(21)) / df['close'].shift(21)).shift(1)
    df_new['return_90'] = ((df['close'] - df['close'].shift(63)) / df['close'].shift(63)).shift(1)                                                
    df_new['return_365'] = ((df['close'] - df['close'].shift(252)) / df['close'].shift(252)).shift(1)
    
    # average of return
    df_new['moving_avg_5'] = df_new['return_1'].rolling(window=5).mean()
    df_new['moving_avg_30'] = df_new['return_1'].rolling(window=21).mean()
    df_new['moving_avg_90'] = df_new['return_1'].rolling(window=63).mean()
    df_new['moving_avg_365'] = df_new['return_1'].rolling(window=252).mean()
    

    # NEUE INDIKATOREN

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

    # Bollinger Bands
    sma = df['close'].rolling(window=20).mean()
    rolling_std = df['close'].rolling(window=20).std()
    df_new['bollinger_upper'] = sma + (rolling_std * 2)
    df_new['bollinger_middle'] = sma
    df_new['bollinger_lower'] = sma - (rolling_std * 2)
    
    # Stochastic Oscillator
    low_min = df['low'].rolling(window=14).min()
    high_max = df['high'].rolling(window=14).max()
    df_new['k_stochastic'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
    df_new['d_stochastic'] = df_new['k_stochastic'].rolling(window=3).mean()
    
    # Average True Range (ATR)
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
    
    # the target
    df_new['close'] = df['close']
    df_new = df_new.dropna(axis=0)
    return df_new

# Visualisation der Features in Form einer Heatmap
def visual1(data, show_graphs, base_path, graph_number, run_number):    
    plt.figure(figsize=(18, 15))  # Reduzierte Größe der Heatmap
    sns.set(font_scale=0.5)  # Leichte Reduzierung der Fontgröße
    ax = sns.heatmap(
        data.corr(),
        annot=True,
        cmap='coolwarm',
        center=0,
        fmt=".1f",  # Reduzieren der Anzahl der Dezimalstellen
        annot_kws={"size": 8},  # Verkleinern der Schriftgröße der Annotationen
        linewidths=.5
    )
    plt.title("Feature Correlation Heatmap", fontsize=14)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    # Graphenname und Nummer für diesen Code
    graph_name = "Heatmap_Korrelation"

    # Speichert den Graphen
    save_graph(base_path, run_number, graph_name, graph_number)
    if show_graphs:
        plt.show()
    else:
        plt.close('all')
        
def scale_prediction_features(data, scaler):
    """Skaliert die Features für den Vorhersagedatensatz mit einem vorhandenen Scaler."""
    features_to_scale = data.drop(['close'], axis=1)  # Entfernen der Zielvariable
    scaled_features = scaler.transform(features_to_scale)  # Verwendung des geladenen Scalers
    data_scaled = pd.DataFrame(scaled_features, columns=features_to_scale.columns, index=features_to_scale.index)
    
    return data_scaled


#==========================================================================================================================================================================================#

def predict_next_days(model, scaled_data, num_days):
    """Macht eine Vorhersage für die nächsten 'num_days' Tage."""
    # Verwenden Sie `iloc` für die Slicing-Operation
    last_n_days = scaled_data.iloc[-num_days:, :].to_numpy()

     # Wählt die letzten 'num_days' Einträge aus den skalierten Daten.
    predictions = model.predict(last_n_days)  # Macht Vorhersagen für die nächsten 'num_days' Tage.
    
    return predictions


def plot_predictions(last_seven_days, predictions, last_seven_dates, show_graphs, graph_path, graph_number, run_number):
    """Plottet die Vorhersagen für die nächsten n Tage im Vergleich zu den letzten sieben Tagen."""
    
    # Tatsächliche Daten der letzten sieben Tage
    actual_data = list(last_seven_days)
    
    # Vorhersagedaten für die nächsten n Tage
    predicted_data = list(predictions)
    
    # Datumsangaben für die tatsächlichen und vorhergesagten Daten
    actual_dates = [pd.Timestamp(date) for date in last_seven_dates]
    last_actual_date = pd.Timestamp(last_seven_dates[-1])
    prediction_dates = [(last_actual_date + BDay(i)).date() for i in range(1, len(predictions) + 1)]
    
    # Größe des Diagramms festlegen
    plt.figure(figsize=(15, 9))
    
    # Tatsächliche Daten in Grün plotten
    plt.plot(actual_dates, actual_data, marker='o', color='g', label='Tatsächliche Werte', linewidth=2)
    
    # Vorhergesagte Daten in Rot plotten
    plt.plot(prediction_dates, predicted_data, marker='o', color='r', label='Vorhersagen', linewidth=2)
    
    # Die letzte tatsächliche Datenpunkte mit dem ersten vorhergesagten Punkt mit einer roten gestrichelten Linie verbinden
    plt.plot([actual_dates[-1], prediction_dates[0]], [actual_data[-1], predicted_data[0]], 'r--', linewidth=2)
 
    # Vertikale Linie zur Anzeige des heutigen Datums, in Schwarz
    today_date = datetime.now().strftime('%Y-%m-%d')  # Aktuelles Datum im gleichen Format wie Ihre Daten
    plt.axvline(pd.Timestamp(today_date), color='black', linestyle='--', linewidth=2, label='Heute')
    
    # Text neben den Punkten hinzufügen
    for i, value in enumerate(actual_data):
        plt.text(actual_dates[i], value, f"{value:.5f}", fontsize=12, ha='right')
    for i, value in enumerate(predicted_data):
        plt.text(prediction_dates[i], value, f"{value:.5f}", fontsize=12, ha='right')
    
    # Titel und Achsenbeschriftungen
    plt.title(f"USD/CHF Vorhersage für die nächsten {len(predictions)} Tage", fontsize=16)
    plt.xlabel("Datum", fontsize=14)
    plt.ylabel("Schlusskurs", fontsize=14)
    
    # Schriftgröße der Achsenticks festlegen
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Rasters und Datumsformate
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
    
    # Automatische Anpassung des Datumsformats
    plt.gcf().autofmt_xdate()
    
    # Legende hinzufügen
    plt.legend(fontsize=12)
    
    save_graph(graph_path, run_number, "Vorhersage", graph_number)
    
    # Diagramm anzeigen
    plt.show()
    
#==========================================================================================================================================================================================#
#==========================================================================================================================================================================================#

def main(AZ_TAGE_PRED, show_graphs, model_directory, selected_model):
    """
    Hauptfunktion zur Ausführung des Vorhersagemodells.
    """
    
    # Konfigurationsdatei laden
    config = load_config()
    kontroll_output_path = os.path.abspath(config['Paths']['kontroll_output_path'])
    graph_path = os.path.abspath(config['Paths']['graphiken_code2'])  
    
    model, scaler, ibkrfile = initialize(model_directory, selected_model)
    
    print("Modell:", model)
    print("Scaler:", scaler)
    print("Pfad zur neuesten Datei:", ibkrfile)
    
    # Daten aus der Excel Datei laden
    eu = pd.read_csv(ibkrfile)
    
    # Konvertiere die 'Date'-Spalte zum Datumsformat
    eu['date'] = pd.to_datetime(eu['date'], format='%Y.%m.%d')
    
    # Füge eine neue Spalte für den Wochentag hinzu
    eu['weekday'] = eu['date'].dt.weekday

    # Features generieren
    data = generate_features(eu) 
    
    # Überprüfen des DataFrames
    print("DataFrame nach generate_features:")
    print(data.head())
    print(data.columns)
    
    # Feature Namen anzeigen
    feature_names = data.columns.tolist()
    print("\n\nFeature-Namen:", feature_names, "\n\n")
    
    # Ermittelt die nächste Run-Nummer einmal für diese Ausführung des Codes
    run_number = get_next_run_number(graph_path)
    graph_number = 1
    
    # Datenvisualisierung mit dem aktualisierten DataFrame
    visual1(data, show_graphs, graph_path, graph_number, run_number)
    graph_number += 1  # Inkrementieren der Graphennummer
    
    # Skalieren der Features
    data_scaled = scale_prediction_features(data, scaler)
    
    # Kontrollausgabe nach der Skalierung
    kontroll_output_scaled_code2 = os.path.join(kontroll_output_path, 'data_scaled_code2.csv')
    pd.DataFrame(data_scaled).to_csv(kontroll_output_scaled_code2, index=False)
    
    # Vorhersage für die nächsten n Tage treffen
    predictions = predict_next_days(model, data_scaled, num_days=AZ_TAGE_PRED)

    print(f"Vorhersagen für die nächsten {len(predictions)} Tage: {predictions}")
        
    # Letzte 7 Tage echte Daten
    last_seven_days = data['close'].iloc[-7:].values

    # Datumsangaben generieren (angenommen, Ihr DataFrame enthält eine 'date'-Spalte)
    last_seven_dates = eu['date'].iloc[-7:].values 
    
    # Graph zeichnen
    plot_predictions(last_seven_days, predictions, last_seven_dates, show_graphs, graph_path, graph_number, run_number)

#==========================================================================================================================================================================================#
#==========================================================================================================================================================================================#    

if __name__ == "__main__":
    
    #User Inputs
    AZ_TAGE_PRED, show_graphs, model_directory, selected_model = get_user_preferences()
    
    #Main
    main(AZ_TAGE_PRED, show_graphs, model_directory, selected_model)