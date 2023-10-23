#code 5

# Importieren von Bibliotheken und Modulen
import yfinance as yf
import os
import pandas as pd
import configparser
import sys

#Code Name
print(f"code 5")
print("\n")  # Einen Zeilenumbruch einfügen

# Python und Paketversionen anzeigen
print('Python version:{}'.format(sys.version))
print('Pandas version:{}'.format(pd.__version__))
print('yfinance version:{}'.format(yf.__version__))
print("\n")  # Einen Zeilenumbruch einfügen


#Konfigurationsdatei laden
def load_config():
    config = configparser.ConfigParser()
    # Pfad zum Verzeichnis, in dem sich dieses Skript befindet
    current_dir = os.path.dirname(__file__)
    # Relativer Pfad zur config.ini-Datei
    config_path = os.path.join(current_dir, '..', '05config', 'config.ini')
    config.read(config_path)
    return config

# Funktion, um neue Daten von Yahoo Finance zu holen
def get_new_data_from_yahoo(start_date, end_date):
    ticker_symbol = "USDCHF=X"
    data = yf.download(ticker_symbol, start=start_date, end=end_date)
    print(data.columns)
    
    # Änderung der Spaltennamen in Kleinbuchstaben
    data.columns = data.columns.str.lower()

    data = modify_data(data)
    return data


# Daten in richtiges Format bringen
def modify_data(df):
    # 'Adj Close' und 'Volume' Spalte entfernen
    columns_to_drop = ['adj Close', 'volume']
    for col in columns_to_drop:
        if col in df.columns:
            df.drop([col], axis=1, inplace=True)

    # Format von "Date" ändern
    df.index = pd.to_datetime(df.index).strftime('%Y.%m.%d')
    
    # Entferne den letzten Eintrag aus dem DataFrame, da dieser falsche close information enthält
    df = df.iloc[:-1] 

    return df


# Funktion, um den nächsten Dateinamen zu erzeugen
def get_next_filename(directory):
    files = os.listdir(directory)
    version_numbers = []

    # Durchlaufe alle Dateien im Verzeichnis
    for file in files:
        if file.startswith("usdchf_") and file.endswith(".csv"):
            version_str = file.rstrip('.csv').split('_')[-1]  # Korrektur hier
            try:
                version_number = float(version_str)
                version_numbers.append(version_number)
            except ValueError:
                continue

    # Bestimme die nächste Version
    if not version_numbers:
        next_version = 1.00
    else:
        next_version = max(version_numbers) + 0.01

    return f"usdchf_{next_version:.2f}.csv"


if __name__ == "__main__":
    start_date = "2022-01-01"
    end_date = "2023-12-31"
    
    config = load_config()
    directory = config.get('Paths', 'yahoo_path')

    # Hole neue Daten
    new_data = get_new_data_from_yahoo(start_date, end_date)
    
    # Bestimme den Namen der neuen Datei
    file_name = get_next_filename(directory)
    path = os.path.join(directory, file_name)
    
    # Speichere die Daten
    new_data.to_csv(path)
    print("\n")  # Einen Zeilenumbruch einfügen
    print(f"Data saved in {path}.")
    print("\n")  # Einen Zeilenumbruch einfügen