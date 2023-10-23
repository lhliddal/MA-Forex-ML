#code 4

# Importieren von Bibliotheken und Modulen
import os
import sys
import configparser
import pandas as pd

# Code Name
print("code 4")
print("\n")  # Einen Zeilenumbruch einfügen

# Python und Paketversionen anzeigen
print('Python version:{}'.format(sys.version))
print('Pandas version:{}'.format(pd.__version__))
print("\n")  # Einen Zeilenumbruch einfügen


# Lade Konfigurationsdatei
config = configparser.ConfigParser()
current_dir = os.path.dirname(__file__)
config_path = os.path.join(current_dir, '..', '05config', 'config.ini')
config.read(config_path)

# Lese den Dateipfad aus der Konfigurationsdatei
bankold_file_path = config.get("Paths", "bankold_file_path")
bankold_path = config.get("Paths", "bankold_path")

# Lade die CSV-Datei
data = pd.read_csv(bankold_file_path)

# Konvertiere die 'Gmt time'-Spalte zum Datumsformat und extrahiere nur das Datum
data['Gmt time'] = pd.to_datetime(data['Gmt time'], format='%d.%m.%Y %H:%M:%S.%f').dt.date


# Entferne Sonntage (Wochentag 6 entspricht Sonntag)
data['weekday'] = pd.to_datetime(data['Gmt time']).dt.weekday
data = data[data['weekday'] != 6]

# Entferne die 'Volume'- und 'weekday'-Spalten
data = data.drop(columns=['Volume', 'weekday'])

# Benenne die Spalten um
data.columns = ['date', 'open', 'high', 'low', 'close']

# Ändere das Datumsformat in 'yyyy.mm.dd' zu 'yyyy.mm.dd'
data['date'] = data['date'].apply(lambda x: x.strftime('%Y.%m.%d'))

# Bestimme den neuen Dateinamen, indem "_pre" entfernt wird
new_file_name = os.path.basename(bankold_file_path).replace("_pre", "")
new_file_path = os.path.join(bankold_path, new_file_name)

# Speichere die aktualisierten Daten in einer neuen CSV-Datei
data.to_csv(new_file_path, index=False)

# Fertig
print(f"Verarbeitete Daten wurden in {new_file_path} gespeichert.")
