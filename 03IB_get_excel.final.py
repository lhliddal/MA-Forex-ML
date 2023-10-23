#code 3

# Importieren von Bibliotheken und Modulen
import os
import re
import time
import sys
import threading
import configparser
import pandas as pd
import ibapi
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import *
from ibapi.common import BarData

# Code Name
print("code 3")
print("\n")  # Einen Zeilenumbruch einfügen

# Python und Paketversionen anzeigen
print('Python version:{}'.format(sys.version))
print('Pandas version:{}'.format(pd.__version__))
print('ibapi version:{}'.format(ibapi.__version__))
print("\n")  # Einen Zeilenumbruch einfügen


# Lade Konfigurationsdatei
config = configparser.ConfigParser()
current_dir = os.path.dirname(os.path.abspath(__file__))  # Aktuelles Verzeichnis des Skripts
config_path = os.path.join(current_dir, '..', '05config', 'config.ini')
config.read(config_path)

# Lese den Yahoo Pfad aus der Konfigurationsdatei
ibrk_path = config.get('Paths', 'ibrk_path')

class IBApi(EWrapper, EClient):
    def __init__(self, bot_instance):
        EClient.__init__(self, self)
        self.bot_instance = bot_instance  # Speichern Sie die Bot-Instanz

    def historicalData(self, reqId, bar: BarData):
        self.bot_instance.on_historical_data(reqId, bar)  # Verwenden Sie die gespeicherte Bot-Instanz

# Funktion, um den nächsten Dateinamen zu erzeugen
def get_next_filename(directory):
    files = os.listdir(directory)
    version_numbers = []

    # Durchläuft alle Dateien im Verzeichnis
    for file in files:
        if file.startswith("usdchf_") and file.endswith(".csv"):
            version_str = file.rstrip('.csv').split('_')[-1]
            try:
                version_number = float(version_str)
                version_numbers.append(version_number)
            except ValueError:
                continue

    # Bestimmt die nächste Version
    if not version_numbers:
        next_version = 2.00
    else:
        next_version = max(version_numbers) + 0.01

    return f"usdchf_{next_version:.2f}.csv"

class Bot:
    ib = None
    historical_data = []
   
    
    def __init__(self):
        self.ib = IBApi(self)
        self.ib.connect("127.0.0.1", 7496, 1)
        
        ib_thread = threading.Thread(target=self.run_loop, daemon=True)
        ib_thread.start()
        time.sleep(1)
        
        contract = Contract()
        contract.symbol = "USD"
        contract.secType = "CASH"
        contract.exchange = "IDEALPRO"
        contract.currency = "CHF"
        
        self.ib.reqHistoricalData(1, contract, "", "3 Y", "1 day", "BID", 1, 1, False, [])
        
        time.sleep(10)  # Wartezeit erhöht, um sicherzustellen, dass alle Daten heruntergeladen werden
        
        df = pd.DataFrame(self.historical_data)
        
        # Spaltenreihenfolge und -namen ändern
        df = df[['Date', 'Open', 'High', 'Low', 'Close']].rename(columns={'Date':'date', 'Open':'open', 'High':'high', 'Low':'low', 'Close':'close'})

        # Ändert den Datentyp der 'Date'-Spalte in datetime
        df['date'] = pd.to_datetime(df['date'])

        # Es werden nur die Daten seit dem 3.1.22 behalten, da dort die Trainingsdaten aufhören
        df_filtered = df[df['date'] >= '2022-01-03']
        
        # Erstellen einer Kopie des gefilterten DataFrames, um Warnungen zu vermeiden
        df_filtered = df_filtered.copy()

        # Ändert das Format der 'Date'-Spalte im gefilterten DataFrame zurück zu 'YYYY.MM.DD'
        df_filtered['date'] = df_filtered['date'].dt.strftime('%Y.%m.%d')

        # Generiert den vollständigen Pfad für die CSV-Datei
        csv_file_name = get_next_filename(ibrk_path)
        csv_file_path = os.path.join(ibrk_path, csv_file_name)

        print(f"Versuche, CSV-Datei zu speichern unter {csv_file_path}...")
        df_filtered.to_csv(csv_file_path, index=False)  # Speichert die CSV-Datei im angegebenen Pfad
        print("CSV-Datei erfolgreich gespeichert.")


    def run_loop(self):
        print("Run loop started.")  
        self.ib.run()
        
    def on_historical_data(self, reqId, bar: BarData):
        print(f"Received data for {bar.date}: Open {bar.open}, Close {bar.close}, High {bar.high}, Low {bar.low}")
        self.historical_data.append({"Date": bar.date, "Open": bar.open, "Close": bar.close, "High": bar.high, "Low": bar.low})

# Bot starten
if __name__ == "__main__":
    print("Bot starting.")
    bot = Bot()
    print("Bot finished.")  
