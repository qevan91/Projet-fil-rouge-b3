import sqlite3
import pandas as pd

DB_NAME = "data/meteo_archive_2010_2026.db"

print("Vérification du schéma actuel...")
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

# Récupérer les colonnes actuelles
cursor.execute("PRAGMA table_info(hourly_weather)")
existing_columns = [row[1] for row in cursor.fetchall()]
print(f"Colonnes existantes : {existing_columns}")

# Définir les nouvelles colonnes à ajouter
new_columns = {
    'apparent_temperature': 'REAL',
    'precipitation': 'REAL',
    'surface_pressure': 'REAL',
    'dew_point_2m': 'REAL',
    'shortwave_radiation': 'REAL',
}

# Ajouter les colonnes manquantes
added = []
for col_name, col_type in new_columns.items():
    if col_name not in existing_columns:
        try:
            cursor.execute(f"ALTER TABLE hourly_weather ADD COLUMN {col_name} {col_type}")
            added.append(col_name)
            print(f"Colonne '{col_name}' ajoutée")
        except sqlite3.OperationalError as e:
            print(f"Erreur sur '{col_name}': {e}")
    else:
        print(f"Colonne '{col_name}' déjà présente")

conn.commit()

if added:
    print(f"\nMigration réussie ! {len(added)} colonne(s) ajoutée(s).")
else:
    print("\nSchéma déjà à jour, aucune modification nécessaire.")

# Vérification finale
cursor.execute("PRAGMA table_info(hourly_weather)")
final_columns = [row[1] for row in cursor.fetchall()]
print(f"\nSchéma final : {final_columns}")

conn.close()
print("\nTu peux maintenant relancer FetchMeteoApi.py")
