import openmeteo_requests
import pandas as pd
import requests_cache
import sqlite3
from retry_requests import retry
from datetime import datetime, timedelta

# Configuration du client API
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

LATITUDE = 48.8534
LONGITUDE = 2.3488
DB_NAME = "data/meteo_archive_2010_2026.db"

end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=10 * 365)).strftime('%Y-%m-%d')

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": start_date,
    "end_date": end_date,
    "hourly": [
        "temperature_2m",
        "apparent_temperature",
        "weather_code",
        "wind_speed_10m",
        "rain",
        "precipitation",
        "relative_humidity_2m",
        "cloud_cover",
        "surface_pressure",
        "dew_point_2m",
        "shortwave_radiation",
    ],
    "timezone": "Europe/Berlin"
}

print("Récupération des données API...")
responses = openmeteo.weather_api(url, params=params)
hourly = responses[0].Hourly()

hourly_data = {"date": pd.date_range(
    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
    freq=pd.Timedelta(seconds=hourly.Interval()),
    inclusive="left"
)}
for i, var_name in enumerate(params["hourly"]):
    hourly_data[var_name] = hourly.Variables(i).ValuesAsNumpy()

df_hourly = pd.DataFrame(data=hourly_data)
df_hourly['date'] = df_hourly['date'].dt.strftime('%Y-%m-%d %H:%M:%S')

print(f"Connexion à {DB_NAME}...")
conn = sqlite3.connect(DB_NAME)

# Vérifier quelles colonnes existent dans la table
try:
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(hourly_weather)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    print(f"Colonnes présentes dans la DB : {len(existing_columns)}")

    df_to_insert = df_hourly[[col for col in df_hourly.columns if col in existing_columns]]

    ignored_cols = set(df_hourly.columns) - existing_columns
    if ignored_cols:
        print(f"Colonnes ignorées (absentes de la DB) : {ignored_cols}")
        print("Lance 'python migrate_db.py' pour ajouter ces colonnes.")

    existing_dates = pd.read_sql('SELECT date FROM hourly_weather', conn)['date']
    df_to_insert = df_to_insert[~df_to_insert['date'].isin(existing_dates)]
    print(f"{len(df_to_insert)} nouvelles lignes à insérer.")

except sqlite3.OperationalError:
    print("La table n'existe pas encore, création initiale.")
    df_to_insert = df_hourly

df_to_insert.to_sql('hourly_weather', conn, if_exists='append', index=False)
print("Base de données à jour !")