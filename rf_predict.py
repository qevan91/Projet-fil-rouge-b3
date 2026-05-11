import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
import os

conn = sqlite3.connect("data/meteo_archive_2010_2026.db")
query = "SELECT * FROM hourly_weather WHERE date IS NOT NULL"
df = pd.read_sql_query(query, conn)
df['date'] = pd.to_datetime(df['date'])

agg_dict = {
    'temperature_2m': ['max', 'min', 'mean'],
    'relative_humidity_2m': 'mean',
    'wind_speed_10m': 'mean',
    'cloud_cover': 'mean',
    'rain': 'sum',
}


optional_cols = {
    'apparent_temperature': 'mean',
    'surface_pressure': 'mean',
    'precipitation': 'sum',
    'dew_point_2m': 'mean',
    'shortwave_radiation': 'mean',
}
for col, func in optional_cols.items():
    if col in df.columns:
        agg_dict[col] = func
        print(f"  ✓ Colonne '{col}' détectée et ajoutée.")
    else:
        print(f"  ✗ Colonne '{col}' absente de la DB (voir FetchMeteoApi.py pour l'ajouter).")

df_daily = df.set_index('date').resample('D').agg(agg_dict).reset_index()

df_daily.columns = ['_'.join(filter(None, col)).strip('_') if isinstance(col, tuple) else col
                    for col in df_daily.columns]

rename_map = {
    'temperature_2m_max': 'temp_max',
    'temperature_2m_min': 'temp_min',
    'temperature_2m_mean': 'temp_mean',
    'relative_humidity_2m_mean': 'humidity_mean',
    'wind_speed_10m_mean': 'wind_mean',
    'cloud_cover_mean': 'cloud_mean',
    'rain_sum': 'rain_sum',
    'apparent_temperature_mean': 'apparent_temp_mean',
    'surface_pressure_mean': 'pressure_mean',
    'precipitation_sum': 'precip_sum',
    'dew_point_2m_mean': 'dew_point_mean',
    'shortwave_radiation_mean': 'radiation_mean',
}
df_daily = df_daily.rename(columns={k: v for k, v in rename_map.items() if k in df_daily.columns})

print(f"\nDonnées journalières : {len(df_daily)} jours ({df_daily['date'].min().date()} → {df_daily['date'].max().date()})")

df_daily['target'] = df_daily['temp_max'].shift(-1)

for lag in range(1, 8):
    df_daily[f'temp_max_lag{lag}'] = df_daily['temp_max'].shift(lag)

for lag in range(1, 4):
    df_daily[f'temp_min_lag{lag}'] = df_daily['temp_min'].shift(lag)

df_daily['temp_max_rolling3'] = df_daily['temp_max'].rolling(3).mean()
df_daily['temp_max_rolling7'] = df_daily['temp_max'].rolling(7).mean()
df_daily['temp_min_rolling3'] = df_daily['temp_min'].rolling(3).mean()

df_daily['amplitude_thermique'] = df_daily['temp_max'] - df_daily['temp_min']
df_daily['amplitude_lag1'] = df_daily['amplitude_thermique'].shift(1)

df_daily['day_of_year'] = df_daily['date'].dt.dayofyear
df_daily['month'] = df_daily['date'].dt.month
df_daily['doy_sin'] = np.sin(2 * np.pi * df_daily['day_of_year'] / 365.25)
df_daily['doy_cos'] = np.cos(2 * np.pi * df_daily['day_of_year'] / 365.25)

df_daily['day_month'] = df_daily['date'].dt.strftime('%m-%d')
normales = df_daily.groupby('day_month')['temp_max'].mean().to_dict()
df_daily['normale_climatologique'] = df_daily['day_month'].map(normales)
df_daily['ecart_normale'] = df_daily['temp_max'] - df_daily['normale_climatologique']

df_daily['flag_canicule'] = ((df_daily['temp_max'] > 33) & (df_daily['temp_min'] > 18)).astype(int)
df_daily['jours_canicule_consecutifs'] = (
    df_daily['flag_canicule']
    .groupby((df_daily['flag_canicule'] != df_daily['flag_canicule'].shift()).cumsum())
    .cumcount() + 1
) * df_daily['flag_canicule']

lag_cols = [c for c in df_daily.columns if 'lag' in c or 'rolling' in c or c == 'target']

optional_features = ['apparent_temp_mean', 'pressure_mean', 'precip_sum', 'dew_point_mean', 'radiation_mean']
optional_num_cols = [c for c in optional_features if c in df_daily.columns]

for col in optional_num_cols:
    if df_daily[col].isna().any():
        monthly_median = df_daily.groupby('month')[col].transform('median')
        df_daily[col] = df_daily[col].fillna(monthly_median)
        df_daily[col] = df_daily[col].fillna(df_daily[col].median())
        print(f"  ↳ '{col}' : NaN imputés par médiane mensuelle")

df_daily = df_daily.dropna(subset=lag_cols)
print(f"Lignes après nettoyage : {len(df_daily)} jours")

base_features = [
    'temp_max', 'temp_min', 'temp_mean',
    'temp_max_lag1', 'temp_max_lag2', 'temp_max_lag3',
    'temp_max_lag4', 'temp_max_lag5', 'temp_max_lag6', 'temp_max_lag7',
    'temp_min_lag1', 'temp_min_lag2', 'temp_min_lag3',
    'temp_max_rolling3', 'temp_max_rolling7', 'temp_min_rolling3',
    'amplitude_thermique', 'amplitude_lag1',
    'humidity_mean', 'wind_mean', 'cloud_mean', 'rain_sum',
    'doy_sin', 'doy_cos', 'month',
    'normale_climatologique', 'ecart_normale',
    'flag_canicule', 'jours_canicule_consecutifs',
]

optional_features = ['apparent_temp_mean', 'pressure_mean', 'precip_sum', 'dew_point_mean', 'radiation_mean']
features = base_features + [f for f in optional_features if f in df_daily.columns]

print(f"\nNombre de features utilisées : {len(features)}")

X = df_daily[features]
y = df_daily['target']

split_index = int(len(df_daily) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

print(f"Entraînement : {len(X_train)} jours | Test : {len(X_test)} jours")

print("\nEntraînement de la Random Forest améliorée...")
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=5,
    max_features=0.7,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

predictions_test = rf_model.predict(X_test)
mae  = mean_absolute_error(y_test, predictions_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions_test))

print(f"\n{'='*50}")
print(f"RÉSULTATS SUR LE JEU DE TEST")
print(f"{'='*50}")
print(f"  MAE  (Erreur Absolue Moyenne) : {mae:.2f}°C")
print(f"  RMSE (Erreur Quadratique)     : {rmse:.2f}°C")
print(f"  (Rappel : ton ancienne version avait MAE = 1.94°C)")

importances = pd.DataFrame({
    'Facteur': features,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(f"\nTop 15 des facteurs les plus importants :")
print(importances.head(15).to_string(index=False))

print("\nGénération des prédictions sur 90 jours...")

last_known = df_daily.iloc[-1].copy()
future_predictions = []

for i in range(90):
    row = {}
    row['temp_max']   = last_known['temp_max']
    row['temp_min']   = last_known['temp_min']
    row['temp_mean']  = last_known['temp_mean']

    # Lags
    for lag in range(1, 8):
        col = f'temp_max_lag{lag}'
        if lag == 1:
            row[col] = last_known['temp_max']
        else:
            row[col] = last_known.get(f'temp_max_lag{lag-1}', last_known['temp_max'])

    for lag in range(1, 4):
        col = f'temp_min_lag{lag}'
        if lag == 1:
            row[col] = last_known['temp_min']
        else:
            row[col] = last_known.get(f'temp_min_lag{lag-1}', last_known['temp_min'])

    row['temp_max_rolling3'] = last_known.get('temp_max_rolling3', last_known['temp_max'])
    row['temp_max_rolling7'] = last_known.get('temp_max_rolling7', last_known['temp_max'])
    row['temp_min_rolling3'] = last_known.get('temp_min_rolling3', last_known['temp_min'])

    row['amplitude_thermique'] = last_known['temp_max'] - last_known['temp_min']
    row['amplitude_lag1'] = last_known.get('amplitude_thermique', row['amplitude_thermique'])

    for col in ['humidity_mean', 'wind_mean', 'cloud_mean', 'rain_sum']:
        row[col] = last_known.get(col, 0)

    future_date = df_daily['date'].iloc[-1] + pd.Timedelta(days=i+1)
    doy = future_date.timetuple().tm_yday
    row['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
    row['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)
    row['month'] = future_date.month

    day_month = future_date.strftime('%m-%d')
    row['normale_climatologique'] = normales.get(day_month, last_known['temp_max'])
    row['ecart_normale'] = last_known['temp_max'] - row['normale_climatologique']
    row['flag_canicule'] = last_known.get('flag_canicule', 0)
    row['jours_canicule_consecutifs'] = last_known.get('jours_canicule_consecutifs', 0)

    for col in optional_features:
        if col in df_daily.columns:
            row[col] = last_known.get(col, 0)

    X_future = pd.DataFrame([row])[features]
    pred_temp_max = rf_model.predict(X_future)[0]

    future_predictions.append({
        'date': future_date.strftime('%Y-%m-%d'),
        'tempMax': round(pred_temp_max, 1),
        'tempMin': round(pred_temp_max - row['amplitude_thermique'], 1),
        'normale': round(row['normale_climatologique'], 1),
        'seuilAlerte': 35,
        'ecartNormale': round(pred_temp_max - row['normale_climatologique'], 1),
    })

    last_known = last_known.copy()
    last_known['temp_max'] = pred_temp_max

chemin_react_data = r"C:\Users\quiat\WebstormProjects\tableau-de-bord-meteo\src\data"
os.makedirs(chemin_react_data, exist_ok=True)

fichier_export = os.path.join(chemin_react_data, "forecast_rf.json")
with open(fichier_export, 'w', encoding='utf-8') as f:
    json.dump(future_predictions, f, ensure_ascii=False, indent=2)

print(f"\nExport terminé vers {fichier_export}")
print(f"   {len(future_predictions)} jours de prédictions générés.")