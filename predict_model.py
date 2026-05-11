import sqlite3
import pandas as pd
from prophet import Prophet
import os

conn = sqlite3.connect("data/meteo_archive_2010_2026.db")
query = """
    SELECT date, temperature_2m 
    FROM hourly_weather 
    WHERE date IS NOT NULL
"""
df = pd.read_sql_query(query, conn)
df['date'] = pd.to_datetime(df['date'])

df_daily = df.set_index('date').resample('D').agg({
    'temperature_2m': ['max', 'min', 'mean']
}).reset_index()

df_daily.columns = ['date', 'temp_max', 'temp_min', 'temp_mean']

df_prophet = df_daily[['date', 'temp_max']].rename(columns={'date': 'ds', 'temp_max': 'y'})

print("Entraînement du modèle Prophet en cours...")
model = Prophet(yearly_seasonality=True, daily_seasonality=False)
model.fit(df_prophet)

future = model.make_future_dataframe(periods=100)
forecast = model.predict(future)

predictions = forecast.tail(90)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
predictions = predictions.rename(columns={
    'ds': 'date',
    'yhat': 'tempMax',
    'yhat_lower': 'tempMin',
    'yhat_upper': 'tempMax_pire_cas'
})

df_daily['day_month'] = df_daily['date'].dt.strftime('%m-%d')
normales = df_daily.groupby('day_month')['temp_max'].mean().to_dict()

predictions['normale'] = predictions['date'].dt.strftime('%m-%d').map(normales)
predictions['seuilAlerte'] = 35

predictions['date'] = predictions['date'].dt.strftime('%Y-%m-%d')

chemin_react_data = r"C:\Users\quiat\WebstormProjects\tableau-de-bord-meteo\src\data"
os.makedirs(chemin_react_data, exist_ok=True)


fichier_export = os.path.join(chemin_react_data, "forecast.json")
predictions.to_json(fichier_export, orient="records")

print(f"Export terminé avec succès vers {fichier_export} !")