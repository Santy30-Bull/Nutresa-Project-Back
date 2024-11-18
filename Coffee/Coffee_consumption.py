# Imports necesarios
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from utils import series_to_supervised  # Importamos la función del archivo utils.py
import os
import numpy as np
from numpy import asarray  # Para manejar arreglos en predicciones
import plotly.express as px  # Para gráficos interactivos

# Configuración warnings
import warnings
warnings.filterwarnings('once')

pd.set_option('display.float_format', '{:.3f}'.format)

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=ImportWarning)

# Obtener la ruta base del script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Carpeta donde está el script actual
DATA_BANK_PATH = os.path.join(BASE_DIR, '..', 'DataBank', 'materia_prima.xlsx')
PREDICTIONS_PATH = os.path.join(BASE_DIR, '..', 'Predictions', 'coffee_prediction.xlsx')

# Cargar los datos
materia_prima = pd.read_excel(DATA_BANK_PATH, decimal=',', header=0, index_col=0)
materia_prima_reset = materia_prima.reset_index()
coffee = materia_prima_reset[['Date', 'cantidad_cafe']].set_index('Date')
data16 = coffee.values

x1 = coffee[['cantidad_cafe']].to_numpy().flatten() #datasets y la columna a predecir

n_step =2
n_obser = 27

x_transform = series_to_supervised(x1, n_obser, 1)
train_data, test_data = x_transform[:-n_step], x_transform[-n_step:]

x_train_no_scaler, y_train_no_scaler = train_data[:,:-1], train_data[:,-1]
x_test_no_scaler, y_test_no_scaler = test_data[:,:-1], test_data[:,-1]

GBR = GradientBoostingRegressor(
              n_estimators = 100,
              loss         = 'squared_error',
              max_features = None,
              random_state = 123
          )

GBR.fit(x_train_no_scaler, y_train_no_scaler)

y_pred_14 = GBR.predict(x_test_no_scaler)

MSE = mean_squared_error(y_test_no_scaler, y_pred_14)
RMSE = MSE**.5
MAE = mean_absolute_error(y_test_no_scaler, y_pred_14)
R2 = r2_score(y_test_no_scaler, y_pred_14)

print(f'MSE: {MSE}')
print(f'RMSE: {RMSE}')
print(f'MAE: {MAE}')
print(f'R2: {R2}')

# plt.figure(figsize=(10, 5))
# plt.plot(y_test_no_scaler, label='test_y',linewidth=1)
# plt.plot(y_pred_14, label='predictions',linewidth=1)
# plt.legend(loc='upper left')
# plt.show()

def prediction_ml(start_date, end_date, freq_date, n_steps):
    forecast_dates = pd.date_range(start=start_date, end=end_date, freq=freq_date)
    data_pred_ml = pd.DataFrame(data16.copy()).reset_index(drop=True) # restablecerá los índices para evitar cualquier problema con índices duplicados
    data_predicted_ml = pd.DataFrame(columns=['Date', 'cantidad_cafe']) #Actualizar el nombre de la columna a pronosticar
    predicted_rows = []  # Lista para almacenar los diccionarios de las filas

    for i in range(len(forecast_dates)):
        x_input_ml = data_pred_ml[-n_steps:].values.flatten()
        x_input = x_input_ml.reshape((1, n_steps))
        yhat_ml = GBR.predict(asarray([x_input_ml])) #Actualizar el nombre del modelo
        data_pred_ml = pd.concat([data_pred_ml, pd.DataFrame(yhat_ml)], ignore_index=True)
        predicted_rows.append({'Date': forecast_dates[i].strftime("%Y-%m-%d"), 'Predict': yhat_ml[0]}) #Actualizar el nombre de la columna a pronosticar

    data_predicted_ml = pd.DataFrame(predicted_rows)
    return data_predicted_ml

data_predicted_ml = prediction_ml('2024-01-31','2030-12-31','ME',27)
test_ml = coffee.copy() #actualizar el nombre del dataset

data_predicted_ml.to_excel(PREDICTIONS_PATH)

# concat_df = pd.concat([coffee,data_predicted_ml], axis=0)
# concat_df = concat_df.reset_index()
# concat_df.drop(['index'],axis=1)

# # Establecemos el índice del DataFrame con una serie de fechas
# concat_df_i = pd.date_range(start='2012-01-31', end='2030-12-31', freq='ME')
# concat_df.set_index(concat_df_i, inplace=True)

# fig = px.line(concat_df, x=concat_df.index, y=["cantidad_cafe", "Predict"], template='plotly_white')

# # Actualizamos los nombres de los ejes usando update_layout
# fig.update_layout(
#     xaxis_title="Date",  # Cambiar nombre al eje X
#     yaxis_title="Values"  # Cambiar nombre al eje Y
# )
# fig.show()