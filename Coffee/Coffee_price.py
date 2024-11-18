# Imports necesarios
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils import series_to_supervised  # Importamos la función del archivo utils.py
import numpy as np
from numpy import asarray  # Para manejar arreglos en predicciones
import plotly.express as px  # Para gráficos interactivos
import os

# Configuración warnings
import warnings
warnings.filterwarnings('once')

pd.set_option('display.float_format', '{:.3f}'.format)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=ImportWarning)

# Obtener la ruta base del script (directorio donde está Coffee_price.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Carpeta donde está el script actual

# Definir la ruta absoluta para los archivos
DATA_BANK_PATH = os.path.join(BASE_DIR, '..', 'DataBank', 'materia_prima.xlsx')
PREDICTIONS_PATH = os.path.join(BASE_DIR, '..', 'Predictions', 'coffee_price_prediction.xlsx')

print(os.access(DATA_BANK_PATH, os.R_OK))  # Verifica si se puede leer
print(os.access(PREDICTIONS_PATH, os.W_OK))  # Verifica si se puede escribir

# Cargar los datos
materia_prima = pd.read_excel(DATA_BANK_PATH, decimal=',', header=0, index_col=0)
materia_prima_reset = materia_prima.reset_index()
coffee_price = materia_prima_reset[['Date', 'Precio_cafe']].set_index('Date')
data9 = coffee_price.values

# Preprocesar datos
x1 = coffee_price[['Precio_cafe']].to_numpy().flatten()
n_step = 40
x_transform = series_to_supervised(x1, 5, 1)
train_data, test_data = x_transform[:-n_step], x_transform[-n_step:]
x_train_no_scaler, y_train_no_scaler = train_data[:, :-1], train_data[:, -1]
x_test_no_scaler, y_test_no_scaler = test_data[:, :-1], test_data[:, -1]

# Modelo de regresión
LR = LinearRegression()
LR.fit(x_train_no_scaler, y_train_no_scaler)
y_pred_4 = LR.predict(x_test_no_scaler)

# Métricas
MSE = mean_squared_error(y_test_no_scaler, y_pred_4)
RMSE = MSE ** 0.5
MAE = mean_absolute_error(y_test_no_scaler, y_pred_4)
R2 = r2_score(y_test_no_scaler, y_pred_4)

print(f'MSE: {MSE}')
print(f'RMSE: {RMSE}')
print(f'MAE: {MAE}')
print(f'R2: {R2}')

#Gráficos
# plt.figure(figsize=(10, 5))
# plt.plot(y_test_no_scaler, label='Actual', linewidth=1)
# plt.plot(y_pred_4, label='Predicted', linewidth=1)
# plt.legend(loc='upper left')
# plt.show()

# Función para realizar la predicción
def prediction_ml(start_date, end_date, freq_date, n_steps):
    forecast_dates = pd.date_range(start=start_date, end=end_date, freq=freq_date)
    data_pred_ml = pd.DataFrame(data9.copy()).reset_index(drop=True)  # restablecerá los índices para evitar cualquier problema con índices duplicados
    data_predicted_ml = pd.DataFrame(columns=['Date', 'Precio_cafe'])  # Actualizar el nombre de la columna a pronosticar
    predicted_rows = []  # Lista para almacenar los diccionarios de las filas

    for i in range(len(forecast_dates)):
        x_input_ml = data_pred_ml[-n_steps:].values.flatten()
        x_input = x_input_ml.reshape((1, n_steps))
        yhat_ml = LR.predict(asarray([x_input_ml]))  # Actualizar el nombre del modelo
        data_pred_ml = pd.concat([data_pred_ml, pd.DataFrame(yhat_ml)], ignore_index=True)
        predicted_rows.append({'Date': forecast_dates[i].strftime("%Y-%m-%d"), 'Predict': yhat_ml[0]})  # Actualizar el nombre de la columna a pronosticar

    data_predicted_ml = pd.DataFrame(predicted_rows)
    return data_predicted_ml

# Realizar la predicción
data_predicted_ml = prediction_ml('2024-01-31', '2030-12-31', 'ME', 5)
test_ml = coffee_price.copy()  # Actualizar el nombre del dataset
# print(data_predicted_ml)

# Guardar el resultado de la predicción en un archivo Excel
data_predicted_ml.to_excel(PREDICTIONS_PATH)

# # Concatenar los datos reales con los datos predichos
# concat_df = pd.concat([coffee_price, data_predicted_ml], axis=0)
# concat_df = concat_df.reset_index()
# concat_df.drop(['index'], axis=1)

# # Establecer el índice con una serie de fechas
# concat_df_i = pd.date_range(start='2012-01-31', end='2030-12-31', freq='ME')
# concat_df.set_index(concat_df_i, inplace=True)

# # Crear gráfico interactivo
# fig = px.line(concat_df, x=concat_df.index, y=["Precio_cafe", "Predict"], template='plotly_white')

# # Actualizar los nombres de los ejes usando update_layout
# fig.update_layout(
#     xaxis_title="Date",  # Cambiar nombre al eje X
#     yaxis_title="Values"  # Cambiar nombre al eje Y
# )
# fig.show()
