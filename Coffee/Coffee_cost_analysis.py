# Imports necesarios
import pandas as pd
import plotly.express as px
import os

# Obtener la ruta base del script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Carpeta donde está el script actual
DATA_BANK_PATH = os.path.join(BASE_DIR, '..', 'DataBank', 'materia_prima.xlsx')
COFFEE_PREDICTIONS_PATH = os.path.join(BASE_DIR, '..', 'Predictions', 'coffee_prediction.xlsx')
COFFEE_PRICE_PREDICTIONS_PATH = os.path.join(BASE_DIR, '..', 'Predictions', 'coffee_price_prediction.xlsx')
COST_ANALYSIS_PATH = os.path.join(BASE_DIR, '..', 'Cost_Analysis', 'total_cost_coffee.xlsx')

# Cargar los datos
materia_prima = pd.read_excel(DATA_BANK_PATH, decimal=',', header=0, index_col=0)
materia_prima_reset = materia_prima.reset_index()
coffee_price = materia_prima_reset[['Date', 'Precio_cafe']].set_index('Date')
coffee = materia_prima_reset[['Date', 'cantidad_cafe']].set_index('Date')

#Cargar el pronóstico del precio de la variable
price = pd.read_excel(COFFEE_PRICE_PREDICTIONS_PATH)
#Renombrar las columnas predict por precio, Date por Date_2
price = price.rename(columns={'Predict':'Precio', 'Date':'Date_2'})
#Cargar el pronóstico del consumo de la variable
predict_coffee = pd.read_excel(COFFEE_PREDICTIONS_PATH)
#Renombrar la columna predict por consumo_predict
predict_coffee = predict_coffee.rename(columns={'Predict':'Consumo_predict'})

#Concatenar los df anteriores, eliminando la columna Date_2, eliminar el índice
cost = pd.concat([predict_coffee, price], axis=1)
cost = cost.drop(['Date_2','Unnamed: 0'], axis=1)

#Crear la columna nueva predict_cost, multiplicando las columnas anteriores
cost['predict_cost'] = cost['Consumo_predict']*cost['Precio']

#Cargar el consumo histórico de la variable
historic_consumption = coffee.reset_index()
#Renombrar la columna del consumo histórico por consumo_histórico
historic_consumption = historic_consumption.rename(columns={'cantidad_cafe':'Consumo_historico'}) #Cambiar el nombre de la columna del df original
#Cargar el precio histórico de la variable
historic_price = coffee_price.reset_index()
#Renombrar las columnas precio por precio histórico, y Date por Date_2
historic_price = historic_price.rename(columns={'Precio_cafe':'Precio_historico', 'Date':'Date_2'}) #Cambiar el nombre de la columna del df original

#Concatenar los df anteriores, eliminando la columna Date_2
cost_2 = pd.concat([historic_consumption, historic_price], axis=1)
cost_2 = cost_2.drop(['Date_2'], axis=1)

#Crear la columna nueva costo_historico, multiplicando las columnas anteriores
cost_2['costo_historico'] = cost_2['Consumo_historico']*cost_2['Precio_historico']

#Concateno los dos costos totales calculados, pronóstico e histórico
concat_df = pd.concat([cost_2, cost], axis=0)
concat_df = concat_df.reset_index()

concat_df.to_excel(COST_ANALYSIS_PATH)

# # Establecemos el índice del DataFrame con una serie de fechas
# concat_df_i = pd.date_range(start='2012-01-31', end='2030-12-31', freq='ME')
# concat_df.set_index(concat_df_i, inplace=True)

# fig = px.line(concat_df, x=concat_df.index, y=["costo_historico", "predict_cost"], template='plotly_white')

# # Actualizamos los nombres de los ejes usando update_layout
# fig.update_layout(
#     xaxis_title="Date",  # Cambiar nombre al eje X
#     yaxis_title="Values"  # Cambiar nombre al eje Y
# )
# fig.show()
