import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pickle
import plotly.express as px
import os

# Función para añadir características de estacionalidad y cantidad de café
def add_seasonality_features(df):
    """ Añadir características de estacionalidad (seno y coseno de los meses) y la cantidad de café """
    df['Mes'] = df.index.month
    df['Sin_Mes'] = np.sin(2 * np.pi * df['Mes'] / 12)
    df['Cos_Mes'] = np.cos(2 * np.pi * df['Mes'] / 12)
    return df

# Función para dividir el conjunto de datos en entrenamiento y prueba
def split_data(df, train_size=120, test_size=48):
    """ Dividir el conjunto de datos en entrenamiento y prueba """
    train = df[['Time', 'Precio_cafe', 'cantidad_cafe', 'Sin_Mes', 'Cos_Mes']][:-test_size]
    test = df[['Time', 'Precio_cafe', 'cantidad_cafe', 'Sin_Mes', 'Cos_Mes']][-test_size:]
    return train, test

# Función para entrenar el modelo de regresión lineal
def train_linear_model(train_data):
    """ Entrenar el modelo de regresión lineal con los datos de entrenamiento """
    X_train = train_data[['Time', 'cantidad_cafe', 'Sin_Mes', 'Cos_Mes']]  # Incluimos 'cantidad_cafe'
    y_train = train_data['Precio_cafe']
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Función para realizar predicciones múltiples
def predict_multiple(model, start_time, n_steps):
    """ Predecir múltiples pasos de tiempo utilizando el modelo entrenado """
    future_times = np.arange(start_time, start_time + n_steps)
    sin_vals = np.sin(2 * np.pi * (future_times % 12 + 1) / 12)
    cos_vals = np.cos(2 * np.pi * (future_times % 12 + 1) / 12)
    
    # Necesitamos usar la columna cantidad_cafe en el futuro, pero supongamos que sigue una tendencia
    future_cantidad_cafe = np.linspace(df['cantidad_cafe'].iloc[-1], df['cantidad_cafe'].iloc[-1], n_steps)  # Suponemos que no cambia mucho
    
    future_data = pd.DataFrame({
        'Time': future_times,
        'cantidad_cafe': future_cantidad_cafe,
        'Sin_Mes': sin_vals,
        'Cos_Mes': cos_vals
    })
    
    predictions = model.predict(future_data)
    future_data['Predicted_Price'] = predictions
    return future_data

# Función para generar fechas futuras para las predicciones
def generate_forecast_dates(start_date, n_steps, freq='M'):
    """ Generar las fechas de predicción futuras """
    forecast_dates = pd.date_range(start=start_date, periods=n_steps, freq=freq)
    return forecast_dates

# Función para guardar el modelo
def save_model(model, filename):
    """ Guardar el modelo usando pickle """
    try:
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"Modelo guardado exitosamente en {filename}")
    except Exception as e:
        print(f"Error al guardar el modelo: {e}")

# Función para cargar el modelo
def load_model(filename):
    """ Cargar el modelo desde un archivo pickle """
    try:
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print(f"Modelo cargado exitosamente desde {filename}")
        return model
    except FileNotFoundError:
        print(f"El archivo {filename} no existe.")
        return None
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None

# Clase principal para el pronóstico del precio del café
class CafePricePredictor:
    def __init__(self, df):
        self.df = df
        self.model = None
    
    def preprocess_data(self):
        """ Preprocesar y limpiar los datos """
        self.df = add_seasonality_features(self.df)
    
    def train_model(self):
        """ Entrenar el modelo de regresión lineal """
        train, _ = split_data(self.df)
        self.model = train_linear_model(train)
    
    def predict(self, start_date, n_steps):
        """ Realizar predicciones para un período futuro """
        forecast_dates = generate_forecast_dates(start_date, n_steps)
        predictions = predict_multiple(self.model, self.df['Time'].iloc[-1] + 1, n_steps)
        return predictions

    def save_model(self, filename):
        save_model(self.model, filename)
    
    def load_model(self, filename):
        self.model = load_model(filename)

# Cargar el dataset .xlsx
df = pd.read_excel("uploads/materia_prima.xlsx")

# Asegúrate de que la columna 'Date' esté en formato de fecha
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Limpieza de datos
df['Precio_cafe'] = df['Precio_cafe'].ffill()  # Rellenar valores nulos con el último valor disponible
df['cantidad_cafe'] = df['cantidad_cafe'].ffill()  # Asegurarse de que 'cantidad_cafe' no tenga valores nulos
df = df[~df.index.duplicated()]  # Eliminar duplicados de fecha

# Eliminación de datos atípicos
q_low = df["Precio_cafe"].quantile(0.01)
q_high = df["Precio_cafe"].quantile(0.99)
df = df[(df["Precio_cafe"] > q_low) & (df["Precio_cafe"] < q_high)]

# Crear la columna 'Time' como una representación del tiempo en meses
df['Time'] = np.arange(len(df))  # Representación del tiempo en meses

# Instanciar la clase CafePricePredictor
predictor = CafePricePredictor(df)
predictor.preprocess_data()

# Entrenar el modelo
predictor.train_model()

# Generar predicciones futuras (por ejemplo, 6 años)
future_predictions = predictor.predict(start_date='2024-01-31', n_steps=72)

# Concatenar los datos reales con las predicciones
data_predicted_ml = pd.DataFrame({
    'Date': pd.date_range(start='2024-01-31', periods=72, freq='M'),
    'Predict': future_predictions['Predicted_Price']
})
data_predicted_ml.set_index('Date', inplace=True)

# Concatenar datos reales y pronosticados
concat_df = pd.concat([df[['Precio_cafe']], data_predicted_ml], axis=1)

# Asegurar que el índice tenga la frecuencia correcta
concat_df = concat_df.asfreq('M')

# Visualización de datos reales y pronosticados con plotly
fig = px.line(concat_df, x=concat_df.index, y=["Precio_cafe", "Predict"], template='plotly_white')
fig.update_layout(title="Evolución del Precio del Café (Real y Predicción)",
                  xaxis_title="Fecha", yaxis_title="Precio del Café")
fig.show()

# Crear el folder 'modelos' si no existe
os.makedirs('modelos', exist_ok=True)

# Guardar el modelo dentro del folder 'modelos'
predictor.save_model('modelos/Precio_cafe.pkl')