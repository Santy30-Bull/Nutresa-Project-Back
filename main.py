from fastapi import FastAPI, HTTPException, Depends, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import HTMLResponse
import subprocess
import os
from dotenv import load_dotenv
import re
import json
import pandas as pd
import plotly.express as px
import bcrypt
from typing import Dict, Any, List
from pathlib import Path
from pydantic import BaseModel

app = FastAPI()

# Cargar variables del archivo .env
load_dotenv()

# Configuración de CORS
origins = [
    os.getenv("ORIGIN")  # Origen del frontend
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Crear un directorio para almacenar los archivos si no existe
UPLOAD_DIR = "DataBank"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Directores y archivos
USER_DIR = "users"
os.makedirs(USER_DIR, exist_ok=True)

USERS_FILE = os.path.join(USER_DIR, "users.json")
if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, 'w') as f:
        json.dump({}, f)  # Si no existe, crear el archivo

class UserCredentials(BaseModel):
    username: str
    password: str

# Función para cargar usuarios desde el archivo JSON
def load_users() -> Dict[str, str]:
    with open(USERS_FILE, "r") as file:
        try:
            users = json.load(file)
        except json.JSONDecodeError:
            users = {}
    return users

# Función para guardar usuarios en el archivo JSON
def save_users(users: Dict[str, str]):
    with open(USERS_FILE, "w") as file:
        json.dump(users, file)

# Validar formato de correo electrónico
def validate_email(email: str) -> bool:
    regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(regex, email) is not None

# Validar contraseña (mínimo 6 caracteres, al menos un número y un carácter especial)
def validate_password(password: str) -> bool:
    regex = r'^(?=.*\d)(?=.*[!@#$%^&*])(?=.*[A-Za-z]).{6,}$'
    return re.match(regex, password) is not None

# Función para encriptar una contraseña
def hash_password(password: str) -> str:
    # Generar un salt y usarlo para encriptar la contraseña
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed_password.decode('utf-8')

# Función para verificar una contraseña
def verify_password(stored_password: str, input_password: str) -> bool:
    # Verificar si la contraseña ingresada coincide con la almacenada
    return bcrypt.checkpw(input_password.encode('utf-8'), stored_password.encode('utf-8'))

# Ruta para registrar un nuevo usuario
@app.post("/register")
async def register(credentials: UserCredentials):
    # Validar formato de email y contraseña
    if not validate_email(credentials.username):
        raise HTTPException(status_code=400, detail="Correo electrónico inválido")
    if not validate_password(credentials.password):
        raise HTTPException(status_code=400, detail="La contraseña debe tener al menos 6 caracteres, un número y un carácter especial.")

    users = load_users()

    if credentials.username in users:
        raise HTTPException(status_code=400, detail="El usuario ya existe")

    # Encriptar la contraseña antes de guardarla
    hashed_password = hash_password(credentials.password)

    # Guardar la contraseña encriptada
    users[credentials.username] = hashed_password

    save_users(users)
    return {"message": "Usuario registrado con éxito"}

# Ruta para iniciar sesión
@app.post("/login")
async def login_user(credentials: UserCredentials):
    users = load_users()

    # Verificar si el usuario existe
    if credentials.username not in users:
        raise HTTPException(status_code=401, detail="Usuario no encontrado")

    # Verificar la contraseña encriptada
    stored_password = users[credentials.username]
    if not verify_password(stored_password, credentials.password):
        raise HTTPException(status_code=401, detail="Contraseña incorrecta")

    return {"message": f"Bienvenido, {credentials.username}!"}




# Parte de carga, eliminación y listado de archivos ------------------------------------------------------------------------------------------------------------------------------------    




# Ruta para subir archivos
@app.post("/upload/")
async def upload_file(files: List[UploadFile] = File(...)):
    file_names = []

    # Verificar que los archivos sean CSV o Excel
    for file in files:
        if not (file.filename.endswith(".csv") or file.filename.endswith(".xlsx")):
            raise HTTPException(status_code=400, detail="Solo se aceptan archivos CSV o Excel")

        # Guardar el archivo en el sistema de archivos
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        try:
            with open(file_location, "wb") as buffer:
                buffer.write(await file.read())  # Leer y escribir el archivo en el disco
            file_names.append(file.filename)  # Agregar el nombre del archivo a la lista
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error al guardar el archivo: {str(e)}")

    return {
        "message": "Archivo(s) procesado(s) y guardado(s) correctamente",
        "files": file_names
    }

# Ruta para listar archivos guardados
@app.get("/files/")
async def list_files():
    try:
        files = os.listdir(UPLOAD_DIR)  # Listar archivos en el directorio
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# Ruta para obtener datos de un archivo por su nombre
@app.get("/file-data/{file_name}")
async def get_file_data(file_name: str) -> Dict[str, Any]:
    file_location = os.path.join(UPLOAD_DIR, file_name)

    # Verificar si el archivo existe
    if not os.path.exists(file_location):
        raise HTTPException(status_code=404, detail="Archivo no encontrado")

    # Leer el archivo usando pandas para obtener encabezados y datos
    try:
        if file_name.endswith(".csv"):
            df = pd.read_csv(file_location)
        elif file_name.endswith(".xlsx"):
            df = pd.read_excel(file_location)
        else:
            raise HTTPException(status_code=400, detail="Formato de archivo no soportado")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al leer el archivo: {str(e)}")

    # Obtener los nombres de los encabezados
    headers = df.columns.tolist()  # Obtener los nombres de las columnas
    data = df.to_dict(orient='records')  # Convertir el DataFrame a un diccionario

    return {
        "headers": headers,
        "data": data
    }

# Ruta para eliminar archivos por nombre
@app.delete("/files/{file_name}")
async def delete_file(file_name: str):
    file_location = os.path.join(UPLOAD_DIR, file_name)
    
    # Verificar si el archivo existe
    if not os.path.exists(file_location):
        raise HTTPException(status_code=404, detail="Archivo no encontrado")

    # Eliminar el archivo
    os.remove(file_location)
    return {"message": f"Archivo '{file_name}' eliminado correctamente"}


# Ruta para eliminar todos los archivos
@app.delete("/files/")
async def delete_all_files():
        try:
            files = os.listdir(UPLOAD_DIR)  # Listar archivos en el directorio
            for file in files:
                file_location = os.path.join(UPLOAD_DIR, file)
                os.remove(file_location)  # Eliminar cada archivo
            return {"message": "Todos los archivos han sido eliminados correctamente"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        

# Para los modelos de Coffee

@app.get("/coffee_price_plot", response_class=HTMLResponse)
async def coffee_price_plot():
    try:
        # Ejecutar el script de Python Coffee_price.py para generar el archivo .xlsx
        print("Ejecutando el script Coffee_price.py...")
        result = subprocess.run(["python", "Coffee/Coffee_price.py"], check=True, capture_output=True)
        print(f"Resultado de la ejecución: {result.stdout.decode()}")
        
        # Verificar si hubo un error en la ejecución
        if result.returncode != 0:
            raise Exception(f"Error en la ejecución de Coffee_price.py: {result.stderr.decode()}")

        # Leer el archivo generado por el script
        print("Leyendo el archivo de predicción generado...")
        prediction_ml_1 = pd.read_excel('./Predictions/coffee_price_prediction.xlsx')

        # Eliminar cualquier columna no útil como 'Unnamed'
        prediction_ml_1 = prediction_ml_1.loc[:, ~prediction_ml_1.columns.str.contains('^Unnamed')]

        if prediction_ml_1.empty:
            raise ValueError("El archivo de predicción está vacío o no se pudo leer correctamente.")
        print(f"Datos de predicción cargados: {prediction_ml_1.head()}")

        # Realizar el renombrado de las columnas
        prediction_ml_1 = prediction_ml_1.rename(columns={'Predict': 'Precio', 'Date': 'Date_2'})
        print(f"Datos de predicción después del renombrado: {prediction_ml_1.head()}")

        # Reemplazar NaN por 0 en las columnas correspondientes
        prediction_ml_1['Precio'] = prediction_ml_1['Precio'].fillna(0)
        prediction_ml_1['Date_2'] = prediction_ml_1['Date_2'].fillna(0)

        # Cargar los datos de materia prima
        print("Cargando datos de materia prima...")
        materia_prima = pd.read_excel('./DataBank/materia_prima.xlsx', decimal=',', header=0, index_col=0)
        materia_prima_reset = materia_prima.reset_index()
        coffee_price = materia_prima_reset[['Date', 'Precio_cafe']].set_index('Date')

        # Reemplazar NaN en los datos de coffee_price
        coffee_price['Precio_cafe'] = coffee_price['Precio_cafe'].fillna(0)
        print(f"Primeras filas de coffee_price: {coffee_price.head()}")

        # Concatenar los datos reales con los predichos
        print("Concatenando los datos reales con los predichos...")
        concat_df = pd.concat([coffee_price, prediction_ml_1], axis=0)
        concat_df = concat_df.reset_index(drop=True)  # Evitar columna extra 'index'
        print(f"Datos concatenados: {concat_df.head()}")
    
        # Establecer el índice como un rango de fechas
        print("Estableciendo el índice de fechas...")
        concat_df_i = pd.date_range(start='2012-01-31', end='2030-12-31', freq='ME')
        concat_df.set_index(concat_df_i, inplace=True)
        print(f"Datos con índice de fechas head: {concat_df.head()}")
        print(f"Datos con índice de fechas tail: {concat_df.tail()}")

        # Crear el gráfico interactivo con Plotly
        print("Generando gráfico interactivo con Plotly...")
        fig = px.line(concat_df, x=concat_df.index, y=["Precio_cafe", "Precio"], template='plotly_white')
        
        # Actualizar los nombres de los ejes
        fig.update_layout(
            xaxis_title="Date",  # Cambiar nombre al eje X
            yaxis_title="Values"  # Cambiar nombre al eje Y
        )

        # Convertir el gráfico a HTML interactivo
        print("Generando HTML interactivo del gráfico...")
        plot_html = fig.to_html(full_html=False)  # Generamos el HTML sin la envoltura completa

        # Devolver el gráfico interactivo en el HTML
        return HTMLResponse(content=plot_html)

    except Exception as e:
        print(f"Error: {str(e)}")
        return HTMLResponse(f"<h1>Error: {str(e)}</h1>", status_code=500)

@app.get("/coffee_consumption_plot", response_class=HTMLResponse)
async def coffee_price_plot():
    try:
        # Ejecutar el script de Python Coffee_consumption.py para generar el archivo .xlsx
        print("Ejecutando el script Coffee_consumption.py...")
        result = subprocess.run(["python", "Coffee/Coffee_consumption.py"], check=True, capture_output=True)
        print(f"Resultado de la ejecución: {result.stdout.decode()}")
        
        # Verificar si hubo un error en la ejecución
        if result.returncode != 0:
            raise Exception(f"Error en la ejecución de Coffee_consumption.py: {result.stderr.decode()}")

        # Leer el archivo generado por el script
        print("Leyendo el archivo de predicción generado...")
        prediction_ml_1 = pd.read_excel('./Predictions/coffee_prediction.xlsx')

        # Eliminar cualquier columna no útil como 'Unnamed'
        prediction_ml_1 = prediction_ml_1.loc[:, ~prediction_ml_1.columns.str.contains('^Unnamed')]

        if prediction_ml_1.empty:
            raise ValueError("El archivo de predicción está vacío o no se pudo leer correctamente.")
        print(f"Datos de predicción cargados: {prediction_ml_1.head()}")

        # Cargar los datos de materia prima
        print("Cargando datos de materia prima...")
        materia_prima = pd.read_excel('./DataBank/materia_prima.xlsx', decimal=',', header=0, index_col=0)
        materia_prima_reset = materia_prima.reset_index()
        coffee = materia_prima_reset[['Date', 'cantidad_cafe']].set_index('Date')
        print(f"Primeras filas de coffee_price: {coffee.head()}")

        # Concatenar los datos reales con los predichos
        print("Concatenando los datos reales con los predichos...")
        concat_df = pd.concat([coffee, prediction_ml_1], axis=0)
        concat_df = concat_df.reset_index(drop=True)  # Evitar columna extra 'index'
        print(f"Datos concatenados: {concat_df.head()}")
    
        # Establecer el índice como un rango de fechas
        print("Estableciendo el índice de fechas...")
        concat_df_i = pd.date_range(start='2012-01-31', end='2030-12-31', freq='ME')
        concat_df.set_index(concat_df_i, inplace=True)
        print(f"Datos con índice de fechas head: {concat_df.head()}")
        print(f"Datos con índice de fechas tail: {concat_df.tail()}")

        # Crear el gráfico interactivo con Plotly
        print("Generando gráfico interactivo con Plotly...")
        fig = px.line(concat_df, x=concat_df.index, y=["cantidad_cafe", "Predict"], template='plotly_white')
        
        # Actualizar los nombres de los ejes
        fig.update_layout(
            xaxis_title="Date",  # Cambiar nombre al eje X
            yaxis_title="Values"  # Cambiar nombre al eje Y
        )

        # Convertir el gráfico a HTML interactivo
        print("Generando HTML interactivo del gráfico...")
        plot_html = fig.to_html(full_html=False)  # Generamos el HTML sin la envoltura completa

        # Devolver el gráfico interactivo en el HTML
        return HTMLResponse(content=plot_html)

    except Exception as e:
        print(f"Error: {str(e)}")
        return HTMLResponse(f"<h1>Error: {str(e)}</h1>", status_code=500)

@app.get("/coffee_cost_analysis_plot", response_class=HTMLResponse)
async def coffee_price_plot():
    try:
        # Ejecutar el script de Python Coffee_cost_analysis.py para generar el archivo .xlsx
        print("Ejecutando el script Coffee_cost_analysis.py...")
        result = subprocess.run(["python", "Coffee/Coffee_cost_analysis.py"], check=True, capture_output=True)
        print(f"Resultado de la ejecución: {result.stdout.decode()}")
        
        # Verificar si hubo un error en la ejecución
        if result.returncode != 0:
            raise Exception(f"Error en la ejecución de Coffee_cost_analysis.py: {result.stderr.decode()}")

        # Leer el archivo generado por el script
        print("Leyendo el archivo de predicción generado...")
        prediction_ml_1 = pd.read_excel('./Cost_Analysis/total_cost_coffee.xlsx')

        # Eliminar cualquier columna no útil como 'Unnamed'
        prediction_ml_1 = prediction_ml_1.loc[:, ~prediction_ml_1.columns.str.contains('^Unnamed')]

        if prediction_ml_1.empty:
            raise ValueError("El archivo de predicción está vacío o no se pudo leer correctamente.")
        print(f"Datos de predicción cargados: {prediction_ml_1.head()}")

        # Establecer el índice como un rango de fechas
        print("Estableciendo el índice de fechas...")
        concat_df_i = pd.date_range(start='2012-01-31', end='2030-12-31', freq='ME')
        prediction_ml_1.set_index(concat_df_i, inplace=True)

        # Crear el gráfico interactivo con Plotly
        print("Generando gráfico interactivo con Plotly...")
        fig = px.line(prediction_ml_1, x=prediction_ml_1.index, y=["costo_historico", "predict_cost"], template='plotly_white')
        
        # Actualizar los nombres de los ejes
        fig.update_layout(
            xaxis_title="Date",  # Cambiar nombre al eje X
            yaxis_title="Values"  # Cambiar nombre al eje Y
        )

        # Convertir el gráfico a HTML interactivo
        print("Generando HTML interactivo del gráfico...")
        plot_html = fig.to_html(full_html=False)  # Generamos el HTML sin la envoltura completa

        # Devolver el gráfico interactivo en el HTML
        return HTMLResponse(content=plot_html)

    except Exception as e:
        print(f"Error: {str(e)}")
        return HTMLResponse(f"<h1>Error: {str(e)}</h1>", status_code=500)


# Se corre usando el comando uvicorn main:app --reload