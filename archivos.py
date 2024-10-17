from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
from typing import List, Dict, Any

app = FastAPI()

# Configuración de CORS
origins = [
    "http://localhost:5173",  # Origen de tu frontend
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Crear un directorio para almacenar los archivos si no existe
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Ruta para subir archivos
@app.post("/upload/")
async def upload_file(files: List[UploadFile] = File(...)):
    file_names = []
    
    for file in files:
        if not (file.filename.endswith(".csv") or file.filename.endswith(".xlsx")):
            raise HTTPException(status_code=400, detail="Solo se aceptan archivos CSV o Excel")

        # Guardar el archivo en el sistema de archivos
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())  # Leer y escribir el archivo en el disco

        file_names.append(file.filename)  # Agregar el nombre del archivo a la lista

    return {
        "message": "Archivo(s) procesado(s) y guardado(s) correctamente",
        "files": file_names
    }

# Ruta para obtener datos de un archivo por su nombre
@app.get("/file-data/{file_name}")
async def get_file_data(file_name: str) -> Dict[str, Any]:
    file_location = os.path.join(UPLOAD_DIR, file_name)

    # Verificar si el archivo existe
    if not os.path.exists(file_location):
        raise HTTPException(status_code=404, detail="Archivo no encontrado")

    # Leer el archivo usando pandas para obtener encabezados y datos
    if file_name.endswith(".csv"):
        df = pd.read_csv(file_location)
    elif file_name.endswith(".xlsx"):
        df = pd.read_excel(file_location)

    # Obtener los nombres de los encabezados
    headers = df.columns.tolist()  # Obtener los nombres de las columnas
    data = df.to_dict(orient='records')  # Convertir el DataFrame a un diccionario

    return {
        "headers": headers,
        "data": data
    }

# Ruta para listar archivos guardados
@app.get("/files/")
async def list_files():
    try:
        files = os.listdir(UPLOAD_DIR)  # Listar archivos en el directorio
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

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


# Se corre usando el comando uvicorn archivos:app --reload