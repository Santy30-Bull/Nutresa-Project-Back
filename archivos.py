from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import os, re
import json
from typing import Dict
import re
from pathlib import Path
from pydantic import BaseModel

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
USER_DIR = "users"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(USER_DIR, exist_ok=True)

# Archivo JSON para almacenar usuarios en la carpeta users
USERS_FILE = os.path.join(USER_DIR, "users.json")
Path(USERS_FILE).touch(exist_ok=True)  # Crear el archivo si no existe

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

    # Guardar la contraseña en texto claro sin encriptación
    users[credentials.username] = credentials.password

    save_users(users)
    return {"message": "Usuario registrado con éxito"}

# Ruta para iniciar sesión
@app.post("/login")
async def login_user(credentials: HTTPBasicCredentials = Depends(HTTPBasic())):
    users = load_users()

    # Verificar si el usuario existe y comparar la contraseña en texto claro
    if credentials.username not in users or credentials.password != users[credentials.username]:
        raise HTTPException(status_code=401, detail="Credenciales incorrectas")

    return {"message": f"Bienvenido, {credentials.username}!"}

# Ruta para obtener o crear un usuario (esto podría no ser necesario)
@app.get("/users/{username}")
async def get_or_create_user(username: str):
    users = load_users()

    # Si el usuario existe, lo devolvemos
    if username in users:
        return {"message": f"Usuario '{username}' encontrado", "username": username}

    # Si no existe, lo creamos con una contraseña predeterminada
    default_password = "password123"
    users[username] = default_password
    save_users(users)

    return {"message": f"Usuario '{username}' creado automáticamente", "username": username, "password": default_password}


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