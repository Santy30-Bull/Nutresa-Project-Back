Write-Host "Instalando dependencias..."

pip install --upgrade numpy
pip install --upgrade pandas
pip install --upgrade seaborn matplotlib plotly
pip install catboost
pip install xgboost
pip install lightgbm
pip install -U scikit-learn
pip install prophet
pip install openpyxl
pip install fastapi
pip install uvicorn
pip install kaleido==0.1.0post1 
pip install bcrypt
pip install python-dotenv

Write-Host "Instalación completada."
