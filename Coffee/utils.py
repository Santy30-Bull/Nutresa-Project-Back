import pandas as pd

# transformar un conjunto de datos de series temporales en un conjunto de datos de aprendizaje supervisado
def series_to_supervised(data, n_in, n_out=1):
    df = pd.DataFrame(data)
    cols = list()
    # serie de entrada: (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # serie de predicción: (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # concatenar datos
    agg = pd.concat(cols, axis=1)
    # quitar valores vacíos
    agg.dropna(inplace=True)
    return agg.values
