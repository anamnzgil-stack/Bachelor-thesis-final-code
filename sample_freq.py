# sample_freq.py

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



def analyze_sampling_frequency(csv_path):
    print(f"\n📊 Analizando archivo: {csv_path}")
    df = pd.read_csv(csv_path) #DataFrame de pandas (pandas separa las columnas por las ",")
    
    # Extraer timestamps
    timestamps = df.iloc[:, 0].astype(float).values 
    # iloc = index location (seleccion por posicion), y astype(float) convierte los valores en float
    # .values: Convierte la columna de pandas en un array de NumPy
    
    # Calcular diferencias entre muestras
    time_diffs = np.diff(timestamps) # Calcula la fierenecia entre cada elemento consecutivo
    
    print("\n📈 Estadísticas de diferencias entre muestras:")
    print(f"  ➤ Media   : {np.mean(time_diffs):.4f} s")
    print(f"  ➤ Mediana : {np.median(time_diffs):.4f} s")
    print(f"  ➤ Mín     : {np.min(time_diffs):.4f} s")
    print(f"  ➤ Máx     : {np.max(time_diffs):.4f} s")
    
    # Calcular frecuencia por segundo
    segundos = np.floor(timestamps).astype(int) #Pasamos los valores a int redondeando hacia abajo
    frecuencia_por_segundo = pd.Series(segundos).value_counts().sort_index()
    # Convertimos el array en una serie de Pandas para poder usar funciones estadisticas
    # value_counts() : Cuenta cuantas veces aparece cada segundo
    # sort_index() : Ordena la cantidad de segundos cronologicamente

    print(f"\n📉 Frecuencia media por segundo: {frecuencia_por_segundo.mean():.2f} Hz")
    print(f"  ➤ Mínimo: {frecuencia_por_segundo.min()} Hz")
    print(f"  ➤ Máximo: {frecuencia_por_segundo.max()} Hz")
    
    # Graficar histograma de diferencias
    plt.figure(figsize=(10, 4))
    plt.hist(time_diffs, bins=100, color='skyblue', edgecolor='k')
    plt.title("Distribución de diferencias entre muestras")
    plt.xlabel("Diferencia entre timestamps (s)")
    plt.ylabel("Frecuencia")
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
    plt.close() 
    # Graficar frecuencia por segundo
    plt.figure(figsize=(12, 4))
    plt.plot(frecuencia_por_segundo.index, frecuencia_por_segundo.values, marker='o', color='green')
    plt.title("Frecuencia de muestreo por segundo")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Muestras por segundo (Hz)")
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
    plt.close()
    return time_diffs, frecuencia_por_segundo


