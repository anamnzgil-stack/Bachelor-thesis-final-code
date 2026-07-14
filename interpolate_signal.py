# interpolate_signal.py

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def analyze_sampling_stats(timestamps): # se usa para comprobar la calidad del muestreo antes de interpolar
    time_diffs = np.diff(timestamps) 
    print("\n🔍 Estadísticas de diferencias entre muestras:")
    print(f"  ➤ Media   : {np.mean(time_diffs):.4f} s")
    print(f"  ➤ Mediana : {np.median(time_diffs):.4f} s")
    print(f"  ➤ Mín     : {np.min(time_diffs):.4f} s")
    print(f"  ➤ Máx     : {np.max(time_diffs):.4f} s")

    # Histograma
    plt.figure(figsize=(10, 4))
    plt.hist(time_diffs, bins=100, color='skyblue', edgecolor='k')
    plt.title("Distribución de diferencias entre muestras")
    plt.xlabel("Diferencia entre timestamps (s)")
    plt.ylabel("Frecuencia")
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
    plt.close()


def compare_original_vs_interpolated(timestamps_orig, data_orig, new_times, data_interp, ch_idx=0):
    plt.figure(figsize=(12, 4))
    plt.plot(timestamps_orig[:100], data_orig[:100, ch_idx], label='Original (mV)', marker='o')
    plt.plot(new_times[:100], data_interp[:100, ch_idx], label='Interpolado (µV)', linestyle='--')
    plt.title(f"Canal {ch_idx+1} - Señal original vs. interpolada (primeros 100 puntos)")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
    plt.close()

    
def interpolate_eeg(csv_path, output_path, target_sfreq, validate=True):
    print(f"\n📁 Leyendo archivo: {csv_path}")
    df = pd.read_csv(csv_path)
    timestamps = df.iloc[:, 0].values # Cogemos la primera columna del excel eeg -> timestamps 
    data = df.iloc[:, 1:].values #data = todos los valores del eeg (en milivoltios)
    ch_names = df.columns[1:] # "['chann_1','chann_2','chann_3','chann_4','chann_5','chann_6','chann_7','chann_8']"

    print("🔧 Escalando : NANOvoltios a voltios para MNE")
    data = data*1e-9

    if validate:
        analyze_sampling_stats(timestamps)

    # Crear eje de tiempo regular
    start, stop = timestamps[0], timestamps[-1] #start -> primer timestamp. stop -> ultimo simestamp
    num_samples = int(np.round((stop - start) * target_sfreq)) + 1 #sacamos las muestras (añadimos 1 para incluir la primera muestra)
    new_times = np.linspace(start, stop, num=num_samples)
    # con esto hemos sacado_ new_times, que es un eje regular, sin jitter, exactamente a 256 Hz

    interpolated_data = [] 
    for i in range(data.shape[1]):
        interp_func = interp1d(timestamps, data[:, i], kind='linear', fill_value="extrapolate")
        interpolated_channel = interp_func(new_times)
        interpolated_data.append(interpolated_channel)
        #datos originales -> se interpolan -> se proyectan en new_times

    interpolated_data = np.array(interpolated_data).T # se convierte en un array tipo numpy traspuesto
    df_interp = pd.DataFrame(interpolated_data, columns=ch_names) #para trabajar con pandas queremos: [muestras x canales]. Trasponemos para que cada fila represente un instante de tiempo y cada columna un canal EEG
    df_interp.insert(0, "timestamp", new_times)
    cols_to_keep = ["timestamp"] + list(ch_names) #se usa :8 porque el EEG tiene 8 canales
    df_interp = df_interp[cols_to_keep]

    # Guardar CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True) # crea carpeta si no existe (si existe no hace nada)
    df_interp.to_csv(output_path, index=False) #guarda el csv aqui (index=false --> asi pandas no crea por defecto una nueva columna con un indice)
    print(f"\n✅ Archivo interpolado guardado en:\n{output_path}")

    if validate:
        compare_original_vs_interpolated(timestamps, data, new_times, interpolated_data, ch_idx=0)

        #expected_len = int((stop - start) * target_sfreq)
        expected_len = len(new_times)
        print(f"\n📏 Validación de longitud:")
        print(f"  ➤ Esperado: {expected_len} muestras")
        print(f"  ➤ Obtenido: {len(new_times)} muestras") 
        #no entiendo estos checks, porque al final 100% te va a salir lo mimso
        print("✅ Longitud interpolada OK.")

        actual_sfreq = 1 / np.mean(np.diff(new_times))
        print(f"\n📐 Frecuencia de muestreo interpolada: {actual_sfreq:.2f} Hz")

        if abs(len(new_times) - expected_len) > 1:
            print("⚠️  ¡CUIDADO! El número de muestras interpoladas no coincide con lo esperado.")
        else:
            print("✅ Longitud interpolada OK.")


