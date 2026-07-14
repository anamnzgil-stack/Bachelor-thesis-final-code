import pandas as pd
import numpy as np
import os
import mne
import re

def segment_eeg_data(eeg_path, markers_path, output_dir="output_segments"):
    os.makedirs(output_dir, exist_ok=True)

    # Cargar EEG
    eeg_df = pd.read_csv(eeg_path)
    timestamps = eeg_df.iloc[:, 0].values
    data = eeg_df.iloc[:, 1:].T.values  # Transpuesta para MNE (channels x samples)

    # Estimar frecuencia de muestreo
    time_diffs = np.diff(timestamps)
    sfreq = 1 / np.median(time_diffs)

    # Crear objeto RawArray de MNE
    ch_names = list(eeg_df.columns[1:])
    #ch_names = [f"ch_{i}" for i in range(1, 9)] 
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info) #data se convierte en un objeto RawArray
    raw._times = timestamps - timestamps[0] #esto hace que  el tiempo empiece en t=0 segundos

    # Cargar marcadores
    markers_df = pd.read_csv(markers_path)

    # Filtrar solo marcadores de tipo back_start o back_end (y reseteamos el indice para que empiece en 0,1,2...)
    task_markers = markers_df[markers_df["marker"].str.contains("BACK_START") | 
                              (markers_df["marker"].str.contains("BACK_END"))].reset_index(drop=True)

    segments = []
    for i in range(len(task_markers) - 1): 
        flag = task_markers.iloc[i]["marker"]
        if "BACK_START" in flag:
            start_time=task_markers.iloc[i]["timestamp"] #timestamp del marcador actual
            end_time = task_markers.iloc[i + 1]["timestamp"] #timestamp del siguiente marcador

            marker = task_markers.iloc[i]["marker"]
            difficulty_match = re.search(r"(\d)_BACK", marker)
            difficulty = difficulty_match.group(1) + "_BACK" if difficulty_match else "UNKNOWN"


            segments.append({
            "start": start_time,
            "end": end_time,
            "difficulty": difficulty,
            "index": len(segments) + 1
            })

    all_segments=[]

    # Cortar y guardar segmentos
    for seg in segments: 
        start_sample = np.argmin(np.abs(timestamps - seg["start"])) #buscamos la muestra del eeg que coincida con el marcador 1_BACK_START
        end_sample = np.argmin(np.abs(timestamps - seg["end"]))

        segment = raw.copy().crop(
            tmin=(timestamps[start_sample] - timestamps[0]),
            tmax=(timestamps[end_sample] - timestamps[0])
        ) #en segments se almacenan las partes del eeg que coincide con los segmentos de cada marcador (START hasta END, ...)

        segment_data = segment.get_data().T
        segment_timestamps = timestamps[start_sample:start_sample + segment_data.shape[0]] # tiempo del segmento

        df = pd.DataFrame(segment_data, columns=ch_names)
        df.insert(0, "timestamp", segment_timestamps)
        #df.head(10)
        df["TASK_DIFFICULTY"] = seg["difficulty"] #añado nueva columna para señalar la dificultad de cada segmento (poder identificarlos)
        all_segments.append(df)
        print(f"Segmento ({seg['difficulty']}) añadido con {df.shape[0]} muestras")
    
    #Concatenar todo
    final_df = pd.concat(all_segments, ignore_index=True)
    final_df.to_csv(os.path.join(output_dir, "segments.csv"), index=False)
    print(f"🎉 Todos los segmentos han sido guardados en:\n{os.path.abspath(output_dir)}")


def extract_calibration_segment(eeg_path, markers_path, output_path,
                                 start_label="eeg_calibration_started", end_label="eeg_calibration_end"):
    print(f"\n🔎 Extrayendo segmento de calibración desde {eeg_path}")

    # Leer EEG interpolado
    eeg_df = pd.read_csv(eeg_path)

    # Leer marcadores
    markers_df = pd.read_csv(markers_path)

    # Buscar timestamps de inicio y fin según 'marker'
    try:
        t_start = markers_df.loc[markers_df['marker'] == start_label, 'timestamp'].values[0] #values[0]: lo convierte en un array y coge el primer valor (es decir, el timestamp)
        t_end = markers_df.loc[markers_df['marker'] == end_label, 'timestamp'].values[0]
    except IndexError:
        print("❌ Error: No se encontraron los marcadores 'eeg_calibration_started' o 'end'.")
        return

    print(f"⏱ Segmento de calibración: {t_start:.3f} s → {t_end:.3f} s")

    # Recortar EEG entre esos tiempos
    segment = eeg_df[(eeg_df["timestamp"] >= t_start) & (eeg_df["timestamp"] <= t_end)].copy()

    for col in segment.columns[1:]:
        segment[col] = segment[col] * 1e-3 #los datos estan en mV, pero mne trabaja en V
    # Crear carpeta destino si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Guardar CSV
    segment.to_csv(output_path, index=False)
    print(f"✅ Segmento guardado en:\n{output_path}")