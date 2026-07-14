import os
import pandas as pd
import numpy as np
import mne

# Canales que sabemos defectuosos (no estan conectados por las dispositcion con las gafas)
FIXED_BAD_CHANNELS = ["chann_20", "chann_24"]

def detect_bad_channels_from_raw(raw, std_factor=2.0,
                                 threshold_samples_ratio=0.95,
                                 saturation_uv=100.0):
    """
    Detecta canales ruidosos directamente desde un Raw continuo, para hacerlo antes del re-referencing (antes lo hacia por epoch)
 
    Metricas por canal:
      - Std (uV): desviacion tipica de la senal completa.
      - muestras saturadas: n de muestras con |amplitud| > saturation_uv (uV).
 
    Un canal es malo si:
      - su Std supera std_factor * mediana(Std de todos los canales), o
      - tiene >= threshold_samples_ratio * n_muestras saturadas.
 
    Args:
        raw (mne.io.Raw): datos continuos (en Voltios, como trabaja MNE).
        std_factor (float): multiplicador sobre la mediana de las Std.
        threshold_samples_ratio (float): proporcion de muestras saturadas que marca un canal.
        saturation_uv (float): umbral de saturacion en uV.
 
    Returns:
        list[str]: nombres de canales malos detectados.
    """
   # data: (n_channels, n_times). MNE guarda en V -> pasamos a uV.
    data_uv = raw.get_data() * 1e6
    ch_names = raw.ch_names
    n_times = data_uv.shape[1]
 
    std_per_ch = data_uv.std(axis=1)                           
    sat_per_ch = (np.abs(data_uv) > saturation_uv).sum(axis=1)  
 
    std_threshold = std_factor * np.median(std_per_ch)
    sat_threshold = threshold_samples_ratio * n_times
 
    bad_channels = []
    for i, ch_name in enumerate(ch_names):
        is_high_std = std_per_ch[i] > std_threshold
        is_saturated = sat_per_ch[i] >= sat_threshold
        if is_high_std or is_saturated:
            bad_channels.append(ch_name)
 
    print(f"   Std mediana={np.median(std_per_ch):.2f} uV -> umbral Std={std_threshold:.2f} uV")
    print(f"   Umbral saturacion: {sat_threshold:.0f} muestras (>{saturation_uv:.0f} uV)")
    return bad_channels
 

def preprocess_segment(csv_path, sfreq, apply_ica,
                       epoch_length=2.0, overlap=1.0, sliding=True):
    """
    Filtros añadidos: FPBanda de [5,45]Hz, F Notch de 50 Hz
    ICA APLICADO
    Creamos y guardamos epochs
    
    """
    df = pd.read_csv(csv_path) #df =  segmento(i=1,2,3...) en tipo panda
    #eeg_cols = [col for col in df.columns if col.startswith("chann_")]
    eeg_cols = [col for col in df.columns
                if col not in ["timestamp", "TASK_DIFFICULTY"] + FIXED_BAD_CHANNELS]
    print("Columnas del CSV:", df.columns.tolist())
    print("EEG cols detectadas:", eeg_cols)
    print(f"Canales fijos excluidos: {FIXED_BAD_CHANNELS}")

    # -- Deteccion de bad channels una vez, sobre TODO el registro continuo --
    # Construimos un raw con todos los segmentos concatenados (orden temporal)
    # para detectar canales globalmente malos de forma consistente.
    full_data = df[eeg_cols].T.values.copy()
    info_full = mne.create_info(ch_names=eeg_cols, sfreq=sfreq, ch_types='eeg')
    raw_full = mne.io.RawArray(full_data, info_full, verbose=False)
    print("\nDetectando bad channels (antes del re-referencing)...")
    auto_bads = detect_bad_channels_from_raw(raw_full)
    print(f"Bad channels detectados: {auto_bads}")
 
    all_epochs = []
    for difficulty, group in df.groupby("TASK_DIFFICULTY"):
        print(f"\n🔄 Preprocesando: {difficulty} ({len(group)} muestras)")
        data = group[eeg_cols].T.values.copy()
        print(f"Rango en V: {data.min()} → {data.max()}")
        #print(f"Rango en µV: {data.min()*1e6} → {data.max()*1e6}")

        info = mne.create_info(ch_names=eeg_cols, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info) #raw = toda la data de segmento menos los timestamps
        print(raw)
        # Fijar tipo EEG explícitamente para que sobreviva al guardado/carga del .fif
        raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names})
        print(raw.get_channel_types())
        # -- Marcar bad channels ANTES de re-referenciar ----------------------
        # Solo los que existan en este raw (todos deberian existir).
        raw.info['bads'] = [ch for ch in auto_bads if ch in raw.ch_names]
        if raw.info['bads']:
            raw.drop_channels(raw.info['bads'])
            print(f"   Marcados como bads (excluidos del promedio): {raw.info['bads']}")
        # Preprocesamiento básico
        raw.set_eeg_reference('average', projection=False)  #se usa como referencia la media de todos los electrodos. False =lo aplica directamente a los datos
        #raw.filter(l_freq=0.5, h_freq=120., fir_design='firwin')
        raw.filter(l_freq=0.5, h_freq=45., fir_design='firwin') #FPBanda de [0.5,45]Hz
        raw.notch_filter(freqs=50., fir_design='firwin') #F Notch de 50 Hz

        if apply_ica:
            raw = apply_ica_to_raw(raw) 
        #── Crear epochs para esta dificultad 
        epochs = create_epochs(raw, epoch_length=epoch_length,
                               overlap=overlap, sliding=sliding,
                               difficulty=difficulty)
        all_epochs.append(epochs)
        print(f"   → {len(epochs)} epochs creados para {difficulty}")

    # ── Concatenar todas las dificultades en un único Epochs ─────────────
    print("\n🔗 Concatenando epochs de todas las dificultades...")
    epochs_all = mne.concatenate_epochs(all_epochs)
    print(f"✅ Total epochs: {len(epochs_all)}")
    print(f"   Dificultades presentes: {epochs_all.metadata['difficulty'].unique().tolist()}")

    # ── Guardar un único .fif ─────────────────────────────────────────────
    epochs_dir = os.path.join(os.path.dirname(csv_path), "epochs")
    os.makedirs(epochs_dir, exist_ok=True)
    epochs_path = os.path.join(epochs_dir, "all_epochs_epo.fif")
    epochs_all.save(epochs_path, overwrite=True)
    print(f"📁 Epochs guardados en: {epochs_path}")

    print(f"Rango en V: {data.min()} → {data.max()}")
    
    # Registrar los bad channels eliminados para trazabilidad
    bads_log = os.path.join(epochs_dir, "removed_bad_channels.txt")
    with open(bads_log, "w") as f:
        f.write("\n".join(FIXED_BAD_CHANNELS + auto_bads))
    print(f"Bad channels eliminados registrados en: {bads_log}")
    
    return epochs_all, epochs_path

'''
        # Guardar Raw .fif por dificultad
        output_dir = os.path.join(os.path.dirname(csv_path), "preprocessed_fif")
        os.makedirs(output_dir, exist_ok=True)

        fif_path = os.path.join(output_dir, f"{difficulty}_preprocessed_raw.fif")
        raw.save(fif_path, overwrite=True)

        print(f"✅ Guardado: {fif_path}")
        all_raws[difficulty] = (raw, fif_path)
    return all_raws
    '''

def apply_ica_to_raw(raw):
    print("🧠 Aplicando ICA para eliminación de artefactos...")
    ica = mne.preprocessing.ICA(n_components=0.999999, random_state=97, max_iter='auto') #lo cambio asi para que no coja todos los valores (incluidos los de chann 5, que son defectuosos)
    ica.fit(raw)
    # Aquí deberías revisar los componentes manualmente si quieres excluir artefactos específicos.
    raw = ica.apply(raw.copy())
    print("✅ ICA aplicado.")
    return raw


def create_epochs(raw, epoch_length, overlap, sliding, difficulty = None):
    sfreq = raw.info['sfreq'] #freq muestreo
    data = raw.get_data()
    n_samples = data.shape[1] #timestamps
    epoch_samples = int(epoch_length * sfreq)

    if sliding:
        step = int((epoch_length - overlap) * sfreq)
        starts = np.arange(0, n_samples - epoch_samples + 1, step)
    else:
        starts = np.arange(0, n_samples, epoch_samples) 

    events = [[int(start), 0, 1] for start in starts if (start + epoch_samples) <= n_samples]
    events = np.array(events)
    event_id = dict(segment=1)
    metadata = pd.DataFrame({
        "difficulty": [difficulty] * len(events) #etiqueta de dificultad
    })

    epochs_data = np.array([
        data[:, e[0]:e[0] + epoch_samples] for e in events  # data[:, start : start + epoch_samples]
    ])

    # resultado de cada trozo: (n_channels, epoch_samples)

    epochs = mne.EpochsArray(epochs_data, raw.info, events,
                             tmin=0.0, event_id=event_id, metadata=metadata)
    print(f"📏 {len(epochs)} epochs de {epoch_length}s "
          f"{'(sliding)' if sliding else '(non-overlapping)'} creados.")
    return epochs
