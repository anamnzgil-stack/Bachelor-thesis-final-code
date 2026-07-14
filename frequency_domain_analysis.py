# analyze_psd_bands.py

import mne
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import os


# Compatibilidad NumPy: np.trapz fue renombrado a np.trapezoid en NumPy 2.0
# y eliminado en versiones recientes. Usamos el que esté disponible.
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


FREQ_BANDS = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 45),
}

def compute_band_power(epochs, fmin=0.5, fmax=45, method='welch', n_fft=None):
    """
    Calcula la potencia en bandas de frecuencia definidas.
    
    Args:
        raw_or_epochs (Raw | Epochs): objeto MNE.
        fmin/fmax (Hz): rango de interés para la PSD.
        method: método de estimación ('welch', 'multitaper', etc)
        'welch' -> Divide la señal en ventanas, calcula la FFT de cada una y promedia (el mas comun en EEG porque reduce el ruido).
        n_fft: tamaño del FFT para resolución espectral (mas gtande = mejor resolucion en frecuencia)
    Returns:
        DataFrame con potencia por banda y canal.
        get_data -> extrae los dato numericos del PSD. Devuelve:
        psd - array de forma (n_channels, n_frecuencias) con la potencia en cada frecuencia para cada canal
        freqs: Array con los valores de frecuencia correspondientes
    """
    print("🧠 Calculando PSD...")
    # Calcular n_times para asegurarse de que n_fft no sea mayor
    n_times = epochs.get_data().shape[2]  # shape: (n_epochs, n_channels, n_times)
    if n_fft is None or n_fft > n_times:
        n_fft = n_times  
        print(f"   n_fft ajustado a {n_fft} (longitud de la epoch)")
    
    # IMPORTANTE: calculamos el spectrum y sacamos ch_names DEL SPECTRUM, no de
    # epochs. compute_psd puede excluir canales (p.ej. los marcados como 'bads'),
    # con lo que el eje de canales del array psd no tiene por que coincidir con
    # epochs.ch_names. Usar spectrum.ch_names garantiza que psd[ep, ch_idx] y
    # ch_names esten alineados y evita el IndexError.
    spectrum = epochs.compute_psd(
        fmin=fmin, fmax=fmax, method=method, n_fft=n_fft, picks="eeg"
    )
    psd, freqs = spectrum.get_data(return_freqs=True)

    ch_names = spectrum.ch_names

    if epochs.metadata is not None and "difficulty" in epochs.metadata.columns:
        difficulties = epochs.metadata["difficulty"].values #si en el metadata de los epochs hay una columna llamada difficulty, entonces la guardamos en difficulties
    else:
        difficulties = ["UNKNOWN"] * len(epochs) 
    
    # Frecuencia máxima global para tratar el borde superior de la última banda
    f_global_max = max(high for (_, high) in FREQ_BANDS.values())

    rows = []
    for ep_idx in range(psd.shape[0]): #recorre cada epoch
        for ch_idx, ch_name in enumerate(ch_names): #recorre cada canal

            # total = suma de las INTEGRALES (área bajo la PSD) de cada banda
            # Integramos con np.trapz -> potencia real
            total_power = 0.0
            band_powers = {}
            for band, (low, high) in FREQ_BANDS.items():
                # Bordes sin solape: [low, high) para que el bin del límite
                # superior (p.ej. 8 Hz) no se cuente dos veces en bandas contiguas.
                # La última banda incluye su borde superior para no perder el bin final.
                if high >= f_global_max:
                    mask = np.logical_and(freqs >= low, freqs <= high)
                else:
                    mask = np.logical_and(freqs >= low, freqs < high)

                # Integral de la PSD sobre las frecuencias de la banda.
                # Si hay <2 puntos en la banda, np.trapz daría 0/error -> fallback a 0.
                if np.count_nonzero(mask) >= 2:
                    band_powers[band] = _trapz(psd[ep_idx, ch_idx, mask], freqs[mask])
                elif np.count_nonzero(mask) == 1:
                    band_powers[band] = float(psd[ep_idx, ch_idx, mask][0])
                else:
                    band_powers[band] = 0.0
                total_power += band_powers[band]

            row = {
                "epoch_index": ep_idx,
                "difficulty":  difficulties[ep_idx],
                "Channel":     ch_name,
            }

            for band in FREQ_BANDS:
                # Absoluta (integral) y relativa (fracción del total)
                row[band] = band_powers[band]
                row[f"{band}_rel"] = band_powers[band] / total_power if total_power > 0 else np.nan

            rows.append(row)  # añadimos el diccionario a la lista de filas
    return pd.DataFrame(rows)  # columnas: epoch_index, difficulty, Channel, Delta..Gamma (abs) y _rel

def detect_bad_channels_from_epochs(epochs, std_factor=2.0, threshold_samples_ratio=0.95,
                                    saturation_uv=100.0):
    """
    [OBSOLETA] La deteccion de bad channels se movio a preprocess.py
    (detect_bad_channels_from_raw), para ejecutarse ANTES del re-referencing.
    Se conserva aqui por compatibilidad, pero ya no se llama en el flujo principal.

    Detecta canales ruidosos directamente desde el objeto Epochs, replicando el
    criterio del time domain (Std y muestras saturadas) sin depender de él.

    Métricas por canal (media sobre todos los epochs):
      - Std (µV): desviación típica de la señal.
      - muestras saturadas: nº de muestras con |amplitud| > saturation_uv (µV).

    Un canal es malo si:
      - su Std supera std_factor * mediana(Std de todos los canales), o
      - tiene >= threshold_samples_ratio * epoch_samples muestras saturadas (de media).

    Args:
        epochs (mne.Epochs): epochs cargados (datos en Voltios, como guarda MNE).
        std_factor (float): multiplicador sobre la mediana de las Std.
        threshold_samples_ratio (float): proporción de muestras saturadas que marca un canal.
        saturation_uv (float): umbral de saturación en µV.

    Returns:
        list[str]: nombres de canales malos.
    """
    # data: (n_epochs, n_channels, n_times). MNE guarda en V -> pasamos a µV.
    data_uv = epochs.get_data() * 1e6
    ch_names = epochs.ch_names
    n_times = data_uv.shape[2]  # epoch_samples (p.ej. 512 a 256 Hz / 2 s)

    # Std por canal y epoch -> media sobre epochs
    std_per_epoch = data_uv.std(axis=2)            # (n_epochs, n_channels)
    mean_std = std_per_epoch.mean(axis=0)          # (n_channels,)

    # Muestras saturadas por canal y epoch -> media sobre epochs
    sat_per_epoch = (np.abs(data_uv) > saturation_uv).sum(axis=2)  # (n_epochs, n_channels)
    mean_sat = sat_per_epoch.mean(axis=0)                          # (n_channels,)

    std_threshold = std_factor * np.median(mean_std)
    sat_threshold = threshold_samples_ratio * n_times

    bad_channels = []
    for i, ch_name in enumerate(ch_names):
        is_high_std = mean_std[i] > std_threshold
        is_saturated = mean_sat[i] >= sat_threshold
        if is_high_std or is_saturated:
            bad_channels.append(ch_name)

    print(f"   Std mediana={np.median(mean_std):.2f} µV → umbral Std={std_threshold:.2f} µV")
    print(f"   Umbral saturación: {sat_threshold:.0f} muestras (>{saturation_uv:.0f} µV)")
    return bad_channels


def run_freq_domain_analysis(epochs_fif_path, output_dir):
    """
    Carga all_epochs_epo.fif, calcula PSD epoch a epoch y guarda:
      - freq_metrics_all_epochs.csv  → una fila por (epoch, canal)
      - freq_metrics_summary.csv     → media por (difficulty, canal)

    Args:
        epochs_fif_path (str): ruta al archivo all_epochs_epo.fif.
        output_dir (str): carpeta de salida.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"📂 Cargando epochs: {epochs_fif_path}")
    epochs = mne.read_epochs(epochs_fif_path, preload=True)
    print(f"   Epochs totales: {len(epochs)}")
    print(f"   Dificultades: {epochs.metadata['difficulty'].value_counts().to_dict()}")

    # Los bad channels ya se ELIMINARON en preprocess.py (antes del re-referencing).
    # Por seguridad, vaciamos info['bads']: cualquier nombre residual ahi haria que
    # compute_psd(picks="eeg") excluyese canales silenciosamente y el array psd
    # tendria menos canales que epochs.ch_names (causa del IndexError).
    if epochs.info['bads']:
        print(f"   ⚠️ info['bads'] residual en el .fif: {epochs.info['bads']} → se limpia")
        epochs.info['bads'] = []
    # NOTA: La deteccion y eliminacion de bad channels (canales fijos chann_20/chann_24
    # y los detectados por Std/saturacion) ahora se realiza en preprocess.py, ANTES del
    # re-referencing por promediado. Por tanto, los epochs cargados aqui ya vienen sin
    # canales malos y el promedio de referencia se calculo solo con canales buenos.
    print(f"   Canales presentes (ya sin bad channels): {len(epochs.ch_names)}")

    df = compute_band_power(epochs)

    # CSV completo (epoch a epoch) - absoluta + relativa
    output_csv = os.path.join(output_dir, "freq_metrics_all_epochs.csv")
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Métricas guardadas en: {output_csv}")

    # Resumen: media por dificultad y canal (absolutas + relativas)
    abs_cols = list(FREQ_BANDS.keys())                 # Delta, Theta, ... (absolutas / integral)
    rel_cols = [f"{b}_rel" for b in FREQ_BANDS.keys()] # Delta_rel, Theta_rel, ... (relativas)
    summary = (
        df.groupby(["difficulty", "Channel"])[abs_cols + rel_cols]
        .mean()
        .reset_index()
    )
    summary_csv = os.path.join(output_dir, "freq_metrics_summary.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"✅ Resumen guardado en: {summary_csv}")

    #print("\n📊 Potencia media por dificultad y canal:")
    #print(summary.to_string(index=False))

    return df