import os
import numpy as np
import pandas as pd
import mne
from pathlib import Path
from frequency_domain_analysis import FREQ_BANDS

'''
Feature extraction: 
PSD Por bandas:
- Evidencia: Cambios en potencia (PSD --> Delta, Theta, Alpha, Beta, Gamma), reflejan carga cognitiva.

Ratios entre bandas importantes:
- Theta/Alpha: Aumenta con mayor carga cognitiva.
- Theta/Beta: Aumenta con mayor carga cognitiva.
- Beta / (alpha + thetha): Aumenta con mayor carga cognitiva. (no estoy segura de este, no creo que sea tan comuun)
'''
# ---------- CONFIG ----------
def compute_band_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula ratios entre bandas de frecuencia.
    Todos aumentan teóricamente con mayor carga cognitiva.

    - Theta/Alpha : el más usado en literatura de workload
    - Theta/Beta  : también correlaciona con carga cognitiva
    - Beta/(Alpha+Theta) : engagement index
    """
    eps = 1e-8  # evita división por cero 
    df["ratio_theta_alpha"] = df["Theta_rel"] / (df["Alpha_rel"] + eps)
    df["ratio_theta_beta"] = df["Theta_rel"] / (df["Beta_rel"]  + eps)
    df["ratio_beta_alpha_theta"] = df["Beta_rel"]  / (df["Alpha_rel"] + df["Theta_rel"] + eps)
    return df

# ---------- MAIN EXTRACTION FUNCTION ----------
def run_feature_extraction(freq_csv: str, time_csv: str = None, output_csv: str = None) -> pd.DataFrame:
    """
    Une los CSVs de frecuencia y tiempo, calcula ratios y guarda el CSV final. 

    Features incluidas:
    - Potencia relativa por banda (Delta_rel...) -> Sin problema de unidades (SE PUEDE CAMBIAR ESTO)

    - Ratios entre bandas
    - Metricas temporales (Stf, Var, Peak-to-Peak)

    Args:
        freq_csv   : ruta a freq_metrics_all_epochs.csv
        time_csv   : ruta a time_metrics_all_epochs.csv (no se incluyen)
        output_csv : ruta donde guardar features_all.csv

    Returns:
        DataFrame con todas las features.
    """
    print("📂 Leyendo CSVs...")
    freq = pd.read_csv(freq_csv)
    #time = pd.read_csv(time_csv)

    print(f"   freq_metrics : {freq.shape[0]} filas, {freq.shape[1]} columnas")
    #print(f"   time_metrics : {time.shape[0]} filas, {time.shape[1]} columnas")

    # Columnas que no queremos duplicar en el merge
    #time_feature_cols = ["Std (µV)", "Var (µV²)", "Peak-to-Peak (µV)"]
    time_feature_cols = [] #no se incluyen al final
    merge_keys        = ["epoch_index", "difficulty", "Channel"]
    # Columnas de potencia relativa (las que usaremos para clasificación)
    rel_band_cols = ["Delta_rel", "Theta_rel", "Alpha_rel", "Beta_rel", "Gamma_rel"]

    df = freq.copy()
    # Inner join: Une solo filas que existen en ambos CSVs
    # (how= "inner"descarta canales excluidos como bad channels en freq pero presentes en time)
   # df = pd.merge(
   #     freq,
   #     time[merge_keys + time_feature_cols],
   #     on=merge_keys,
   #     how="inner"
   # )df = pd.merge(
    print(f"\n✅ Merge completado: {df.shape[0]} filas, {df.shape[1]} columnas")



    # Calcular ratios entre bandas
    df = compute_band_ratios(df)
    print("✅ Ratios calculados: ratio_theta_alpha, ratio_theta_beta, ratio_beta_alpha_theta")
    # Eliminar potencias absolutas (dinalmente nos quedamos solo con relativas y ratios, mejor para LOSO)
    abs_band_cols = list(FREQ_BANDS.keys())  # ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    df = df.drop(columns=[c for c in abs_band_cols if c in df.columns])

    # Ordenar columnas de forma lógica
    col_order = (
        merge_keys
        + rel_band_cols
        + ["ratio_theta_alpha", "ratio_theta_beta", "ratio_beta_alpha_theta"]
        + time_feature_cols
    )
    df = df[col_order]

    # Guardar
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n💾 Features guardadas en: {output_csv}")
    print(f"Shape final: {df.shape[0]} filas × {df.shape[1]} columnas")
    print(f"Columnas: {df.columns.tolist()}")

    # Resumen rápido por dificultad
    print("\n📊 Media de features clave por dificultad:")
    summary_cols = ["Theta_rel", "Alpha_rel", "ratio_theta_alpha", "ratio_theta_beta"]
    print(df.groupby("difficulty")[summary_cols].mean().round(4).to_string())

    return df
