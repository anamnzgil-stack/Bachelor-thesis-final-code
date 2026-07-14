

from segment_data import segment_eeg_data, extract_calibration_segment
from sample_freq import analyze_sampling_frequency#, print_test
from interpolate_signal import interpolate_eeg
from preprocess import preprocess_segment
from frequency_domain_analysis import run_freq_domain_analysis
from feature_extraction import run_feature_extraction
#from exploratory_analysis_features import compare_eo_ec, compare_eo_vs_tasks, analyze_difficulty_levels
#from feature_eeg_selection import run_feature_selection
#from extract_eeg_features import run_eeg_feat_extraction  # importa el módulo


import os
import numpy as np
import pandas as pd
import mne
from pathlib import Path


target_sfreq = 256
participants = [f"participant_{i:02d}" for i in range(1, 25)]  # participant_01 ... participant_24

project_root = "C://Users//Usuario//Desktop//UNI//TFG//python"

def run_sample_freq(participant_id):
    csv_path = f"{project_root}//participantes//{participant_id}//eeg_data.csv"
    
    analyze_sampling_frequency(csv_path)

def run_interpolation(participant_id):
    input_csv = f"{project_root}//participantes//{participant_id}//eeg_data.csv"
    output_csv = f"{project_root}//output_segments//{participant_id}//interpolated//eeg_data_interp_scaled.csv"
    
    interpolate_eeg(input_csv, output_csv, target_sfreq, validate = True)

def run_extract_calibration(participant_id):
    eeg_path = f"{project_root}//output_segments//{participant_id}//interpolated//eeg_data_interp_scaled.csv"
    markers_path = f"{project_root}//participantes/{participant_id}//psychopymarkers_data.csv"
    output_path = f"{project_root}//output_segments//{participant_id}//calibration//calibration.csv"

    extract_calibration_segment(eeg_path, markers_path, output_path)

def run_segmentation(participant_id):
    eeg_path = f"{project_root}//output_segments//{participant_id}//interpolated//eeg_data_interp_scaled.csv"
    markers_path = f"{project_root}//participantes//{participant_id}//psychopymarkers_data.csv"
    #markers_path = f"{project_root}//EEG_Noise_Analysis//data//session_12//{participant_id}//psychopymarkers_data_modified_calibration.csv"
    #output_dir = f"{project_root}//output_segments//math_analysis//segmented//math_data"
    output_dir = f"{project_root}//output_segments//{participant_id}//segmented"
    
    segment_eeg_data(eeg_path, markers_path, output_dir)


def run_preprocessing(participant_id,
                      apply_ica=False,
                      epoch_length=2.0,
                      overlap=1.0,
                      sliding=True):



    csv_path = f"{project_root}//output_segments//{participant_id}//segmented//segments.csv"

    epochs_all, epochs_path = preprocess_segment(
        csv_path,
        sfreq=target_sfreq,
        apply_ica=apply_ica,
        epoch_length=epoch_length,
        overlap=overlap,
        sliding=sliding,
    )

    print(f"\n🎉 Preprocesamiento completado.")
    print(f"   Archivo guardado en: {epochs_path}")
    print(f"   Epochs totales: {len(epochs_all)}")
    print(f"   Metadata:\n{epochs_all.metadata['difficulty'].value_counts()}")  
    


def run_time_domain(participant_id):
    
    epochs_fif = f"{project_root}//output_segments//{participant_id}//segmented//epochs//all_epochs_epo.fif"
    #input_dir = f"{project_root}//output_segments//{participant_id}//segmented//preprocessed_fif"
    output_dir = f"{project_root}//output_segments//{participant_id}//metrics_time"  
    run_time_domain_analysis(epochs_fif, output_dir, plot=True) 
    #plot=True para abrir la ventana interactiva de MNE
    
def run_freq_domain(participant_id):  
    epochs_fif= f"{project_root}//output_segments//{participant_id}//segmented//epochs//all_epochs_epo.fif"
    #fif_dir = f"{project_root}//output_segments//{participant_id}//segmented//preprocessed_fif"
    output_dir = f"{project_root}//output_segments//{participant_id}//freq_analysis"
    run_freq_domain_analysis(epochs_fif, output_dir)


################
##      EEG FEATURE EXTRACTION
#########################



def run_feat_extraction(participant_id):
    freq_csv = Path(f"{project_root}//output_segments//{participant_id}//freq_analysis//freq_metrics_all_epochs.csv")
    #time_csv = Path(f"{project_root}//output_segments//{participant_id}//metrics_time//time_metrics_all_epochs.csv")
    output_csv = Path(f"{project_root}//output_segments//{participant_id}//features//features_eeg_all.csv")

    #input_dir = Path(r"C:\Users\ainar\Documents\PhD\BACSI\simulador_28_julio\analysis\EEG_Noise_Analysis\output_segments\math_analysis\segmented\math_data\epochs")
    #output_csv = Path(r"C:\Users\ainar\Documents\PhD\BACSI\simulador_28_julio\analysis\EEG_Noise_Analysis\output_segments\math_analysis\segmented\math_data\features\features_eeg_all.csv")
    
    run_feature_extraction(freq_csv, None, output_csv)


def run_eeg_feature_selection(participant_id):
    features_csv = f"{project_root}//output_segments//{participant_id}//features//features_eeg_all.csv"
    markers_csv = f"{project_root}//participantes/{participant_id}//psychopymarkers_data.csv"
    output_csv = f"{project_root}//output_segments//{participant_id}//features//features_selection.csv"

    
    run_feature_selection(features_csv, markers_csv, output_csv)

########################################################## FEATURE EXTRACTION ########################################################################
'''
def run_feat_extraction(): # ESTO NO SE LO QUE ES
    #base_dir = "C://Users//ainar//Documents//PhD//BACSI//simulador_28_julio//analysis//EEG_Noise_Analysis//output_segments//math_analysis//segmented//calibration_data"
    epochs_dir = f"{project_root}//output_segments//math_analysis//segmented//calibration_data//EC_ON//epochs"
    output_csv = f"{project_root}//arithmetics_analysis_results//all_features_calib_EC_ON.csv"
    #output_csv = os.path.join(base_dir, f"{participant_id}_all_features.csv")
    run_feature_extraction(epochs_dir, output_csv)

'''
########################################################### ANALYSIS ########################################################################




def main():

    RUN_SAMPLE_FREQ = True
    RUN_INTERPOLATION = True
    RUN_SEGMENTATION = True
    RUN_CALIB_SEGMENTATION = True
    RUN_PREPROCESSING = True
    RUN_TIME_DOMAIN = False
    RUN_FREQ_DOMAIN = True
    RUN_FEATURE_EXTRACTION = True
    RUN_FEATURE_SELECTION = False

    for id in participants:
        print(f"\n{'='*70}")
        print(f"  PROCESANDO: {id}")
        print(f"{'='*70}")

        try:
            if RUN_SAMPLE_FREQ:
                run_sample_freq(id)
            if RUN_INTERPOLATION:
                run_interpolation(id)
            if RUN_SEGMENTATION:
                run_segmentation(id)
            if RUN_PREPROCESSING:
                run_preprocessing(id, apply_ica=False,
                                  epoch_length=2.0, overlap=1.0, sliding=True)
            if RUN_FREQ_DOMAIN:
                run_freq_domain(id)
            if RUN_FEATURE_EXTRACTION:
                run_feat_extraction(id)

        except Exception as e:
            print(f"❌ Error en {id}: {e}")
            continue  # sigue con el siguiente participante aunque falle uno

"""
    ########
   # RUN_FEATURE_EXTRACTION = False
    RUN_EC_VS_EO_ANALYSIS = False
    RUN_EO_VS_TASKS_ANALYSIS = False
    RUN_DIFFICULTY_ANALYSIS = False
     #######
    if RUN_SAMPLE_FREQ:
        run_sample_freq()
    if RUN_INTERPOLATION:
        run_interpolation()
    if RUN_SEGMENTATION:
        run_segmentation()
    if RUN_CALIB_SEGMENTATION:
        run_extract_calibration()
    if RUN_PREPROCESSING:
        run_preprocessing(participant_id, target_sfreq, apply_ica=False, epoch_length=2.0, overlap=1.0, sliding=True)
    if RUN_TIME_DOMAIN:
        run_time_domain()
    if RUN_FREQ_DOMAIN:
        run_freq_domain()
    if RUN_FEATURE_EXTRACTION:
        run_feat_extraction()
    if RUN_FEATURE_SELECTION:
        run_eeg_feature_selection()


    #######################################3    
   # if RUN_FEATURE_EXTRACTION:
   #     run_feat_extraction()
    if RUN_EC_VS_EO_ANALYSIS:
        ec_vs_eo_analysis()
    if RUN_EO_VS_TASKS_ANALYSIS:
        run_eo_vs_tasks_analysis()
    if RUN_DIFFICULTY_ANALYSIS:
        run_difficulty_analysis()


"""
        

if __name__ == "__main__":
    main()
