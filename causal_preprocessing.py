import os
import numpy as np
import warnings

# Ignore RuntimeWarnings for dividing by zero in empty slices (handled in code)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 1. Define the exact feature list based on your structure
ALL_FEATURES = [
    'insar_cum', 'insar_diff', 'coh_avg', 'hgt', 'U.geo', 'mask', 'vstd',
    'total_precipitation_sum', 'total_evaporation_sum', 'runoff_sum',
    'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2',
    'volumetric_soil_water_layer_3', 'volumetric_soil_water_layer_4',
    'temperature_2m', 'skin_temperature', 'soil_temperature_level_1',
    'soil_temperature_level_4', 'surface_net_solar_radiation_sum',
    'surface_sensible_heat_flux_sum', 'surface_pressure',
    'u_component_of_wind_10m', 'v_component_of_wind_10m',
    'dewpoint_temperature_2m', 'leaf_area_index_high_vegetation',
    'leaf_area_index_low_vegetation', 'bdod_gcm3', 'clay_pct',
    'phh2o_pH', 'sand_pct', 'silt_pct', 'soc_dgkg'
]

# 2. UPDATED INDICES: 
# Removed index 24 (leaf_area_index_high_vegetation) due to Zero Variance in arid regions (e.g., Semnan)
DROP_INDICES = [0, 3, 4, 5, 24, 26, 27, 28, 29, 30, 31] 
RAW_KEEP_INDICES = [1, 2, 6] # insar_diff, coh_avg, vstd
DESEASON_KEEP_INDICES = list(range(7, 24)) + [25] # ERA5 features skipping index 24

# 3. Define file paths
FILE_PATHS = [
    r"C:\Users\DFMRendering\Desktop\subsidence\Causal\Isfahan\Merged_Dataset_3D.npz",
    r"C:\Users\DFMRendering\Desktop\subsidence\Causal\Jiroft\Merged_Dataset_3D.npz",
    r"C:\Users\DFMRendering\Desktop\subsidence\Causal\Lake Urmia Tabriz\Merged_Dataset_3D.npz",
    r"C:\Users\DFMRendering\Desktop\subsidence\Causal\Marvdasht\Merged_Dataset_3D.npz",
    r"C:\Users\DFMRendering\Desktop\subsidence\Causal\Nishabur\Merged_Dataset_3D.npz",
    r"C:\Users\DFMRendering\Desktop\subsidence\Causal\Qazvin-Alborz-Tehran\Merged_Dataset_3D.npz",
    r"C:\Users\DFMRendering\Desktop\subsidence\Causal\Rafsanjan\Merged_Dataset_3D.npz",
    r"C:\Users\DFMRendering\Desktop\subsidence\Causal\Semnan\Merged_Dataset_3D.npz"
]

def apply_deseasonalization(data_3d):
    """
    Applies Monthly Z-Score Anomaly deseasonalization.
    Assumes data_3d is shaped (Time, Pixels)
    """
    T, P = data_3d.shape
    deseasonalized = np.zeros_like(data_3d, dtype=np.float64)
    
    for p in range(P):
        pixel_ts = data_3d[:, p]
        for m in range(12):
            month_indices = np.arange(m, T, 12)
            if len(month_indices) == 0:
                continue
                
            month_data = pixel_ts[month_indices]
            mean_val = np.mean(month_data)
            std_val = np.std(month_data)
            
            if std_val == 0 or np.isnan(std_val):
                deseasonalized[month_indices, p] = 0.0
            else:
                deseasonalized[month_indices, p] = (month_data - mean_val) / std_val
                
    return deseasonalized

def process_city_dataset(file_path):
    print(f"Processing: {os.path.basename(os.path.dirname(file_path))}...")
    
    # Load Data
    data_dict = np.load(file_path)
    array_key = list(data_dict.keys())[0]
    raw_data = data_dict[array_key]
    
    # Extract and Process Features
    final_features_data = []
    final_features_names = []
    
    # A. Raw Features
    for idx in RAW_KEEP_INDICES:
        final_features_data.append(raw_data[:, :, idx])
        final_features_names.append(ALL_FEATURES[idx])
        
    # B. Deseasonalized Features
    for idx in DESEASON_KEEP_INDICES:
        feature_data = raw_data[:, :, idx]
        deseasonalized_data = apply_deseasonalization(feature_data)
        final_features_data.append(deseasonalized_data)
        final_features_names.append(ALL_FEATURES[idx])
        
    # Stack features (Time, Pixels, Features)
    processed_data = np.stack(final_features_data, axis=2)
    
    # Transpose for Tigramite -> (Pixels, Time, Features)
    ready_for_tigramite = np.transpose(processed_data, (1, 0, 2))
    
    # Save NPZ
    dir_name = os.path.dirname(file_path)
    save_path = os.path.join(dir_name, "JPCMCI_Ready_Dataset.npz")
    np.savez_compressed(save_path, data=ready_for_tigramite, features=np.array(final_features_names))
    
    # Generate validation report
    report_path = os.path.join(dir_name, "JPCMCI_Metadata_Report.txt")
    generate_validation_report(report_path, os.path.basename(dir_name), raw_data.shape, ready_for_tigramite, final_features_names)
    print(f"   -> Saved Data & Report in: {dir_name}\n")

def generate_validation_report(report_path, city_name, old_shape, final_data, feature_names):
    new_shape = final_data.shape
    report = f"""====================================================
J-PCMCI+ CAUSAL DISCOVERY DATASET REPORT & VALIDATION
====================================================
City/Region: {city_name}

--- 1. STRUCTURAL TRANSFORMATIONS ---
Original Shape (Time, Pixels, Features): {old_shape}
Final Shape (Pixels, Time, Features):    {new_shape}
Total Features Retained: {len(feature_names)}

* NOTE ON EXCLUDED FEATURES: 
  - 'leaf_area_index_high_vegetation' was explicitly removed because it has 
    Zero Variance in arid regions (e.g., Semnan). Retaining it would cause 
    a Singular Matrix error in Tigramite's covariance calculations.
  - All static variables (Soil, Height) were removed for the same reason.

--- 2. DATASET HEALTH & STATISTICAL VALIDATION ---
Legend:
- Status 'OK': No NaNs, No Infs, Variance > 0.
- Mean ~ 0 for ERA5 features indicates successful deseasonalization.

Idx | Feature Name                   | Min       | Max       | Mean      | Std       | Status
--------------------------------------------------------------------------------------------------
"""
    # Calculate stats for each feature
    # final_data shape is (P, T, V)
    for i, f_name in enumerate(feature_names):
        feature_slice = final_data[:, :, i]
        
        has_nan = np.isnan(feature_slice).any()
        has_inf = np.isinf(feature_slice).any()
        f_min = np.nanmin(feature_slice)
        f_max = np.nanmax(feature_slice)
        f_mean = np.nanmean(feature_slice)
        f_std = np.nanstd(feature_slice)
        
        status = "OK"
        if has_nan:
            status = "WARNING: Contains NaN!"
        elif has_inf:
            status = "WARNING: Contains Inf!"
        elif f_std == 0:
            status = "CRITICAL: Zero Variance!"
            
        # Formatting the row
        row = f"{i:<3} | {f_name:<28} | {f_min:<9.4f} | {f_max:<9.4f} | {f_mean:<9.4f} | {f_std:<9.4f} | {status}\n"
        report += row

    report += """
--- 3. HOW TO LOAD IN TIGRAMITE ---
import numpy as np
from tigramite.dataframe import DataFrame

loaded = np.load('JPCMCI_Ready_Dataset.npz')
data_array = loaded['data']       # Shape: (Samples, Time, Variables)
feature_names = loaded['features']

dataframe = DataFrame(data_array, var_names=feature_names)
"""
    with open(report_path, 'w') as f:
        f.write(report)

# Execute the pipeline
for path in FILE_PATHS:
    if os.path.exists(path):
        process_city_dataset(path)
    else:
        print(f"WARNING: File not found at {path}")

print("All datasets successfully preprocessed and validated for J-PCMCI+!")
