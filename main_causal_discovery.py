import os
import sys
import numpy as np
import pickle
from tigramite.data_processing import DataFrame  
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

# ==============================================================================
# TEST MODE TOGGLE
# ==============================================================================
TEST_MODE = False
TEST_SAMPLE_SIZE = 5
# ==============================================================================

# 1. Define Paths to the "Ready" datasets
BASE_PATHS = [
    r"C:\Users\DFMRendering\Desktop\subsidence\Causal\Isfahan",
    r"C:\Users\DFMRendering\Desktop\subsidence\Causal\Jiroft",
    r"C:\Users\DFMRendering\Desktop\subsidence\Causal\Lake Urmia Tabriz",
    r"C:\Users\DFMRendering\Desktop\subsidence\Causal\Marvdasht",
    r"C:\Users\DFMRendering\Desktop\subsidence\Causal\Nishabur",
    r"C:\Users\DFMRendering\Desktop\subsidence\Causal\Qazvin-Alborz-Tehran",
    r"C:\Users\DFMRendering\Desktop\subsidence\Causal\Rafsanjan",
    r"C:\Users\DFMRendering\Desktop\subsidence\Causal\Semnan"
]

FILE_PATHS = [os.path.join(path, "JPCMCI_Ready_Dataset.npz") for path in BASE_PATHS]

# Output Directory
OUTPUT_DIR = r"C:\Users\DFMRendering\Desktop\subsidence\Causal\Final_True_JCI_Results"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==============================================================================
# TERMINAL LOGGER
# ==============================================================================
class Logger(object):
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a") 
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

log_path = os.path.join(OUTPUT_DIR, "0_Terminal_Full_Log.txt")
sys.stdout = Logger(log_path)
# ==============================================================================

print("\n" + "="*80)
print("STARTING PURE LINEAR CAUSAL DISCOVERY PIPELINE (TRUE JCI + CITY-SPECIFIC)")
print("="*80)
print("\nStep 1: Loading and Padding Datasets...")

max_time = 0
features = None
datasets = []

# Map cities to a numeric ID for the Context Variable
city_to_id = {os.path.basename(os.path.dirname(p)): idx for idx, p in enumerate(FILE_PATHS)}

for file_path in FILE_PATHS:
    if os.path.exists(file_path):
        data_dict = np.load(file_path)
        city_data = data_dict['data'] 
        features = data_dict['features']
        
        T = city_data.shape[1]
        if T > max_time:
            max_time = T
        
        city_name = os.path.basename(os.path.dirname(file_path))
        datasets.append((city_name, city_data))
    else:
        print(f"File not found: {file_path}")

print(f"Maximum Timeline found: {max_time} months.")

MISSING_FLAG = -9999.0
padded_data_list = []

for city_name, data in datasets:
    P, T, V = data.shape
    padded = np.full((P, max_time, V), MISSING_FLAG, dtype=np.float64)
    padded[:, :T, :] = data
    padded_data_list.append(padded)
    print(f" - {city_name} (ID: {city_to_id[city_name]}) padded from {T} to {max_time} months.")

joint_data = np.concatenate(padded_data_list, axis=0)

if TEST_MODE:
    print(f"\n[WARNING] TEST MODE IS ACTIVE! Subsetting data to only {TEST_SAMPLE_SIZE} samples.")
    joint_data = joint_data[:TEST_SAMPLE_SIZE, :, :]

Total_P, Final_T, Total_V = joint_data.shape
print(f"\nStep 2: Joint Dataset Created! Total Pixels: {Total_P}, Time: {Final_T}, Features: {Total_V}")

feature_list = features.tolist()
target_name = 'insar_diff'
context_name = 'City_ID'

# ==============================================================================
# --- ULTRA-REFINED EXPERT KNOWLEDGE FEATURE REDUCTION ---
#
# METHODOLOGICAL JUSTIFICATION FOR FEATURE PRUNING:
# The original dataset contained 21 variables. To maximize the statistical power 
# of the PCMCI+ (ParCorr) algorithm and prevent spurious causal links or false 
# "Not Detected" results due to Multicollinearity, the feature space was strictly 
# pruned down to 9 core physical variables.
#
# EXCLUDED FEATURES (12 Variables) & REASONS:
# 1. Non-Physical / Sensor Noise: 
#    - 'coh_avg', 'vstd': These are InSAR remote sensing quality/error metrics, 
#      not physical drivers of land subsidence.
# 2. Weak/Irrelevant Physical Linkage:
#    - 'u_component_of_wind_10m', 'v_component_of_wind_10m', 'surface_pressure': 
#      Atmospheric wind vectors and surface pressure do not directly drive 
#      hydro-mechanical aquifer compaction.
# 3. Severe Multicollinearity (Redundancy):
#    - 'temperature_2m', 'dewpoint_temperature_2m': Highly collinear with 'skin_temperature'.
#    - 'volumetric_soil_water_layer_2', 'volumetric_soil_water_layer_3': Highly collinear 
#      with bounding soil layers (1 & 4).
#    - 'surface_sensible_heat_flux_sum': Highly collinear with solar radiation.
#
# RETAINED CORE FEATURES (9 Variables):
# 1. 'insar_diff'                        --> Target variable (Subsidence).
# 2. 'total_precipitation_sum'           --> Primary hydrological input (Recharge).
# 3. 'total_evaporation_sum'             --> Primary hydrological output.
# 4. 'runoff_sum'                        --> Surface water dynamics.
# 5. 'volumetric_soil_water_layer_1'     --> Shallow moisture (immediate surface interactions).
# 6. 'volumetric_soil_water_layer_4'     --> Deep moisture (proxy for deep aquifer state).
# 7. 'skin_temperature'                  --> Primary thermal forcing (best thermodynamic proxy).
# 8. 'surface_net_solar_radiation_sum'   --> Primary energy driver.
# 9. 'leaf_area_index_low_vegetation'    --> Crucial proxy for agricultural groundwater pumping.
# ==============================================================================

balanced_physical_vars = [
    'insar_diff', 
    'total_precipitation_sum',
    'total_evaporation_sum',
    'runoff_sum',
    'volumetric_soil_water_layer_1',
    'volumetric_soil_water_layer_4',
    'skin_temperature',
    'surface_net_solar_radiation_sum',
    'leaf_area_index_low_vegetation'
]

# Keep only the variables that exist in the dataset
final_features_base = [v for v in balanced_physical_vars if v in feature_list]
final_feature_indices = [feature_list.index(v) for v in final_features_base]

print(f"\n[FEATURE SELECTION] Reduced features from {Total_V} to {len(final_features_base)} relevant variables.")

# Apply reduction to the Joint dataset
joint_data_reduced = joint_data[:, :, final_feature_indices]

# --- CREATE THE CONTEXT VARIABLE (City_ID) FOR JOINT JCI ---
# We need to create an array of shape (Total_P, Final_T, 1) to hold the City_ID
city_id_array = np.zeros((Total_P, Final_T, 1), dtype=np.float64)

current_p_idx = 0
for city_name, data in datasets:
    P, _, _ = data.shape
    if TEST_MODE:
        P = min(P, TEST_SAMPLE_SIZE - current_p_idx)
        if P <= 0: break
    
    city_id = city_to_id[city_name]
    city_id_array[current_p_idx:current_p_idx+P, :, 0] = city_id
    current_p_idx += P

# Concatenate the context variable to the joint dataset
joint_data_with_context = np.concatenate([joint_data_reduced, city_id_array], axis=2)

final_features_with_context = final_features_base + [context_name]
final_target_idx_context = final_features_with_context.index(target_name)
context_idx = final_features_with_context.index(context_name)

df_joint = DataFrame(
    joint_data_with_context, 
    var_names=final_features_with_context, 
    missing_flag=MISSING_FLAG,
    analysis_mode='multiple'
)

insar_vars = ['insar_diff'] 
insar_indices_context = [final_features_with_context.index(v) for v in insar_vars if v in final_features_with_context]

TAU_MAX = 6
ALPHAS = [0.01, 0.02, 0.05, 0.1] 
MAX_CONDS = 5 

# ==============================================================================
# PHASE 1: TRUE JOINT CAUSAL DISCOVERY (JCI LINEAR)
# ==============================================================================
phase1_pickle_path = os.path.join(OUTPUT_DIR, "1_Main_Linear_Results.pkl")

if os.path.exists(phase1_pickle_path) and not TEST_MODE:
    print(f"\n[RESUME] Found existing True JCI Phase 1 results at: {phase1_pickle_path}")
    print("Skipping Main Linear Causal Discovery and loading existing data...")
    
    with open(phase1_pickle_path, 'rb') as f:
        results = pickle.load(f)
        
    p_matrix = results['p_matrix']
    val_matrix = results['val_matrix']
    graph = results['graph']
    
else:
    print("\nStep 3: Initializing and Running True Joint J-PCMCI+ (Linear) with Context Variable ...")
    
    parcorr = ParCorr(significance='analytic')
    pcmci = PCMCI(dataframe=df_joint, cond_ind_test=parcorr, verbosity=1)

    link_assumptions = pcmci._set_link_assumptions(None, tau_min=0, tau_max=TAU_MAX)

    # --- APPLY JCI AND PHYSICAL CONSTRAINTS ---
    for j, t_name in enumerate(final_features_with_context):
        
        # JCI Rule 1: Nothing can cause the Context (City_ID)
        if t_name == context_name:
            # Delete all incoming links to City_ID from any lag
            for i in range(len(final_features_with_context)):
                for tau in range(0, TAU_MAX + 1):
                    if (i, -tau) in link_assumptions[j]:
                        del link_assumptions[j][(i, -tau)]
        else:
            # JCI Rule 2: Context is static. It only affects things contemporaneously (tau=0)
            for tau in range(1, TAU_MAX + 1):
                if (context_idx, -tau) in link_assumptions[j]:
                    del link_assumptions[j][(context_idx, -tau)]

            # Physical Rule: InSAR cannot cause climate
            if t_name not in insar_vars: 
                for i in insar_indices_context:       
                    for tau in range(1, TAU_MAX + 1): 
                        if (i, -tau) in link_assumptions[j]:
                            del link_assumptions[j][(i, -tau)]

    results = pcmci.run_pcmciplus(tau_max=TAU_MAX, pc_alpha=ALPHAS, link_assumptions=link_assumptions, max_conds_px=MAX_CONDS)

    print("\nMain Algorithm Finished! Extracting and saving results...")
    with open(phase1_pickle_path, 'wb') as f:
        pickle.dump(results, f)

    p_matrix = results['p_matrix']
    val_matrix = results['val_matrix']
    graph = results['graph']

try:
    optimal_alpha = float(results['pc_alpha'])
except (KeyError, TypeError):
    optimal_alpha = ALPHAS[0]

report_path = os.path.join(OUTPUT_DIR, "1_Main_Linear_Report.txt")

report = f"""====================================================
TRUE J-PCMCI+ CAUSAL DISCOVERY: MACRO JOINT LINEAR RESULTS
====================================================
Test Mode Active: {TEST_MODE}
Total Datasets (Cities): {len(datasets)}
Total Independent Samples (Pixels): {Total_P}
Maximum Time-Lags Checked (tau_max): {TAU_MAX} months
Tested Alpha Grid: {ALPHAS}
OPTIMAL ALPHA SELECTED (via AIC): {optimal_alpha}
Context Variable Used: {context_name} (City_ID)

--- STRICT CAUSAL DRIVERS FOR SUBSIDENCE ({target_name}) ---
(Sorted by strongest effect / Partial Correlation)
-------------------------------------------------------------------------
"""

significant_links = []

for i, var_name in enumerate(final_features_with_context):
    for tau in range(TAU_MAX + 1):
        if graph[i, final_target_idx_context, tau] == '-->': 
            significant_links.append({
                'var': var_name,
                'tau': tau,
                'val': val_matrix[i, final_target_idx_context, tau],
                'pval': p_matrix[i, final_target_idx_context, tau]
            })

significant_links.sort(key=lambda x: abs(x['val']), reverse=True)

if not significant_links:
    report += "\nNO SIGNIFICANT DIRECTED CAUSAL LINKS FOUND."
else:
    for link in significant_links:
        direction = f"tau={link['tau']:<2}"
        report += f"{link['var']:<32} ({direction}) ----> {target_name:<12} | {link['val']:>8.4f} | {link['pval']:.3e}\n"

with open(report_path, 'w') as f:
    f.write(report)
print(f"Joint Linear Report saved to: {report_path}")

# ==============================================================================
# PHASE 2: CITY-SPECIFIC CAUSAL DISCOVERY (LOCAL MODELS)
# ==============================================================================
print("\n" + "="*80)
print("Step 4: Running City-Specific Analysis (Panel-PCMCI+ Local Models)")
print("="*80)

# Re-define target and insar indices for the base features (without context)
final_target_idx_base = final_features_base.index(target_name)
insar_indices_base = [final_features_base.index(v) for v in insar_vars if v in final_features_base]

for city_name, city_data_raw in datasets:
    print(f"\n--- Processing City: {city_name} ---")
    city_out_dir = os.path.join(OUTPUT_DIR, f"City_{city_name}")
    if not os.path.exists(city_out_dir):
        os.makedirs(city_out_dir)
        
    if TEST_MODE:
        city_data_raw = city_data_raw[:TEST_SAMPLE_SIZE, :, :]
        
    P_city, T_city, V_city = city_data_raw.shape
    print(f"City Dataset Loaded! Pixels: {P_city}, Time: {T_city}")
    
    # Apply the balanced feature reduction to the individual city data (NO CONTEXT VARIABLE HERE)
    city_data_reduced = city_data_raw[:, :, final_feature_indices]

    df_city = DataFrame(
        city_data_reduced, 
        var_names=final_features_base, 
        missing_flag=MISSING_FLAG,
        analysis_mode='multiple'
    )
    
    city_phase1_pkl = os.path.join(city_out_dir, f"1_{city_name}_Linear_Results.pkl")
    city_report_txt = os.path.join(city_out_dir, f"1_{city_name}_Linear_Report.txt")
    
    if os.path.exists(city_phase1_pkl) and not TEST_MODE:
        print(f"[RESUME] Loading existing Linear Results for {city_name}...")
        with open(city_phase1_pkl, 'rb') as f:
            city_results = pickle.load(f)
        c_p_matrix = city_results['p_matrix']
        c_val_matrix = city_results['val_matrix']
        c_graph = city_results['graph']
        try:
            c_optimal_alpha = float(city_results['pc_alpha'])
        except (KeyError, TypeError):
            c_optimal_alpha = ALPHAS[0]
    else:
        print(f"Running Linear ParCorr for {city_name}...")
        
        c_parcorr = ParCorr(significance='analytic')
        c_pcmci = PCMCI(dataframe=df_city, cond_ind_test=c_parcorr, verbosity=0) 
        
        c_link_assumptions = c_pcmci._set_link_assumptions(None, tau_min=0, tau_max=TAU_MAX)
        for j, t_name in enumerate(final_features_base):
            if t_name not in insar_vars: 
                for i in insar_indices_base:       
                    for tau in range(1, TAU_MAX + 1): 
                        if (i, -tau) in c_link_assumptions[j]:
                            del c_link_assumptions[j][(i, -tau)]
                            
        city_results = c_pcmci.run_pcmciplus(tau_max=TAU_MAX, pc_alpha=ALPHAS, link_assumptions=c_link_assumptions, max_conds_px=MAX_CONDS)
        
        with open(city_phase1_pkl, 'wb') as f:
            pickle.dump(city_results, f)
            
        c_p_matrix = city_results['p_matrix']
        c_val_matrix = city_results['val_matrix']
        c_graph = city_results['graph']
        try:
            c_optimal_alpha = float(city_results['pc_alpha'])
        except (KeyError, TypeError):
            c_optimal_alpha = ALPHAS[0]

    # Generate City Linear Report
    c_report = f"""====================================================
CITY: {city_name} | PANEL-PCMCI+ LOCAL LINEAR RESULTS
====================================================
Total Independent Samples (Pixels): {P_city}
Tested Alpha Grid: {ALPHAS}
OPTIMAL ALPHA SELECTED (via AIC): {c_optimal_alpha}

--- STRICT CAUSAL DRIVERS FOR SUBSIDENCE ({target_name}) ---
Format: [Driver] (tau) ----> [Target] | Effect Size | p-value
-------------------------------------------------------------------------
"""
    c_significant_links = []
    for i, var_name in enumerate(final_features_base):
        for tau in range(TAU_MAX + 1):
            if c_graph[i, final_target_idx_base, tau] == '-->': 
                c_significant_links.append({
                    'var': var_name, 'tau': tau,
                    'val': c_val_matrix[i, final_target_idx_base, tau], 'pval': c_p_matrix[i, final_target_idx_base, tau]
                })

    c_significant_links.sort(key=lambda x: abs(x['val']), reverse=True)
    if not c_significant_links:
        c_report += "\nNO SIGNIFICANT DIRECTED CAUSAL LINKS FOUND."
    else:
        for link in c_significant_links:
            c_report += f"{link['var']:<32} (tau={link['tau']:<2}) ----> {target_name:<12} | {link['val']:>8.4f} | {link['pval']:.3e}\n"

    with open(city_report_txt, 'w') as f:
        f.write(c_report)
    print(f"Saved {city_name} Linear Report.")

print("\n" + "="*80)
print(f"ALL PROCESSES COMPLETED SUCCESSFULLY! Check the results folder:\n{OUTPUT_DIR}")
print("="*80)

sys.stdout.flush()
