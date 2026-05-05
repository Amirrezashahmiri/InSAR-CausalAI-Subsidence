# 🌍 A Data-Driven Causal AI Framework for Land Subsidence Attribution
### Integrating InSAR and Hydro-Climatic Data

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Framework: Tigramite](https://img.shields.io/badge/Framework-Tigramite-orange.svg)](https://github.com/jakobrunge/tigramite)

---

## 📌 Overview

This repository provides the **official implementation** of a novel **Causal AI framework** for attributing land subsidence to its underlying hydro-climatic and anthropogenic drivers.

By integrating **MT-InSAR deformation time-series** with **ERA5-Land climate variables**, the framework employs the **J-PCMCI+ algorithm** to uncover **time-lagged causal relationships** across heterogeneous basins.

---

## 🧠 Key Contributions

- Causal inference instead of correlation-based modeling  
- Detection of time-lagged interactions  
- Multi-basin scalable framework  
- Integration of physical constraints into causal discovery  

---

## 🧩 Methodological Innovations

### 🔹 Avoiding Simpson’s Paradox
Uses **Joint Causal Inference (JCI)** with contextual variables:

```
C_ID
```

---

### 🔹 Identification of Hidden Anthropogenic Drivers

Thermodynamic variables used as proxies for groundwater extraction:

- Radiation  
- Skin Temperature  
- Leaf Area Index (LAI)  

---

### 🔹 Geomechanical Diagnostics

| System Type | Feedback | Interpretation |
|------------|--------|----------------|
| Poroelastic | Negative | Resilient system |
| Inelastic | Positive | Irreversible collapse |

---

## 👨‍🔬 Authors

- **Amirreza Shahmiri** — Sultan Qaboos University  
  📧 amirreza.shahmiri@gmail.com  

- **Masoud Ebrahimi Derakhshan** — Iran University of Science and Technology  

- **Seyed Mostafa Siadatmousavi** *(Corresponding Author)* — Iran University of Science and Technology  

---

## ⚙️ Repository Structure

```
scripts/
│
├── 01_gee_data_extraction.js
├── 02_data_alignment_fusion.py
├── 03_causal_preprocessing.py
└── 04_main_causal_discovery.py
```

---

### 🛰️ 1. GEE Data Extraction
**File:** `01_gee_data_extraction.js`

- Extracts ERA5-Land data from Google Earth Engine  
- Aligns temporal resolution with InSAR acquisitions  
- Outputs GeoTIFF stacks  

---

### 🔄 2. Data Alignment & Fusion
**File:** `02_data_alignment_fusion.py`

- Spatial and temporal resampling  
- Integrates:
  - InSAR (HDF5)
  - ERA5 (TIFF)
  - SoilGrids  

**Output:**
```
(Time, Pixels, Features)
```

---

### 🧹 3. Causal Preprocessing
**File:** `03_causal_preprocessing.py`

- Monthly Z-score deseasonalization  
- Multicollinearity reduction  
- Feature selection  
- Conversion to Tigramite format  

---

### 🧠 4. Causal Discovery Engine
**File:** `04_main_causal_discovery.py`

- Implements **True JCI**  
- Uses **J-PCMCI+ algorithm**  
- Enforces physical constraints (e.g., InSAR cannot cause climate)  

**Outputs:**
- Global causal graph  
- Basin-specific causal networks  

---

## 🚀 Getting Started

### 📦 Requirements

- Python 3.8+
- Tigramite
- numpy
- pandas
- scipy
- rasterio
- h5py

---

### 🔧 Installation

```bash
git clone https://github.com/YourUsername/InSAR-CausalAI-Subsidence.git
cd InSAR-CausalAI-Subsidence
pip install -r requirements.txt
```

---

## 📊 Key Findings

### 🌱 Poroelastic Spring Effect
Observed in resilient basins (e.g., Marvdasht, Urmia)  
→ Negative autoregressive feedback stabilizes deformation  

---

### ⚠️ Inelastic Collapse
Observed in Tehran and Bardsir  
→ Positive feedback indicates irreversible compaction  

---

### 💧 AET Paradox

- Reduces subsidence in water-rich basins  
- Acts as proxy for groundwater extraction in arid regions  

---

### 🌧️ Precipitation Duality

- Recharges healthy aquifers  
- In collapsed systems:
  - Causes elastic mass loading  
  - Exacerbates subsidence  

---

## 📖 Citation

```
Shahmiri, A., Derakhshan, M. E., & Siadatmousavi, S. M. (2025).
A Data-Driven Causal AI Framework Integrating InSAR and Hydro-Climatic Data 
for Land Subsidence Attribution. (Under Review)
```

---

## ⚖️ License

This project is licensed under the MIT License.  
See the LICENSE file for details.

---

## ⭐ Contributing

Contributions, issues, and feature requests are welcome.

---

## 📬 Contact

📧 amirreza.shahmiri@gmail.com
