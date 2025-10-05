# 🌌 Project NOVA: Exoplanet Classifier & Analyzer

**AI-powered exoplanet classification and mass prediction** using NASA Kepler data.  
Includes an interactive Streamlit web app for scientists and enthusiasts to explore exoplanet candidates.

---

## 🚀 Overview
Project NOVA performs two main tasks:  
1. **Exoplanet Classification** – predicts whether a candidate is a real exoplanet or a false positive using a RandomForest classifier trained on confirmed Kepler data.  
2. **Property Analysis** – estimates missing planet mass using a RandomForest regressor and computes derived metrics such as density, planet type (Rocky/Icy/Gaseous), and habitability index.

The pipeline integrates robust ML models with interactive visualizations to help researchers quickly analyze exoplanet candidates.

> **Note:** The model is designed to predict mass and compute habitability, but due to time constraints and working solo, this functionality is partially demonstrated in the current demo. The underlying architecture fully supports these predictions.

---

## 🧩 Files
| File | Description |
|------|-------------|
| `app.py` | Streamlit web app for classification & regression |
| `AI_Pipeline.py` | Notebook for model training and evaluation (optional) |
| `planet_pipeline.joblib` | Serialized models + feature metadata |
| `requirements.txt` | Python package dependencies |
| `LICENSE` | MIT license |
| `.gitignore` | Files ignored by GitHub |

---

## 📄 Required CSV Columns for App Input
To use the CSV upload feature in the Streamlit app, your file should include the following columns:

| Column | Notes |
|--------|-------|
| koi_period | Orbital period of the planet candidate |
| koi_duration | Transit duration |
| koi_depth | Transit depth |
| koi_prad | Planet radius in Earth radii |
| koi_teq | Planet equilibrium temperature |
| koi_insol | Insolation flux |
| koi_model_snr | Signal-to-noise ratio from transit model |
| koi_steff | Stellar effective temperature |
| koi_srad | Stellar radius |
| koi_slogg | Stellar surface gravity |
| koi_kepmag | Kepler magnitude |
| koi_fpflag_nt | Non-transit false positive flag |
| koi_fpflag_ss | Stellar system false positive flag |
| koi_fpflag_co | Centroid offset false positive flag |
| koi_fpflag_ec | Eclipsing binary false positive flag |
| mass | Optional: planet mass (Earth masses). If not provided, regression model will predict it for probable planets |

> **Important:** Missing or misnamed columns may result in prediction errors. Columns not listed are ignored.

---

## 🎥 Demo
[30-second Project Demo](https://youtu.be/BSkNEZ1Y5jg) – interactive exploration: upload CSV or enter parameters manually, then visualize probability, habitability, and RA/DEC maps.

---

## ✨ Features
- **Safe CSV Upload & Manual Input** – handle both full datasets and custom parameters  
- **Exoplanet Classification** – probability of being a real planet  
- **Missing Mass Prediction** – regression fills gaps for probable planets  
- **Derived Metrics** – density, planet type, habitability index  
- **Interactive Visualizations** – probability histograms, habitability vs temperature scatter, RA/DEC galaxy maps  

---

## 🛰️ Data & Resources
- Dataset: [NASA Kepler Exoplanet Search Results](https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results?resource=download)  
- Frameworks & Libraries: Python, Scikit-learn, Pandas, Streamlit, Plotly  

---

## 🏆 Credits
Developed by **Melika Rezaeyan** for the NASA Space Apps Challenge 2025.  
Data courtesy of NASA Kepler Mission.  
