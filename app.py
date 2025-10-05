import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px

# ----------------------------
# Load models
# ----------------------------
artifacts = joblib.load("planet_pipeline.joblib")
clf = artifacts['classifier']
reg = artifacts['regressor']
features_clf = artifacts['features_clf']
features_reg = artifacts['features_reg']

# ----------------------------
# Column aliases (extend as needed)
# ----------------------------
column_aliases = {
    # classifier
    'koi_period':'koi_period', 'pl_orbper':'koi_period',
    'koi_duration':'koi_duration',
    'koi_depth':'koi_depth',
    'koi_prad':'koi_prad', 'pl_rade':'koi_prad', 'pl_radj':'koi_prad',
    'koi_teq':'koi_teq', 'pl_eqt':'koi_teq',
    'koi_insol':'koi_insol', 'pl_insol':'koi_insol',
    'koi_model_snr':'koi_model_snr', 'snr':'koi_model_snr',
    'koi_steff':'koi_steff', 'st_teff':'koi_steff',
    'koi_srad':'koi_srad', 'st_rad':'koi_srad',
    'koi_slogg':'koi_slogg', 'st_logg':'koi_slogg',
    'koi_kepmag':'koi_kepmag', 'kep_mag':'koi_kepmag',
    'koi_fpflag_nt':'koi_fpflag_nt', 'fpflag_nt':'koi_fpflag_nt',
    'koi_fpflag_ss':'koi_fpflag_ss', 'fpflag_ss':'koi_fpflag_ss',
    'koi_fpflag_co':'koi_fpflag_co', 'fpflag_co':'koi_fpflag_co',
    'koi_fpflag_ec':'koi_fpflag_ec', 'fpflag_ec':'koi_fpflag_ec',
    # mass for density
    'mass':'mass', 'pl_bmasse':'mass', 'st_mass':'mass',
    # regression features
    'pl_rade':'pl_rade', 'pl_orbsmax':'pl_orbsmax', 'pl_eqt':'pl_eqt',
    'st_mass':'st_mass', 'st_rad':'st_rad', 'st_teff':'st_teff'
}

# ----------------------------
# Helper functions
# ----------------------------
def ensure_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def map_columns(df):
    df_renamed = df.copy()
    for col in df.columns:
        if col in column_aliases:
            df_renamed.rename(columns={col: column_aliases[col]}, inplace=True)
    return df_renamed

def probability_label(prob):
    if prob >= 0.85: return "CONFIRMED"
    elif prob >= 0.5: return "Probably a planet"
    elif prob <= 0.15: return "FALSE POSITIVE"
    else: return "Probably not a planet"

def compute_density(row):
    try:
        if not pd.isna(row.get('mass')) and not pd.isna(row.get('koi_prad')):
            radius_m = row['koi_prad'] * 6371e3
            volume = (4/3) * np.pi * radius_m**3
            mass_kg = row['mass'] * 5.972e24
            return float(mass_kg / volume)
    except:
        return np.nan
    return np.nan

def planet_type(row):
    if not pd.isna(row.get('density')):
        d = row['density']
        if d >= 5000: return "Rocky"
        elif d >= 1500: return "Icy"
        else: return "Gaseous"
    return np.nan

def compute_habitability(row):
    if pd.isna(row.get('koi_teq')) or pd.isna(row.get('koi_prad')):
        return np.nan
    temp_score = 1 - abs(row['koi_teq'] - 288)/100
    size_score = 1 - abs(row['koi_prad'] - 1)/2
    density_score = 0
    if not pd.isna(row.get('density')):
        density_score = (row['density'] - 1000)/5000
        density_score = min(max(density_score,0),1)
    scores = [s for s in [temp_score, size_score, density_score] if not pd.isna(s)]
    return float(np.mean(scores)) if scores else np.nan

def clean_features(df, features):
    X = pd.DataFrame()
    for f in features:
        if f in df.columns:
            X[f] = pd.to_numeric(df[f], errors='coerce')
        else:
            X[f] = 0
    if not X.empty:
        X = X.fillna(X.median())
        X = X.fillna(0)
    return X

# ---------- small helper change ----------
def probability_label(prob):
    # handle NaN safely
    if pd.isna(prob):
        return "Insufficient data"
    if prob >= 0.85: return "CONFIRMED"
    elif prob >= 0.5: return "Probably a planet"
    elif prob <= 0.15: return "FALSE POSITIVE"
    else: return "Probably not a planet"


# ---------- robust fill_missing_mass ----------
def fill_missing_mass(df, mask_planet):
    mask_missing = mask_planet & df['mass'].isna()
    if not mask_missing.any():
        return df

    sub = df.loc[mask_missing]
    if sub.empty:
        return df

    X_reg = clean_features(sub, features_reg)

    if X_reg.empty:
        print("⚠️ Regression skipped: no regression features available")
        return df

    try:
        preds_log = reg.predict(X_reg.values)
        preds = np.expm1(preds_log)   # remove if regressor was trained on raw mass
        df.loc[sub.index, 'mass'] = preds
        print(f"✅ Regression filled {len(preds)} masses")
    except Exception as e:
        print(f"❌ Regression failed: {e}")

    return df



# ---------- robust compute_and_display ----------
def compute_and_display(df_input):
    """
    Safe compute + annotate pipeline.
    - always returns a DataFrame (never None)
    - never calls streamlit directly (pure function)
    - adds these columns: prob_planet, prediction, density, planet_type, habitability_index
    """
    # work on a copy
    df = map_columns(df_input.copy())

    # ensure 'mass' exists
    if 'mass' not in df.columns:
        df['mass'] = np.nan

    # coerce numeric for relevant columns (only if present)
    cols_to_coerce = list(dict.fromkeys(features_clf + features_reg + ['mass','koi_prad','koi_teq','density']))
    for c in cols_to_coerce:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # --- Classification features (cleaned) ---
    X_clf = clean_features(df, features_clf)

    # If no classifier features available -> return dataframe with placeholder columns
    if X_clf.shape[1] == 0:
        df['prob_planet'] = np.nan
        df['prediction'] = "Insufficient features"
        df['density'] = np.nan
        df['planet_type'] = np.nan
        df['habitability_index'] = np.nan
        return df

    # If there are zero rows, just return empty annotated df
    if X_clf.shape[0] == 0:
        df['prob_planet'] = np.nan
        df['prediction'] = "No rows"
        df['density'] = np.nan
        df['planet_type'] = np.nan
        df['habitability_index'] = np.nan
        return df

    # fill NaNs with medians (then zero)
    try:
        X_clf = X_clf.fillna(X_clf.median()).fillna(0)
    except Exception:
        X_clf = X_clf.fillna(0)

    # Make sure we have 2D numpy array
    try:
        probs = clf.predict_proba(X_clf.values)[:, 1]
    except Exception as e:
        # If prediction fails, attach safe placeholders and return
        df['prob_planet'] = np.nan
        df['prediction'] = f"Prediction failed: {str(e)}"
        df['density'] = np.nan
        df['planet_type'] = np.nan
        df['habitability_index'] = np.nan
        return df

    # assign probabilities
    df['prob_planet'] = probs
    df['prediction'] = df['prob_planet'].apply(probability_label)

    # mask probable planets (>= 0.5)
    mask_planet = df['prob_planet'] >= 0.5
    mask_planet = mask_planet.fillna(False)  # ensure boolean no-NaN

    # fill missing mass for probable planets
    try:
        df = fill_missing_mass(df, mask_planet)
    except Exception:
        # don't let regression failure break the pipeline
        pass

    # ensure numeric before any derived calcs
    for c in ['mass', 'koi_prad', 'koi_teq']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # initialize derived columns
    df['density'] = np.nan
    df['planet_type'] = np.nan
    df['habitability_index'] = np.nan

    # compute density/type/habitability for probable planets
    if mask_planet.any():
        try:
            df.loc[mask_planet, 'density'] = df.loc[mask_planet].apply(compute_density, axis=1)
        except Exception:
            df.loc[mask_planet, 'density'] = np.nan

        try:
            df.loc[mask_planet, 'planet_type'] = df.loc[mask_planet].apply(planet_type, axis=1)
        except Exception:
            df.loc[mask_planet, 'planet_type'] = np.nan

        try:
            df.loc[mask_planet, 'habitability_index'] = df.loc[mask_planet].apply(compute_habitability, axis=1)
        except Exception:
            df.loc[mask_planet, 'habitability_index'] = np.nan

    # reorder columns: keep predictions at the end
    base_cols = ['prob_planet', 'prediction', 'planet_type', 'habitability_index', 'density']
    other_cols = [c for c in df.columns if c not in base_cols]
    df = df[other_cols + base_cols]

    return df



# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Planet Detector & Analysis", layout="wide")
st.title("🌌 Planet Detector & Habitability Analyzer")

st.sidebar.header("Input Options")
input_mode = st.sidebar.radio("Choose input method:", ["Upload CSV", "Manual Input"], index=0)

# ----------------------------
# CSV input
# ----------------------------
if input_mode == "Upload CSV":
    st.subheader("Upload CSV")
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())
        if 'mass' not in df.columns:
            df['mass'] = np.nan

        if st.button("Predict All"):
            try:
                df_output = compute_and_display(df)
                st.subheader("Prediction Summary")
                st.dataframe(df_output[['prob_planet','prediction','planet_type','habitability_index','density']])

                st.subheader("Probability Distribution")
                fig1 = px.histogram(df_output, x='prob_planet', nbins=20,
                                    title="Probability of Planet Being Real")
                st.plotly_chart(fig1)

                if 'koi_teq' in df_output.columns:
                    st.subheader("Habitability vs Temperature")
                    fig2 = px.scatter(df_output, x='koi_teq', y='habitability_index',
                                      color='prediction', hover_data=['koi_prad'])
                    st.plotly_chart(fig2)

                if 'ra' in df_output.columns and 'dec' in df_output.columns:
                    st.subheader("Galaxy Map (RA vs DEC)")
                    fig3 = px.scatter(df_output, x='ra', y='dec',
                                      size='prob_planet', color='habitability_index',
                                      hover_data=['kepoi_name'])
                    st.plotly_chart(fig3)

                df_output.to_csv("predicted_planets.csv", index=False)
                st.success("Saved predictions to predicted_planets.csv")

                if not any(col in df.columns for col in features_reg):
                    st.warning("⚠️ No regression features found in your CSV. Mass imputation may be skipped.")

            except Exception as e:
                st.error(f"Prediction failed. Check your CSV data.\nError: {str(e)}")

# ----------------------------
# Manual input
# ----------------------------
else:
    st.subheader("Manual Input")
    user_input = {}

    for f in features_clf:
        user_input[f] = st.sidebar.number_input(f"Enter value for {f}", value=0.0)

    # Regression features
    for f in features_reg:
        user_input[f] = st.sidebar.number_input(f"Enter value for {f}", value=0.0)

    # Optional mass
    mass_val = st.sidebar.number_input("Optional: Enter mass (M_Earth)", value=0.0)
    user_input['mass'] = mass_val if mass_val > 0 else np.nan

    features_df = pd.DataFrame([user_input])

    if st.button("Predict"):
        try:
            out = compute_and_display(features_df)
            st.write(f"Probability: {out['prob_planet'].iloc[0]:.2f}")
            st.write(f"Prediction: {out['prediction'].iloc[0]}")
            st.dataframe(out[['prob_planet','prediction','planet_type','habitability_index','density']])
        except Exception as e:
            st.error(f"Prediction failed. Check your input values.\nError: {str(e)}")
