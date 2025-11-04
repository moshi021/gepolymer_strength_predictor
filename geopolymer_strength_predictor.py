"""Geopolymer Strength Predictor
"""

import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
from streamlit_pdf_viewer import pdf_viewer # Added for displaying PDFs

# Set page config
st.set_page_config(
    page_title="Geopolymer Strength Predictor",
    layout="wide"
)

# --- Apply custom CSS for white background and black text ---
# This forces a light-theme look, regardless of system settings.
st.markdown("""
<style>
    /* Main app background */
    [data-testid="stAppViewContainer"] {
        background-color: #FFFFFF;
    }

    /* All text in the main app area */
    [data-testid="stAppViewContainer"] * {
        color: #000000;
    }

    /* Explicitly set headers */
    h1, h2, h3, h4, h5, h6 {
        color: #000000;
    }

    /* --- FIX for number inputs --- */
    /* Target the text inside the number input box */
    [data-testid="stNumberInput"] input {
        color: #FFFFFF; /* Make the text white */
    }

    /* Target the + and - buttons (SVGs) inside the number input */
    [data-testid="stNumberInput"] button svg {
        fill: #FFFFFF; /* Make the + and - icons white */
    }
</style>
""", unsafe_allow_html=True)


# --- Function to Display PDF (REMOVED) ---
# The old display_pdf function is no longer needed.

# --- Model Training ---
@st.cache_data(show_spinner="Training model...")
def train_model(data_path="etai final.csv"):
    """
    Loads data, cleans it, and trains the XGBoost model.
    Returns the trained model and the feature data (X).
    """
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        st.error(f"Error: The data file ({data_path}) was not found.")
        st.info("Please make sure 'etai final.csv' is in the same folder as this app.")
        return None, None, None

    data.drop(['Total GHG emission', 'Total Cost(USD)'], axis=1, inplace=True)
    data.dropna(inplace=True)

    features = [
        'Fly Ash', 'GGBFS', 'NaOH_Molarity', 'NaOH amount', 'Sodium Silicate',
        'Extra Water', 'Coarse Agg', 'Fine Agg', 'Coarse/Fine Agg',
        'SuperPlasticizer', 'Curing Time(h)', 'Curing Temp(C)', 'Age of Testing (day)'
    ]
    target = 'Compressive Strength (MPa)'

    X = data[features]
    y = data[target]

    # This fix is kept as good practice, although the error it solved (in SHAP)
    # is no longer relevant since SHAP is removed.
    base_score_val = float(y.mean())

    model = xgb.XGBRegressor(
        random_state=42,
        base_score=base_score_val
    )

    try:
        model.fit(X, y)
    except Exception as e:
        st.error(f"An error occurred during model training: {e}")
        return None, None, None

    return model, X, y

# --- Main App Interface ---

# Load and train the model
model, features_data, target_data = train_model()

if model and features_data is not None and target_data is not None:
    st.title("🧱 Geopolymer Compressive Strength Predictor")
    st.markdown("### Model $R^2 \\approx 95\\%$")
    st.markdown("Use the controls below to input material properties and predict the compressive strength.")

    st.divider()

    # --- Input Fields ---
    st.header("Input Parameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Binders & Aggregates (kg/m³)")
        fly_ash = st.number_input("Fly Ash (kg/m³)", min_value=0.0, value=282.0)
        ggbfs = st.number_input("GGBFS (kg/m³)", min_value=0.0, value=0.0)
        coarse_agg = st.number_input("Coarse Agg (kg/m³)", min_value=0.0, value=1204.0)
        fine_agg = st.number_input("Fine Agg (kg/m³)", min_value=0.0, value=648.0)
        c_f_agg = st.number_input("Coarse/Fine Agg Ratio", min_value=0.0, value=1.85, format="%.2f")

    with col2:
        st.subheader("Activators & Additives")
        naoh_molarity = st.number_input("NaOH Molarity (M)", min_value=0.0, value=10.0)
        naoh_amount = st.number_input("NaOH amount (kg/m³)", min_value=0.0, value=52.0)
        na_silicate = st.number_input("Sodium Silicate (kg/m³)", min_value=0.0, value=131.0)
        extra_water = st.number_input("Extra Water (kg/m³)", min_value=0.0, value=9.0)
        superplasticizer = st.number_input("SuperPlasticizer (kg/m³)", min_value=0.0, value=6.0)

    with col3:
        st.subheader("Curing & Testing")
        curing_time = st.number_input("Curing Time (h)", min_value=0.0, value=100.0)
        curing_temp = st.number_input("Curing Temp (°C)", min_value=0.0, value=38.0)
        age_testing = st.number_input("Age of Testing (day)", min_value=1.0, value=28.0)


    # --- Prediction Display ---
    st.divider()
    st.header("Prediction Result")

    input_data = pd.DataFrame(
        [[
            fly_ash, ggbfs, naoh_molarity, naoh_amount, na_silicate,
            extra_water, coarse_agg, fine_agg, c_f_agg,
            superplasticizer, curing_time, curing_temp, age_testing
        ]],
        columns=features_data.columns
    )

    prediction = model.predict(input_data)

    pred_col, _ = st.columns([1, 2])
    with pred_col:
        st.metric(
            label="Predicted Compressive Strength",
            value=f"{prediction[0]:.2f} MPa"
        )

    st.divider()

    # --- Model Performance Plots (from PDF) ---
    st.header("Model Performance")
    
    plot_col1, plot_col2 = st.columns(2)

    with plot_col1:
        st.subheader("Predicted vs. Actual")
        # Use the new pdf_viewer component
        try:
            pdf_viewer("pred_strength.pdf", height=500)
        except FileNotFoundError:
            st.error("Error: The file 'pred_strength (1).pdf' was not found.")

    with plot_col2:
        st.subheader("SHAP Summary Plot")
        # Use the new pdf_viewer component
        try:
            pdf_viewer("shap.pdf", height=500)
        except FileNotFoundError:
            st.error("Error: The file 'shap.pdf' was not found.")

    # --- Add Footer/Credit ---
    st.divider()
    st.markdown("---")
    st.markdown(
        """
        Built by **Md. Mashiur Rahman**
        
        [LinkedIn](https://www.linkedin.in/mashiur-rahman-ruet/) | [Email](mailto:2013021@student.ruet.ac.bd)
        """
    )
else:
    st.error("Application could not start. Please ensure the 'etai final.csv' file is available and the model can be trained.")

