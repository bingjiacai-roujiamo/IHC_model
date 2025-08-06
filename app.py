import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
import os
import base64
from io import BytesIO
import lightgbm as lgb

# Set page config
st.set_page_config(
    page_title="Hepatitis B Surface Antigen Clearance Prediction",
    page_icon="🧬",
    layout="wide"
)

# Function to load model and preprocessing objects
@st.cache_resource
def load_model_and_preprocessors():
    model_dir = "model"
    
    # Load model and scaler
    lgb_model = joblib.load(os.path.join(model_dir, "LGB_model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    
    # Define the features used by the new model
    final_selected_features = ['HBsAgbaseline', 'ALT12w_HBsAg12w', 'HBsAg24wdecline_1']
    
    return lgb_model, scaler, final_selected_features

# Load model and preprocessors
lgb_model, scaler, final_selected_features = load_model_and_preprocessors()

# Helper function to create feature name mapping
feature_name_mapping = {
    'HBsAgbaseline': 'Baseline HBsAg',
    'ALT12w_HBsAg12w': 'Week 12 ALT / HBsAg Ratio',
    'HBsAg24wdecline_1': 'HBsAg Decline (log10) ≥1'
}

# Function to prepare input data for prediction
def prepare_input_data(baseline_hbsag, week12_hbsag, week12_alt, week24_hbsag):
    # Adjust very low values of week12_hbsag and week24_hbsag
    week12_hbsag = 0.01 if week12_hbsag <= 0.05 else week12_hbsag
    week24_hbsag = 0.01 if week24_hbsag <= 0.05 else week24_hbsag
    
    # Calculate derived features
    alt_hbsag_ratio = week12_alt / week12_hbsag if week12_hbsag > 0 else 0
    
    # Calculate HBsAg_d1 value
    try:
        if week24_hbsag < baseline_hbsag:
            log_diff = np.log10(baseline_hbsag / week24_hbsag)
            hbsag_d1 = 1 if log_diff >= 1 else 0
        else:
            hbsag_d1 = 0
    except:
        hbsag_d1 = 0   # Default to 0 if calculation fails
    
    # Create input dataframe with the required features
    input_df = pd.DataFrame({
        'HBsAgbaseline': [baseline_hbsag],
        'ALT12w_HBsAg12w': [alt_hbsag_ratio],
        'HBsAg24wdecline_1': [hbsag_d1]
    })
    
    # Create a copy for display (unscaled)
    display_df = input_df.copy()
    
    # Standardize the features using the scaler
    input_df_scaled = pd.DataFrame(
        scaler.transform(input_df),
        columns=input_df.columns
    )
    
    return input_df_scaled, display_df

# Function to make prediction
def predict(input_df):
    # Get prediction probability
    pred_proba = lgb_model.predict_proba(input_df)[0, 1]
    return pred_proba

# Function to generate SHAP explanation
def generate_shap_explanation(input_df, display_df):
    # Create the explainer
    explainer = shap.TreeExplainer(lgb_model)
    
    # Calculate SHAP values
    shap_values = explainer(input_df)
    
    # Map column names for better visualization
    feature_names = [feature_name_mapping.get(col, col) for col in input_df.columns]
    
    # Create a SHAP explanation for the waterfall plot
    example_shap_values = shap.Explanation(
        values=shap_values.values[0, :],
        base_values=explainer.expected_value,
        data=display_df.values,
        feature_names=feature_names
    )
    
    # Create the waterfall plot
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(example_shap_values, show=False)
    plt.title("Feature Impact on Prediction", fontsize=14)
    plt.tight_layout()
    
    # Convert the plot to a base64 encoded image
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return image_base64

# UI
st.title("Hepatitis B Surface Antigen Clearance Prediction")
st.write("This tool predicts the probability of hepatitis B surface antigen clearance at 48 weeks based on baseline and 12-week measurements.")

with st.container():
    st.subheader("Patient Measurements")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        baseline_hbsag = st.number_input("Baseline HBsAg (IU/mL)", 
                                         min_value=0.0, 
                                         max_value=25000.0, 
                                         value=10.0, 
                                         step=1.0)
    
    with col2:
        week12_hbsag = st.number_input("Week 12 HBsAg (IU/mL)", 
                                       min_value=0.0, 
                                       max_value=25000.0, 
                                       value=10.0, 
                                       step=1.0)
        st.caption("ℹ️ Please enter 0.05 if your Week 12 HBsAg value is ≤ 0.05. This tool will adjust the very low values of week12_hbsag to 0.01.")

    with col3:
        week24_hbsag = st.number_input("Week 24 HBsAg (IU/mL)", 
                                       min_value=0.0, 
                                       max_value=25000.0, 
                                       value=10.0, 
                                       step=1.0)
        st.caption("ℹ️ Please enter 0.05 if your Week 24 HBsAg value is ≤ 0.05. This tool will adjust the very low values of week24_hbsag to 0.01.")
    
    with col4:
        week12_alt = st.number_input("Week 12 ALT (IU/L)", 
                                     min_value=0, 
                                     max_value=5000, 
                                     value=40, 
                                     step=1)

if st.button("Calculate Prediction"):
    # Prepare input data
    input_df, display_df = prepare_input_data(baseline_hbsag, week12_hbsag, week12_alt, week24_hbsag)
    
    # Calculate derived features for display
    alt_hbsag_ratio = week12_alt / week12_hbsag if week12_hbsag > 0 else 0
    
    # Calculate HBsAg_d1 value for display
    try:
        if week24_hbsag < baseline_hbsag:
            log_diff = np.log10(baseline_hbsag / week24_hbsag)
            hbsag_d1 = "Yes" if log_diff >= 1 else "No"
        else:
            hbsag_d1 = "No"
    except:
        hbsag_d1 = "No"  
    
    # Make prediction
    prediction = predict(input_df)
    
    # Display results
    st.subheader("Prediction Results")
    
    # Show probability with gauge visualization
    col1, col2 = st.columns([1, 1])
    with col1:
        st.metric("Probability of HBsAg Clearance at 48 Weeks", f"{prediction:.1%}")
        
        if prediction < 0.3:
            st.error("Low probability of HBsAg clearance")
        elif prediction < 0.7:
            st.warning("Moderate probability of HBsAg clearance")
        else:
            st.success("High probability of HBsAg clearance")
    
    with col2:
        # Create feature table
        st.subheader("Calculated Features")
        feature_data = {
            "Feature": [
                "Baseline HBsAg (IU/mL)", 
                "Week 12 ALT / HBsAg Ratio", 
                "log10(Baseline/Week24 HBsAg) ≥1"
            ],
            "Value": [
                f"{baseline_hbsag:.2f}",
                f"{alt_hbsag_ratio:.4f}",
                hbsag_d1
            ]
        }
        st.table(pd.DataFrame(feature_data))
    
    # Generate and display SHAP explanation
    shap_image = generate_shap_explanation(input_df, display_df)
    st.subheader("Model Explanation (SHAP Values)")
    st.markdown(f"<img src='data:image/png;base64,{shap_image}' style='width: 100%;'>", unsafe_allow_html=True)
    
    # Add interpretation guide
    st.info("""
    **Interpretation Guide:**
    - Features in red push the prediction toward HBsAg clearance (higher probability)
    - Features in blue push the prediction away from HBsAg clearance (lower probability)
    - The width of each bar shows how strongly that feature affects the prediction
    """)
    
    # Disclaimer
    st.caption("Note: This is a prediction model and should be used as a tool to aid clinical decision-making, not as a replacement for clinical judgment.")

# Footer
st.markdown("---")
st.caption("© 2025 - HBV Clearance Prediction Tool")
