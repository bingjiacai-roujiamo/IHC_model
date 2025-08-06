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

# Set page configuration for Streamlit
st.set_page_config(
    page_title="Hepatitis B Surface Antigen Clearance Prediction",
    page_icon="🧬",
    layout="wide"
)

# Function to load model and preprocessing objects
@st.cache_resource
def load_model_and_preprocessors():
    model_dir = "model"
    
    # Load the trained model and preprocessing objects
    lgb_model = joblib.load(os.path.join(model_dir, "LGB_model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    final_selected_features = joblib.load(os.path.join(model_dir, "final_selected_features.pkl"))
    numeric_features = joblib.load(os.path.join(model_dir, "numeric_features.pkl"))
    categorical_features = joblib.load(os.path.join(model_dir, "categorical_features.pkl"))
    
    return lgb_model, scaler, final_selected_features, numeric_features, categorical_features

# Load the model and preprocessors
lgb_model, scaler, final_selected_features, numeric_features, categorical_features = load_model_and_preprocessors()

# Define display names for features (used in SHAP plot)
display_feature_names = {
    'HBsAgbaseline': "Baseline HBsAg (IU/mL)",
    'ALT12w_HBsAg12w': "Week 12 ALT / Week 12 HBsAg Ratio",
    'HBsAg24wdecline_1': "HBsAg Decline ≥ 1 log10 at Week 24"
}

# Function to prepare input data for prediction
def prepare_input_data(baseline_hbsag, week12_hbsag, week12_alt, week24_hbsag):
    # Calculate derived features
    alt_hbsag_ratio = week12_alt / week12_hbsag if week12_hbsag > 0 else 0
    
    # Calculate HBsAg24wdecline_1 (binary: 1 if decline ≥ 1 log10, 0 otherwise)
    try:
        if week24_hbsag < baseline_hbsag:
            log_diff = np.log10(baseline_hbsag / week24_hbsag)
            hbsag_d1 = 1 if log_diff >= 1 else 0
        else:
            hbsag_d1 = 0
    except:
        hbsag_d1 = 0  # Default to 0 if calculation fails
    
    # Create input dataframe with the three required features
    input_df = pd.DataFrame({
        'HBsAgbaseline': [baseline_hbsag],
        'ALT12w_HBsAg12w': [alt_hbsag_ratio],
        'HBsAg24wdecline_1': [hbsag_d1]
    })
    
    # Ensure only the selected features are used
    input_df = input_df[final_selected_features]
    
    # Keep an unscaled copy for display purposes
    display_df = input_df.copy()
    
    # Standardize numeric features for model input
    numeric_cols = [col for col in input_df.columns if col in numeric_features]
    
    if numeric_cols:
        # Create a temporary dataframe with all numeric features for scaling
        temp_df = pd.DataFrame(0, index=[0], columns=numeric_features)
        
        # Fill in the input values
        for feature in numeric_cols:
            temp_df[feature] = input_df[feature]
        
        # Apply the scaler
        temp_df_scaled = pd.DataFrame(
            scaler.transform(temp_df),
            index=temp_df.index,
            columns=temp_df.columns
        )
        
        # Update input_df with scaled values
        for feature in numeric_cols:
            input_df[feature] = temp_df_scaled[feature]
    
    return input_df, display_df

# Function to make prediction
def predict(input_df):
    # Get the probability of HBsAg clearance (positive class)
    pred_proba = lgb_model.predict_proba(input_df)[0, 1]
    return pred_proba

# Function to generate SHAP explanation
def generate_shap_explanation(input_df, display_df):
    # Create SHAP explainer
    explainer = shap.TreeExplainer(lgb_model)
    
    # Calculate SHAP values
    shap_values = explainer(input_df)
    
    # Use unscaled values for display
    display_values = display_df.values[0]
    
    # Map feature names to display names
    feature_names = [display_feature_names.get(col, col) for col in input_df.columns]
    
    # Create SHAP explanation object for visualization
    example_shap_values = shap.Explanation(
        values=shap_values.values[0, :, 1],
        base_values=explainer.expected_value[1],
        data=display_values,
        feature_names=feature_names
    )
    
    # Generate waterfall plot
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(example_shap_values, show=False)
    plt.title("Feature Impact on Prediction", fontsize=14)
    plt.tight_layout()
    
    # Convert plot to base64 image
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return image_base64

# Streamlit UI
st.title("Hepatitis B Surface Antigen Clearance Prediction")
st.write("This tool predicts the probability of hepatitis B surface antigen clearance at 48 weeks based on baseline, week 12, and week 24 measurements.")

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
        st.caption("ℹ️ Enter 0.05 if Week 12 HBsAg is ≤ 0.05. Very low values will be adjusted to 0.01.")
    
    with col3:
        week24_hbsag = st.number_input("Week 24 HBsAg (IU/mL)", 
                                       min_value=0.0, 
                                       max_value=25000.0, 
                                       value=10.0, 
                                       step=1.0)
        st.caption("ℹ️ Enter 0.05 if Week 24 HBsAg is ≤ 0.05. Very low values will be adjusted to 0.01.")
    
    with col4:
        week12_alt = st.number_input("Week 12 ALT (IU/L)", 
                                     min_value=0, 
                                     max_value=5000, 
                                     value=40, 
                                     step=1)

if st.button("Calculate Prediction"):
    # Adjust very low HBsAg values
    week12_hbsag_adj = 0.01 if week12_hbsag <= 0.05 else week12_hbsag
    week24_hbsag_adj = 0.01 if week24_hbsag <= 0.05 else week24_hbsag
    
    # Prepare input data
    input_df, display_df = prepare_input_data(baseline_hbsag, week12_hbsag_adj, week12_alt, week24_hbsag_adj)
    
    # Extract feature values for display
    alt_hbsag_ratio = display_df['ALT12w_HBsAg12w'].values[0]
    hbsag_d1 = display_df['HBsAg24wdecline_1'].values[0]
    hbsag_d1_display = "Yes" if hbsag_d1 == 1 else "No"
    
    # Make prediction
    prediction = predict(input_df)
    
    # Display results
    st.subheader("Prediction Results")
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
        st.subheader("Model Features")
        feature_data = {
            "Feature": ["Baseline HBsAg", "Week 12 ALT / Week 12 HBsAg Ratio", "HBsAg Decline ≥ 1 log10 at Week 24"],
            "Value": [f"{baseline_hbsag:.2f} IU/mL", f"{alt_hbsag_ratio:.4f}", hbsag_d1_display]
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
    - The width of each bar shows the strength of that feature's impact on the prediction
    """)
    
    # Disclaimer
    st.caption("Note: This is a prediction tool to assist clinical decision-making, not a substitute for clinical judgment.")

# Footer
st.markdown("---")
st.caption("© 2025 - HBV Clearance Prediction Tool")
