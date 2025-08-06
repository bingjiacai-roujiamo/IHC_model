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

# 页面配置
st.set_page_config(
    page_title="HBsAg Clearance Prediction",
    page_icon="🧬",
    layout="wide"
)

# 加载模型及预处理器
@st.cache_resource
def load_model_and_preprocessors():
    model_dir = "model"
    lgb_model = joblib.load(os.path.join(model_dir, "LGB_model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    final_selected_features = joblib.load(os.path.join(model_dir, "final_selected_features.pkl"))
    numeric_features = joblib.load(os.path.join(model_dir, "numeric_features.pkl"))
    return lgb_model, scaler, final_selected_features, numeric_features

lgb_model, scaler, final_selected_features, numeric_features = load_model_and_preprocessors()

# 构造输入数据
def prepare_input_data(baseline_hbsag, week12_hbsag, week12_alt, week24_hbsag):
    week12_hbsag = 0.01 if week12_hbsag <= 0.05 else week12_hbsag
    week24_hbsag = 0.01 if week24_hbsag <= 0.05 else week24_hbsag

    alt_hbsag_ratio = week12_alt / week12_hbsag if week12_hbsag > 0 else 0
    log_decline = np.log10(baseline_hbsag / week24_hbsag) if week24_hbsag > 0 else 0
    hbsag_decline_flag = 1 if log_decline >= 1 else 0

    input_df = pd.DataFrame({
        'HBsAgbaseline': [baseline_hbsag],
        'ALT12w_HBsAg12w': [alt_hbsag_ratio],
        'HBsAg24wdecline_1': [hbsag_decline_flag]
    })

    display_df = input_df.copy()

    if numeric_features:
        temp_df = pd.DataFrame(0, index=[0], columns=numeric_features)
        for feature in input_df.columns:
            if feature in numeric_features:
                temp_df[feature] = input_df[feature]
        temp_df_scaled = pd.DataFrame(
            scaler.transform(temp_df),
            columns=numeric_features
        )
        for feature in numeric_features:
            input_df[feature] = temp_df_scaled[feature]

    return input_df, display_df, alt_hbsag_ratio, log_decline, hbsag_decline_flag

# 模型预测
def predict(input_df):
    return lgb_model.predict_proba(input_df)[0, 1]

# SHAP解释图
def generate_shap_explanation(input_df, display_df):
    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer(input_df)

    example_shap = shap.Explanation(
        values=shap_values.values[0, :],
        base_values=explainer.expected_value,
        data=display_df.values[0],
        feature_names=input_df.columns.tolist()
    )

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(example_shap, show=False)
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    return image_base64

# 页面UI
st.title("HBsAg Clearance Prediction")
st.write("This tool predicts the probability of HBsAg clearance at 48 weeks based on patient data.")

with st.container():
    st.subheader("Enter Patient Data")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        baseline_hbsag = st.number_input("Baseline HBsAg (IU/mL)", 0.0, 25000.0, 10.0, step=1.0)

    with col2:
        week12_hbsag = st.number_input("Week 12 HBsAg (IU/mL)", 0.0, 25000.0, 10.0, step=1.0)
        st.caption("ℹ️ If ≤ 0.05, please enter 0.05 (automatically adjusted to 0.01 for computation).")

    with col3:
        week24_hbsag = st.number_input("Week 24 HBsAg (IU/mL)", 0.0, 25000.0, 10.0, step=1.0)

    with col4:
        week12_alt = st.number_input("Week 12 ALT (IU/L)", 0, 5000, 40, step=1)

if st.button("Predict"):
    input_df, display_df, ratio, log_decline, flag = prepare_input_data(
        baseline_hbsag, week12_hbsag, week12_alt, week24_hbsag
    )

    prob = predict(input_df)

    st.subheader("Prediction Result")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.metric("Predicted Probability", f"{prob:.1%}")
        if prob < 0.3:
            st.error("Low likelihood of HBsAg clearance")
        elif prob < 0.7:
            st.warning("Moderate likelihood of HBsAg clearance")
        else:
            st.success("High likelihood of HBsAg clearance")

    with col2:
        st.subheader("Feature Calculation")
        st.table(pd.DataFrame({
            "Feature": ["Baseline HBsAg", "Week 12 ALT / Week 12 HBsAg", "log10(Baseline HBsAg / Week 24 HBsAg) ≥ 1"],
            "Value": [f"{baseline_hbsag:.2f}", f"{ratio:.4f}", "Yes" if flag == 1 else "No"]
        }))

    # SHAP解释图
    shap_image = generate_shap_explanation(input_df, display_df)
    st.subheader("Model Interpretation")
    st.markdown(f"<img src='data:image/png;base64,{shap_image}' style='width: 100%;'>", unsafe_allow_html=True)
    st.info("""
    **Interpretation:**
    - Red bars increase probability
    - Blue bars decrease probability
    - Width reflects impact magnitude
    """)

st.markdown("---")
st.caption("© 2025 - HBV Clearance Prediction Tool")


