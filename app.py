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

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os

# 页面设置
st.set_page_config(
    page_title="HBsAg Clearance Prediction",
    page_icon="🧬",
    layout="wide"
)

# 加载模型
@st.cache_resource
def load_model():
    model = joblib.load("model/LGB_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    final_features = joblib.load("model/final_selected_features.pkl")
    numeric_features = joblib.load("model/numeric_features.pkl")
    return model, scaler, final_features, numeric_features

model, scaler, final_features, numeric_features = load_model()

# 数据准备
def prepare_input(baseline_hbsag, week12_hbsag, week12_alt, week24_hbsag):
    week12_hbsag = 0.01 if week12_hbsag <= 0.05 else week12_hbsag
    week24_hbsag = 0.01 if week24_hbsag <= 0.05 else week24_hbsag

    alt_hbsag_ratio = week12_alt / week12_hbsag
    log_diff = np.log10(baseline_hbsag / week24_hbsag)
    hbsag_decline_1 = 1 if log_diff >= 1 else 0

    df = pd.DataFrame({
        "HBsAgbaseline": [baseline_hbsag],
        "ALT12w_HBsAg12w": [alt_hbsag_ratio],
        "HBsAg24wdecline_1": [hbsag_decline_1]
    })

    # 标准化
    numeric_df = pd.DataFrame(0, index=[0], columns=numeric_features)
    for col in numeric_features:
        numeric_df[col] = df[col]
    scaled = pd.DataFrame(scaler.transform(numeric_df), columns=numeric_df.columns)

    for col in numeric_features:
        df[col] = scaled[col]

    return df, alt_hbsag_ratio, log_diff

# SHAP解释图
def plot_shap(input_df, original_values):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(input_df)

    shap_exp = shap.Explanation(
        values=shap_values.values[0, :],
        base_values=explainer.expected_value,
        data=original_values,
        feature_names=input_df.columns
    )

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_exp, show=False)
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode()
    return image_base64

# 页面输入
st.title("HBsAg Clearance Prediction at 48 Weeks")

col1, col2, col3, col4 = st.columns(4)
with col1:
    baseline_hbsag = st.number_input("Baseline HBsAg (IU/mL)", 0.0, 25000.0, 10.0)
with col2:
    week12_hbsag = st.number_input("Week 12 HBsAg (IU/mL)", 0.0, 25000.0, 10.0)
with col3:
    week24_hbsag = st.number_input("Week 24 HBsAg (IU/mL)", 0.0, 25000.0, 10.0)
with col4:
    week12_alt = st.number_input("Week 12 ALT (IU/L)", 0, 5000, 40)

# 预测按钮
if st.button("Predict"):
    input_df, alt_hbsag_ratio, log_diff = prepare_input(baseline_hbsag, week12_hbsag, week12_alt, week24_hbsag)
    prob = model.predict_proba(input_df)[0, 1]

    st.subheader("Prediction Result")
    st.metric("Probability of HBsAg Clearance", f"{prob:.1%}")

    if prob < 0.3:
        st.error("Low probability")
    elif prob < 0.7:
        st.warning("Moderate probability")
    else:
        st.success("High probability")

    st.subheader("Calculated Variables")
    st.write(pd.DataFrame({
        "Variable": ["Baseline HBsAg", "ALT/Week12_HBsAg", "log10(Baseline/Week24) ≥ 1"],
        "Value": [f"{baseline_hbsag:.2f}", f"{alt_hbsag_ratio:.4f}", "Yes" if log_diff >= 1 else "No"]
    }))

    shap_img = plot_shap(input_df, input_df.values[0])
    st.subheader("SHAP Explanation")
    st.markdown(f"<img src='data:image/png;base64,{shap_img}' style='width: 100%;'>", unsafe_allow_html=True)

    st.caption("Note: This is a prediction tool to assist clinical decisions.")
