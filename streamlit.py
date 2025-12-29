#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# 解决matplotlib绘图潜在后端问题
plt.switch_backend('Agg')


# -------------------------- 1. 路径配置（替换为相对路径，兼容所有环境） --------------------------
MODEL_PATH = "svm_best_model.pkl"  # 仅保留文件名，同级目录直接读取
SCALER_PATH = "scaler.pkl"          # 仅保留文件名，同级目录直接读取
FEATURE_NAMES = ["hb", "tt", "siri", "afr", "cea_log", "lvi", "t_stage", "n_stage"]

# -------------------------- 2. 加载模型和标准化器 --------------------------
try:
    # 加载线性SVM模型并验证
    model = joblib.load(MODEL_PATH)
    if not (isinstance(model, SVC) and model.kernel == "linear"):
        st.error("加载的模型不是线性SVM，请检查模型文件路径！")
        st.stop()

    # 加载标准化器
    scaler = joblib.load(SCALER_PATH)
    st.success("模型和标准化器加载成功！")
    st.info(f"特征顺序：{FEATURE_NAMES}")
except Exception as e:
    st.error(f"加载失败：{str(e)}，请检查文件路径是否正确！")
    st.stop()

# -------------------------- 3. 网页输入界面 --------------------------
st.title("Gastric Cancer Liver Metastasis Predictor")
st.subheader("Input Feature Values")

user_input_dict = {}

# 连续型特征输入
user_input_dict["hb"] = st.number_input("Hemoglobin (Hb, g/L)", min_value=30.0, max_value=200.0, value=120.0, step=0.1)
user_input_dict["tt"] = st.number_input("Thrombin Time (TT, sec)", min_value=5.0, max_value=30.0, value=13.0, step=0.1)
user_input_dict["siri"] = st.number_input("Systemic Inflammatory Response Index (SIRI)", min_value=0.0, max_value=10.0, value=1.0, step=0.01)
user_input_dict["afr"] = st.number_input("Neutrophil/Lymphocyte Ratio (AFR)", min_value=0.1, max_value=20.0, value=2.5, step=0.01)

# CEA原始值自动转换为cea_log
cea_original = st.number_input("CEA Original Value (ng/mL)", min_value=0.0, max_value=1000.0, value=1.47, step=0.1)
cea_log = np.log10(cea_original + 1)
user_input_dict["cea_log"] = cea_log
st.caption(f"CEA原始值={cea_original} → cea_log≈{cea_log:.6f}")

# 分类型特征输入
user_input_dict["lvi"] = st.selectbox("Lymphovascular Invasion (LVI)", options=[0, 1], format_func=lambda x: "No (0)" if x==0 else "Yes (1)")
user_input_dict["t_stage"] = st.selectbox("T-stage", options=[1, 2, 3, 4], format_func=lambda x: f"T{x}")
user_input_dict["n_stage"] = st.selectbox("N-stage", options=[0, 1, 2, 3], format_func=lambda x: f"N{x}")

# -------------------------- 4. 构造模型输入 --------------------------
try:
    user_input_list = [user_input_dict[feat] for feat in FEATURE_NAMES]
    input_data = np.array(user_input_list).reshape(1, -1)
except KeyError as e:
    st.error(f"特征缺失：{str(e)}，请检查特征名称是否一致！")
    st.stop()

# -------------------------- 5. 预测逻辑 --------------------------
if st.button("Predict Liver Metastasis Risk"):
    # 标准化输入数据
    input_data_scaled = scaler.transform(input_data)

    # 模型预测
    predicted_class = model.predict(input_data_scaled)[0]
    predicted_proba = model.predict_proba(input_data_scaled)[0]

    # 展示预测结果
    st.write("### Prediction Result")
    if predicted_class == 1:
        risk_prob = predicted_proba[1] * 100
        st.error(f"**Liver Metastasis Risk: High Risk**")
        st.write(f"Probability of Liver Metastasis: {risk_prob:.1f}%")
    else:
        risk_prob = predicted_proba[0] * 100
        st.success(f"**Liver Metastasis Risk: Low Risk**")
        st.write(f"Probability of No Liver Metastasis: {risk_prob:.1f}%")

    # 临床建议
    st.write("### Clinical Advice")
    advice_high = (
        "1. Complete enhanced abdominal CT/MRI within 1 month to confirm metastasis;\n"
        "2. Monitor serological indicators (Hb, CEA) every 2 weeks;\n"
        "3. Consult an oncologist for adjuvant therapy (targeted/chemotherapy);\n"
        "4. Maintain a high-protein diet to improve anemia."
    )
    advice_low = (
        "1. Follow up (abdominal ultrasound + serology) every 3 months;\n"
        "2. Avoid alcohol/spicy foods to reduce gastric irritation;\n"
        "3. Keep regular schedule & moderate exercise;\n"
        "4. Seek medical help for abdominal pain/jaundice/weight loss."
    )
    st.write(advice_high if predicted_class == 1 else advice_low)

    # SHAP Force Plot
    st.write("### Model Interpretation (SHAP Force Plot)")
    masker = shap.maskers.Independent(data=input_data_scaled, max_samples=100)
    explainer = shap.LinearExplainer(model=model, masker=masker, feature_names=FEATURE_NAMES)
    shap_values = explainer.shap_values(input_data_scaled)

    # 绘制并展示力图
    plt.figure(figsize=(12, 5))
    shap.force_plot(
        base_value=explainer.expected_value,
        shap_values=shap_values,
        features=pd.DataFrame(input_data_scaled, columns=FEATURE_NAMES),
        matplotlib=True,
        show=False,
        plot_cmap="RdBu_r"
    )
    plt.title("SHAP Force Plot: Feature Impact on Prediction", fontsize=12)
    plt.tight_layout()
    plt.savefig("svm_shap_force_plot.png", bbox_inches="tight", dpi=300)
    st.image("svm_shap_force_plot.png")
    st.caption("Red: Increase metastasis risk; Blue: Reduce metastasis risk; Length: Contribution degree")


