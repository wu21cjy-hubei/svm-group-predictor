import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# 模拟训练数据（简化模型）
data = pd.read_excel("base.xlsx")
data = data[[
    'group',
    'Vertebral intraosseous abscess',
    'involved/normal',
    'Endplate inflammatory reaction line',
    'CRP',
    'N%']]

X = data.drop(columns=["group"])
y = data["group"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
model.fit(X_scaled, y_encoded)

# Streamlit 应用界面
st.title("简易病情预测器")
st.write("请输入以下指标，预测对应的 group（疾病类别）")

# 用户输入
vi_abscess = st.selectbox("Vertebral intraosseous abscess", [0, 1])
involved_ratio = st.number_input("involved/normal", min_value=0.0, max_value=5.0, step=0.01)
ep_line = st.selectbox("Endplate inflammatory reaction line", [0, 1])
crp = st.number_input("CRP (C-反应蛋白值)", min_value=0.0, step=0.1)
n_percent = st.number_input("N% (中性粒细胞百分比)", min_value=0.0, max_value=100.0, step=0.1)

# 模型预测按钮
if st.button("预测 group"):
    input_data = np.array([[vi_abscess, involved_ratio, ep_line, crp, n_percent]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    # 获取概率分布
    probas = model.predict_proba(input_scaled)[0]
    class_labels = label_encoder.inverse_transform(np.arange(len(probas)))

    # 展示结果
    st.success(f"预测结果：group {predicted_label}")
    st.subheader("预测概率：")
    for label, prob in zip(class_labels, probas):
        st.write(f"Group {label}: {prob:.2%}")