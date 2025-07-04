import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import pickle
import pandas as pd

# ---------------------
# 页面设置
# ---------------------

st.set_page_config(page_title="PVL 预测系统 (PVL Prediction System)", layout="wide")
st.title("PVL 预测系统 (PVL Prediction System)")
st.write("🔬 本系统可加载 MRI 图像，手动勾画 ROI，提取影像特征并结合临床信息预测 PVL 风险。")
st.write("This system allows you to load MRI, draw ROI, extract radiomics features, and predict PVL risk based on clinical data.")

st.markdown(
    """
    <div style="border: 2px solid #ff4b4b; padding: 12px; border-radius: 10px; background-color: #2c2c2c;">
        <strong style="color: #ff4b4b; font-size: 18px;">⚠️ Disclaimer / 免责声明</strong><br>
        <span style="color: white;">
        This web application is intended <strong>for research and educational purposes only</strong>.<br>
        Please <strong>do not upload any real patient data</strong> containing Protected Health Information (PHI).<br>
        All processing is performed client-side, and <strong>no data is stored or transmitted to a server</strong>.<br>
        The tool is <strong>not approved for clinical use</strong>, and should not be used to guide medical decisions.<br><br>
        本网站仅供科研与教学使用，请勿上传任何含有真实患者身份信息的影像资料。<br>
        所有数据处理均在本地浏览器中完成，<strong>不会上传或存储到服务器</strong>。<br>
        本工具<strong>尚未获得临床认证</strong>，不得用于指导实际医疗决策。
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------
# 初始化状态变量
# ---------------------
if 'radscore' not in st.session_state:
    st.session_state.radscore = None
if 'sequence_features' not in st.session_state:
    st.session_state.sequence_features = {}

# ---------------------
# 工具函数
# ---------------------
def load_dicom_series(folder_path):
    try:
        if not os.path.exists(folder_path):
            st.warning(f"❌ 文件夹不存在: {folder_path}")
            return []
        files = [f for f in os.listdir(folder_path) if not f.startswith('.') and os.path.isfile(os.path.join(folder_path, f))]
        if not files:
            st.warning(f"⚠️ 未找到DICOM文件: {folder_path}")
            return []
        slices = [pydicom.dcmread(os.path.join(folder_path, f)) for f in files]
        slices.sort(key=lambda x: float(x.InstanceNumber) if hasattr(x, 'InstanceNumber') else 0)
        images = [s.pixel_array.astype(np.float32) for s in slices]
        return images
    except Exception as e:
        st.error(f"❌ 加载DICOM序列失败: {str(e)}")
        return []

def extract_features_from_roi(image, mask):
    roi_pixels = image[mask > 0]
    features = {
        "mean": float(np.mean(roi_pixels)),
        "std": float(np.std(roi_pixels)),
        "max": float(np.max(roi_pixels)),
        "min": float(np.min(roi_pixels)),
        "median": float(np.median(roi_pixels)),
        "energy": float(np.sum(roi_pixels ** 2)),
        "iqr": float(np.percentile(roi_pixels, 75) - np.percentile(roi_pixels, 25)),
        "skewness": float(pd.Series(roi_pixels.flatten()).skew()),
        "kurtosis": float(pd.Series(roi_pixels.flatten()).kurt()),
        "uniformity": float(np.sum(np.square(np.histogram(roi_pixels, bins=32, range=(0, 1), density=True)[0]))),
    }
    return features

def calculate_radscore(features_dict):
    weights = {
        "T2WI_mean": 0.15, "T2WI_std": 0.12, "T1WI_max": 0.08, "T1WI_min": -0.07,
        "T1WI_median": 0.1, "T2WI_energy": 0.1, "FLAIR_iqr": 0.07,
        "FLAIR_skewness": -0.05, "FLAIR_kurtosis": 0.06, "FLAIR_uniformity": 0.12
    }
    radscore = sum(features_dict.get(k, 0) * w for k, w in weights.items())
    return radscore

# ---------------------
# 侧边栏 - 图像路径和处理
# ---------------------
st.sidebar.header("📂 上传 DICOM 序列 (Upload DICOM Folders)")
t1_path = st.sidebar.text_input("T1WI 文件夹路径 (T1WI folder path)")
t2_path = st.sidebar.text_input("T2WI 文件夹路径 (T2WI folder path)")
flair_path = st.sidebar.text_input("FLAIR 文件夹路径 (FLAIR folder path)")

st.sidebar.markdown("---")
st.sidebar.header("🛠️ 图像预处理 (Image Preprocessing)")
normalize = st.sidebar.checkbox("归一化 (Normalize)", True)
contrast = st.sidebar.slider("对比度调整 (Contrast)", 0.0, 2.0, 1.0)

# 加载图像
t1_images = load_dicom_series(t1_path) if t1_path else []
t2_images = load_dicom_series(t2_path) if t2_path else []
flair_images = load_dicom_series(flair_path) if flair_path else []

# ---------------------
# 图像展示
# ---------------------
cols = st.columns(3)
image_arrays = [t1_images, t2_images, flair_images]
sequence_labels = ["T1WI", "T2WI", "FLAIR"]
current_images = []

for i, (images, label) in enumerate(zip(image_arrays, sequence_labels)):
    with cols[i]:
        if images:
            st.markdown(f"### {label}")
            idx = st.slider(f"{label} 当前帧 (Frame)", 0, len(images) - 1, 0, key=label)
            current_image = images[idx]
            if normalize:
                current_image = (current_image - current_image.min()) / (current_image.max() - current_image.min())
            current_image = np.clip(current_image * contrast, 0, 1)
            st.image(current_image, caption=f"{label} 第 {idx+1} 张", use_column_width=True)
            current_images.append(current_image)
        else:
            current_images.append(None)

# ---------------------
# 手动勾画 ROI 并提取特征
# ---------------------
st.markdown("---")
st.markdown("## ✍️ 手动勾画 ROI (Draw ROI Manually for Each Sequence)")

for i, label in enumerate(sequence_labels):
    if current_images[i] is not None:
        st.markdown(f"### 🧠 {label} - 勾画 ROI")
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=2,
            stroke_color="white",
            background_image=Image.fromarray((current_images[i] * 255).astype(np.uint8)).convert("RGB"),
            height=current_images[i].shape[0],
            width=current_images[i].shape[1],
            drawing_mode="freedraw",
            key=f"canvas_{label}"
        )

        if canvas_result.image_data is not None:
            if st.button(f"💾 保存 {label} ROI"):
                mask = np.array(canvas_result.image_data[:, :, 0] > 0, dtype=np.uint8)
                np.save(f"roi_{label}.npy", mask)
                st.success(f"✅ {label} ROI已保存")

            if st.button(f"🔍 提取 {label} 特征"):
                mask = np.array(canvas_result.image_data[:, :, 0] > 0, dtype=np.uint8)
                mask = np.array(Image.fromarray(mask).resize(
                    (current_images[i].shape[1], current_images[i].shape[0]),
                    resample=Image.NEAREST
                ))

                features = extract_features_from_roi(current_images[i], mask)
                prefixed_features = {f"{label}_{k}": v for k, v in features.items()}
                st.session_state.sequence_features.update(prefixed_features)
                st.success(f"✅ 已提取 {label} 特征")

# ---------------------
# RadScore 计算
# ---------------------
if st.button("🎯 计算 RadScore (Calculate RadScore)"):
    if not st.session_state.sequence_features:
        st.warning("⚠️ 请至少绘制一个 ROI 并提取特征")
    else:
        try:
            st.session_state.radscore = calculate_radscore(st.session_state.sequence_features)
            st.success(f"✅ RadScore = {st.session_state.radscore:.4f}")
        except Exception as e:
            st.error(f"❌ RadScore计算失败: {str(e)}")

# ---------------------
# 临床信息输入
# ---------------------
st.markdown("---")
st.markdown("## 🩺 临床信息 (Clinical Information)")

col1, col2 = st.columns(2)
with col1:
    ga = st.number_input("胎龄(天) (Gestational Age in days)", 140, 300, 259)
    hypoglycemia = st.selectbox("新生儿低血糖 (Neonatal Hypoglycaemia)", ["否 (No)", "是 (Yes)"])
    ischemia = st.selectbox("缺血缺氧史 (History of ischaemia and hypoxia)", ["否 (No)", "是 (Yes)"])
    infection = st.selectbox("母婴感染 (Maternal or neonatal infection)", ["否 (No)", "是 (Yes)"])

with col2:
    ventricles = st.selectbox("侧脑室扩大变形 (Enlarged deformity of lateral ventricles)", ["否 (No)", "是 (Yes)"])
    myelination = st.selectbox("髓鞘化延迟 (Delayed myelination)", ["否 (No)", "是 (Yes)"])
    abnormal_signal = st.selectbox("侧脑室周围异常信号 (Abnormal signal around lateral ventricle)", ["否 (No)", "是 (Yes)"])

# ---------------------
# PVL 预测逻辑
# ---------------------
if st.button("🚀 预测 PVL 风险"):
    try:
        if not st.session_state.sequence_features:
            st.warning("⚠️ 请先提取至少一个序列的 ROI 特征")
            st.stop()

        # 构造临床特征（使用与模型训练一致的字段名）
        clinical_features = {
            "gestational_age(days)": ga,
            "Neonatal_hypoglycaemia": 1 if hypoglycemia == "是 (Yes)" else 0,
            "History_of_ischaemia_and_hypoxia": 1 if ischemia == "是 (Yes)" else 0,
            "Maternal_or_neonatal_infection": 1 if infection == "是 (Yes)" else 0,
            "Enlarged_deformity_of_the_lateral_ventricles": 1 if ventricles == "是 (Yes)" else 0,
            "Delayed_myelination": 1 if myelination == "是 (Yes)" else 0,
            "Abnormal_signal_around_lateral_ventricle": 1 if abnormal_signal == "是 (Yes)" else 0,
        }

        # 添加 RadScore
        if st.session_state.radscore is None:
            st.warning("⚠️ 请先计算 RadScore")
            st.stop()
        clinical_features["RadScore"] = st.session_state.radscore

        # 最终模型输入特征
        all_features = clinical_features

        # 加载模型
        with open("logistic_model.pkl", "rb") as f:
            model_data = pickle.load(f)

        if isinstance(model_data, dict):
            model = model_data.get("model")
            feature_names = model_data.get("features")
            if model is None or feature_names is None:
                raise ValueError("模型文件缺少 'model' 或 'features' 键")
        else:
            raise ValueError("模型文件格式错误，期望为包含 'model' 和 'features' 的字典")

        # 检查是否缺失所需特征
        missing_features = [f for f in feature_names if f not in all_features]
        if missing_features:
            st.error(f"❌ 缺失以下模型所需特征: {missing_features}")
            st.stop()

        # 构造输入
        input_df = pd.DataFrame([[all_features[f] for f in feature_names]], columns=feature_names)

        # 预测
        pred_proba = model.predict_proba(input_df)[:, 1][0]
        st.success(f"✅ PVL 预测风险为: {pred_proba:.3f}")

        # 展示调试信息
        with st.expander("🛠️ 调试模式 - 查看模型输入特征 (Debug: Show Input Features)"):
            st.write("📋 模型使用特征 (Features used by model):", feature_names)
            st.write("📋 实际输入特征字典:", all_features)
            st.dataframe(input_df)

    except Exception as e:
        st.error(f"❌ 模型预测失败: {str(e)}")
