"""NeuroScan-AI: Streamlit Clinical Decision Support & Explainability Interface.

Architected and developed by Adarsh Dwivedi (@Adarsh-Intelligence).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from src import config as C
from src import data_loader as D
from src import gradcam as G
from src import preprocessing as P

# ----------------------------------------------------------------------------- Page Config
st.set_page_config(
    page_title="NeuroScan-AI | Brain Tumor MRI Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------- Custom CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .badge-dev {
        display: inline-block;
        background: rgba(99, 102, 241, 0.25);
        color: #a5b4fc;
        border: 1px solid rgba(165, 180, 252, 0.3);
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
        backdrop-filter: blur(8px);
    }
    
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
    }
    
    .metric-label {
        font-size: 0.82rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
    
    .diagnosis-box {
        border-radius: 14px;
        padding: 1.5rem;
        margin: 1.2rem 0;
        border: 1px solid;
    }
    
    .diag-glioma {
        background: rgba(239, 68, 68, 0.12);
        border-color: rgba(239, 68, 68, 0.4);
        color: #fca5a5;
    }
    
    .diag-meningioma {
        background: rgba(245, 158, 11, 0.12);
        border-color: rgba(245, 158, 11, 0.4);
        color: #fcd34d;
    }
    
    .diag-no_tumor {
        background: rgba(16, 185, 129, 0.12);
        border-color: rgba(16, 185, 129, 0.4);
        color: #6ee7b7;
    }
    
    .diag-pituitary {
        background: rgba(139, 92, 246, 0.12);
        border-color: rgba(139, 92, 246, 0.4);
        color: #c4b5fd;
    }
    
    .info-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------- Helpers & Caching
CLASS_DETAILS = {
    "glioma": {
        "color": "#EF4444",
        "css": "diag-glioma",
        "title": "Glioma",
        "desc": "A primary brain tumor originating from glial cells (astrocytes, oligodendrocytes, or ependymal cells). Typically infiltrative within brain parenchyma.",
        "icon": "⚠️",
    },
    "meningioma": {
        "color": "#F59E0B",
        "css": "diag-meningioma",
        "title": "Meningioma",
        "desc": "Tumor arising from the meninges (membranous layers surrounding the brain and spinal cord). Mostly benign and extra-axial, showing distinct dural tail sign.",
        "icon": "🔶",
    },
    "no_tumor": {
        "color": "#10B981",
        "css": "diag-no_tumor",
        "title": "No Tumor (Normal MRI)",
        "desc": "Healthy brain parenchyma with no identifiable focal neoplastic mass or abnormal contrast enhancement.",
        "icon": "✅",
    },
    "pituitary": {
        "color": "#8B5CF6",
        "css": "diag-pituitary",
        "title": "Pituitary Tumor",
        "desc": "Neoplasm localized in the sella turcica / pituitary gland region. Commonly adenomas that may impact optic chiasm or endocrine regulation.",
        "icon": "🟣",
    },
}


@st.cache_resource(show_spinner=False)
def load_keras_model(model_path: str):
    import keras

    return keras.models.load_model(model_path)


@st.cache_data(show_spinner=False)
def get_available_models() -> list[str]:
    if not C.MODELS_DIR.exists():
        return []
    models = [p.parent.name for p in C.MODELS_DIR.glob("*/model.keras")]
    return sorted(models)


@st.cache_data(show_spinner=False)
def get_class_names_and_mapping():
    try:
        return D.load_class_mapping()
    except Exception:
        classes = ["glioma", "meningioma", "no_tumor", "pituitary"]
        return classes, {c: i for i, c in enumerate(classes)}


@st.cache_data(show_spinner=False)
def get_sample_images():
    """Pick 1 real test sample for each class."""
    try:
        test_df = D.load_split("test")
        samples = {}
        for cls in sorted(test_df["class_name"].unique()):
            row = test_df[test_df["class_name"] == cls].iloc[0]
            samples[cls] = str(C.PROJECT_ROOT / row["filepath"])
        return samples
    except Exception:
        return {}


# ----------------------------------------------------------------------------- Header
st.markdown(
    """
    <div class="main-header">
        <div class="badge-dev">👨‍💻 Developed by Adarsh Dwivedi</div>
        <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.02em;">
            🧠 NeuroScan-AI
        </h1>
        <p style="margin: 0.6rem 0 0 0; color: #cbd5e1; font-size: 1.05rem; max-width: 850px;">
            Deep learning-based clinical decision support system for brain tumor MRI categorization 
            with visual Grad-CAM feature attribution heatmaps.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------- Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Model & Settings")
    available_models = get_available_models()

    if not available_models:
        st.error("No saved model found in `artifacts/models/`.")
        st.stop()

    # Pre-select full models first if available, otherwise smoke model
    default_idx = 0
    for pref in ["efficientnet", "custom_cnn"]:
        if pref in available_models:
            default_idx = available_models.index(pref)
            break

    selected_model_name = st.selectbox(
        "Active Deep Learning Model",
        options=available_models,
        index=default_idx,
        help="Select which trained checkpoint to load for inference",
    )

    model_dir = C.MODELS_DIR / selected_model_name
    model_path = model_dir / "model.keras"

    with st.spinner("Loading neural network weights..."):
        model = load_keras_model(str(model_path))

    class_names, class_mapping = get_class_names_and_mapping()

    # Read training summary if available
    summary_file = model_dir / "training_summary.json"
    if summary_file.exists():
        with open(summary_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        st.caption(f"**Architecture**: `{selected_model_name}`")
        if meta.get("smoke"):
            st.warning("⚠️ **Note**: This is a 1-epoch sanity smoke checkpoint.")
        if "params" in meta:
            st.caption(f"**Total Parameters**: `{meta['params']:,}`")

    st.markdown("---")
    st.markdown("### 🔬 Explainability (Grad-CAM)")
    cam_alpha = st.slider("Heatmap Blend Opacity (α)", min_value=0.1, max_value=0.9, value=0.45, step=0.05)
    colormap_choice = st.selectbox("Colormap", ["JET", "TURBO", "VIRIDIS", "MAGMA"], index=0)

    st.markdown("---")
    st.markdown("### 👨‍💻 Author Information")
    st.markdown(
        """
        **Adarsh Dwivedi**  
        • GitHub: [@Adarsh-Intelligence](https://github.com/Adarsh-Intelligence)  
        • Repo: [NeuroScan-AI](https://github.com/Adarsh-Intelligence/NeuroScan-AI)  
        """
    )

# ----------------------------------------------------------------------------- Tabs
tab_diag, tab_batch, tab_metrics, tab_about = st.tabs(
    ["🔬 MRI Diagnosis & Grad-CAM", "📂 Dataset Explorer", "📊 Model Performance", "ℹ️ Architecture & Methodology"]
)

# ============================================================================= TAB 1: Diagnosis
with tab_diag:
    col_input, col_pred = st.columns([1, 1.2], gap="large")

    with col_input:
        st.markdown("#### 1. Provide an MRI Scan")
        input_mode = st.radio("Choose Input Method:", ["Select Pre-loaded Sample", "Upload Your Own Scan"], horizontal=True)

        input_image_arr = None
        true_label = None

        if input_mode == "Select Pre-loaded Sample":
            samples = get_sample_images()
            if samples:
                selected_sample = st.selectbox(
                    "Choose an MRI slice from test set:",
                    options=list(samples.keys()),
                    format_func=lambda c: f"Patient Scan — {CLASS_DETAILS[c]['icon']} {c.replace('_', ' ').title()}",
                )
                sample_file = samples[selected_sample]
                true_label = selected_sample
                input_image_arr = P.load_image(sample_file)
                st.caption(f"📍 Loaded from: `{sample_file}`")
            else:
                st.info("Splits not prepared yet. Please upload an MRI scan below.")

        else:
            uploaded_file = st.file_uploader("Upload Brain MRI scan (JPG, PNG, JPEG):", type=["jpg", "jpeg", "png", "bmp"])
            if uploaded_file is not None:
                img_pil = Image.open(uploaded_file).convert("RGB")
                img_pil = img_pil.resize((224, 224), Image.Resampling.BILINEAR)
                input_image_arr = np.asarray(img_pil, dtype=np.uint8)

        if input_image_arr is not None:
            st.image(input_image_arr, caption="Input MRI (Resized 224×224 RGB)", use_container_width=True)

    with col_pred:
        st.markdown("#### 2. Diagnostic Prediction & Grad-CAM")
        if input_image_arr is None:
            st.info("👈 Please select or upload an MRI scan on the left to run inference.")
        else:
            with st.spinner("Analyzing scan and computing gradients..."):
                cam, class_idx, probs = G.gradcam_heatmap(model, input_image_arr.astype(np.float32))
                pred_class = class_names[class_idx]
                confidence = float(probs[class_idx])

                # Colormap conversion for overlay
                cv2_cmap = getattr(cv2, f"COLORMAP_{colormap_choice}", cv2.COLORMAP_JET)
                heat = cv2.resize(cam.astype("float32"), (input_image_arr.shape[1], input_image_arr.shape[0]))
                heat = cv2.applyColorMap(np.uint8(255 * heat), cv2_cmap)[..., ::-1]
                blended = np.uint8(np.clip((1 - cam_alpha) * input_image_arr + cam_alpha * heat, 0, 255))

            details = CLASS_DETAILS.get(pred_class, {"title": pred_class, "css": "diag-glioma", "desc": "", "icon": "🔍"})

            # Result Banner
            st.markdown(
                f"""
                <div class="diagnosis-box {details['css']}">
                    <div style="font-size: 0.9rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">
                        Diagnosis Assessment
                    </div>
                    <div style="font-size: 1.8rem; font-weight: 800; margin: 0.3rem 0;">
                        {details['icon']} {details['title']}
                    </div>
                    <div style="font-size: 1.05rem;">
                        Model Confidence: <strong>{confidence * 100:.2f}%</strong>
                        {f' | Ground Truth: <strong>{true_label.replace("_", " ").title()}</strong>' if true_label else ''}
                    </div>
                    <p style="margin: 0.6rem 0 0 0; font-size: 0.88rem; opacity: 0.9;">
                        {details['desc']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Class Probability Distribution
            st.markdown("##### Class Probability Distribution")
            prob_df = pd.DataFrame({
                "Class": [c.replace("_", " ").title() for c in class_names],
                "Probability": [float(p) for p in probs],
            })
            st.bar_chart(prob_df.set_index("Class"), color="#6366F1", height=180)

            # Grad-CAM Side-by-Side
            st.markdown("##### Visual Attention (Grad-CAM Overlay)")
            gcol1, gcol2 = st.columns(2)
            with gcol1:
                st.image(blended, caption="Grad-CAM Overlay", use_container_width=True)
            with gcol2:
                st.image(heat, caption=f"Heatmap ({colormap_choice})", use_container_width=True)

            st.caption(
                "💡 **Clinical Context**: Grad-CAM highlights coarse convolutional feature activation regions that contributed "
                "most strongly to this classification score. It is an explanatory attribution, not an automated surgical segmentation."
            )

# ============================================================================= TAB 2: Dataset Explorer
with tab_batch:
    st.markdown("### 📂 Dataset Overview & Leakage Prevention")

    summary_path = C.EXPLORATION_DIR / "dataset_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            dsum = json.load(f)

        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{dsum["total_files"]:,}</div><div class="metric-label">Total MRI Images</div></div>', unsafe_allow_html=True)
        with mcol2:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{len(dsum["class_names"])}</div><div class="metric-label">Tumor Classes</div></div>', unsafe_allow_html=True)
        with mcol3:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{dsum["duplicate_groups"]:,}</div><div class="metric-label">Independent Patient Groups</div></div>', unsafe_allow_html=True)
        with mcol4:
            st.markdown(f'<div class="metric-card"><div class="metric-val">0</div><div class="metric-label">Cross-Split Leakage</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        dist_col1, dist_col2 = st.columns(2)
        with dist_col1:
            st.markdown("#### New Leakage-Safe Stratified Splits")
            split_img = C.EXPLORATION_DIR / "class_distribution_new_split.png"
            if split_img.exists():
                st.image(str(split_img), use_container_width=True)

        with dist_col2:
            st.markdown("#### Original Roboflow Split (Showed Leakage)")
            orig_img = C.EXPLORATION_DIR / "class_distribution_original.png"
            if orig_img.exists():
                st.image(str(orig_img), use_container_width=True)

        st.markdown(
            """
            > **Why this matters**: In medical imaging, the same patient scan often produces multiple adjacent slices or Roboflow augmentations. 
            > Placing copies in both `train` and `test` yields artificially inflated accuracy. NeuroScan-AI isolates duplicate clusters into 
            > mutually exclusive folds using **StratifiedGroupKFold**.
            """
        )

# ============================================================================= TAB 3: Model Performance
with tab_metrics:
    st.markdown("### 📊 Test Set Evaluation & Benchmarks")
    metric_files = sorted(C.METRICS_DIR.glob("*/test_metrics.json"))

    if not metric_files:
        st.info("No test metrics found yet. Run `python -m src.evaluate --model <model_name>` to generate complete test metrics.")
    else:
        perf_data = []
        for mf in metric_files:
            with open(mf, "r", encoding="utf-8") as f:
                d = json.load(f)
            perf_data.append({
                "Model": mf.parent.name,
                "Accuracy": f"{d.get('accuracy', 0)*100:.2f}%",
                "Macro F1": f"{d.get('macro_f1', 0):.4f}",
                "Parameters": f"{d.get('params', 0):,}",
                "Size (MB)": f"{d.get('model_size_mb', 0):.1f} MB",
                "Latency (Batch 1)": f"{d.get('inference', {}).get('batch1_ms_median', 0):.1f} ms",
            })
        st.dataframe(pd.DataFrame(perf_data), use_container_width=True)

        # Show confusion matrix for selected model if exists
        cm_path = C.PLOTS_DIR / selected_model_name / "test_confusion_matrix.png"
        if cm_path.exists():
            st.markdown("#### Confusion Matrix")
            st.image(str(cm_path), width=550)

# ============================================================================= TAB 4: About
with tab_about:
    st.markdown("### ℹ️ Architecture & Clinical Pipeline Details")
    st.markdown(
        """
        #### 1. Custom CNN
        - 4 convolutional blocks: `[Conv2D(3×3) → BatchNorm → ReLU] × 2 → MaxPooling → Dropout`
        - Filter sizes: 32 → 64 → 128 → 256
        - Fully convolutional classification head: `GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.4) → Dense(4, Softmax)`
        
        #### 2. EfficientNetB0 Transfer Learning
        - Two-stage progressive training strategy:
          1. **Stage A (Feature Extraction)**: ImageNet backbone weights completely frozen. Top head trained with `lr=1e-3`.
          2. **Stage B (Fine-Tuning)**: Blocks 6 & 7 unfreezed (`lr=1e-5`) with `BatchNormalization` layers preserved in inference mode.
          3. Selection decided strictly on **Validation Macro-F1**.
        
        #### 3. Diagnostic Classes Covered
        - **Glioma**: Intra-axial brain neoplasms arising from supporting glial cells.
        - **Meningioma**: Extra-axial neoplasms originating from arachnoid cap cells of the meninges.
        - **Pituitary Adenoma**: Sellar masses often causing endocrine abnormalities and bitemporal hemianopsia.
        - **No Tumor**: Healthy brain MRI slices without mass effect or pathological enhancement.
        
        ---
        **Developed by:** [Adarsh Dwivedi](https://github.com/Adarsh-Intelligence)  
        **Repository:** [NeuroScan-AI](https://github.com/Adarsh-Intelligence/NeuroScan-AI)
        """
    )
