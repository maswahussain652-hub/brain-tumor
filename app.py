"""
NeuroScan 3D: Production Medical AI Cockpit.
3D Brain Tumor Detection, Multiplanar Slice Exploration,
and Interactive 3D Mesh Surface Rendering using Streamlit and Plotly.
"""

from typing import Dict, Any
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from config import (
    DEFAULT_VOLUME_SHAPE,
    DEFAULT_VOXEL_SPACING,
    CLINICAL_CASES,
    THEME,
    MESH_CONFIG,
)
from data_loader import SyntheticBrainGenerator
from inference import MedicalInferenceEngine


# --- Page Configuration ---
st.set_page_config(
    page_title="NeuroScan 3D | Medical AI Cockpit",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Dark Medical CSS Styling ---
st.markdown(
    f"""
    <style>
    /* Sleek Obsidian Black & High-Tech Cyber-Medical Core */
    .stApp {{
        background-color: {THEME.DARK_BG_COLOR};
        color: {THEME.TEXT_LIGHT};
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
    }}
    
    /* Top Header Box with Obsidian Glassmorphism */
    .header-box {{
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.88), rgba(7, 10, 15, 0.96));
        border: 1px solid rgba(0, 242, 254, 0.35);
        border-radius: 18px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 0 25px rgba(0, 242, 254, 0.12), 0 12px 36px rgba(0, 0, 0, 0.75);
        backdrop-filter: blur(12px);
    }}
    
    .header-title {{
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #FFFFFF 0%, #38BDF8 50%, #00F2FE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }}
    
    .header-subtitle {{
        font-size: 0.95rem;
        color: #94A3B8;
        margin-top: 6px;
    }}

    .brain-logo-badge {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 2.4rem;
        background: linear-gradient(135deg, rgba(0, 242, 254, 0.15), rgba(56, 189, 248, 0.25));
        border: 1px solid rgba(0, 242, 254, 0.5);
        border-radius: 16px;
        padding: 8px 14px;
        box-shadow: 0 0 20px rgba(0, 242, 254, 0.35);
    }}
    
    /* High-Tech Obsidian Black Metric Cards */
    .metric-card {{
        background: linear-gradient(145deg, #111827, #0B0F19);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 14px;
        text-align: center;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.60);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }}
    .metric-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(0, 242, 254, 0.25);
    }}
    
    .metric-card-vol {{
        border: 1px solid rgba(255, 51, 102, 0.45);
    }}
    .metric-card-vol:hover {{
        border-color: #FF3366;
        box-shadow: 0 0 22px rgba(255, 51, 102, 0.35);
    }}
    
    .metric-card-diam {{
        border: 1px solid rgba(245, 158, 11, 0.45);
    }}
    .metric-card-diam:hover {{
        border-color: #F59E0B;
        box-shadow: 0 0 22px rgba(245, 158, 11, 0.35);
    }}
    
    .metric-card-centroid {{
        border: 1px solid rgba(56, 189, 248, 0.45);
    }}
    .metric-card-centroid:hover {{
        border-color: #38BDF8;
        box-shadow: 0 0 22px rgba(56, 189, 248, 0.35);
    }}
    
    .metric-card-dice {{
        border: 1px solid rgba(0, 242, 254, 0.55);
    }}
    .metric-card-dice:hover {{
        border-color: #00F2FE;
        box-shadow: 0 0 24px rgba(0, 242, 254, 0.45);
    }}
    
    .metric-card-risk {{
        border: 1px solid rgba(168, 85, 247, 0.45);
    }}
    .metric-card-risk:hover {{
        border-color: #A855F7;
        box-shadow: 0 0 22px rgba(168, 85, 247, 0.35);
    }}
    
    .metric-title {{
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        margin-bottom: 6px;
    }}
    .metric-value {{
        font-size: 1.65rem;
        font-weight: 700;
        color: #F8FAFC;
    }}
    .metric-sub {{
        font-size: 0.75rem;
        color: #64748B;
        margin-top: 4px;
    }}
    
    /* Neon Status Badges */
    .badge {{
        display: inline-block;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }}
    .badge-cyan {{
        background: rgba(0, 242, 254, 0.15);
        color: #00F2FE;
        border: 1px solid rgba(0, 242, 254, 0.45);
        box-shadow: 0 0 10px rgba(0, 242, 254, 0.2);
    }}
    .badge-blue {{
        background: rgba(56, 189, 248, 0.15);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.45);
    }}
    .badge-ruby {{
        background: rgba(255, 51, 102, 0.18);
        color: #FF3366;
        border: 1px solid rgba(255, 51, 102, 0.45);
        box-shadow: 0 0 10px rgba(255, 51, 102, 0.2);
    }}
    .badge-green {{
        background: rgba(16, 185, 129, 0.18);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.45);
    }}
    .badge-amber {{
        background: rgba(245, 158, 11, 0.18);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.45);
    }}
    
    /* Glowing Neon Pill Buttons */
    .stButton > button, .stDownloadButton > button {{
        background: linear-gradient(135deg, #00F2FE 0%, #0284C7 100%) !important;
        color: #070A0F !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 9999px !important;
        padding: 10px 24px !important;
        box-shadow: 0 0 18px rgba(0, 242, 254, 0.40) !important;
        transition: all 0.2s ease !important;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        background: linear-gradient(135deg, #38BDF8 0%, #00F2FE 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 0 28px rgba(0, 242, 254, 0.65) !important;
    }}
    
    /* Sleek Dark Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: #0A0E17 !important;
        border-right: 1px solid rgba(0, 242, 254, 0.20) !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_inference_engine() -> MedicalInferenceEngine:
    """Instantiates and caches the PyTorch 3D UNet inference engine."""
    return MedicalInferenceEngine()


def create_3d_mesh_figure(
    brain_mesh: Dict[str, Any],
    tumor_mesh: Dict[str, Any],
    show_brain: bool,
    brain_opacity: float,
    show_tumor: bool,
    tumor_opacity: float,
    wireframe: bool,
) -> go.Figure:
    """Builds interactive Plotly 3D mesh figure with lighting shaders."""
    fig = go.Figure()

    # Add Brain Cortex Surface Mesh
    if show_brain and brain_mesh.get("has_mesh", False):
        v_b = brain_mesh["vertices"]
        f_b = brain_mesh["faces"]
        fig.add_trace(
            go.Mesh3d(
                x=v_b[:, 2],
                y=v_b[:, 1],
                z=v_b[:, 0],
                i=f_b[:, 0],
                j=f_b[:, 1],
                k=f_b[:, 2],
                name="Brain Cortex Surface",
                color=THEME.BRAIN_COLOR,
                opacity=brain_opacity,
                flatshading=False,
                lighting=dict(
                    ambient=0.45,
                    diffuse=0.60,
                    fresnel=0.35,
                    specular=0.50,
                    roughness=0.40,
                ),
                showscale=False,
                hoverinfo="name",
            )
        )

    # Add Pathological Tumor Mesh
    if show_tumor and tumor_mesh.get("has_mesh", False):
        v_t = tumor_mesh["vertices"]
        f_t = tumor_mesh["faces"]
        fig.add_trace(
            go.Mesh3d(
                x=v_t[:, 2],
                y=v_t[:, 1],
                z=v_t[:, 0],
                i=f_t[:, 0],
                j=f_t[:, 1],
                k=f_t[:, 2],
                name="3D Tumor Lesion",
                color=THEME.TUMOR_COLOR,
                opacity=tumor_opacity,
                flatshading=wireframe,
                lighting=dict(
                    ambient=0.60,
                    diffuse=0.85,
                    fresnel=0.80,
                    specular=1.00,
                    roughness=0.20,
                ),
                showscale=False,
                hoverinfo="name",
            )
        )

    # Sleek obsidian space black scene camera and bounds
    axis_config = dict(
        showbackground=True,
        backgroundcolor="#05080E",
        gridcolor="#111A29",
        showgrid=True,
        zeroline=False,
        showticklabels=True,
        tickfont=dict(size=10, color="#64748B"),
        title="",
    )

    fig.update_layout(
        paper_bgcolor=THEME.DARK_BG_COLOR,
        plot_bgcolor=THEME.DARK_BG_COLOR,
        scene=dict(
            xaxis=dict(**axis_config, title="Sagittal (X)"),
            yaxis=dict(**axis_config, title="Coronal (Y)"),
            zaxis=dict(**axis_config, title="Axial (Z)"),
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.6, y=1.6, z=1.3),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        margin=dict(l=0, r=0, b=0, t=20),
        legend=dict(
            font=dict(color="#F8FAFC", size=11),
            bgcolor="rgba(11, 15, 25, 0.90)",
            bordercolor="rgba(0, 242, 254, 0.35)",
            borderwidth=1,
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02,
        ),
        height=580,
    )

    return fig


def render_slice_figure(
    volume: np.ndarray,
    prob_map: np.ndarray,
    mask: np.ndarray,
    plane: str,
    slice_idx: int,
    alpha_overlay: float,
    show_gt: bool,
    gt_mask: np.ndarray,
) -> go.Figure:
    """Constructs 2D multiplanar slice plot with overlaid probability mask."""
    # Slicing according to selected orthogonal anatomical plane
    if plane == "Axial (Transverse)":
        slice_idx = np.clip(slice_idx, 0, volume.shape[0] - 1)
        mri_slice = volume[slice_idx, :, :]
        prob_slice = prob_map[slice_idx, :, :]
        mask_slice = mask[slice_idx, :, :]
        gt_slice = gt_mask[slice_idx, :, :]
        xlabel, ylabel = "Left-Right (X)", "Anterior-Posterior (Y)"
    elif plane == "Coronal (Frontal)":
        slice_idx = np.clip(slice_idx, 0, volume.shape[1] - 1)
        mri_slice = volume[:, slice_idx, :]
        prob_slice = prob_map[:, slice_idx, :]
        mask_slice = mask[:, slice_idx, :]
        gt_slice = gt_mask[:, slice_idx, :]
        xlabel, ylabel = "Left-Right (X)", "Superior-Inferior (Z)"
    else:  # Sagittal (Lateral)
        slice_idx = np.clip(slice_idx, 0, volume.shape[2] - 1)
        mri_slice = volume[:, :, slice_idx]
        prob_slice = prob_map[:, :, slice_idx]
        mask_slice = mask[:, :, slice_idx]
        gt_slice = gt_mask[:, :, slice_idx]
        xlabel, ylabel = "Anterior-Posterior (Y)", "Superior-Inferior (Z)"

    fig = go.Figure()

    # 1. Base Grayscale MRI Image
    fig.add_trace(
        go.Heatmap(
            z=mri_slice,
            colorscale="gray",
            showscale=False,
            hoverinfo="z",
            name="MRI T1/T2",
        )
    )

    # 2. Predicted Tumor Probability Heatmap Overlay (Vibrant Neon Ruby/Flame)
    active_overlay = prob_slice.copy()
    active_overlay[mask_slice == 0] = np.nan  # Mask out background

    if np.any(mask_slice > 0):
        fig.add_trace(
            go.Heatmap(
                z=active_overlay,
                colorscale=[
                    [0.0, "rgba(255, 51, 102, 0.0)"],
                    [0.4, "rgba(255, 60, 0, 0.70)"],
                    [1.0, "rgba(255, 0, 85, 0.95)"],
                ],
                zmin=0.0,
                zmax=1.0,
                opacity=alpha_overlay,
                showscale=True,
                colorbar=dict(
                    title=dict(text="Tumor Prob", font=dict(color="#00F2FE", size=10)),
                    tickfont=dict(color="#94A3B8", size=9),
                    len=0.7,
                    thickness=12,
                ),
                name="AI Prediction",
                hoverinfo="z",
            )
        )

    # 3. Ground Truth Contour if enabled (Glowing Cyan Border)
    if show_gt and np.any(gt_slice > 0):
        fig.add_trace(
            go.Contour(
                z=gt_slice,
                contours_coloring="none",
                line=dict(color="#00F2FE", width=2),
                showscale=False,
                name="Ground Truth Border",
                hoverinfo="none",
            )
        )

    fig.update_layout(
        paper_bgcolor="#05080E",
        plot_bgcolor="#030508",
        xaxis=dict(
            title=dict(text=xlabel, font=dict(color="#94A3B8", size=11)),
            tickfont=dict(color="#64748B", size=9),
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text=ylabel, font=dict(color="#94A3B8", size=11)),
            tickfont=dict(color="#64748B", size=9),
            showgrid=False,
            zeroline=False,
            autorange="reversed",
        ),
        margin=dict(l=40, r=20, b=40, t=30),
        height=480,
    )
    return fig


def main():
    # --- Sidebar Controls ---
    st.sidebar.markdown(
        """
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 24px; padding: 12px 14px; background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(0, 242, 254, 0.3); border-radius: 14px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);">
            <span style="font-size: 2.2rem; filter: drop-shadow(0 0 10px rgba(0, 242, 254, 0.6));">🧠</span>
            <div>
                <h3 style="margin: 0; color: #00F2FE; font-size: 1.25rem; font-weight: 800; letter-spacing: 0.5px;">NEUROSCAN 3D</h3>
                <span style="color: #94A3B8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em;">Neural AI Cockpit</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.subheader("Clinical Case Selection")
    case_names = list(CLINICAL_CASES.keys())
    selected_case_name = st.sidebar.selectbox(
        "Patient Scan Profile",
        options=case_names,
        index=0,
    )

    custom_seed = st.sidebar.number_input(
        "Simulation Random Seed",
        min_value=1,
        max_value=99999,
        value=101,
        step=1,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("AI Segmentation Engine")

    prob_threshold = st.sidebar.slider(
        "Tumor Confidence Cutoff",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Probability threshold to classify a voxel as neoplastic tumor tissue.",
    )

    smooth_toggle = st.sidebar.toggle(
        "Post-inference Gaussian Smoothing",
        value=True,
        help="Applies volumetric Gaussian filtering to reduce voxel quantization artifacts.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("3D Mesh Visualization")

    col_s1, col_s2 = st.sidebar.columns(2)
    with col_s1:
        show_brain = st.toggle("Brain Cortex", value=True)
    with col_s2:
        show_tumor = st.toggle("Tumor Mesh", value=True)

    brain_opacity = st.sidebar.slider(
        "Brain Cortex Opacity",
        min_value=0.04,
        max_value=0.50,
        value=THEME.BRAIN_OPACITY,
        step=0.02,
    )

    tumor_opacity = st.sidebar.slider(
        "Tumor Mesh Opacity",
        min_value=0.20,
        max_value=1.00,
        value=THEME.TUMOR_OPACITY,
        step=0.05,
    )

    wireframe_mode = st.sidebar.checkbox("Wireframe Shading", value=False)
    step_size = st.sidebar.select_slider(
        "Marching Cubes Resolution",
        options=[1, 2],
        value=1,
        format_func=lambda x: "High Fidelity (Step 1)" if x == 1 else "Fast Render (Step 2)",
    )

    # --- Header Display with Glowing Brain Logo ---
    inference_engine = get_inference_engine()
    device_label = "CUDA GPU" if inference_engine.device.type == "cuda" else "CPU Accelerator"

    st.markdown(
        f"""
        <div class="header-box">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
                <div style="display: flex; align-items: center; gap: 18px;">
                    <div class="brain-logo-badge">🧠</div>
                    <div>
                        <h1 class="header-title">NeuroScan 3D &mdash; Precision Medical AI</h1>
                        <p class="header-subtitle">Volumetric 3D UNet Brain Pathology &middot; Marching Cubes Iso-Surface Mesh &middot; Multiplanar Reconstruction</p>
                    </div>
                </div>
                <div style="display: flex; gap: 8px;">
                    <span class="badge badge-cyan">Hardware: {device_label}</span>
                    <span class="badge badge-green">PyTorch 3D UNet: Ready</span>
                    <span class="badge badge-blue">Voxel Grid: 64&times;64&times;64</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Data Loading & Inference Execution ---
    with st.spinner("Generating 3D MRI scan and executing PyTorch 3D UNet inference..."):
        generator = SyntheticBrainGenerator(
            shape=DEFAULT_VOLUME_SHAPE,
            spacing=DEFAULT_VOXEL_SPACING,
            random_seed=custom_seed,
        )
        case_data = generator.generate_case(selected_case_name)
        volume = case_data["volume"]
        gt_mask = case_data["ground_truth_mask"]

        # Run Neural Segmentation
        prob_map, pred_mask = inference_engine.segment_volume(
            volume=volume,
            probability_threshold=prob_threshold,
            apply_smoothing=smooth_toggle,
        )

        # Calculate clinical metrics
        metrics = inference_engine.calculate_clinical_metrics(
            pred_mask=pred_mask,
            gt_mask=gt_mask,
            spacing=DEFAULT_VOXEL_SPACING,
        )

        # Extract 3D Isosurfaces
        brain_mesh = inference_engine.extract_marching_cubes_mesh(
            volume_or_mask=case_data["brain_envelope_mask"],
            iso_level=0.50,
            spacing=DEFAULT_VOXEL_SPACING,
            step_size=step_size,
        )

        tumor_mesh = inference_engine.extract_marching_cubes_mesh(
            volume_or_mask=prob_map,
            iso_level=prob_threshold,
            spacing=DEFAULT_VOXEL_SPACING,
            step_size=step_size,
        )

    # --- Metric Cards Row (High-Tech Obsidian Dark) ---
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card metric-card-vol">
                <div class="metric-title">Tumor Volume</div>
                <div class="metric-value" style="color: #FF3366;">{metrics['volume_cm3']} <span style="font-size: 1rem; color:#94A3B8;">cm³</span></div>
                <div class="metric-sub">{metrics['volume_mm3']} mm³ ({metrics['voxel_count']} voxels)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card metric-card-diam">
                <div class="metric-title">Maximum Diameter</div>
                <div class="metric-value" style="color: #F59E0B;">{metrics['longest_diameter_mm']} <span style="font-size: 1rem; color:#94A3B8;">mm</span></div>
                <div class="metric-sub">3D Feret Bounding Dimension</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        cz, cy, cx = metrics["centroid_mm"]
        st.markdown(
            f"""
            <div class="metric-card metric-card-centroid">
                <div class="metric-title">Centroid Coordinates</div>
                <div class="metric-value" style="color: #38BDF8; font-size: 1.25rem;">({cx}, {cy}, {cz})</div>
                <div class="metric-sub">Anatomical Space (mm)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        dice_text = f"{metrics['dice_score'] * 100:.1f}%" if metrics["dice_score"] is not None else "N/A"
        iou_text = f"{metrics['iou_score'] * 100:.1f}%" if metrics["iou_score"] is not None else "N/A"
        st.markdown(
            f"""
            <div class="metric-card metric-card-dice">
                <div class="metric-title">Model Dice Score</div>
                <div class="metric-value" style="color: #00F2FE;">{dice_text}</div>
                <div class="metric-sub" style="color: #38BDF8;">IoU (Jaccard): {iou_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col5:
        risk = metrics["risk_tier"]
        badge_style = "badge-ruby" if "High" in risk else ("badge-green" if "Negative" in risk else "badge-amber")
        st.markdown(
            f"""
            <div class="metric-card metric-card-risk">
                <div class="metric-title">Clinical Assessment</div>
                <div style="margin-top: 6px;"><span class="badge {badge_style}">{risk}</span></div>
                <div class="metric-sub">{case_data['title'].split(':')[0]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # --- Core Interactive Displays: 2D MPR Slice Viewer & 3D Mesh ---
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("### 🔬 Multiplanar Cross-Section (MPR)")
        mpr_tabs = st.tabs(["Axial (Transverse)", "Coronal (Frontal)", "Sagittal (Lateral)"])

        slice_alpha = st.slider(
            "Tumor Probability Heatmap Alpha",
            min_value=0.0,
            max_value=1.0,
            value=0.65,
            step=0.05,
        )
        show_gt_border = st.checkbox("Show Ground Truth Annotation Contour", value=True)

        for tab, plane_name, max_slice, default_slice in zip(
            mpr_tabs,
            ["Axial (Transverse)", "Coronal (Frontal)", "Sagittal (Lateral)"],
            [volume.shape[0] - 1, volume.shape[1] - 1, volume.shape[2] - 1],
            [
                int(metrics["centroid_voxel"][0]) if metrics["voxel_count"] > 0 else volume.shape[0] // 2,
                int(metrics["centroid_voxel"][1]) if metrics["voxel_count"] > 0 else volume.shape[1] // 2,
                int(metrics["centroid_voxel"][2]) if metrics["voxel_count"] > 0 else volume.shape[2] // 2,
            ],
        ):
            with tab:
                curr_slice = st.slider(
                    f"Slice Index ({plane_name})",
                    min_value=0,
                    max_value=max_slice,
                    value=default_slice,
                    key=f"slider_{plane_name}",
                )
                slice_fig = render_slice_figure(
                    volume=volume,
                    prob_map=prob_map,
                    mask=pred_mask,
                    plane=plane_name,
                    slice_idx=curr_slice,
                    alpha_overlay=slice_alpha,
                    show_gt=show_gt_border,
                    gt_mask=gt_mask,
                )
                st.plotly_chart(slice_fig, use_container_width=True)

    with col_right:
        st.markdown("### 🧊 Interactive 3D Mesh Surface Exploration")
        st.caption("Rotate (left click & drag), pan (right click), or zoom (scroll) to inspect 3D lesion topography.")

        mesh_3d_fig = create_3d_mesh_figure(
            brain_mesh=brain_mesh,
            tumor_mesh=tumor_mesh,
            show_brain=show_brain,
            brain_opacity=brain_opacity,
            show_tumor=show_tumor,
            tumor_opacity=tumor_opacity,
            wireframe=wireframe_mode,
        )
        st.plotly_chart(mesh_3d_fig, use_container_width=True)

        # Mesh details
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            if tumor_mesh["has_mesh"]:
                st.info(
                    f"**Tumor Mesh**: {len(tumor_mesh['vertices']):,} vertices | "
                    f"{len(tumor_mesh['faces']):,} faces | "
                    f"Area: {tumor_mesh['surface_area_mm2']} mm²"
                )
            else:
                st.info("No active tumor surface detected above threshold.")
        with col_m2:
            if brain_mesh["has_mesh"]:
                st.info(
                    f"**Brain Envelope**: {len(brain_mesh['vertices']):,} vertices | "
                    f"{len(brain_mesh['faces']):,} faces"
                )

    st.markdown("---")

    # --- Clinical Findings & Data Export Section ---
    st.subheader("📋 Diagnostic Summary & Clinical Mesh Export")
    col_exp1, col_exp2 = st.columns([2, 1], gap="medium")

    with col_exp1:
        st.markdown(f"**Patient Presentation & Imaging Notes:**")
        st.write(case_data["description"])
        st.markdown(
            f"""
            - **Primary Neoplasm Prediction**: {selected_case_name.split(':')[1] if ':' in selected_case_name else selected_case_name}
            - **Estimated Tumor Load**: `{metrics['volume_cm3']} cm³` across `{metrics['voxel_count']}` segmented voxels.
            - **Segmentation Fidelity**: PyTorch UNet3D achieved a **Dice Score of {dice_text}** and **IoU of {iou_text}** relative to reference ground truth annotations.
            """
        )

    with col_exp2:
        st.markdown("**Download 3D Assets & Reports**")
        if tumor_mesh["has_mesh"]:
            obj_content = inference_engine.export_mesh_obj(
                tumor_mesh["vertices"],
                tumor_mesh["faces"],
            )
            st.download_button(
                label="📥 Download 3D Tumor Mesh (.obj)",
                data=obj_content,
                file_name="brain_tumor_3d_mesh.obj",
                mime="text/plain",
                help="Export standard Wavefront OBJ 3D mesh compatible with 3D Slicer, Blender, and 3D printing.",
            )
        else:
            st.button("📥 Download 3D Tumor Mesh (.obj)", disabled=True)

        import json

        report_json = json.dumps(
            {
                "case_title": case_data["title"],
                "description": case_data["description"],
                "volume_shape": list(case_data["shape"]),
                "voxel_spacing_mm": list(case_data["spacing"]),
                "metrics": metrics,
            },
            indent=2,
        )
        st.download_button(
            label="📄 Download Diagnostic Report (JSON)",
            data=report_json,
            file_name="neuroscan_diagnostic_report.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
