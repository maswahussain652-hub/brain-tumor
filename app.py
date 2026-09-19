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
    /* Dark Theme Core */
    .stApp {{
        background-color: {THEME.DARK_BG_COLOR};
        color: {THEME.TEXT_LIGHT};
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}
    
    /* Top Header */
    .header-box {{
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(6, 182, 212, 0.25);
        border-radius: 12px;
        padding: 20px 28px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
    }}
    
    .header-title {{
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #38BDF8, #22D3EE, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }}
    
    .header-subtitle {{
        font-size: 0.95rem;
        color: {THEME.TEXT_MUTED};
        margin-top: 6px;
    }}
    
    /* Metric Cards */
    .metric-card {{
        background: rgba(17, 24, 39, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }}
    .metric-card:hover {{
        border-color: {THEME.PRIMARY_CYAN};
        transform: translateY(-2px);
    }}
    .metric-title {{
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {THEME.TEXT_MUTED};
        margin-bottom: 6px;
    }}
    .metric-value {{
        font-size: 1.6rem;
        font-weight: 700;
        color: #F1F5F9;
    }}
    .metric-sub {{
        font-size: 0.75rem;
        color: {THEME.PRIMARY_CYAN};
        margin-top: 4px;
    }}
    
    /* Badges */
    .badge {{
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }}
    .badge-gpu {{
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }}
    .badge-cpu {{
        background: rgba(59, 130, 246, 0.15);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }}
    .badge-critical {{
        background: rgba(239, 68, 68, 0.2);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }}
    .badge-safe {{
        background: rgba(34, 197, 94, 0.2);
        color: #4ADE80;
        border: 1px solid rgba(34, 197, 94, 0.4);
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

    # Clean dark scene camera and bounds
    axis_config = dict(
        showbackground=True,
        backgroundcolor="#070A12",
        gridcolor="#1E293B",
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
            font=dict(color="#CBD5E1", size=11),
            bgcolor="rgba(15, 23, 42, 0.75)",
            bordercolor="rgba(255, 255, 255, 0.1)",
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

    # 2. Predicted Tumor Probability Heatmap Overlay
    active_overlay = prob_slice.copy()
    active_overlay[mask_slice == 0] = np.nan  # Mask out background

    if np.any(mask_slice > 0):
        fig.add_trace(
            go.Heatmap(
                z=active_overlay,
                colorscale=[
                    [0.0, "rgba(255, 80, 80, 0.0)"],
                    [0.5, "rgba(255, 60, 0, 0.65)"],
                    [1.0, "rgba(220, 20, 60, 0.95)"],
                ],
                zmin=0.0,
                zmax=1.0,
                opacity=alpha_overlay,
                showscale=True,
                colorbar=dict(
                    title=dict(text="Tumor Prob", font=dict(color="#94A3B8", size=10)),
                    tickfont=dict(color="#94A3B8", size=9),
                    len=0.7,
                    thickness=12,
                ),
                name="AI Prediction",
                hoverinfo="z",
            )
        )

    # 3. Ground Truth Contour if enabled
    if show_gt and np.any(gt_slice > 0):
        fig.add_trace(
            go.Contour(
                z=gt_slice,
                contours_coloring="none",
                line=dict(color="#10B981", width=2),
                showscale=False,
                name="Ground Truth Border",
                hoverinfo="none",
            )
        )

    fig.update_layout(
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#070A12",
        xaxis=dict(
            title=dict(text=xlabel, font=dict(color="#64748B", size=11)),
            tickfont=dict(color="#64748B", size=9),
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text=ylabel, font=dict(color="#64748B", size=11)),
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
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
            <span style="font-size: 2rem;">🧠</span>
            <div>
                <h3 style="margin: 0; color: #38BDF8; font-size: 1.25rem;">NeuroScan 3D</h3>
                <span style="color: #64748B; font-size: 0.75rem;">Medical AI System</span>
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

    # --- Header Display ---
    inference_engine = get_inference_engine()
    device_label = "CUDA GPU" if inference_engine.device.type == "cuda" else "CPU Accelerator"
    badge_class = "badge-gpu" if inference_engine.device.type == "cuda" else "badge-cpu"

    st.markdown(
        f"""
        <div class="header-box">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                <div>
                    <h1 class="header-title">NeuroScan 3D &mdash; Volumetric Brain Tumor AI</h1>
                    <p class="header-subtitle">Deep 3D UNet Pathology Segmentation &middot; Marching Cubes Iso-Surface Extraction &middot; Orthogonal MPR</p>
                </div>
                <div style="display: flex; gap: 8px;">
                    <span class="badge {badge_class}">Hardware: {device_label}</span>
                    <span class="badge badge-gpu">PyTorch 3D UNet: Ready</span>
                    <span class="badge" style="background: rgba(129, 140, 248, 0.2); color: #818CF8; border: 1px solid rgba(129, 140, 248, 0.3);">Voxel Grid: 64&times;64&times;64</span>
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

    # --- Metric Cards Row ---
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Tumor Volume</div>
                <div class="metric-value">{metrics['volume_cm3']} <span style="font-size: 1rem; color:#94A3B8;">cm³</span></div>
                <div class="metric-sub">{metrics['volume_mm3']} mm³ ({metrics['voxel_count']} voxels)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Maximum Diameter</div>
                <div class="metric-value">{metrics['longest_diameter_mm']} <span style="font-size: 1rem; color:#94A3B8;">mm</span></div>
                <div class="metric-sub">3D Feret Bounding Dimension</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        cz, cy, cx = metrics["centroid_mm"]
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Centroid Coordinates</div>
                <div class="metric-value" style="font-size: 1.25rem;">({cx}, {cy}, {cz})</div>
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
            <div class="metric-card">
                <div class="metric-title">Model Dice Score</div>
                <div class="metric-value" style="color: #38BDF8;">{dice_text}</div>
                <div class="metric-sub">IoU (Jaccard): {iou_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col5:
        risk = metrics["risk_tier"]
        badge_style = "badge-critical" if "High" in risk else ("badge-safe" if "Negative" in risk else "badge-cpu")
        st.markdown(
            f"""
            <div class="metric-card">
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
