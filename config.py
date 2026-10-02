"""
Central Configuration for 3D Medical AI Brain Tumor Detection & Mesh Rendering.
Defines volume dimensions, intensity thresholds, deep learning parameters,
and visualization palettes.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

# Base Project Directories
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "weights"
CACHE_DIR = BASE_DIR / "cache"
EXPORTS_DIR = BASE_DIR / "exports"

# Create directories if they do not exist
for directory in [MODELS_DIR, CACHE_DIR, EXPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# 3D Volume Parameters
DEFAULT_VOLUME_SHAPE: Tuple[int, int, int] = (64, 64, 64)
DEFAULT_VOXEL_SPACING: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # millimeters

# Tissue Intensity Profiles for Synthetic Multimodal MRI Simulation
@dataclass(frozen=True)
class TissueIntensity:
    BACKGROUND: float = 0.0
    SKULL: float = 0.20
    CSF: float = 0.12         # Cerebrospinal fluid
    GREY_MATTER: float = 0.55
    WHITE_MATTER: float = 0.75
    VENTRICLES: float = 0.10
    EDEMA: float = 0.82       # Vasogenic peritumoral edema
    TUMOR_ENHANCING: float = 0.95
    TUMOR_NECROTIC: float = 0.28

TISSUE_PROFILES = TissueIntensity()

# 3D UNet Deep Learning Hyperparameters
@dataclass(frozen=True)
class ModelConfig:
    IN_CHANNELS: int = 1
    OUT_CHANNELS: int = 1
    BASE_FILTERS: int = 16
    DEPTH: int = 3
    DROPOUT_PROB: float = 0.10
    NORM_TYPE: str = "instance"  # 'instance' or 'batch'
    ACTIVATION: str = "leaky_relu"
    LEAKY_SLOPE: float = 0.1

MODEL_CONFIG = ModelConfig()

# Marching Cubes 3D Mesh Extraction Parameters
@dataclass(frozen=True)
class MeshConfig:
    BRAIN_ISO_VALUE: float = 0.18
    TUMOR_ISO_VALUE: float = 0.45
    STEP_SIZE: int = 1  # 1 for maximum fidelity, 2 for ultra-fast rendering
    SMOOTHING_ITERATIONS: int = 2

MESH_CONFIG = MeshConfig()

# Visualization Theme & Plotly 3D Styling (Botanical Forest, Terracotta & Warm Sand Palette)
@dataclass(frozen=True)
class VisualTheme:
    DARK_BG_COLOR: str = "#0D1F18"        # Deep Forest Pine Green
    CONTAINER_BG: str = "#152E24"         # Rich Deep Sage
    CARD_BG: str = "#1C3B2F"              # Dark Forest Card
    CARD_BROWN: str = "#5C3720"           # Earthy Terracotta Card
    CARD_CHESTNUT: str = "#442716"        # Deep Chestnut
    PRIMARY_GOLD: str = "#DEBA8A"         # Warm Golden Wheat / Sand
    ACCENT_SAGE: str = "#7EA878"          # Fresh Leaf Sage
    TERRACOTTA_RED: str = "#D96B43"       # Terracotta / Coral Accent
    AMBER_WARM: str = "#E5B887"           # Warm Sand Amber
    TEXT_LIGHT: str = "#F7EEDB"           # Warm Ivory Cream
    TEXT_MUTED: str = "#9EB3A6"           # Soft Sage Muted
    
    # 3D Mesh Plotly Colors
    BRAIN_COLOR: str = "rgb(95, 160, 125)"    # Translucent Organic Jade / Sage
    BRAIN_OPACITY: float = 0.16
    TUMOR_COLOR: str = "rgb(217, 107, 67)"    # Warm Terracotta Red / Coral
    TUMOR_OPACITY: float = 0.90
    EDEMA_COLOR: str = "rgb(222, 186, 138)"   # Warm Golden Wheat
    EDEMA_OPACITY: float = 0.40

THEME = VisualTheme()

# Standard Pre-defined Clinical Cases
CLINICAL_CASES = {
    "Case 1: Frontal Lobe High-Grade Glioblastoma": {
        "tumor_type": "glioblastoma",
        "location": "frontal",
        "radius": 9.5,
        "irregularity": 0.35,
        "edema_extent": 1.45,
        "has_necrotic_core": True,
        "description": "Aggressive, infiltrative lesion in right frontal cortex with central necrosis and surrounding vasogenic edema.",
    },
    "Case 2: Temporal Lobe Oligodendroglioma": {
        "tumor_type": "oligodendroglioma",
        "location": "temporal",
        "radius": 7.5,
        "irregularity": 0.22,
        "edema_extent": 1.25,
        "has_necrotic_core": False,
        "description": "Heterogeneous mass in left temporal lobe with moderate mass effect and cortical expansion.",
    },
    "Case 3: Parietal Convexity Meningioma": {
        "tumor_type": "meningioma",
        "location": "parietal",
        "radius": 8.0,
        "irregularity": 0.10,
        "edema_extent": 1.15,
        "has_necrotic_core": False,
        "description": "Well-circumscribed extra-axial mass with distinct dural tail and minimal parenchymal infiltration.",
    },
    "Case 4: Healthy Baseline Control (Negative Scan)": {
        "tumor_type": "none",
        "location": "none",
        "radius": 0.0,
        "irregularity": 0.0,
        "edema_extent": 1.0,
        "has_necrotic_core": False,
        "description": "Normal brain anatomy without focal mass lesion, acute infarction, or intracranial hemorrhage.",
    },
}
