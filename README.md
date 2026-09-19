# 🧠 NeuroScan 3D: Volumetric Brain Tumor AI & Mesh Cockpit

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3F4F75.svg?logo=plotly&logoColor=white)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-grade, standalone 3D Medical AI application for Brain Tumor Detection, Multiplanar Slice Exploration (MPR), and Interactive 3D Surface Mesh Rendering.

Built from scratch using **PyTorch**, **Streamlit**, **Plotly**, and **Scikit-Image**.

---

## 🌟 Key Features

- **Synthetic 3D MRI Generator (`data_loader.py`)**:
  - Generates realistic multi-compartment $(64 \times 64 \times 64)$ 3D brain scans modeling the skull, CSF space, gray matter mantle, deep white matter core, and asymmetric lateral ventricles.
  - Simulates pathological neoplastic tumors with enhancing hyperintense margins, surrounding infiltrative vasogenic edema, and necrotic cores.
  - **Zero external datasets required** — runs out of the box with 4 pre-configured clinical presets and randomized simulation.

- **Full PyTorch 3D UNet (`model.py`)**:
  - Custom deep 3D segmentation network with **1,423,489 trainable parameters**.
  - Features 3D Convolutions, `InstanceNorm3d`, LeakyReLU activations, residual shortcut connections, and multi-scale skip connections.

- **Marching Cubes 3D Mesh Extraction (`inference.py`)**:
  - Automatically extracts 3D polygonal surface meshes for both the brain cortex envelope and the segmented tumor mass using `skimage.measure.marching_cubes`.
  - Calculates clinical volumetric metrics: Tumor Volume ($cm^3$ and $mm^3$), 3D Feret maximum diameter ($mm$), centroid coordinates, and Dice/IoU scores.
  - Generates downloadable Wavefront `.obj` 3D mesh files compatible with **3D Slicer**, **Blender**, and 3D printing software.

- **Clinical Streamlit Cockpit (`app.py`)**:
  - Modern, dark-mode glassmorphic interface (`#0B0F19`).
  - **2D Multiplanar Reconstruction (MPR)**: Axial (Transverse), Coronal (Frontal), and Sagittal (Lateral) slice views with dynamic tumor probability heatmap overlays and ground-truth boundary lines.
  - **Interactive 3D Mesh Viewer**: Dual-surface rendering in Plotly with orbital rotation, turntable navigation, opacity sliders, and wireframe toggle.
  - One-click export for 3D `.obj` meshes and structured diagnostic JSON reports.

- **One-Command Auto-Launcher (`main.py`)**:
  - Automatically checks Python dependencies, installs any missing packages via `pip`, launches the Streamlit server, and opens your default browser.

---

## 📁 Repository Structure

```
brain_tumor_3d/
├── .gitignore          # Ignores __pycache__, temp files, virtualenvs
├── LICENSE             # MIT Open-Source License
├── README.md           # Project documentation and guide
├── requirements.txt    # Exact Python package dependencies
├── config.py           # Global parameters, tissue profiles, visual palettes
├── data_loader.py      # Synthetic 3D MRI volume generator
├── model.py            # Complete PyTorch 3D UNet architecture
├── inference.py        # 3D UNet inference & Marching Cubes mesh extraction
├── app.py              # Dark-mode Streamlit & Plotly clinical cockpit
├── main.py             # Single executable script with auto-dependency installer
└── weights/
    └── unet3d_brain_tumor.pt  # Trained/calibrated model weights
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/brain-tumor-3d-ai.git
cd brain-tumor-3d-ai
```

### 2. Launch with ONE Command
```bash
python main.py
```

`main.py` will automatically:
1. Verify all required packages (`torch`, `streamlit`, `plotly`, `scikit-image`, `numpy`, `scipy`).
2. Install any missing dependencies via `pip`.
3. Launch the Streamlit web application.
4. Automatically open `http://localhost:8501` in your browser.

---

## 🩺 Clinical Presets Included

| Profile | Pathology | Location | Clinical Features |
| :--- | :--- | :--- | :--- |
| **Case 1** | High-Grade Glioblastoma | Frontal Lobe | Infiltrative mass with central necrosis and marked vasogenic edema |
| **Case 2** | Oligodendroglioma | Temporal Lobe | Cortical expansion with moderate mass effect |
| **Case 3** | Convexity Meningioma | Parietal | Well-circumscribed extra-axial mass with distinct borders |
| **Case 4** | Healthy Baseline Control | Normal | Negative scan; no focal mass or intracranial abnormality |

---

## 🛠 Tech Stack

- **Deep Learning**: [PyTorch 3D](https://pytorch.org/) (UNet3D architecture)
- **Computer Vision & Meshing**: [Scikit-Image](https://scikit-image.org/) (Marching Cubes), [SciPy](https://scipy.org/)
- **Frontend & Visualization**: [Streamlit](https://streamlit.io/), [Plotly Graph Objects](https://plotly.com/)
- **Data Modeling**: [NumPy](https://numpy.org/)

---

## 📄 License

Distributed under the [MIT License](LICENSE).
