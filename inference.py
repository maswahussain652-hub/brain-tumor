"""
Inference & 3D Geometric Mesh Extraction Engine.
Manages PyTorch 3D UNet execution, automated weight calibration,
Marching Cubes 3D surface mesh generation, and clinical volumetric analysis.
"""

from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import io
import numpy as np
from scipy.ndimage import gaussian_filter, binary_opening
from skimage.measure import marching_cubes
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    MODELS_DIR,
    DEFAULT_VOXEL_SPACING,
    MESH_CONFIG,
    DEFAULT_VOLUME_SHAPE,
    CLINICAL_CASES,
)
from model import UNet3D, create_unet3d
from data_loader import SyntheticBrainGenerator


class DiceLoss(nn.Module):
    """Smooth Soft Dice Loss for 3D binary segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        return 1.0 - dice


class MedicalInferenceEngine:
    """End-to-end inference and 3D surface mesh extraction pipeline."""

    def __init__(self, weights_filename: str = "unet3d_brain_tumor.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_path = MODELS_DIR / weights_filename
        self.model = create_unet3d(self.device)
        self._ensure_weights()

    def _ensure_weights(self) -> None:
        """Loads cached weights or performs fast calibration on synthetic volumes."""
        if self.weights_path.exists():
            try:
                state_dict = torch.load(self.weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                return
            except Exception:
                pass  # Fall back to quick calibration if file is corrupted

        self._calibrate_weights(num_epochs=12)

    def _calibrate_weights(self, num_epochs: int = 12) -> None:
        """
        Fast on-device calibration on diverse synthetic cases.
        Ensures the 3D UNet reliably recognizes hyperintense lesions,
        edema boundaries, and complex tumor topologies immediately out-of-the-box.
        """
        self.model.train()
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion_dice = DiceLoss()
        criterion_bce = nn.BCEWithLogitsLoss()

        gen = SyntheticBrainGenerator(
            shape=DEFAULT_VOLUME_SHAPE,
            spacing=DEFAULT_VOXEL_SPACING,
            random_seed=101,
        )

        # Build diverse training cases
        training_cases = []
        for case_name, params in CLINICAL_CASES.items():
            sample = gen.generate_case(params)
            vol_t = (
                torch.from_numpy(sample["volume"])
                .unsqueeze(0)
                .unsqueeze(0)
                .to(self.device)
            )
            mask_t = (
                torch.from_numpy(sample["ground_truth_mask"])
                .unsqueeze(0)
                .unsqueeze(0)
                .float()
                .to(self.device)
            )
            training_cases.append((vol_t, mask_t))

        for _ in range(num_epochs):
            for vol_t, mask_t in training_cases:
                optimizer.zero_grad()
                logits = self.model(vol_t)
                loss = 0.6 * criterion_dice(logits, mask_t) + 0.4 * criterion_bce(
                    logits, mask_t
                )
                loss.backward()
                optimizer.step()

        self.model.eval()
        try:
            torch.save(self.model.state_dict(), self.weights_path)
        except Exception:
            pass

    def segment_volume(
        self,
        volume: np.ndarray,
        probability_threshold: float = 0.50,
        apply_smoothing: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs 3D volumetric inference.
        Returns:
            probability_map: float32 ndarray in [0.0, 1.0]
            binary_mask: uint8 ndarray (0 or 1)
        """
        self.model.eval()
        tensor_in = (
            torch.from_numpy(volume.astype(np.float32))
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            probs_t = self.model.predict_probability(tensor_in)
            prob_map = probs_t.squeeze().cpu().numpy()

        if apply_smoothing:
            prob_map = gaussian_filter(prob_map, sigma=0.6)

        binary_mask = (prob_map >= probability_threshold).astype(np.uint8)

        # Light morphological opening to remove isolated spurious voxel noise
        if binary_mask.sum() > 8:
            binary_mask = binary_opening(binary_mask, structure=np.ones((2, 2, 2))).astype(np.uint8)

        return prob_map, binary_mask

    @staticmethod
    def extract_marching_cubes_mesh(
        volume_or_mask: np.ndarray,
        iso_level: float,
        spacing: Tuple[float, float, float] = DEFAULT_VOXEL_SPACING,
        step_size: int = MESH_CONFIG.STEP_SIZE,
    ) -> Dict[str, Any]:
        """
        Extracts 3D polygonal surface mesh using Marching Cubes algorithm.
        Returns vertices, faces, vertex normals, and surface area.
        Safely returns empty arrays if no isosurface exists.
        """
        data = volume_or_mask.astype(np.float32)
        v_min, v_max = float(data.min()), float(data.max())

        # If data is completely below iso_level, return empty mesh safely
        if v_max <= iso_level or (v_max - v_min) < 1e-4:
            return {
                "vertices": np.zeros((0, 3), dtype=np.float32),
                "faces": np.zeros((0, 3), dtype=np.int32),
                "normals": np.zeros((0, 3), dtype=np.float32),
                "surface_area_mm2": 0.0,
                "has_mesh": False,
            }

        try:
            verts, faces, normals, _ = marching_cubes(
                volume=data,
                level=iso_level,
                spacing=spacing,
                step_size=step_size,
                allow_degenerate=False,
            )

            # Compute approximate surface area from triangular faces
            v0 = verts[faces[:, 0]]
            v1 = verts[faces[:, 1]]
            v2 = verts[faces[:, 2]]
            cross_prod = np.cross(v1 - v0, v2 - v0)
            area = float(0.5 * np.sum(np.linalg.norm(cross_prod, axis=1)))

            return {
                "vertices": verts.astype(np.float32),
                "faces": faces.astype(np.int32),
                "normals": normals.astype(np.float32),
                "surface_area_mm2": round(area, 2),
                "has_mesh": len(verts) > 0,
            }
        except Exception:
            return {
                "vertices": np.zeros((0, 3), dtype=np.float32),
                "faces": np.zeros((0, 3), dtype=np.int32),
                "normals": np.zeros((0, 3), dtype=np.float32),
                "surface_area_mm2": 0.0,
                "has_mesh": False,
            }

    @staticmethod
    def calculate_clinical_metrics(
        pred_mask: np.ndarray,
        gt_mask: Optional[np.ndarray],
        spacing: Tuple[float, float, float] = DEFAULT_VOXEL_SPACING,
    ) -> Dict[str, Any]:
        """Calculates volumetric, geometric, and overlap validation metrics."""
        voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
        pred_voxel_count = int(np.sum(pred_mask == 1))
        pred_volume_mm3 = pred_voxel_count * voxel_volume_mm3
        pred_volume_cm3 = pred_volume_mm3 / 1000.0

        if pred_voxel_count > 0:
            coords = np.argwhere(pred_mask == 1)
            centroid_z, centroid_y, centroid_x = coords.mean(axis=0)
            centroid = (float(centroid_z), float(centroid_y), float(centroid_x))
            centroid_mm = (
                round(float(centroid_z * spacing[0]), 1),
                round(float(centroid_y * spacing[1]), 1),
                round(float(centroid_x * spacing[2]), 1),
            )

            z_min, y_min, x_min = coords.min(axis=0)
            z_max, y_max, x_max = coords.max(axis=0)
            dim_mm = (
                (z_max - z_min + 1) * spacing[0],
                (y_max - y_min + 1) * spacing[1],
                (x_max - x_min + 1) * spacing[2],
            )
            longest_diameter_mm = float(np.max(dim_mm))

            # Risk categorization
            if pred_volume_cm3 > 15.0:
                risk_tier = "High Volumetric Burden / Critical Mass Effect"
            elif pred_volume_cm3 > 5.0:
                risk_tier = "Moderate Volumetric Burden"
            else:
                risk_tier = "Low Volumetric Burden / Early Stage"
        else:
            centroid = (0.0, 0.0, 0.0)
            centroid_mm = (0.0, 0.0, 0.0)
            longest_diameter_mm = 0.0
            risk_tier = "Negative / No Mass Detected"

        # Overlap metrics if ground truth is available
        if gt_mask is not None:
            gt_voxel_count = int(np.sum(gt_mask == 1))
            intersection = int(np.sum((pred_mask == 1) & (gt_mask == 1)))
            union = int(np.sum((pred_mask == 1) | (gt_mask == 1)))

            if gt_voxel_count == 0 and pred_voxel_count == 0:
                dice = 1.0
                iou = 1.0
            else:
                dice = (2.0 * intersection) / (pred_voxel_count + gt_voxel_count + 1e-6)
                iou = intersection / (union + 1e-6)
        else:
            dice = None
            iou = None

        return {
            "voxel_count": pred_voxel_count,
            "volume_mm3": round(pred_volume_mm3, 2),
            "volume_cm3": round(pred_volume_cm3, 3),
            "longest_diameter_mm": round(longest_diameter_mm, 2),
            "centroid_voxel": centroid,
            "centroid_mm": centroid_mm,
            "risk_tier": risk_tier,
            "dice_score": round(float(dice), 4) if dice is not None else None,
            "iou_score": round(float(iou), 4) if iou is not None else None,
        }

    @staticmethod
    def export_mesh_obj(vertices: np.ndarray, faces: np.ndarray) -> str:
        """
        Serializes 3D mesh vertices and triangle faces into standard Wavefront OBJ.
        """
        buffer = io.StringIO()
        buffer.write("# 3D Medical AI Generated Mesh Export\n")
        buffer.write(f"# Vertices: {len(vertices)}, Faces: {len(faces)}\n")

        for v in vertices:
            buffer.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")

        # OBJ face indices are 1-based
        for f in faces:
            buffer.write(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n")

        return buffer.getvalue()
