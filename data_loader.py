"""
Synthetic 3D MRI Brain & Tumor Volume Generator.
Generates realistic multi-compartment anatomical 3D brain scans with
pathological lesions (edema, enhancing rim, necrotic core) and ground truth masks
without requiring external datasets.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from scipy.ndimage import gaussian_filter

from config import (
    DEFAULT_VOLUME_SHAPE,
    DEFAULT_VOXEL_SPACING,
    TISSUE_PROFILES,
    CLINICAL_CASES,
)


class SyntheticBrainGenerator:
    """Generates anatomically plausible 3D MRI brain scans with pathological lesions."""

    def __init__(
        self,
        shape: Tuple[int, int, int] = DEFAULT_VOLUME_SHAPE,
        spacing: Tuple[float, float, float] = DEFAULT_VOXEL_SPACING,
        random_seed: Optional[int] = None,
    ):
        self.depth, self.height, self.width = shape
        self.shape = shape
        self.spacing = spacing
        self.rng = np.random.default_rng(random_seed)

        # Coordinate grids centered at (0, 0, 0)
        z = np.linspace(-1.0, 1.0, self.depth, dtype=np.float32)
        y = np.linspace(-1.0, 1.0, self.height, dtype=np.float32)
        x = np.linspace(-1.0, 1.0, self.width, dtype=np.float32)
        self.Z, self.Y, self.X = np.meshgrid(z, y, x, indexing="ij")

    def _generate_brain_anatomy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Creates skull, CSF, gray matter, white matter, and ventricular cavities."""
        volume = np.zeros(self.shape, dtype=np.float32)

        # Scalp and Skull envelope
        skull_dist = (
            (self.X / 0.82) ** 2 + (self.Y / 0.90) ** 2 + (self.Z / 0.78) ** 2
        )
        skull_mask = (skull_dist <= 1.0) & (skull_dist > 0.86)
        volume[skull_mask] = TISSUE_PROFILES.SKULL

        # Intracranial space (Brain + CSF)
        brain_dist = (
            (self.X / 0.77) ** 2 + (self.Y / 0.85) ** 2 + (self.Z / 0.73) ** 2
        )
        parenchyma_mask = brain_dist <= 0.88
        csf_mask = (brain_dist <= 0.98) & (~parenchyma_mask)

        volume[csf_mask] = TISSUE_PROFILES.CSF
        volume[parenchyma_mask] = TISSUE_PROFILES.GREY_MATTER

        # Deep White Matter Core
        wm_dist = (
            (self.X / 0.60) ** 2 + (self.Y / 0.68) ** 2 + (self.Z / 0.56) ** 2
        )
        white_matter_mask = wm_dist <= 0.85
        volume[white_matter_mask] = TISSUE_PROFILES.WHITE_MATTER

        # Cortical folds / sulci texture via high-frequency harmonics
        gyri_texture = 0.08 * (
            np.sin(self.X * 16.0) * np.cos(self.Y * 16.0) * np.sin(self.Z * 14.0)
        )
        volume[parenchyma_mask] += gyri_texture[parenchyma_mask]

        # Bilateral Lateral Ventricles
        # Left ventricle
        ventricle_left = (
            ((self.X - 0.12) / 0.08) ** 2
            + ((self.Y - 0.02) / 0.32) ** 2
            + ((self.Z - 0.02) / 0.18) ** 2
        )
        # Right ventricle
        ventricle_right = (
            ((self.X + 0.12) / 0.08) ** 2
            + ((self.Y - 0.02) / 0.32) ** 2
            + ((self.Z - 0.02) / 0.18) ** 2
        )
        ventricles_mask = (ventricle_left <= 1.0) | (ventricle_right <= 1.0)
        volume[ventricles_mask] = TISSUE_PROFILES.VENTRICLES

        # Brain envelope mask for 3D surface extraction
        brain_envelope_mask = brain_dist <= 0.96

        return volume, brain_envelope_mask

    def _get_anatomical_center(self, location: str) -> Tuple[float, float, float]:
        """Resolves anatomical compartment into 3D normalized coordinates (z, y, x)."""
        loc = location.lower()
        if loc == "frontal":
            # Anterior superior right hemisphere
            return (0.22, 0.40, 0.28)
        elif loc == "temporal":
            # Lateral inferior middle left hemisphere
            return (-0.18, -0.05, -0.42)
        elif loc == "parietal":
            # Superior posterior left hemisphere
            return (0.42, -0.32, -0.25)
        elif loc == "occipital":
            # Posterior inferior right hemisphere
            return (-0.15, -0.55, 0.22)
        elif loc == "central":
            return (0.10, 0.05, 0.12)
        else:
            # Default or random location within parenchyma
            rz = float(self.rng.uniform(-0.25, 0.35))
            ry = float(self.rng.uniform(-0.35, 0.35))
            rx = float(self.rng.uniform(-0.35, 0.35))
            return (rz, ry, rx)

    def _inject_tumor(
        self,
        volume: np.ndarray,
        tumor_type: str,
        location: str,
        radius_voxels: float,
        irregularity: float,
        edema_extent: float,
        has_necrotic_core: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Injects realistic pathological lesion with edema, enhancing rim, and necrosis."""
        if tumor_type.lower() == "none" or radius_voxels <= 0:
            return volume, np.zeros(self.shape, dtype=np.uint8)

        cz, cy, cx = self._get_anatomical_center(location)
        norm_radius = radius_voxels / (self.width / 2.0)

        # Distance field from lesion center
        dz = self.Z - cz
        dy = self.Y - cy
        dx = self.X - cx
        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        # Realistic irregular boundary via spherical harmonic perturbations
        if irregularity > 0:
            angles_phi = np.arctan2(dy, dx)
            angles_theta = np.arccos(np.clip(dz / (dist + 1e-6), -1.0, 1.0))
            harmonic_perturbation = irregularity * (
                0.55 * np.sin(3 * angles_phi) * np.cos(2 * angles_theta)
                + 0.35 * np.cos(4 * angles_phi + 1.2) * np.sin(3 * angles_theta)
                + 0.20 * np.sin(5 * angles_phi)
            )
            effective_radius = norm_radius * (1.0 + harmonic_perturbation)
        else:
            effective_radius = norm_radius

        # Tumor core mask (active enhancing tumor + necrotic center)
        tumor_mask = (dist <= effective_radius).astype(np.uint8)

        # Edema region (infiltrative FLAIR hyperintensity surrounding lesion)
        edema_radius = effective_radius * edema_extent
        edema_mask = ((dist <= edema_radius) & (tumor_mask == 0)).astype(bool)

        # Apply edema signal
        volume[edema_mask] = TISSUE_PROFILES.EDEMA

        # Apply active enhancing tumor shell
        volume[tumor_mask == 1] = TISSUE_PROFILES.TUMOR_ENHANCING

        # Apply central necrosis if high-grade lesion
        if has_necrotic_core and radius_voxels > 5.0:
            necrotic_radius = effective_radius * 0.48
            necrotic_mask = (dist <= necrotic_radius).astype(bool)
            volume[necrotic_mask] = TISSUE_PROFILES.TUMOR_NECROTIC

        return volume, tumor_mask

    def _apply_mri_artifacts(self, volume: np.ndarray) -> np.ndarray:
        """Simulates realistic magnetic resonance imaging noise and B1 bias field."""
        # 1. Low-frequency B1 field inhomogeneity (bias field)
        bias_field = (
            1.0
            + 0.12 * np.sin(self.X * 2.5)
            + 0.08 * np.cos(self.Y * 2.0)
            + 0.05 * np.sin(self.Z * 1.8)
        )
        volume = volume * bias_field

        # 2. Gaussian smoothing to mimic realistic scanner point-spread function (PSF)
        volume = gaussian_filter(volume, sigma=0.6)

        # 3. Additive Rician/Gaussian noise
        noise_sigma = 0.022
        noise_real = self.rng.normal(0, noise_sigma, self.shape)
        noise_imag = self.rng.normal(0, noise_sigma, self.shape)
        noisy_volume = np.sqrt((volume + noise_real) ** 2 + noise_imag**2)

        # 4. Normalize to strictly [0.0, 1.0]
        v_min, v_max = noisy_volume.min(), noisy_volume.max()
        if v_max > v_min:
            normalized_volume = (noisy_volume - v_min) / (v_max - v_min)
        else:
            normalized_volume = noisy_volume

        return normalized_volume.astype(np.float32)

    def generate_case(
        self,
        case_name_or_dict: Any,
    ) -> Dict[str, Any]:
        """
        Generates a complete 3D MRI volume, tumor segmentation mask,
        brain outer envelope, and anatomical metadata.
        """
        if isinstance(case_name_or_dict, str):
            case_params = CLINICAL_CASES.get(
                case_name_or_dict,
                CLINICAL_CASES["Case 1: Frontal Lobe High-Grade Glioblastoma"],
            )
            case_title = case_name_or_dict
        else:
            case_params = case_name_or_dict
            case_title = case_params.get("title", "Custom Simulated Patient")

        # Step 1: Base anatomy
        volume, brain_envelope = self._generate_brain_anatomy()

        # Step 2: Pathological tumor injection
        volume, tumor_mask = self._inject_tumor(
            volume=volume,
            tumor_type=case_params.get("tumor_type", "glioblastoma"),
            location=case_params.get("location", "frontal"),
            radius_voxels=case_params.get("radius", 8.5),
            irregularity=case_params.get("irregularity", 0.25),
            edema_extent=case_params.get("edema_extent", 1.35),
            has_necrotic_core=case_params.get("has_necrotic_core", True),
        )

        # Step 3: Realistic MRI scanner physics & artifacts
        final_mri = self._apply_mri_artifacts(volume)

        # Step 4: Metadata calculation
        voxel_volume_mm3 = (
            self.spacing[0] * self.spacing[1] * self.spacing[2]
        )
        tumor_voxel_count = int(np.sum(tumor_mask))
        tumor_volume_mm3 = tumor_voxel_count * voxel_volume_mm3
        tumor_volume_cm3 = tumor_volume_mm3 / 1000.0

        if tumor_voxel_count > 0:
            coords = np.argwhere(tumor_mask == 1)
            centroid_z, centroid_y, centroid_x = coords.mean(axis=0)
            centroid = (float(centroid_z), float(centroid_y), float(centroid_x))
            # Bounding box
            z_min, y_min, x_min = coords.min(axis=0)
            z_max, y_max, x_max = coords.max(axis=0)
            extent_mm = (
                (z_max - z_min + 1) * self.spacing[0],
                (y_max - y_min + 1) * self.spacing[1],
                (x_max - x_min + 1) * self.spacing[2],
            )
            longest_diameter_mm = float(np.max(extent_mm))
        else:
            centroid = (0.0, 0.0, 0.0)
            longest_diameter_mm = 0.0

        return {
            "title": case_title,
            "description": case_params.get("description", "Simulated 3D MRI Brain Scan"),
            "volume": final_mri,
            "ground_truth_mask": tumor_mask.astype(np.uint8),
            "brain_envelope_mask": brain_envelope.astype(np.uint8),
            "spacing": self.spacing,
            "shape": self.shape,
            "metrics": {
                "voxel_count": tumor_voxel_count,
                "volume_mm3": round(tumor_volume_mm3, 2),
                "volume_cm3": round(tumor_volume_cm3, 3),
                "longest_diameter_mm": round(longest_diameter_mm, 2),
                "centroid": centroid,
            },
        }


def load_dataset_case(case_name: str, seed: int = 42) -> Dict[str, Any]:
    """Helper entrypoint to load a case by name with reproducible random seed."""
    generator = SyntheticBrainGenerator(
        shape=DEFAULT_VOLUME_SHAPE,
        spacing=DEFAULT_VOXEL_SPACING,
        random_seed=seed,
    )
    return generator.generate_case(case_name)
