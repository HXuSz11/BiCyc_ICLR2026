# vicreg.py
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn

from typing import Dict, Tuple
from torchvision import transforms as T



# ------------------------------------------------------------------
# Dataset-specific metadata: (image_size, mean, std, blur_kernel_A, blur_kernel_B)
# ------------------------------------------------------------------
DATASET_META: Dict[str, Tuple[int, Tuple[float, ...], Tuple[float, ...], int, int]] = {
    "cifar100"    : (32,  (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761), 3, 3),
    "tinyimagenet": (64,  (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262), 5, 5),
    "miniimagenet": (84,  (0.4720, 0.4540, 0.4039), (0.2760, 0.2679, 0.2822), 9, 5),
    "imagenet1k"  : (224, (0.485, 0.456, 0.406),    (0.229, 0.224, 0.225),   23, 23),
}

# ------------------------------------------------------------------
# Helper builders
# ------------------------------------------------------------------
def _color_jitter() -> T.ColorJitter:
    return T.ColorJitter(0.4, 0.4, 0.2, 0.1)

def _gaussian_blur(k: int) -> T.GaussianBlur:
    k |= 1                    # ensure kernel size is odd
    return T.GaussianBlur(kernel_size=k, sigma=(0.1, 2.0))

def _base_transforms(img_size: int) -> list:
    """Common pipeline shared by View-A and View-B."""
    return [
        T.RandomResizedCrop(img_size, scale=(0.2, 1.0), ratio=(3/4, 4/3)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([_color_jitter()], p=0.8),
        T.RandomGrayscale(p=0.2),
    ]

# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
def get_vicreg_viewA(dataset: str) -> T.Compose:
    """
    VICReg *View-A* transform for a given dataset.

    Parameters
    ----------
    dataset : str
        One of: 'cifar100', 'tinyimagenet', 'miniimagenet', 'imagenet1k'.

    Returns
    -------
    torchvision.transforms.Compose
    """
    if dataset not in DATASET_META:
        raise ValueError(f"Unsupported dataset '{dataset}'. "
                         f"Available: {list(DATASET_META)}")

    img_size, mean, std, blur_A, _ = DATASET_META[dataset]
    transforms_A = (
        _base_transforms(img_size)
        + [
            T.RandomApply([_gaussian_blur(blur_A)], p=1.0),  # always blur
            T.RandomSolarize(128, p=0.0),                    # no solarize
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    return T.Compose(transforms_A)


def get_vicreg_viewB(dataset: str) -> T.Compose:
    """
    VICReg *View-B* transform for a given dataset.

    Parameters
    ----------
    dataset : str
        One of: 'cifar100', 'tinyimagenet', 'miniimagenet', 'imagenet1k'.

    Returns
    -------
    torchvision.transforms.Compose
    """
    if dataset not in DATASET_META:
        raise ValueError(f"Unsupported dataset '{dataset}'. "
                         f"Available: {list(DATASET_META)}")

    img_size, mean, std, _, blur_B = DATASET_META[dataset]
    transforms_B = (
        _base_transforms(img_size)
        + [
            T.RandomApply([_gaussian_blur(blur_B)], p=0.1),  # low-probability blur
            T.RandomSolarize(128, p=0.2),                    # solarize with p=0.2
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    return T.Compose(transforms_B)


class VICRegLoss(nn.Module):
    """
    Full VICReg loss (Bardes et al., 2022).

    Parameters
    ----------
    inv_weight : float
        Weight on the invariance term (mean-squared error between views).
    var_weight : float
        Weight on the variance term (std per dimension ≥ gamma).
    cov_weight : float
        Weight on the covariance term (off-diagonal penalty).
    gamma : float
        Desired lower bound for per-dimension standard deviation.
    eps : float
        Numerical stability for std calculation.
    """
    def __init__(
        self,
        inv_weight: float = 25.0,
        var_weight: float = 25.0,
        cov_weight: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.inv_weight = inv_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.gamma = gamma
        self.eps = eps

    @staticmethod
    def _variance_term(z: torch.Tensor, gamma: float, eps: float) -> torch.Tensor:
        """Hinge on std ≥ gamma, averaged over feature dims."""
        std = torch.sqrt(z.var(dim=0) + eps)      # (D,)
        return torch.mean(F.relu(gamma - std))

    @staticmethod
    def _covariance_term(z: torch.Tensor) -> torch.Tensor:
        """Sum of squared off-diagonals of covariance, normalized by D."""
        B, D = z.shape
        z = z - z.mean(dim=0, keepdim=True)
        cov = (z.T @ z) / (B - 1)                 # D × D
        off_diag = cov.flatten()[
            ~torch.eye(D, dtype=torch.bool, device=z.device).flatten()
        ]
        return (off_diag ** 2).sum() / D

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute VICReg loss for two batches of embeddings.

        Parameters
        ----------
        z1, z2 : Tensor, shape (B, D)
            Embeddings from two augmented views of the same batch.

        Returns
        -------
        loss : scalar Tensor
        """
        # -------- 1. Invariance (MSE) --------
        inv_loss = F.mse_loss(z1, z2)

        # -------- 2. Variance (per view) ----
        var_loss = (
            self._variance_term(z1, self.gamma, self.eps) +
            self._variance_term(z2, self.gamma, self.eps)
        ) / 2.0

        # -------- 3. Covariance (per view) --
        cov_loss = (
            self._covariance_term(z1) +
            self._covariance_term(z2)
        ) / 2.0

        # -------- Weighted sum --------------
        loss = (
            self.inv_weight * inv_loss +
            self.var_weight * var_loss +
            self.cov_weight * cov_loss
        )
        return loss