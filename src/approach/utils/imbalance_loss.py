import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None, reduction: str = 'mean'):
        """
        gamma: focusing parameter ≥ 0
        alpha: a float or a list/Tensor of per‑class weights; if None, no class weighting
        reduction: 'mean' | 'sum' | 'none'
        """
        super().__init__()
        self.gamma = gamma
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=torch.float)
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # compute standard CE loss per sample
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)                    # pt = probability of the true class
        # optionally apply per‑class alpha
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
        else:
            alpha_t = 1.0
        # focal term
        loss = alpha_t * (1 - pt).pow(self.gamma) * ce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        

class CBLoss(nn.Module):
    """
    Class-Balanced Loss with a selectable base loss.
    Args
    ----
    samples_per_class : 1-D list/array - total examples per class in *the whole training set*
    beta              : float in (0,1) - smoothing factor (paper default 0.9999)
    loss_type         : 'ce'  | 'focal'
    gamma             : focusing parameter for focal loss
    reduction         : 'mean' | 'sum' | 'none'
    """
    def __init__(self,
                 samples_per_class,
                 beta: float = 0.9999,
                 loss_type: str = 'ce',
                 gamma: float = 2.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.register_buffer(
            'weights', self._compute_cb_weights(samples_per_class, beta)
        )
        loss_type = loss_type.lower()
        assert loss_type in ('ce', 'focal')
        self.loss_type = loss_type
        self.gamma = gamma
        self.reduction = reduction

    @staticmethod
    def _compute_cb_weights(samples_per_class, beta):
        # Effective number: (1 - beta^n)/(1 - beta)
        n = torch.tensor(samples_per_class, dtype=torch.float)
        eff_num = 1.0 - beta ** n
        w = (1.0 - beta) / eff_num
        w = w / w.sum() * len(samples_per_class)   # normalise so sum = C
        return w

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        device = logits.device
        weights = self.weights.to(device)[targets]         # per-sample weight

        if self.loss_type == 'ce':
            loss = F.cross_entropy(
                logits, targets, weight=self.weights.to(device),
                reduction=self.reduction
            )
            return loss                                     # already weighted

        # ----- CB-Focal -----
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        pt = probs.gather(1, targets.view(-1, 1)).squeeze(1)  # p_t
        focal_factor = (1.0 - pt).pow(self.gamma)

        ce = F.nll_loss(log_probs, targets, reduction='none') # per-sample CE
        loss = weights * focal_factor * ce

        if   self.reduction == 'mean': return loss.mean()
        elif self.reduction == 'sum' : return loss.sum()
        else                         : return loss