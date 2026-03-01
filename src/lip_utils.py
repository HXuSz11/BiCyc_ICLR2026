from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import numpy as np
    from scipy.stats import genextreme
    _HAS_SCIPY = True
except Exception:
    import numpy as np
    _HAS_SCIPY = False


def _assert_conv_inshape(conv: nn.Conv2d, in_shape: Tuple[int,int,int], where: str = ""):
    C_in = in_shape[0]
    if C_in != conv.in_channels:
        msg = (f"[FastLip shape error{': '+where if where else ''}] "
               f"conv expects in_channels={conv.in_channels}, got {C_in}. "
               f"in_shape={in_shape}, conv={conv}")
        raise RuntimeError(msg)
# ----------------------------- Low-level utils --------------------------------

def _conv2d_power_norm(conv: nn.Conv2d, in_shape: Tuple[int, int, int], iters: int = 30) -> float:
    """
    Robust spectral (L2) operator-norm estimator for Conv2d via power iteration.

    Key fix:
    - Use the exact adjoint used by autograd, torch.nn.grad.conv2d_input, instead of
      conv_transpose2d with guessed output_padding. This avoids silent dimension drift
      and version-specific pitfalls.

    Assumes eval-time (weights frozen). Bias is ignored (does not affect Lipschitz).
    """
    from torch.nn.grad import conv2d_input
    _assert_conv_inshape(conv, in_shape, where="conv power-iteration")
    C_in, H_in, W_in = in_shape
    W = conv.weight
    s, p, d, g = conv.stride, conv.padding, conv.dilation, conv.groups
    device = W.device
    Wd = W.detach().double() if W.dtype == torch.float32 else W.detach()

    u = torch.randn(1, C_in, H_in, W_in, device=device, dtype=Wd.dtype)
    u = u / (u.norm() + 1e-12)
    for _ in range(iters):
        v = F.conv2d(u, Wd, None, s, p, d, g)
        v = v / (v.norm() + 1e-12)
        u = conv2d_input((1, C_in, H_in, W_in), Wd, v, stride=s, padding=p, dilation=d, groups=g)
        u = u / (u.norm() + 1e-12)
    Av = F.conv2d(u, Wd, None, s, p, d, g)
    return float(Av.norm().item())


def _linear_power_norm(fc: nn.Linear, iters: int = 50) -> float:
    """
    Spectral norm of a Linear layer via power iteration on W (bias ignored).
    Numerically stable and dtype-agnostic.
    """
    W = fc.weight.detach()
    device = W.device
    dtype = torch.float64 if W.dtype == torch.float32 else W.dtype
    Wd = W.to(dtype)

    v = torch.randn(Wd.shape[1], device=device, dtype=dtype)
    v = v / (v.norm() + 1e-12)
    for _ in range(iters):
        u = Wd @ v
        nu = u.norm()
        if nu.item() == 0.0:
            return 0.0
        u = u / nu
        v = Wd.t() @ u
        nv = v.norm()
        if nv.item() == 0.0:
            return 0.0
        v = v / nv
    return float((Wd @ v).norm().item())


def _bn_lipschitz(bn: nn.modules.batchnorm._BatchNorm) -> float:
    """
    Eval-mode BatchNorm Lipschitz constant (channelwise affine or non-affine).
    L = max_c |gamma_c| / sqrt(var_c + eps)  if affine=True
      = max_c 1 / sqrt(var_c + eps)         if affine=False
    """
    if bn.training:
        raise ValueError("BatchNorm must be in eval() when computing Lipschitz bounds.")
    var = bn.running_var.detach()
    if bn.affine:
        w = bn.weight.detach().abs()
        scale = w / torch.sqrt(var + bn.eps)
    else:
        scale = 1.0 / torch.sqrt(var + bn.eps)
    return float(scale.max().item())


# ------------------------ Spectral Norm Product (global) ----------------------

@dataclass
class GlobalBoundConfig:
    input_size: Tuple[int, int, int]       # (C,H,W), e.g., (3,32,32) for CIFAR
    conv_power_iters: int = 30
    linear_power_iters: int = 50

def spectral_product_bound_resnet18(model: nn.Module, cfg: GlobalBoundConfig) -> Dict[str, float]:
    """
    Global L2-Lipschitz upper bound by multiplying operator norms across the network.
    Residual addition uses L(f+g) ≤ L(f) + L(g). ReLU/Pooling are ≤1-Lipschitz.

    Fixes vs. previous version:
    - Use the exact adjoint-based conv power iteration.
    - Count AdaptiveAvgPool2d as 1/sqrt(H*W) (tightens a lot).
    - Handle optional BN after any extra bottleneck conv.
    - Assert BN eval-mode inside _bn_lipschitz.
    """
    model.eval()

    def out_dim(L_in, k, p, s, d):
        return (L_in + 2 * p - d * (k - 1) - 1) // s + 1

    def conv_bn_lip(conv: nn.Conv2d, bn: Optional[nn.BatchNorm2d], in_shape: Tuple[int, int, int]):
        s_conv = _conv2d_power_norm(conv, in_shape, iters=cfg.conv_power_iters)
        s_bn = _bn_lipschitz(bn) if bn is not None else 1.0

        C_in, H_in, W_in = in_shape
        C_out = conv.out_channels
        H_out = out_dim(H_in, conv.kernel_size[0], conv.padding[0], conv.stride[0], conv.dilation[0])
        W_out = out_dim(W_in, conv.kernel_size[1], conv.padding[1], conv.stride[1], conv.dilation[1])
        return s_conv * s_bn, (C_out, H_out, W_out)

    parts: Dict[str, float] = {}
    lip = 1.0
    C, H, W = cfg.input_size

    # Stem
    s, (C, H, W) = conv_bn_lip(model.conv1, model.bn1, (C, H, W))
    lip *= s
    parts["stem_conv1_bn1"] = s
    parts["stem_relu"] = 1.0

    # BasicBlock bound
    def basic_block_lip(block: nn.Module, in_shape: Tuple[int, int, int]):
        s1, shape1 = conv_bn_lip(block.conv1, block.bn1, in_shape)
        s2, shape2 = conv_bn_lip(block.conv2, block.bn2, shape1)
        s_main = s1 * s2

        # Skip path
        if getattr(block, "downsample", None) is not None:
            s_skip = 1.0
            shape_skip = in_shape
            for m in block.downsample:
                if isinstance(m, nn.Conv2d):
                    s_tmp, shape_skip = conv_bn_lip(m, None, shape_skip)
                    s_skip *= s_tmp
                elif isinstance(m, nn.BatchNorm2d):
                    s_skip *= _bn_lipschitz(m)
            sb = s_main + s_skip  # L(f+g) <= L(f)+L(g)
            return sb, shape2
        else:
            sb = s_main + 1.0  # identity skip (1-Lip)
            return sb, shape2

    # Layers 1..4
    for lname in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(model, lname)
        for i, block in enumerate(layer):
            sb, (C, H, W) = basic_block_lip(block, (C, H, W))
            parts[f"{lname}.block{i}"] = sb
            lip *= sb

    # Optional 'bottleneck' Conv2d before avgpool
    if hasattr(model, "bottleneck") and isinstance(model.bottleneck, nn.Conv2d):
        s_b, (C, H, W) = conv_bn_lip(model.bottleneck, None, (C, H, W))
        parts["bottleneck_conv"] = s_b
        lip *= s_b
        # If there is BN after bottleneck (e.g. model.bottleneck_bn)
        if hasattr(model, "bottleneck_bn") and isinstance(model.bottleneck_bn, nn.BatchNorm2d):
            s_bn_b = _bn_lipschitz(model.bottleneck_bn)
            parts["bottleneck_bn"] = s_bn_b
            lip *= s_bn_b

    # AdaptiveAvgPool2d to 1×1: L2 operator is 1/sqrt(H*W)
    parts["avgpool"] = 1.0 / math.sqrt(max(1, H * W))
    lip *= parts["avgpool"]

    # FC
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        s_fc = _linear_power_norm(model.fc, cfg.linear_power_iters)
        parts["fc"] = s_fc
        lip *= s_fc
    else:
        parts["fc"] = 1.0

    parts["global_bound"] = lip
    return parts


# ------------------------------ CLEVER Score ----------------------------------

@dataclass
class CleverConfig:
    norm: str = "l2"         # 'l2' or 'linf'
    radius: float = 0.3
    n_batches: int = 20
    batch_size: int = 64
    targeted: bool = False
    target_class: Optional[int] = None   # if targeted=True and provided, use this class
    fit_evt: bool = True                 # requires SciPy; fallback = empirical
    seed: int = 0

# def _random_ball(shape, norm: str, radius: float, device):
#     if norm == "l2":
#         z = torch.randn(shape, device=device)
#         z = z / (z.view(z.size(0), -1).norm(p=2, dim=1, keepdim=True) + 1e-12)
#         # radius * U^(1/d) ~ uniform in L2-ball
#         d = z[0].numel()
#         u = torch.rand(z.size(0), 1, device=device)
#         r = radius * (u ** (1.0 / d))
#         return z * r.view(-1, 1, 1, 1)
#     elif norm == "linf":
#         return (torch.rand(shape, device=device) * 2 - 1) * radius
#     else:
#         raise ValueError("norm must be 'l2' or 'linf'")

def _random_ball(shape, norm, radius, device):
    if norm == "l2":
        z = torch.randn(shape, device=device)  # (B,C,H,W)
        norms = z.view(z.size(0), -1).norm(p=2, dim=1)  # (B,)
        norms = norms.view(-1, 1, 1, 1)                 # (B,1,1,1)  <-- key!
        z = z / (norms + 1e-12)
        # radius * U^(1/d) for uniform in L2-ball
        d = z[0].numel()
        u = torch.rand(z.size(0), device=device).pow(1.0 / d).view(-1, 1, 1, 1)
        return radius * z * u

    elif norm in ("linf", "l_inf", "Linf", "L∞"):
        # uniform in Linf-ball
        return (torch.rand(shape, device=device) * 2.0 - 1.0) * radius

    else:
        raise ValueError(f"Unsupported norm: {norm}")

@torch.no_grad()
def _top2_margin(logits: torch.Tensor):
    top2 = torch.topk(logits, k=2, dim=1)
    top1 = top2.indices[:, 0]
    top2i = top2.indices[:, 1]
    margin = logits[torch.arange(logits.size(0)), top1] - logits[torch.arange(logits.size(0)), top2i]
    return margin, top1, top2i

def clever_score(model: nn.Module, x: torch.Tensor, cfg: CleverConfig) -> Dict[str, float]:
    """
    CLEVER-style local robustness / Lipschitz proxy for a single input x (B=1).

    Key fixes:
    - Estimate local Lipschitz of the **logit margin** g(x) = f_y(x) - f_j(x),
      not the cross-entropy loss, which introduces softmax scaling.
    - EVT fit done on block maxima via array_split (stable chunking).
    """
    assert x.size(0) == 1, "Use batch size 1 for CLEVER."
    model.eval()
    device = x.device

    # Base logits and margin at x
    x0 = x.detach().clone().requires_grad_(False)
    logits0 = model(x0)
    margin0, top1_idx, top2_idx = _top2_margin(logits0)
    y = int(top1_idx.item())
    margin_val = float(margin0.item())

    # choose comparison class j
    if cfg.targeted:
        j = int(cfg.target_class) if cfg.target_class is not None else int(top2_idx.item())
    else:
        j = int(top2_idx.item())

    # sample perturbations and compute ||∇(f_y - f_j)||_p
    torch.manual_seed(cfg.seed)
    if hasattr(np, "random"):
        try:
            np.random.seed(cfg.seed)
        except Exception:
            pass

    grad_norms = []
    for _ in range(cfg.n_batches):
        delta = _random_ball((cfg.batch_size, *x.shape[1:]), cfg.norm, cfg.radius, device)
        x_pert = (x + delta).clamp(0., 1.).detach()
        x_pert.requires_grad_(True)
        out = model(x_pert)
        # scalar margin per sample
        m = out[:, y] - out[:, j]
        # sum to get a scalar for autograd over batch
        grad = torch.autograd.grad(m.sum(), x_pert, retain_graph=False, create_graph=False)[0]
        g = grad.view(grad.size(0), -1)
        norms = g.norm(p=2, dim=1) if cfg.norm == "l2" else g.norm(p=float('inf'), dim=1)
        grad_norms.append(norms.detach().cpu().numpy())

    grad_norms = np.concatenate(grad_norms, axis=0)

    # EVT or empirical fallback
    if cfg.fit_evt and _HAS_SCIPY and grad_norms.size >= max(50, cfg.n_batches):
        blocks = np.array_split(grad_norms, cfg.n_batches)
        block_max = np.array([b.max() for b in blocks if b.size > 0])
        c, loc, scale = genextreme.fit(block_max)
        L_est = float(genextreme.ppf(0.999, c, loc=loc, scale=scale))
        method = "GEV(0.999)"
    else:
        L_est = float(np.quantile(grad_norms, 0.999)) if grad_norms.size >= 1000 else float(grad_norms.max())
        method = "empirical_max"

    return {
        "predicted_class": y,
        "competitor_class": j,
        "margin_logit_top1_minus_competitor": margin_val,
        "local_grad_norm_estimate": L_est,
        "clever_score": margin_val / (L_est + 1e-12),
        "estimation": method,
        "samples": int(grad_norms.size),
    }


# ----------- Fast-Lip (practical): IBP-style local L2 upper bound --------------

@dataclass
class FastLipConfig:
    eps: float = 0.3
    norm: str = "l2"                 # 'l2' or 'linf' (we use inscribed box for 'l2')
    input_clamp: Tuple[float, float] = (0.0, 1.0)

@torch.no_grad()
def _relu_bounds(xl: torch.Tensor, xu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Convex relaxation for ReLU intervals (exact interval propagation).
    return F.relu(xl), F.relu(xu)

@torch.no_grad()
def _bn_bounds(xl: torch.Tensor, xu: torch.Tensor, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Interval propagation through BatchNorm in eval mode.
    y = a*x + c per channel, where:
      if affine: a = gamma / sqrt(var + eps), c = beta - a*mu
      else:      a = 1 / sqrt(var + eps),     c = - a*mu
    """
    if bn.training:
        raise ValueError("BatchNorm must be in eval() for IBP bounds.")
    var = bn.running_var
    mean = bn.running_mean
    if bn.affine:
        a_c = bn.weight / torch.sqrt(var + bn.eps)
        c_c = bn.bias - a_c * mean
    else:
        a_c = 1.0 / torch.sqrt(var + bn.eps)
        c_c = - a_c * mean
    a = a_c.view(1, -1, 1, 1)
    c = c_c.view(1, -1, 1, 1)
    y1, y2 = a * xl + c, a * xu + c
    return torch.minimum(y1, y2), torch.maximum(y1, y2)

@torch.no_grad()
def _conv_bounds(xl: torch.Tensor, xu: torch.Tensor, conv: nn.Conv2d) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Standard IBP conv bound: split weights into positive/negative parts.
    """
    W = conv.weight
    b = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels, device=W.device, dtype=W.dtype)
    W_pos = torch.clamp(W, min=0.0)
    W_neg = torch.clamp(W, max=0.0)

    yl = F.conv2d(xl, W_pos, None, conv.stride, conv.padding, conv.dilation, conv.groups) + \
         F.conv2d(xu, W_neg, None, conv.stride, conv.padding, conv.dilation, conv.groups) + b.view(1, -1, 1, 1)
    yu = F.conv2d(xu, W_pos, None, conv.stride, conv.padding, conv.dilation, conv.groups) + \
         F.conv2d(xl, W_neg, None, conv.stride, conv.padding, conv.dilation, conv.groups) + b.view(1, -1, 1, 1)
    return yl, yu

@torch.no_grad()
def _ibp_bounds_through_block(block: nn.Module, l: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # main path: conv1->bn1->relu->conv2->bn2
    xl, xu = _conv_bounds(l, u, block.conv1)
    xl, xu = _bn_bounds(xl, xu, block.bn1)
    xl, xu = _relu_bounds(xl, xu)
    xl, xu = _conv_bounds(xl, xu, block.conv2)
    xl, xu = _bn_bounds(xl, xu, block.bn2)

    # skip path
    if getattr(block, "downsample", None) is not None:
        lds, uds = l, u
        for m in block.downsample:
            if isinstance(m, nn.Conv2d):
                lds, uds = _conv_bounds(lds, uds, m)
            elif isinstance(m, nn.BatchNorm2d):
                lds, uds = _bn_bounds(lds, uds, m)
            elif isinstance(m, nn.ReLU):
                lds, uds = _relu_bounds(lds, uds)
        xl, xu = xl + lds, xu + uds
    else:
        xl, xu = xl + l, xu + u

    # torchvision BasicBlock applies ReLU after residual add
    if hasattr(block, "relu") and isinstance(block.relu, nn.ReLU):
        xl, xu = _relu_bounds(xl, xu)

    return xl, xu

def fastlip_local_bound_resnet18(model: nn.Module, x: torch.Tensor, cfg: FastLipConfig) -> Dict[str, float]:
    assert x.size(0) == 1, "Use batch size 1."
    model.eval()

    # build interval around x (unchanged)
    l0, u0 = x.detach().clone(), x.detach().clone()
    if cfg.norm == "linf":
        l0 = torch.clamp(l0 - cfg.eps, *cfg.input_clamp)
        u0 = torch.clamp(u0 + cfg.eps, *cfg.input_clamp)
    elif cfg.norm == "l2":
        r = cfg.eps / math.sqrt(x.numel())
        l0 = torch.clamp(l0 - r, *cfg.input_clamp)
        u0 = torch.clamp(u0 + r, *cfg.input_clamp)
    else:
        raise ValueError("norm must be 'l2' or 'linf'")

    xl, xu = l0, u0
    local_L = 1.0

    def out_dim(L_in, k, p, s, d):
        return (L_in + 2 * p - d * (k - 1) - 1) // s + 1

    # analytic shapes (do NOT read from xl/xu for conv spectral calls)
    C, H, W = x.shape[1:]

    # stem: conv1 -> bn1 -> relu
    _assert_conv_inshape(model.conv1, (C, H, W), where="stem.conv1")
    xl, xu = _conv_bounds(xl, xu, model.conv1)
    local_L *= _conv2d_power_norm(model.conv1, (C, H, W), 15)
    H = out_dim(H, model.conv1.kernel_size[0], model.conv1.padding[0], model.conv1.stride[0], model.conv1.dilation[0])
    W = out_dim(W, model.conv1.kernel_size[1], model.conv1.padding[1], model.conv1.stride[1], model.conv1.dilation[1])
    C = model.conv1.out_channels

    xl, xu = _bn_bounds(xl, xu, model.bn1)
    local_L *= _bn_lipschitz(model.bn1)
    xl, xu = _relu_bounds(xl, xu)

    # layers 1..4
    for lname in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(model, lname)
        for block in layer:
            # --- snapshot pre-block shape for conv1 & downsample ---
            Cin_blk, Hin_blk, Win_blk = C, H, W

            # IBP through the whole block (intervals only)
            xl, xu = _ibp_bounds_through_block(block, xl, xu)

            # ------- spectral factors with correct shapes -------
            # conv1 (uses pre-block shape)
            _assert_conv_inshape(block.conv1, (Cin_blk, Hin_blk, Win_blk), where=f"{lname}.conv1")
            s1 = _conv2d_power_norm(block.conv1, (Cin_blk, Hin_blk, Win_blk), 10) * _bn_lipschitz(block.bn1)
            H1 = out_dim(Hin_blk, block.conv1.kernel_size[0], block.conv1.padding[0],
                         block.conv1.stride[0], block.conv1.dilation[0])
            W1 = out_dim(Win_blk, block.conv1.kernel_size[1], block.conv1.padding[1],
                         block.conv1.stride[1], block.conv1.dilation[1])
            C1 = block.conv1.out_channels

            # conv2 (uses conv1 output shape)
            _assert_conv_inshape(block.conv2, (C1, H1, W1), where=f"{lname}.conv2")
            s2 = _conv2d_power_norm(block.conv2, (C1, H1, W1), 10) * _bn_lipschitz(block.bn2)
            H2 = out_dim(H1, block.conv2.kernel_size[0], block.conv2.padding[0],
                         block.conv2.stride[0], block.conv2.dilation[0])
            W2 = out_dim(W1, block.conv2.kernel_size[1], block.conv2.padding[1],
                         block.conv2.stride[1], block.conv2.dilation[1])
            C2 = block.conv2.out_channels
            s_main = s1 * s2

            # skip path (if present) starts from pre-block shape
            if getattr(block, "downsample", None) is not None:
                s_skip = 1.0
                Cds, Hds, Wds = Cin_blk, Hin_blk, Win_blk
                for m in block.downsample:
                    if isinstance(m, nn.Conv2d):
                        _assert_conv_inshape(m, (Cds, Hds, Wds), where=f"{lname}.downsample.conv")
                        s_skip *= _conv2d_power_norm(m, (Cds, Hds, Wds), 10)
                        Hds = out_dim(Hds, m.kernel_size[0], m.padding[0], m.stride[0], m.dilation[0])
                        Wds = out_dim(Wds, m.kernel_size[1], m.padding[1], m.stride[1], m.dilation[1])
                        Cds = m.out_channels
                    elif isinstance(m, nn.BatchNorm2d):
                        s_skip *= _bn_lipschitz(m)
                local_L *= (s_main + s_skip)
            else:
                local_L *= (s_main + 1.0)

            # --- commit new analytic shape AFTER finishing the block ---
            C, H, W = C2, H2, W2
            # (final ReLU after residual add is 1-Lip)

    # optional bottleneck before avgpool
    if hasattr(model, "bottleneck") and isinstance(model.bottleneck, nn.Conv2d):
        _assert_conv_inshape(model.bottleneck, (C, H, W), where="bottleneck")
        local_L *= _conv2d_power_norm(model.bottleneck, (C, H, W), 15)
        H = out_dim(H, model.bottleneck.kernel_size[0], model.bottleneck.padding[0],
                    model.bottleneck.stride[0], model.bottleneck.dilation[0])
        W = out_dim(W, model.bottleneck.kernel_size[1], model.bottleneck.padding[1],
                    model.bottleneck.stride[1], model.bottleneck.dilation[1])
        C = model.bottleneck.out_channels
        if hasattr(model, "bottleneck_bn") and isinstance(model.bottleneck_bn, nn.BatchNorm2d):
            local_L *= _bn_lipschitz(model.bottleneck_bn)

    # AdaptiveAvgPool2d → 1×1 contributes 1/sqrt(H*W)
    local_L *= 1.0 / math.sqrt(max(1, H * W))

    # FC
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        local_L *= _linear_power_norm(model.fc, 30)

    return {
        "fastlip_local_L2_upper_bound": float(local_L),
        "note": "IBP intervals + spectral per-block (pre-block shape for conv1 & skip); avgpool=1/sqrt(HW)."
    }



# ------------------------------- LipSDP stub ----------------------------------

def lipsdp_upper_bound_resnet18(*args, **kwargs):
    """
    Full LipSDP (SDP with quadratic constraints) on a ResNet18 is intractable with
    generic solvers. Prefer scalable relaxations (spectral product; CLEVER/IBP locals;
    or layer-wise SDP approximations).
    """
    raise RuntimeError(
        "LipSDP on a full ResNet18 is not practical with off-the-shelf SDP solvers.\n"
        "Use scalable relaxations (spectral product, CLEVER/IBP-style locals, or layer-wise SDP approximations)."
    )





# from __future__ import annotations
# from dataclasses import dataclass
# from typing import Tuple, Optional, Dict
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # Optional SciPy for EVT fit in CLEVER (falls back to empirical max if missing)
# try:
#     import numpy as np
#     from scipy.stats import genextreme
#     _HAS_SCIPY = True
# except Exception:
#     import numpy as np
#     _HAS_SCIPY = False


# # ----------------------------- Low-level utils --------------------------------

# def _conv2d_power_norm(conv: nn.Conv2d, in_shape: Tuple[int, int, int], iters: int = 30) -> float:
#     """
#     Robust spectral (L2) operator-norm estimator for Conv2d via power iteration.
#     - Uses positional args for conv_transpose2d and explicitly computes output_padding
#       so the transpose maps back to the original (H, W), avoiding version-specific
#       keyword parsing issues and shape mismatches.
#     - Bias is ignored (affine offset does not affect Lipschitz).
#     """
#     assert in_shape is not None and len(in_shape) == 3, "in_shape must be (C_in, H_in, W_in)"
#     device = conv.weight.device
#     C_in, H_in, W_in = in_shape

#     # Unpack conv hyperparams
#     kH, kW = conv.kernel_size
#     sH, sW = conv.stride
#     pH, pW = conv.padding
#     dH, dW = conv.dilation
#     g = conv.groups

#     # Start with random unit vector in input space
#     u = torch.randn(1, C_in, H_in, W_in, device=device)
#     u = u / (u.norm() + 1e-12)

#     for _ in range(iters):
#         # v = A u / ||A u||
#         v = F.conv2d(u, conv.weight, None, (sH, sW), (pH, pW), (dH, dW), g)
#         vn = v.norm()
#         if vn.item() == 0.0:
#             return 0.0
#         v = v / vn

#         # Compute output_padding that makes A^T v map back to (H_in, W_in)
#         H_out, W_out = v.size(2), v.size(3)
#         H_rec = (H_out - 1) * sH - 2 * pH + dH * (kH - 1) + 1  # transposed conv output size w/o output_padding
#         W_rec = (W_out - 1) * sW - 2 * pW + dW * (kW - 1) + 1
#         opH = H_in - H_rec
#         opW = W_in - W_rec
#         # Legal: 0 <= output_padding < stride (PyTorch requirement; dilation doesn’t increase op range)
#         max_opH = max(1, sH) - 1
#         max_opW = max(1, sW) - 1
#         opH = int(max(0, min(opH, max_opH)))
#         opW = int(max(0, min(opW, max_opW)))

#         # u = A^T v / ||A^T v||   (positional args: input, weight, bias, stride, padding, output_padding, groups, dilation)
#         u = F.conv_transpose2d(v, conv.weight, None, (sH, sW), (pH, pW), (opH, opW), g, (dH, dW))
#         un = u.norm()
#         if un.item() == 0.0:
#             return 0.0
#         u = u / un

#     # Rayleigh quotient estimate of the top singular value
#     Av = F.conv2d(u, conv.weight, None, (sH, sW), (pH, pW), (dH, dW), g)
#     return float(Av.norm().item())


# def _linear_power_norm(fc: nn.Linear, iters: int = 50) -> float:
#     """Spectral norm of a Linear layer via power iteration on W (bias ignored)."""
#     W = fc.weight.detach()
#     device = W.device
#     v = torch.randn(W.shape[1], device=device)
#     v = v / (v.norm() + 1e-12)
#     for _ in range(iters):
#         u = (W @ v); un = u.norm()
#         if un.item() == 0.0:
#             return 0.0
#         u = u / un
#         v = (W.t() @ u); vn = v.norm()
#         if vn.item() == 0.0:
#             return 0.0
#         v = v / vn
#     return float((W @ v).norm().item())


# def _bn_lipschitz(bn: nn.modules.batchnorm._BatchNorm) -> float:
#     """
#     BatchNorm in EVAL mode: max_c |gamma_c| / sqrt(var_c + eps). Call model.eval() first.
#     """
#     if not bn.affine:
#         return 1.0
#     w = bn.weight.detach()
#     var = bn.running_var.detach()
#     scale = torch.abs(w) / torch.sqrt(var + bn.eps)
#     return float(scale.max().item())


# # ------------------------ Spectral Norm Product (global) ----------------------

# @dataclass
# class GlobalBoundConfig:
#     input_size: Tuple[int, int, int]       # (C,H,W), e.g., (3,32,32) for CIFAR
#     conv_power_iters: int = 30
#     linear_power_iters: int = 50

# def spectral_product_bound_resnet18(model: nn.Module, cfg: GlobalBoundConfig) -> Dict[str, float]:
#     """
#     Global L2-Lipschitz upper bound by multiplying operator norms across the network.
#     Residual add uses L(x+y) ≤ L(x) + L(y). ReLU/Pooling are 1-Lipschitz.
#     Compatible with CIFAR-style ResNet18 and your extra 'bottleneck' Conv2d before avgpool.
#     Returns per-stage factors + 'global_bound'.
#     """
#     model.eval()

#     def conv_bn_lip(conv: nn.Conv2d, bn: Optional[nn.BatchNorm2d], in_shape: Tuple[int, int, int]):
#         s_conv = _conv2d_power_norm(conv, in_shape, iters=cfg.conv_power_iters)
#         s_bn = _bn_lipschitz(bn) if bn is not None else 1.0
#         C_out = conv.out_channels
#         C_in, H_in, W_in = in_shape

#         def out_dim(L_in, k, p, s, d):
#             # Standard conv2d output shape formula
#             return (L_in + 2 * p - d * (k - 1) - 1) // s + 1

#         H_out = out_dim(H_in, conv.kernel_size[0], conv.padding[0], conv.stride[0], conv.dilation[0])
#         W_out = out_dim(W_in, conv.kernel_size[1], conv.padding[1], conv.stride[1], conv.dilation[1])
#         return s_conv * s_bn, (C_out, H_out, W_out)

#     parts: Dict[str, float] = {}
#     lip = 1.0
#     C, H, W = cfg.input_size

#     # Stem
#     s, (C, H, W) = conv_bn_lip(model.conv1, model.bn1, (C, H, W))
#     lip *= s
#     parts["stem_conv1_bn1"] = s
#     parts["stem_relu"] = 1.0  # 1-Lip
#     # (No initial maxpool in your print; if present elsewhere, it's ≤1 so we ignore.)

#     # BasicBlock bound
#     def basic_block_lip(block: nn.Module, in_shape: Tuple[int, int, int]):
#         s1, shape1 = conv_bn_lip(block.conv1, block.bn1, in_shape)
#         s2, shape2 = conv_bn_lip(block.conv2, block.bn2, shape1)
#         s_main = s1 * s2
#         # Skip path
#         if getattr(block, "downsample", None) is not None:
#             s_skip = 1.0
#             shape_skip = in_shape
#             for m in block.downsample:
#                 if isinstance(m, nn.Conv2d):
#                     s_tmp, shape_skip = conv_bn_lip(m, None, shape_skip)
#                     s_skip *= s_tmp
#                 elif isinstance(m, nn.BatchNorm2d):
#                     s_skip *= _bn_lipschitz(m)
#             return s_main + s_skip, shape2
#         else:
#             return s_main + 1.0, shape2  # identity skip (1-Lip)

#     # Layers 1..4
#     for lname in ["layer1", "layer2", "layer3", "layer4"]:
#         layer = getattr(model, lname)
#         for i, block in enumerate(layer):
#             sb, (C, H, W) = basic_block_lip(block, (C, H, W))
#             parts[f"{lname}.block{i}"] = sb
#             lip *= sb

#     # Optional 'bottleneck' 1×1 Conv2d before avgpool in your model
#     if hasattr(model, "bottleneck") and isinstance(model.bottleneck, nn.Conv2d):
#         s_b, (C, H, W) = conv_bn_lip(model.bottleneck, None, (C, H, W))
#         parts["bottleneck"] = s_b
#         lip *= s_b

#     # AdaptiveAvgPool2d: ≤1 (we treat as 1)
#     parts["avgpool"] = 1.0

#     # FC
#     if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
#         s_fc = _linear_power_norm(model.fc, cfg.linear_power_iters)
#         parts["fc"] = s_fc
#         lip *= s_fc
#     else:
#         parts["fc"] = 1.0

#     parts["global_bound"] = lip
#     return parts


# # ------------------------------ CLEVER Score ----------------------------------

# @dataclass
# class CleverConfig:
#     norm: str = "l2"         # 'l2' or 'linf'
#     radius: float = 0.3
#     n_batches: int = 20
#     batch_size: int = 64
#     targeted: bool = False
#     fit_evt: bool = True     # requires SciPy; fallback = empirical max
#     seed: int = 0

# def _random_ball(shape, norm: str, radius: float, device):
#     if norm == "l2":
#         z = torch.randn(shape, device=device)
#         z = z / (z.view(z.size(0), -1).norm(p=2, dim=1, keepdim=True) + 1e-12)
#         u = torch.rand(z.size(0), 1, device=device)
#         # radius * U^(1/d) ≈ uniform in L2-ball (adequate for CLEVER sampling)
#         r = radius * (u ** (1.0 / (z[0].numel())))
#         return z * r.view(-1, 1, 1, 1)
#     elif norm == "linf":
#         return (torch.rand(shape, device=device) * 2 - 1) * radius
#     else:
#         raise ValueError("norm must be 'l2' or 'linf'")

# @torch.no_grad()
# def _top2_margin(logits: torch.Tensor):
#     top2 = torch.topk(logits, k=2, dim=1)
#     top1 = top2.indices[:, 0]; top2i = top2.indices[:, 1]
#     margin = logits[torch.arange(logits.size(0)), top1] - logits[torch.arange(logits.size(0)), top2i]
#     return margin, top1, top2i

# def clever_score(model: nn.Module, x: torch.Tensor, cfg: CleverConfig) -> Dict[str, float]:
#     """
#     CLEVER-style local robustness / Lipschitz proxy for a single input x (B=1).
#     Returns {'predicted_class', 'margin_logit_top1_minus_top2', 'local_grad_norm_estimate',
#              'clever_score', 'estimation', 'samples'} with EVT or empirical max.
#     """
#     model.eval()
#     assert x.size(0) == 1, "Use batch size 1 for CLEVER."
#     device = x.device

#     x0 = x.detach().clone().requires_grad_(True)
#     logits = model(x0)
#     margin, top1_idx, top2_idx = _top2_margin(logits)
#     y = int(top1_idx.item())
#     margin_val = float(margin.item())

#     grad_norms = []
#     torch.manual_seed(cfg.seed)
#     for _ in range(cfg.n_batches):
#         delta = _random_ball((cfg.batch_size, *x.shape[1:]), cfg.norm, cfg.radius, device)
#         x_pert = (x + delta).clamp(0., 1.).detach().requires_grad_(True)
#         out = model(x_pert)
#         if cfg.targeted:
#             t = torch.full((cfg.batch_size,), int(top2_idx.item()), device=device, dtype=torch.long)
#             loss = F.cross_entropy(out, t, reduction='sum')
#         else:
#             t = torch.full((cfg.batch_size,), y, device=device, dtype=torch.long)
#             loss = F.cross_entropy(out, t, reduction='sum')
#         grad = torch.autograd.grad(loss, x_pert, retain_graph=False, create_graph=False)[0]
#         g = grad.view(grad.size(0), -1)
#         norms = g.norm(p=2, dim=1) if cfg.norm == "l2" else g.norm(p=float('inf'), dim=1)
#         grad_norms.append(norms.detach().cpu().numpy())

#     grad_norms = np.concatenate(grad_norms, axis=0)

#     # EVT fit on block maxima (if SciPy available); else empirical 99.9%/max
#     if cfg.fit_evt and _HAS_SCIPY and grad_norms.size >= 50:
#         blocks = grad_norms.reshape(-1, max(1, len(grad_norms)//cfg.n_batches))[:cfg.n_batches]
#         block_max = blocks.max(axis=1)
#         c, loc, scale = genextreme.fit(block_max)
#         L_est = float(genextreme.ppf(0.999, c, loc=loc, scale=scale))
#         method = "GEV(0.999)"
#     else:
#         L_est = float(np.quantile(grad_norms, 0.999)) if grad_norms.size >= 1000 else float(grad_norms.max())
#         method = "empirical_max"

#     return {
#         "predicted_class": y,
#         "margin_logit_top1_minus_top2": margin_val,
#         "local_grad_norm_estimate": L_est,
#         "clever_score": margin_val / (L_est + 1e-12),
#         "estimation": method,
#         "samples": int(grad_norms.size),
#     }


# # ----------- Fast-Lip (practical): IBP-style local L2 upper bound --------------

# @dataclass
# class FastLipConfig:
#     eps: float = 0.3
#     norm: str = "l2"                 # 'l2' or 'linf' (box for IBP when 'l2')
#     input_clamp: Tuple[float, float] = (0.0, 1.0)

# @torch.no_grad()
# def _relu_bounds(xl: torch.Tensor, xu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     return F.relu(xl), F.relu(xu)

# @torch.no_grad()
# def _bn_bounds(xl: torch.Tensor, xu: torch.Tensor, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
#     # y = a*x + c per-channel, where a = gamma / sqrt(var + eps), c = beta - a*mu
#     a = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).view(1, -1, 1, 1)
#     c = (bn.bias - bn.weight * bn.running_mean / torch.sqrt(bn.running_var + bn.eps)).view(1, -1, 1, 1)
#     y1, y2 = a * xl + c, a * xu + c
#     return torch.minimum(y1, y2), torch.maximum(y1, y2)

# @torch.no_grad()
# def _conv_bounds(xl: torch.Tensor, xu: torch.Tensor, conv: nn.Conv2d) -> Tuple[torch.Tensor, torch.Tensor]:
#     W = conv.weight
#     b = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels, device=W.device)
#     W_pos = torch.clamp(W, min=0.0)
#     W_neg = torch.clamp(W, max=0.0)
#     yl = F.conv2d(xl, W_pos, None, conv.stride, conv.padding, conv.dilation, conv.groups) + \
#          F.conv2d(xu, W_neg, None, conv.stride, conv.padding, conv.dilation, conv.groups) + b.view(1, -1, 1, 1)
#     yu = F.conv2d(xu, W_pos, None, conv.stride, conv.padding, conv.dilation, conv.groups) + \
#          F.conv2d(xl, W_neg, None, conv.stride, conv.padding, conv.dilation, conv.groups) + b.view(1, -1, 1, 1)
#     return yl, yu

# @torch.no_grad()
# def _ibp_bounds_through_block(block: nn.Module, l: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     # main path: conv1->bn1->relu->conv2->bn2
#     xl, xu = _conv_bounds(l, u, block.conv1)
#     xl, xu = _bn_bounds(xl, xu, block.bn1)
#     xl, xu = _relu_bounds(xl, xu)
#     xl, xu = _conv_bounds(xl, xu, block.conv2)
#     xl, xu = _bn_bounds(xl, xu, block.bn2)

#     # skip path
#     if getattr(block, "downsample", None) is not None:
#         lds, uds = l, u
#         for m in block.downsample:
#             if isinstance(m, nn.Conv2d):
#                 lds, uds = _conv_bounds(lds, uds, m)
#             elif isinstance(m, nn.BatchNorm2d):
#                 lds, uds = _bn_bounds(lds, uds, m)
#             elif isinstance(m, nn.ReLU):
#                 lds, uds = _relu_bounds(lds, uds)
#         xl, xu = xl + lds, xu + uds
#     else:
#         xl, xu = xl + l, xu + u

#     # torchvision BasicBlock applies a ReLU after the residual addition
#     if hasattr(block, "relu") and isinstance(block.relu, nn.ReLU):
#         xl, xu = _relu_bounds(xl, xu)

#     return xl, xu

# @torch.no_grad()
# def fastlip_local_bound_resnet18(model: nn.Module, x: torch.Tensor, cfg: FastLipConfig) -> Dict[str, float]:
#     """
#     Practical Fast-Lip-like local L2 *upper bound* at input x via interval bound propagation (IBP).
#     - Builds a local box around x (Linf; for L2 uses inscribed box).
#     - Propagates intervals through ResNet18.
#     - Multiplies per-block operator norms (with residual sum rule).
#     Note: Uncertain ReLUs are treated as slope=1 (conservative).
#     """
#     assert x.size(0) == 1, "Use batch size 1."
#     model.eval()

#     # Build interval around x
#     l0, u0 = x.detach().clone(), x.detach().clone()
#     if cfg.norm == "linf":
#         l0 = torch.clamp(l0 - cfg.eps, *cfg.input_clamp)
#         u0 = torch.clamp(u0 + cfg.eps, *cfg.input_clamp)
#     elif cfg.norm == "l2":
#         r = cfg.eps / math.sqrt(x.numel())  # inscribed Linf box radius
#         l0 = torch.clamp(l0 - r, *cfg.input_clamp)
#         u0 = torch.clamp(u0 + r, *cfg.input_clamp)
#     else:
#         raise ValueError("norm must be 'l2' or 'linf'")

#     local_L = 1.0

#     # Stem: conv1 -> bn1 -> relu
#     xl, xu = _conv_bounds(l0, u0, model.conv1)
#     local_L *= _conv2d_power_norm(model.conv1, x.shape[1:], 15)
#     xl, xu = _bn_bounds(xl, xu, model.bn1)
#     local_L *= _bn_lipschitz(model.bn1)
#     xl, xu = _relu_bounds(xl, xu)

#     # Layers 1..4
#     for lname in ["layer1", "layer2", "layer3", "layer4"]:
#         layer = getattr(model, lname)
#         for block in layer:
#             # interval propagation through the block (with residual + final ReLU)
#             xl, xu = _ibp_bounds_through_block(block, xl, xu)

#             # conservative per-block Lipschitz factor (product on main; + skip)
#             s1 = _conv2d_power_norm(block.conv1, (xl.size(1), xl.size(2), xl.size(3)), 10) * _bn_lipschitz(block.bn1)
#             s2 = _conv2d_power_norm(block.conv2, (xl.size(1), xl.size(2), xl.size(3)), 10) * _bn_lipschitz(block.bn2)
#             s_main = s1 * s2

#             if getattr(block, "downsample", None) is not None:
#                 s_skip = 1.0
#                 for m in block.downsample:
#                     if isinstance(m, nn.Conv2d):
#                         s_skip *= _conv2d_power_norm(m, (xl.size(1), xl.size(2), xl.size(3)), 10)
#                     elif isinstance(m, nn.BatchNorm2d):
#                         s_skip *= _bn_lipschitz(m)
#             else:
#                 s_skip = 1.0

#             local_L *= (s_main + s_skip)

#     # Optional bottleneck (present in your model)
#     if hasattr(model, "bottleneck") and isinstance(model.bottleneck, nn.Conv2d):
#         xl, xu = _conv_bounds(xl, xu, model.bottleneck)
#         # (No BN on bottleneck in your print; if BN exists, apply _bn_bounds and _bn_lipschitz)
#         local_L *= _conv2d_power_norm(model.bottleneck, (xl.size(1), xl.size(2), xl.size(3)), 15)

#     # AdaptiveAvgPool2d: ≤1 (no change to local_L). We do not change intervals here.
#     # FC
#     if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
#         local_L *= _linear_power_norm(model.fc, 30)

#     return {
#         "fastlip_local_L2_upper_bound": float(local_L),
#         "note": "IBP-like conservative local bound (Fast-Lip inspired). Handles residual ReLU and bottleneck."
#     }


# # ------------------------------- LipSDP stub ----------------------------------

# def lipsdp_upper_bound_resnet18(*args, **kwargs):
#     """
#     Full LipSDP (SDP with quadratic constraints) on a ResNet18 is intractable with
#     generic solvers. Prefer scalable relaxations (spectral product; IBP/linear bound
#     propagation; or layer-wise SDP approximations).
#     """
#     raise RuntimeError(
#         "LipSDP on a full ResNet18 is not practical with off-the-shelf SDP solvers.\n"
#         "Use scalable relaxations (spectral product, CLEVER/IBP-style locals, or layer-wise SDP approximations)."
#     )
