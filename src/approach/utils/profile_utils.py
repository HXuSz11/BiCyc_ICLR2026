# -*- coding: utf-8 -*-
"""
profile_utils.py

Utilities for FLOPs counting and latency measurement that respect
task phase (t=0 vs. t>0), your distillation/adapter/KD settings,
and optional PASS-style rotation at t=0.

Conventions used in papers:
- Training FLOPs ≈ 3 × forward FLOPs (backward for conv/linear is ~2× forward).
- Inference FLOPs typically refers to backbone forward with batch=1.
"""

from typing import Callable, Dict, Tuple, Any
from contextlib import contextmanager
import time

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, parameter_count


# ------------------------------------------------------------
# Precise GPU timing via CUDA events (safe & minimal)
# ------------------------------------------------------------

@contextmanager
def cuda_timer(sync: bool = True):
    """
    Precise GPU timer (or CPU fallback).
    Usage:
        with cuda_timer() as get_ms:
            # code to measure
        elapsed_ms = get_ms()
    """
    if not torch.cuda.is_available():
        start = time.perf_counter()
        yield lambda: (time.perf_counter() - start) * 1000.0
        return

    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)
    if sync:
        torch.cuda.synchronize()
    starter.record()
    yield lambda: starter.elapsed_time(ender)
    ender.record()
    if sync:
        torch.cuda.synchronize()


def avg_ms(fn: Callable[[], None], warmup: int = 50, iters: int = 200) -> float:
    """
    Run `fn` multiple times and return average latency (ms).
    """
    for _ in range(max(0, warmup)):
        fn()
    acc = 0.0
    for _ in range(max(1, iters)):
        with cuda_timer() as get_ms:
            fn()
        acc += get_ms()
    return acc / max(1, iters)


# ------------------------------------------------------------
# FLOPs helpers
# ------------------------------------------------------------

def forward_flops_and_params(model: nn.Module, example_inputs: torch.Tensor) -> Tuple[int, Dict[str, Any]]:
    """
    Count forward FLOPs and parameters for a single module (e.g., backbone).
    Returns (total_flops, {"params": dict of layer_name -> num_params}).
    """
    model.eval()
    with torch.no_grad():
        flops = FlopCountAnalysis(model, example_inputs).total()
        params = parameter_count(model)
    return int(flops), {"params": params}


def _attach_to_graph(anchor: torch.Tensor, hook_scalar: torch.Tensor) -> torch.Tensor:
    """
    Make sure sub-branches are kept in the traced graph without changing outputs:
    anchor' = anchor + 0.0 * reduce(hook)
    Avoid in-place ops (no add_) to prevent aten::add_ warnings.
    """
    hook = hook_scalar
    if hook.ndim > 0:
        hook = hook.mean()
    return anchor + 0.0 * hook


class ProfileGraph(nn.Module):
    """
    Build an exact forward-only subgraph for FLOPs counting that matches what is active
    at a given task `t` under your settings.

    mode ∈ {"train_t0", "train_tpos", "pta", "infer"}:
      - "train_t0"  : training forward at task 0 (no old-model branches).
      - "train_tpos": training forward at tasks t>0 (include D/A + cycle, KD logits, proto-contrast, etc. as enabled).
      - "pta"       : adapter-only mapping A(z_old) at tasks t>0 (used only if you enable PTA).
      - "infer"     : inference forward (usually backbone-only; you may optionally hang a tiny head to unify topology).

    Notes:
    - This module never changes model parameters. It only “touches” ops to count FLOPs.
    - Old branches are attached to the output via _attach_to_graph(...), so fvcore keeps them.
    - IMPORTANT: We compute old_model(x) AT MOST ONCE and reuse it across branches to avoid double-counting.
    """

    def __init__(self, appr, t: int, mode: str = "train_tpos"):
        super().__init__()
        self.appr = appr
        self.t = t
        self.mode = mode

        self.backbone = appr.model
        self.old_model = getattr(appr, "old_model", None)  # set by the training loop after t=0
        self.S = appr.S

        # Prefer persisted modules (trained), otherwise build fresh counting stubs (same topology).
        self.distiller = getattr(appr, "distiller_module", None)
        self.adapter   = getattr(appr, "adapter_module", None)
        if self.distiller is None and (appr.distillation in ("projected", "projected_bi")):
            self.distiller = appr._build_projection(appr.distiller_type).to(appr.device)
        if self.adapter is None:
            self.adapter = appr._build_projection(appr.adapter_type).to(appr.device)

        self.old_heads = list(appr.heads) if len(appr.heads) > 0 else None

        # Tiny head to exercise CE path for current task (does not affect your real model).
        num_cls_t = appr.classes_in_tasks[t] if t < len(appr.classes_in_tasks) else None
        self.head_t = nn.Linear(self.S, num_cls_t).to(appr.device) if (num_cls_t is not None) else None

        # Caches to avoid repeated compute (and double-counting)
        self._cached_z_old = None
        self._cached_d_new = None
        self._cached_a_old = None

        self.to(appr.device).eval()

    def _get_z_old(self, x: torch.Tensor) -> torch.Tensor:
        if self._cached_z_old is None:
            # old backbone forward computed ONCE per graph execution
            with torch.no_grad():
                self._cached_z_old = self.old_model(x)
        return self._cached_z_old

    def _get_d_new(self, z_new: torch.Tensor) -> torch.Tensor:
        if self._cached_d_new is None and self.distiller is not None:
            self._cached_d_new = self.distiller(z_new)
        return self._cached_d_new

    def _get_a_old(self, z_old: torch.Tensor) -> torch.Tensor:
        if self._cached_a_old is None and self.adapter is not None:
            self._cached_a_old = self.adapter(z_old)
        return self._cached_a_old

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Always one NEW backbone forward
        z_new = self.backbone(x)

        # Inference path: usually backbone-only; optionally hang a head to keep a consistent end node.
        if self.mode == "infer":
            if self.head_t is not None:
                logits = self.head_t(z_new)
                z_new = _attach_to_graph(z_new, logits)
            return z_new

        # PTA path: by design we include A(z_old); note that z_new is still computed above
        if self.mode == "pta":
            assert self.t > 0 and self.old_model is not None and self.adapter is not None
            z_old = self._get_z_old(x)              # computed once
            z_new = _attach_to_graph(z_new, self._get_a_old(z_old))
            return z_new

        # Training paths include the per-task CE head to match training compute.
        if self.head_t is not None:
            logits_t = self.head_t(z_new)
            z_new = _attach_to_graph(z_new, logits_t)

        # t=0: no old-model branches (D/A/KD/proto/transfer)
        if self.t == 0:
            return z_new

        # t>0: attach branches according to settings.

        # (1) projected / projected_bi: D(new), and for bi also A(old) + two cycle legs
        if self.appr.distillation in ("projected", "projected_bi") and \
           (self.distiller is not None) and (self.old_model is not None):
            z_old = self._get_z_old(x)                    # ONE old backbone forward
            d_new = self._get_d_new(z_new)                # D(z_new)
            z_new = _attach_to_graph(z_new, d_new)
            if self.appr.distillation == "projected_bi" and (self.adapter is not None):
                a_old = self._get_a_old(z_old)            # A(z_old)
                z_new = _attach_to_graph(z_new, a_old)
                # cycle legs (use cached d_new/a_old)
                z_new = _attach_to_graph(z_new, self.adapter(d_new))
                z_new = _attach_to_graph(z_new, self.distiller(a_old))

        # (2) KD logits over old heads (forward-only; reuse cached z_old; softmax/gates are negligible)
        if getattr(self.appr, "use_aux_logit_kd", False) and self.old_heads:
            z_old = self._get_z_old(x)                    # REUSE cached old features
            with torch.no_grad():
                tlogits = torch.cat([h(z_old) for h in self.old_heads], dim=1)
            slogits = torch.cat([h(z_new) for h in self.old_heads], dim=1)
            z_new = _attach_to_graph(z_new, tlogits.mean() + slogits.mean())

        # (3) Proto-contrast: in 'maha' mode we need D(z_new); reuse cache
        if getattr(self.appr, "proto_contrast", False):
            if getattr(self.appr, "proto_contrast_metric", "cos") == "maha" and self.distiller is not None:
                z_new = _attach_to_graph(z_new, self._get_d_new(z_new))

        # (4) Transfer-backbone: MSE(z_new, A(z_old)) leg; reuse cached z_old / a_old
        if getattr(self.appr, "use_transfer_backbone", False) and \
           self.adapter is not None and self.old_model is not None:
            z_old = self._get_z_old(x)
            z_new = _attach_to_graph(z_new, self._get_a_old(z_old))

        return z_new


def count_forward_flops(
    appr,
    t: int,
    mode: str,
    H: int,
    W: int,
    C: int = 3,
    rotation_task0: bool = False
) -> int:
    """
    Count forward FLOPs (as integer) for the exact subgraph active at (t, mode).

    mode ∈ {"train_t0","train_tpos","pta","infer"}.
    If PASS rotation is enabled at t=0, set `rotation_task0=True` to multiply by 4.
    """
    if mode == "train_t0" and t != 0:
        raise ValueError("mode=train_t0 should only be used at t=0")
    if mode == "train_tpos" and t == 0:
        raise ValueError("mode=train_tpos should only be used at t>0")

    graph = ProfileGraph(appr, t, mode=("train_tpos" if (mode == "train_tpos" and t > 0) else mode)).eval()
    x = torch.randn(1, C, H, W, device=appr.device)
    with torch.no_grad():
        flops = FlopCountAnalysis(graph, x).total()

    if rotation_task0 and (mode == "train_t0") and (t == 0):
        flops *= 4  # 4-way rotation at task 0

    return int(flops)


# ------------------------------------------------------------
# Latency helpers
# ------------------------------------------------------------

def measure_inference_latency_ms(appr, H: int, W: int, C: int = 3, bs: int = 1,
                                 warmup: int = 50, iters: int = 200) -> float:
    """
    Measure inference latency (ms/iter) for backbone.forward with batch=bs.
    Paper reporting usually uses batch=1 and per-image latency.
    """
    x = torch.randn(bs, C, H, W, device=appr.device)

    @torch.no_grad()
    def _one():
        _ = appr.model(x)

    return avg_ms(_one, warmup=warmup, iters=iters)


def measure_train_iter_latency_ms(step_fn: Callable[[], None],
                                  warmup: int = 20, iters: int = 100) -> float:
    """
    Measure training iteration latency (ms/iter).
    Pass a closure `step_fn()` that performs ONE full training step:
    forward + loss + backward + optimizer.step.
    """
    return avg_ms(step_fn, warmup=warmup, iters=iters)
