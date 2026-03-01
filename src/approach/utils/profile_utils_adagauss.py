# -*- coding: utf-8 -*-
"""
profile_utils.py  (baseline version for the “projected / logit / feature / none” method)

What this file does
-------------------
1) Counts FLOPs for the *exact* forward subgraph that is active at task t:
   - t = 0 (no old-model branches): backbone + a tiny head to exercise CE path.
   - t > 0:
       • distillation == "projected": backbone + Distiller(D) + old_model forward
       • distillation == "logit"    : backbone + old_model forward + old heads on (z_old, z_new)
       • distillation == "feature"  : backbone + old_model forward
       • distillation == "none"     : backbone only (plus tiny head)
   PASS-style 4-way rotation at t=0 can be reflected by multiplying FLOPs by 4.

2) Measures latency (ms/iter) for inference (backbone.forward) and training step (you pass a closure).

Paper-friendly convention
-------------------------
- Training FLOPs ≈ 3 × forward FLOPs  (backward for conv/linear is ~2× forward).
- Inference FLOPs is usually reported for backbone forward with batch=1.

Safe graph attachment
---------------------
To ensure fvcore keeps auxiliary branches (distiller, old_model, old heads) in the traced graph,
we *attach* their scalar summaries to the main output via:  out' = out + 0.0 * summary.
This prevents pruning without changing outputs. Avoid in-place ops (no add_).
"""

from typing import Callable, Dict, Tuple, Any
from contextlib import contextmanager
import time

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, parameter_count


# ------------------------------------------------------------------
# Precise GPU timing via CUDA events (safe & minimal)
# ------------------------------------------------------------------

@contextmanager
def cuda_timer(sync: bool = True):
    """
    Precise GPU timer (CPU fallback).
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


# ------------------------------------------------------------------
# FLOPs helpers
# ------------------------------------------------------------------

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


def _attach_to_graph(anchor: torch.Tensor, hook_tensor: torch.Tensor) -> torch.Tensor:
    """
    Keep auxiliary branches inside the traced graph without changing outputs:
        anchor' = anchor + 0.0 * reduce(hook_tensor)
    Avoid in-place ops (no add_) to prevent aten::add_ warnings.
    """
    hook = hook_tensor
    if hook.ndim > 0:
        hook = hook.mean()
    return anchor + 0.0 * hook


class ProfileGraph(nn.Module):
    """
    Build a forward-only subgraph for FLOPs counting that matches what is active
    at task `t` for your baseline method:

      distillation ∈ {"projected", "logit", "feature", "none"}.
      mode ∈ {"train_t0", "train_tpos", "infer"}.

    Notes:
    - This module never changes model parameters. It only “touches” ops to count FLOPs.
    - old_model and old heads are attached via _attach_to_graph(...) so fvcore keeps them.
    """

    def __init__(self, appr, t: int, mode: str = "train_tpos"):
        super().__init__()
        self.appr = appr
        self.t = t
        self.mode = mode

        self.backbone = appr.model
        self.old_model = getattr(appr, "old_model", None)  # set by your training loop after t=0
        self.S = appr.S

        # Distiller stub (only for "projected")
        self.distiller = None
        if getattr(appr, "distillation", "none") == "projected":
            # Prefer a persisted module if you ever save it, else build a counting stub (linear/MLP).
            # Here we build a minimal linear head that matches your train_backbone projected distiller shape.
            if getattr(appr, "distiller_type", "mlp") == "mlp":
                self.distiller = nn.Sequential(
                    nn.Linear(self.S, self.S * getattr(appr, "multiplier", 32), bias=True),
                    nn.GELU(),
                    nn.Linear(self.S * getattr(appr, "multiplier", 32), self.S, bias=True),
                )
            else:
                self.distiller = nn.Linear(self.S, self.S, bias=True)
            self.distiller.to(appr.device)

        # Old heads (for "logit" KD)
        self.old_heads = list(appr.heads) if len(appr.heads) > 0 else None

        # Tiny head to exercise CE path for current task (does not affect your real model).
        num_cls_t = appr.classes_in_tasks[t] if t < len(appr.classes_in_tasks) else None
        self.head_t = nn.Linear(self.S, num_cls_t).to(appr.device) if (num_cls_t is not None) else None

        self.to(appr.device).eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_new = self.backbone(x)

        # Inference path: usually backbone-only; optionally hang a head to keep a consistent end node.
        if self.mode == "infer":
            if self.head_t is not None:
                logits = self.head_t(z_new)
                z_new = _attach_to_graph(z_new, logits)
            return z_new

        # Training paths include the per-task CE head to match training compute.
        if self.head_t is not None:
            logits_t = self.head_t(z_new)
            z_new = _attach_to_graph(z_new, logits_t)

        # t=0: no old-model branches (distiller / old_model / old heads)
        if self.t == 0:
            return z_new

        # t>0: attach branches according to distillation type
        distill_mode = getattr(self.appr, "distillation", "none")

        # (A) projected: use Distiller(D) on z_new and old_model forward on x
        if distill_mode == "projected" and (self.distiller is not None) and (self.old_model is not None):
            z_old_hat = self.distiller(z_new)         # D(z_new)
            z_new = _attach_to_graph(z_new, z_old_hat)
            with torch.no_grad():
                z_old = self.old_model(x)             # old forward
            z_new = _attach_to_graph(z_new, z_old)

        # (B) feature: only need old_model forward (MSE(features, old_features))
        elif distill_mode == "feature" and (self.old_model is not None):
            with torch.no_grad():
                z_old = self.old_model(x)
            z_new = _attach_to_graph(z_new, z_old)

        # (C) logit: need old_model forward + old heads applied to both z_old and z_new
        elif distill_mode == "logit" and (self.old_model is not None) and (self.old_heads is not None) and len(self.old_heads) > 0:
            with torch.no_grad():
                z_old = self.old_model(x)
                tlogits = torch.cat([h(z_old) for h in self.old_heads], dim=1)  # teacher logits (old)
            slogits = torch.cat([h(z_new) for h in self.old_heads], dim=1)      # student logits (new)
            z_new = _attach_to_graph(z_new, tlogits.mean() + slogits.mean())

        # (D) none: nothing else to attach
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
    Count forward FLOPs (as integer) for the subgraph active at (t, mode).

    mode ∈ {"train_t0","train_tpos","infer"}.
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


# ------------------------------------------------------------------
# Latency helpers
# ------------------------------------------------------------------

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
