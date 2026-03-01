import torch
from .cflat_util import enable_running_stats, disable_running_stats
from torch.distributed import ReduceOp
import contextlib
import math

class C_Flat(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, model, cflat=False, rho=0.2, lamb=0.2, adaptive=False, perturb_eps=1e-12,
                 grad_reduce='mean', 
                 rho_scheduler="none", 
                 rho_min=0.0,          
                 T_max=0,              
                 lr_min=0.0,           
                 **kwargs):
        defaults = dict(adaptive=adaptive, **kwargs)
        super(C_Flat, self).__init__(params, defaults)
        self.perturb_eps = perturb_eps
        self.model = model
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.adaptive = adaptive
        self.get_grad_reduce(grad_reduce)

        self.cflat = cflat
        self.rho = rho
        self.lamb = lamb

        self.rho_scheduler = rho_scheduler.lower()
        self.rho_initial = rho 
        self.rho_min = rho_min
        self.T_max = T_max
        self.current_step = 0

        if self.rho_scheduler == 'sync_with_lr':
            self._init_sync_with_lr(lr_min)
        elif self.rho_scheduler in ['cosine', 'linear']:
            if T_max <= 0:
                 raise ValueError("T_max (total steps) must be > 0 for cosine or linear rho_scheduler.")
        elif self.rho_scheduler != "none":
            raise ValueError(f"Unknown rho_scheduler: {self.rho_scheduler}")

    def _init_sync_with_lr(self, lr_min):
        self.lr_min = lr_min
        try:
            self.lr_initial = self.base_optimizer.param_groups[0]['lr']
        except (KeyError, IndexError):
            raise ValueError("Could not determine initial learning rate from base_optimizer for 'sync_with_lr'. Ensure base_optimizer is initialized with 'lr'.")

    @torch.no_grad()
    def _update_rho(self):
        if self.rho_scheduler == "none":
            return
        elif self.rho_scheduler == 'sync_with_lr':
            self._update_rho_from_lr()
        elif self.rho_scheduler in ['cosine', 'linear']:
            self._update_rho_step_based()

    def _update_rho_from_lr(self):
        try:
            current_lr = self.base_optimizer.param_groups[0]['lr']
        except (KeyError, IndexError):
            return 

        # rho_i = rho_min + (rho_max - rho_min) * (lr_i - lr_min) / (lr_max - lr_min)
        
        lr_range = self.lr_initial - self.lr_min
        
        if lr_range > 1e-9: 
            ratio = max(0.0, min(1.0, (current_lr - self.lr_min) / lr_range))
            self.rho = self.rho_min + (self.rho_initial - self.rho_min) * ratio
        elif current_lr <= self.lr_min:
            self.rho = self.rho_min

    def _update_rho_step_based(self):
        if self.current_step >= self.T_max:
            self.rho = self.rho_min
            return

        progress = self.current_step / self.T_max

        if self.rho_scheduler == "linear":
            self.rho = self.rho_min + (self.rho_initial - self.rho_min) * (1 - progress)
        elif self.rho_scheduler == "cosine":
            self.rho = self.rho_min + 0.5 * (self.rho_initial - self.rho_min) * (1 + math.cos(math.pi * progress))

    def get_grad_reduce(self, grad_reduce: str):
        if grad_reduce.lower() == 'mean':
            if hasattr(ReduceOp, 'AVG'):
                self.grad_reduce = ReduceOp.AVG
                self.manual_average = False
            else:  # PyTorch <= 1.11.0 does not have AVG, need to manually average across processes
                self.grad_reduce = ReduceOp.SUM
                self.manual_average = True
        elif grad_reduce.lower() == 'sum':
            self.grad_reduce = ReduceOp.SUM
            self.manual_average = False
        else:
            raise ValueError('"grad_reduce" should be one of ["mean", "sum"].')

    @torch.no_grad()
    def perturb_weights(self, perturb_idx: int):
        grad_norm = self._grad_norm(weight_adaptive=self.adaptive)
        scale = self.rho / (grad_norm + self.perturb_eps)

        if perturb_idx == 0:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    self.state[p]["g_0"] = p.grad.data.clone()
                    e_w = p.grad * scale.to(p)          # e_w
                    p.add_(e_w)                         # w + e_w
                    self.state[p]['e_w_0'] = e_w

        elif perturb_idx == 1:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    self.state[p]["g_2"] = p.grad.data.clone()
                    e_w = p.grad * scale.to(p)
                    p.add_(e_w)
                    self.state[p]['e_w_1_2'] += e_w

        else:
            raise ValueError('"perturb_idx" should be one of [0, 1].')

    @torch.no_grad()
    def grad_norm_ascent(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["g_1"] = p.grad.data.clone()
                p.grad.data -= self.state[p]["g_0"]

        grad_norm = self._grad_norm(weight_adaptive=self.adaptive)
        scale = self.rho / (grad_norm + self.perturb_eps)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                if self.adaptive:
                    e_w *= torch.pow(p, 2)
                p.add_(e_w)
                self.state[p]['e_w_1_2'] = e_w

    #      zero_grad, base_step, state_dict, load_state_dict, maybe_no_sync, set_closure) ...
    
    @torch.no_grad()
    def unperturb(self, perturb_key: str):
        for group in self.param_groups:
            for p in group['params']:
                if perturb_key in self.state[p].keys():
                    p.data.sub_(self.state[p][perturb_key])

    @torch.no_grad()
    def gradient_aggregation(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                p.grad.data = self.state[p]['g_1'] + self.lamb * (p.grad.data.detach().clone() - self.state[p]['g_2'])

    @torch.no_grad()
    def _grad_norm(self, weight_adaptive: bool = False):
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p.data) if weight_adaptive else 1.0) * p.grad).norm(p=2)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    @torch.no_grad()
    def _sync_grad(self):
        if torch.distributed.is_initialized():  # synchronize final gardients
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    if self.manual_average:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
                        world_size = torch.distributed.get_world_size()
                        p.grad.div_(float(world_size))
                    else:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
        return

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none)

    def base_step(self):
        self.base_optimizer.step()

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    def maybe_no_sync(self):
        if torch.distributed.is_initialized():
            return self.model.no_sync()
        else:
            return contextlib.ExitStack()

    @torch.no_grad()
    def set_closure(self, loss_fn):
        # create self.forward_backward_func, which is a function such that
        # self.forward_backward_func() automatically performs forward and backward passes.

        def get_grad():
            self.zero_grad()
            with torch.enable_grad():
                logits, loss_list = loss_fn()
                total_loss = sum(loss_list)
            total_loss.backward()
            return logits, loss_list

        self.forward_backward_func = get_grad

    def step(self, delay=False, closure=None):
        
        if self.cflat:
            self._update_rho()
        
        self.current_step += 1

        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_backward_func

        with self.maybe_no_sync():
            logits, loss_list = get_grad()

            if self.cflat:
                self.perturb_weights(perturb_idx=0)

                disable_running_stats(self.model)
                get_grad()

                self.unperturb(perturb_key="e_w_0")
                self.grad_norm_ascent()
                get_grad()

                self.perturb_weights(perturb_idx=1)
                get_grad()

                self.gradient_aggregation()

                self.unperturb(perturb_key="e_w_1_2")

        self._sync_grad()

        if not delay:
            self.base_optimizer.step()

        enable_running_stats(self.model)

        return logits, loss_list