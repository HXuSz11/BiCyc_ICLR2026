import copy
import random
import torch
import torch.nn.functional as F
import numpy as np

from argparse import ArgumentParser
from itertools import compress
from torch import nn
from torchmetrics import Accuracy

from .mvgb import ClassMemoryDataset, ClassDirectoryDataset
from .models.resnet18 import resnet18
from .incremental_learning import Inc_Learning_Appr
from .criterions.ce import CE


# -------------------- Rotation augmentation (PASS-style) --------------------

def compute_rotations(images, targets, total_classes):
    """PASS-style self-rotation for task 0."""
    images_rot = torch.cat([torch.rot90(images, k, (2, 3)) for k in range(1, 4)])
    images = torch.cat((images, images_rot))
    target_rot = torch.cat([(targets + total_classes * k) for k in range(1, 4)])
    targets = torch.cat((targets, target_rot))
    return images, targets


# -------------------- Anti-collapse regularizer --------------------

def loss_ac(features, beta):
    """Anti-collapse regularizer via Cholesky diagonal (truncated)."""
    cov = torch.cov(features.T)
    cholesky = torch.linalg.cholesky(cov + 1e-6 * torch.eye(cov.shape[0], device=cov.device))
    ch_diag = torch.diag(cholesky)
    loss = -torch.clamp(ch_diag, max=beta).mean()
    return loss, torch.det(cov)


# -------------------- Linear head: pseudo sampling dataset --------------------

class SampledDataset(torch.utils.data.Dataset):
    """Samples pseudo-features from per-class Gaussians to train a linear head."""
    def __init__(self, distributions, samples: int, total_classes: int):
        self.distributions = distributions
        self.samples = int(samples)
        self.total_classes = int(total_classes)

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        y = random.randint(0, self.total_classes - 1)
        x = self.distributions[y].sample()
        return x, y


# ----------------------------- Main Approach -----------------------------

class Appr(Inc_Learning_Appr):
    """
    Minimal code to support your scripts + classifier {bayes, linear}:
      - Backbone: ResNet18 only (+ pretrained + 224)
      - Criterion: CE only
      - Distillation: projected_bi only (Bi-directional + cycle + optional pair-preserve)
      - Classifier: bayes (Mahalanobis) or linear (trained on pseudo-samples)
      - Optional: FP-CE at task0 (+ optional grad-norm)
      - Optional: DA-lr / DA-wd overrides for distiller/adapter (used by script #1)
      - Distiller/Adapter fixed to MLP
      - No MixUp/CutMix
      - No lambda-iso
    """

    def __init__(
        self,
        model,
        device,
        nepochs=200,
        lr=0.05,
        lr_min=1e-4,
        lr_factor=3,
        lr_patience=5,
        clipgrad=1,
        momentum=0,
        wd=0,
        multi_softmax=False,
        wu_nepochs=0,
        wu_lr_factor=1,
        fix_bn=False,
        eval_on_train=False,
        num_tasks=5,
        nc_first_task=20,
        datasets="tiny",
        logger=None,

        # core
        N=10000,
        alpha=1.0,
        lr_backbone=0.01,
        lr_adapter=0.01,
        beta=1.0,
        distillation="projected_bi",
        use_224=False,
        S=64,
        rotation=False,
        criterion="ce",
        lamb=10,
        adaptation_strategy="full",
        pretrained_net=False,
        normalize=False,
        shrink=0.0,
        multiplier=32,
        classifier="bayes",
        nnet="resnet18",

        # Bi-directional knobs
        lambda_bi=None,
        lambda_cycle=1.0,
        pair_preserve=False,
        lambda_pair=0.1,

        # FP-CE knobs (task0 only)
        rho_ce=0.0,
        lambda_fp_ce=0.0,
        fp_use_grad_norm=False,
        lambda_fp_gn=0.0,

        # distiller/adapter optimizer override
        DA_lr=0.0,
        DA_wd=0.0,
    ):
        super(Appr, self).__init__(
            model, device, nepochs, lr, lr_min, lr_factor, lr_patience,
            clipgrad, momentum, wd, multi_softmax, wu_nepochs, wu_lr_factor,
            fix_bn, eval_on_train, logger, exemplars_dataset=None
        )

        # enforce supported modes
        self.nnet = str(nnet).lower()
        if self.nnet != "resnet18":
            raise ValueError(f"[Minimal Appr] Only resnet18 supported, got nnet={self.nnet}")

        self.criterion_type = str(criterion).lower()
        if self.criterion_type != "ce":
            raise ValueError(f"[Minimal Appr] Only criterion=ce supported, got {self.criterion_type}")
        self.criterion = CE

        self.classifier = str(classifier).lower()
        if self.classifier not in ("bayes", "linear"):
            raise ValueError(f"[Minimal Appr] classifier must be bayes or linear, got {self.classifier}")

        self.distillation = str(distillation).lower()
        if self.distillation != "projected_bi":
            raise ValueError(f"[Minimal Appr] Only distillation=projected_bi supported, got {self.distillation}")

        # core hyperparameters
        self.N = int(N)
        self.S = int(S)
        self.lamb = float(lamb)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.lr_backbone = float(lr_backbone)
        self.lr_adapter = float(lr_adapter)
        self.multiplier = int(multiplier)
        self.shrink = float(shrink)
        self.adaptation_strategy = str(adaptation_strategy).lower()
        self.pretrained = bool(pretrained_net)
        self.is_normalization = bool(normalize)
        self.is_rotation = bool(rotation)

        # Bi-directional weights
        self.lambda_bi = self.lamb if (lambda_bi is None) else float(lambda_bi)
        self.lambda_cycle = float(lambda_cycle)
        self.pair_preserve = bool(pair_preserve)
        self.lambda_pair = float(lambda_pair)

        # FP-CE knobs
        self.rho_ce = float(rho_ce)
        self.lambda_fp_ce = float(lambda_fp_ce)
        self.fp_use_grad_norm = bool(fp_use_grad_norm)
        self.lambda_fp_gn = float(lambda_fp_gn)

        # DA optimizer knobs
        self.DA_lr = float(DA_lr)
        self.DA_wd = float(DA_wd)

        # build backbone
        self.model = resnet18(num_features=self.S, is_224=use_224)
        if self.pretrained:
            state_dict = torch.load("./resnet18-f37072fd.pth")
            state_dict.pop("fc.weight", None)
            state_dict.pop("fc.bias", None)
            self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device, non_blocking=True)

        # buffers / bookkeeping
        self.train_data_loaders, self.val_data_loaders = [], []
        self.means = torch.empty((0, self.S), device=self.device)
        self.covs = torch.empty((0, self.S, self.S), device=self.device)
        self.covs_raw = torch.empty((0, self.S, self.S), device=self.device)
        self.covs_inverted = None

        self.task_offset = [0]
        self.classes_in_tasks = []

        # warm-start modules
        self.distiller_module = None
        self.adapter_module = None
        self.old_model = None

        # linear head
        self.pseudo_head = None

    @staticmethod
    def extra_parser(args):
        """Parse only args that appear in (or affect) your scripts."""
        parser = ArgumentParser()

        # core
        parser.add_argument('--N', type=int, default=10000)
        parser.add_argument('--S', type=int, default=64)
        parser.add_argument('--alpha', type=float, default=1.0)
        parser.add_argument('--beta', type=float, default=1.0)
        parser.add_argument('--lamb', type=float, default=10)
        parser.add_argument('--lr-backbone', type=float, default=0.01)
        parser.add_argument('--lr-adapter', type=float, default=0.01)
        parser.add_argument('--multiplier', type=int, default=32)
        parser.add_argument('--shrink', type=float, default=0.0)
        parser.add_argument('--adaptation-strategy', type=str, choices=["none", "mean", "diag", "full"], default="full")

        parser.add_argument('--criterion', type=str, choices=["ce"], default="ce")
        parser.add_argument('--nnet', type=str, choices=["resnet18"], default="resnet18")
        parser.add_argument('--classifier', type=str, choices=["bayes", "linear"], default="bayes")
        parser.add_argument('--distillation', type=str, choices=["projected_bi"], default="projected_bi")

        parser.add_argument('--use-224', action='store_true', default=False)
        parser.add_argument('--pretrained-net', action='store_true', default=False)
        parser.add_argument('--normalize', action='store_true', default=False)
        parser.add_argument('--rotation', action='store_true', default=False)

        # Bi-directional knobs
        parser.add_argument('--lambda-bi', type=float, default=None)
        parser.add_argument('--lambda-cycle', type=float, default=1.0)
        parser.add_argument('--pair-preserve', action='store_true', default=False)
        parser.add_argument('--lambda-pair', type=float, default=0.1)

        # FP-CE knobs
        parser.add_argument('--rho-ce', type=float, default=0.0)
        parser.add_argument('--lambda-fp-ce', type=float, default=0.0)
        parser.add_argument('--fp-use-grad-norm', action='store_true', default=False)
        parser.add_argument('--lambda-fp-gn', type=float, default=0.0)

        # DA optimizer knobs
        parser.add_argument('--DA-lr', type=float, default=0.0)
        parser.add_argument('--DA-wd', type=float, default=0.0)

        return parser.parse_known_args(args)

    # -------------------- builders --------------------

    def _build_mlp(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.S, self.multiplier * self.S),
            nn.GELU(),
            nn.Linear(self.multiplier * self.S, self.S),
        )

    # -------------------- FP helpers --------------------

    def _fp_make_adv(self, z: torch.Tensor, loss_fn, rho: float) -> torch.Tensor:
        if rho <= 0:
            return None
        z_det = z.detach().requires_grad_(True)
        loss_dir = loss_fn(z_det)
        g = torch.autograd.grad(loss_dir, z_det, only_inputs=True, retain_graph=False)[0]
        g = g / (g.norm(dim=1, keepdim=True) + 1e-12)
        return (z_det + rho * g).detach()

    def _fp_gradnorm(self, z: torch.Tensor, loss_fn) -> torch.Tensor:
        z_req = z.detach().requires_grad_(True)
        loss_pert = loss_fn(z_req)
        grad = torch.autograd.grad(loss_pert, z_req, create_graph=True, only_inputs=True)[0]
        return grad.norm(dim=1).mean()

    # -------------------- BiCyc losses --------------------

    def _pairwise_geom_loss(self, Z_ref: torch.Tensor, Z_hat: torch.Tensor, metric: str = "cos") -> torch.Tensor:
        if Z_ref.shape[0] > 128:
            idx = torch.randperm(Z_ref.shape[0], device=Z_ref.device)[:128]
            Z_ref = Z_ref[idx]
            Z_hat = Z_hat[idx]

        if metric == "cos":
            ref = F.normalize(Z_ref, dim=1) @ F.normalize(Z_ref, dim=1).t()
            hat = F.normalize(Z_hat, dim=1) @ F.normalize(Z_hat, dim=1).t()
            return F.mse_loss(hat, ref)

        ref = torch.cdist(Z_ref, Z_ref, p=2)
        hat = torch.cdist(Z_hat, Z_hat, p=2)
        return F.mse_loss(hat, ref)

    def distill_bidirectional(self, t, base_loss, features, distiller, adapter, images):
        """
        projected_bi:
          - new->old:   D(z_new) ≈ z_old
          - old->new:   A(z_old) ≈ z_new.detach()
          - cycles:     A(D(z_new)) ≈ z_new.detach(), D(A(z_old)) ≈ z_old.detach()
          - optional:   pair-preserve
        """
        if t == 0:
            return base_loss, torch.zeros((), device=self.device)

        with torch.no_grad():
            z_old = self.old_model(images)

        z_new = features
        z_old_hat = distiller(z_new)
        z_new_hat = adapter(z_old)

        L_new2old = F.mse_loss(z_old_hat, z_old)
        L_old2new = F.mse_loss(z_new_hat, z_new.detach())

        L_cycle_new = F.mse_loss(adapter(z_old_hat.detach()), z_new.detach())
        L_cycle_old = F.mse_loss(distiller(z_new_hat.detach()), z_old.detach())

        L_pair = torch.zeros((), device=self.device)
        if self.pair_preserve and self.lambda_pair > 0:
            L_pair = self._pairwise_geom_loss(z_new.detach(), z_new_hat) \
                   + self._pairwise_geom_loss(z_old.detach(), z_old_hat)

        kd = self.lambda_bi * (L_new2old + L_old2new) \
           + self.lambda_cycle * (L_cycle_new + L_cycle_old) \
           + self.lambda_pair * L_pair

        return base_loss + kd, kd

    # -------------------- main loop --------------------

    def train_loop(self, t, trn_loader, val_loader, all_val_loader=None):
        num_classes_in_t = len(np.unique(trn_loader.dataset.labels))
        self.classes_in_tasks.append(num_classes_in_t)

        self.train_data_loaders.append(trn_loader)
        self.val_data_loaders.append(val_loader)

        self.old_model = copy.deepcopy(self.model).eval()
        self.task_offset.append(num_classes_in_t + self.task_offset[-1])

        print("### Training backbone ###")
        self.train_backbone(t, trn_loader, val_loader, num_classes_in_t)

        if t > 0 and self.adaptation_strategy != "none":
            print("### Adapting prototypes ###")
            self.adapt_distributions(t, trn_loader, val_loader)

        print("### Creating new prototypes ###")
        self.create_distributions(t, trn_loader, val_loader, num_classes_in_t)

        # Precompute inverted covariances for Bayes classifier
        covs = self.covs.clone()
        for i in range(covs.shape[0]):
            covs[i] = self._symmetrize(self.shrink_cov(covs[i], 3))
        if self.is_normalization:
            covs = self.norm_cov(covs)
        self.covs_inverted = torch.linalg.inv(covs)

        # Optional linear classifier
        if self.classifier == "linear":
            self.train_linear_head()

    def train_backbone(self, t, trn_loader, val_loader, num_classes_in_t):
        trn_loader = torch.utils.data.DataLoader(
            trn_loader.dataset,
            batch_size=trn_loader.batch_size,
            num_workers=trn_loader.num_workers,
            shuffle=True,
            drop_last=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_loader.dataset,
            batch_size=val_loader.batch_size,
            num_workers=val_loader.num_workers,
            shuffle=False,
            drop_last=True
        )

        local_C = num_classes_in_t
        if t == 0 and self.is_rotation:
            local_C = 4 * num_classes_in_t
            trn_loader = torch.utils.data.DataLoader(
                trn_loader.dataset,
                batch_size=max(1, trn_loader.batch_size // 4),
                num_workers=trn_loader.num_workers,
                shuffle=True,
                drop_last=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_loader.dataset,
                batch_size=max(1, val_loader.batch_size // 4),
                num_workers=trn_loader.num_workers,
                shuffle=False,
                drop_last=False
            )

        distiller = self._build_mlp().to(self.device, non_blocking=True)
        adapter = self._build_mlp().to(self.device, non_blocking=True)
        criterion = self.criterion(local_C, self.S, self.device, smoothing=0.0)

        flat_params = list(self.model.parameters()) + list(criterion.parameters()) \
                    + list(distiller.parameters()) + list(adapter.parameters())

        # Param groups for pretrained mode (keeps your prior behavior)
        param_groups = [
            {"params": list(self.model.parameters())[:-1], "lr": self.lr_backbone},
            {"params": list(self.model.parameters())[-1:] + list(criterion.parameters())},

            {"params": list(distiller.parameters()),
             **({"lr": self.DA_lr} if self.DA_lr > 0 else {}),
             **({"weight_decay": self.DA_wd} if self.DA_wd > 0 else {})},

            {"params": list(adapter.parameters()),
             **({"lr": self.DA_lr} if self.DA_lr > 0 else {"lr": self.lr_adapter}),
             **({"weight_decay": self.DA_wd} if self.DA_wd > 0 else {})},
        ]

        optimizer, lr_scheduler = self.get_optimizer(param_groups if self.pretrained else flat_params, t, self.wd)

        for epoch in range(self.nepochs):
            self.model.train()
            criterion.train()
            distiller.train()
            adapter.train()

            train_total, train_hits = 0, 0.0
            loss_sum, kd_sum = 0.0, 0.0

            for images, targets in trn_loader:
                if t == 0 and self.is_rotation:
                    images, targets = compute_rotations(images, targets, num_classes_in_t)

                targets = targets - self.task_offset[t]

                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                bsz = images.size(0)
                train_total += bsz

                optimizer.zero_grad()

                features = self.model(images)
                if epoch < int(self.nepochs * 0.01):
                    features = features.detach()

                cls_loss, logits = criterion(features, targets)

                total_loss, kd_loss = self.distill_bidirectional(
                    t=t, base_loss=cls_loss, features=features,
                    distiller=distiller, adapter=adapter, images=images
                )

                # FP-CE (task0 only)
                if (t == 0) and (self.rho_ce > 0) and (self.lambda_fp_ce > 0):
                    def _ce_loss_fn(z_):
                        l, _ = criterion(z_, targets)
                        return l

                    z_adv = self._fp_make_adv(features, _ce_loss_fn, rho=self.rho_ce)
                    if z_adv is not None:
                        loss_adv, _ = criterion(z_adv, targets)
                        total_loss = total_loss + self.lambda_fp_ce * loss_adv

                        if self.fp_use_grad_norm and self.lambda_fp_gn > 0:
                            gn = self._fp_gradnorm(z_adv, _ce_loss_fn)
                            total_loss = total_loss + self.lambda_fp_gn * gn

                if self.alpha > 0:
                    ac, _ = loss_ac(features, self.beta)
                    total_loss = total_loss + self.alpha * ac

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(flat_params, 1.0)
                optimizer.step()

                with torch.no_grad():
                    if logits is not None:
                        train_hits += float((logits.argmax(1) == targets).sum())
                    loss_sum += float(bsz * cls_loss.detach())
                    kd_sum += float(bsz * kd_loss.detach())

            lr_scheduler.step()
            train_acc = train_hits / max(train_total, 1e-8)
            print(f"Epoch {epoch:03d} | loss {loss_sum / max(train_total,1):.4f} "
                  f"| kd {kd_sum / max(train_total,1):.4f} | acc {100 * train_acc:.2f}")

        self.distiller_module = copy.deepcopy(distiller).eval()
        self.adapter_module = copy.deepcopy(adapter).eval()

    # -------------------- linear head training --------------------

    def get_pseudo_head_optimizer(self, parameters, milestones=(15,)):
        optimizer = torch.optim.SGD(parameters, lr=0.1, weight_decay=5e-4, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
        return optimizer, scheduler

    def train_linear_head(self, samples=10000, epochs=30, batch_size=128):
        """Train a linear head on pseudo-samples from current Gaussians."""
        total_classes = self.task_offset[-1]
        if total_classes <= 0:
            raise RuntimeError("No classes available to train linear head.")

        dists = []
        for c in range(total_classes):
            dists.append(self._safe_mvn(self.means[c], self.covs[c]))

        dataset = SampledDataset(dists, samples=samples, total_classes=total_classes)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True)

        head = nn.Linear(self.S, total_classes).to(self.device)
        optimizer, scheduler = self.get_pseudo_head_optimizer(head.parameters())

        for ep in range(epochs):
            head.train()
            total_loss = 0.0
            total_seen = 0
            for x, y in loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                logits = head(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()

                bsz = x.size(0)
                total_loss += float(loss.detach()) * bsz
                total_seen += bsz

            scheduler.step()
            print(f"[linear head] epoch {ep:03d} | loss {total_loss / max(total_seen,1):.4f}")

        self.pseudo_head = head.eval()

    # -------------------- prototypes & adaptation --------------------

    @torch.no_grad()
    def create_distributions(self, t, trn_loader, val_loader, num_classes_in_t):
        self.model.eval()
        transforms = val_loader.dataset.transform

        new_means = torch.zeros((num_classes_in_t, self.S), device=self.device)
        new_covs = torch.zeros((num_classes_in_t, self.S, self.S), device=self.device)
        new_covs_raw = torch.zeros((num_classes_in_t, self.S, self.S), device=self.device)

        for c in range(num_classes_in_t):
            train_indices = torch.tensor(trn_loader.dataset.labels) == c + self.task_offset[t]

            if isinstance(trn_loader.dataset.images, list):
                train_images = list(compress(trn_loader.dataset.images, train_indices))
                ds = ClassDirectoryDataset(train_images, transforms)
            else:
                ds = ClassMemoryDataset(trn_loader.dataset.images[train_indices], transforms)

            loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=trn_loader.num_workers, shuffle=False)

            from_ = 0
            class_features = torch.full((2 * len(ds), self.S), fill_value=0.0, device=self.device)
            for images in loader:
                bsz = images.size(0)
                images = images.to(self.device, non_blocking=True)

                feats = self.model(images)
                class_features[from_:from_ + bsz] = feats

                feats_flip = self.model(torch.flip(images, dims=(3,)))
                class_features[from_ + bsz:from_ + 2 * bsz] = feats_flip
                from_ += 2 * bsz

            mu = class_features.mean(dim=0)
            cov_raw = torch.cov(class_features.T)
            cov = self.shrink_cov(cov_raw, self.shrink)
            if self.adaptation_strategy == "diag":
                cov = torch.diag(torch.diag(cov))

            new_means[c] = mu
            new_covs[c] = cov
            new_covs_raw[c] = cov_raw

        self.means = torch.cat([self.means, new_means], dim=0)
        self.covs = torch.cat([self.covs, new_covs], dim=0)
        self.covs_raw = torch.cat([self.covs_raw, new_covs_raw], dim=0)

    def adapt_distributions(self, t, trn_loader, val_loader):
        trn_loader = torch.utils.data.DataLoader(
            trn_loader.dataset,
            batch_size=trn_loader.batch_size,
            num_workers=trn_loader.num_workers,
            shuffle=True,
            drop_last=True
        )

        self.model.eval()

        adapter = self._build_mlp().to(self.device, non_blocking=True)
        if self.adapter_module is not None:
            try:
                adapter.load_state_dict(self.adapter_module.state_dict(), strict=True)
                print("[adapt_distributions] Warm-start adapter from backbone training.")
            except Exception as e:
                print(f"[adapt_distributions] Warm-start failed: {e}")

        optimizer, lr_scheduler = self.get_adapter_optimizer(adapter.parameters())
        old_means = self.means.clone()
        old_covs = self.covs.clone()

        for epoch in range(self.nepochs // 2):
            adapter.train()
            for images, _ in trn_loader:
                images = images.to(self.device, non_blocking=True)
                optimizer.zero_grad()

                with torch.no_grad():
                    z_new = self.model(images)
                    z_old = self.old_model(images)

                z_old_to_new = adapter(z_old)
                loss = F.mse_loss(z_old_to_new, z_new)
                if self.alpha > 0:
                    ac, _ = loss_ac(z_old_to_new, self.beta)
                    loss = loss + self.alpha * ac

                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                optimizer.step()

            lr_scheduler.step()
            if epoch % 10 == 9:
                print(f"[adapt_distributions] epoch {epoch:03d}")

        adapter.eval()
        with torch.no_grad():
            if self.adaptation_strategy == "mean":
                self.means = adapter(self.means)

            if self.adaptation_strategy in ("full", "diag"):
                for c in range(self.means.size(0)):
                    dist = self._safe_mvn(old_means[c], old_covs[c])
                    samples = dist.sample((self.N,))
                    samples_new = adapter(samples)

                    mu_new = samples_new.mean(0)
                    cov_new = torch.cov(samples_new.T)
                    cov_new = self.shrink_cov(cov_new, self.shrink)
                    if self.adaptation_strategy == "diag":
                        cov_new = torch.diag(torch.diag(cov_new))

                    self.means[c] = mu_new
                    self.covs[c] = cov_new

    # -------------------- optimizers --------------------

    def get_optimizer(self, parameters, t, wd):
        milestones = (int(0.3 * self.nepochs), int(0.6 * self.nepochs), int(0.9 * self.nepochs))
        lr = self.lr
        if t > 0 and not self.pretrained:
            lr *= 0.1
        optimizer = torch.optim.SGD(parameters, lr=lr, weight_decay=wd, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
        return optimizer, scheduler

    def get_adapter_optimizer(self, parameters, milestones=(30, 60, 90)):
        optimizer = torch.optim.SGD(parameters, lr=self.lr_adapter, weight_decay=5e-4, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
        return optimizer, scheduler

    # -------------------- evaluation (bayes or linear) --------------------

    @torch.no_grad()
    def eval(self, t, val_loader):
        self.model.eval()

        tag_acc = Accuracy("multiclass", num_classes=self.means.shape[0])
        taw_acc = Accuracy("multiclass", num_classes=self.classes_in_tasks[t])
        offset = self.task_offset[t]

        for images, target in val_loader:
            images = images.to(self.device, non_blocking=True)
            features = self.model(images)

            if self.classifier == "linear":
                if self.pseudo_head is None:
                    raise RuntimeError("classifier=linear but pseudo_head is None. Did train_linear_head() run?")
                logits = self.pseudo_head(features)
                tag_preds = torch.argmax(logits, dim=1)
                taw_preds = torch.argmax(logits[:, offset: offset + self.classes_in_tasks[t]], dim=1) + offset
            else:
                if self.is_normalization:
                    diff = F.normalize(features.unsqueeze(1), p=2, dim=-1) - F.normalize(self.means.unsqueeze(0), p=2, dim=-1)
                else:
                    diff = features.unsqueeze(1) - self.means.unsqueeze(0)

                res = diff.unsqueeze(2) @ self.covs_inverted.unsqueeze(0)
                res = res @ diff.unsqueeze(3)
                dist = res.squeeze(2).squeeze(2)

                tag_preds = torch.argmin(dist, dim=1)
                taw_preds = torch.argmin(dist[:, offset: offset + self.classes_in_tasks[t]], dim=1) + offset

            tag_acc.update(tag_preds.cpu(), target)
            taw_acc.update(taw_preds.cpu(), target)

        return 0, float(taw_acc.compute()), float(tag_acc.compute())

    # -------------------- covariance utilities --------------------

    @torch.no_grad()
    def shrink_cov(self, cov, alpha1=1.0, alpha2=0.0):
        if alpha2 == -1.0:
            return cov + alpha1 * torch.eye(cov.shape[0], device=self.device)
        diag_mean = torch.mean(torch.diagonal(cov))
        I = torch.eye(cov.shape[0], device=self.device)
        mask = I == 0.0
        off_diag_mean = torch.mean(cov[mask]) if mask.any() else torch.tensor(0.0, device=self.device)
        return cov + (alpha1 * diag_mean * I) + (alpha2 * off_diag_mean * (1 - I)) + 1e-8 * I

    @torch.no_grad()
    def norm_cov(self, cov):
        diag = torch.diagonal(cov, dim1=1, dim2=2)
        std = torch.sqrt(diag + 1e-12)
        cov = cov / (std.unsqueeze(2) @ std.unsqueeze(1) + 1e-12)
        return cov

    def _symmetrize(self, cov: torch.Tensor) -> torch.Tensor:
        return 0.5 * (cov + cov.T)

    def _safe_cholesky(self, cov: torch.Tensor, init_jitter: float = 1e-6, max_tries: int = 6):
        cov = self._symmetrize(cov)
        I = torch.eye(cov.size(-1), device=cov.device, dtype=cov.dtype)
        jitter = 0.0
        for k in range(max_tries):
            try:
                return torch.linalg.cholesky(cov + jitter * I)
            except RuntimeError:
                jitter = init_jitter if k == 0 else jitter * 10.0
        vals, vecs = torch.linalg.eigh(cov)
        vals = torch.clamp(vals, min=init_jitter)
        cov_spd = (vecs * vals) @ vecs.T
        return torch.linalg.cholesky(self._symmetrize(cov_spd))

    def _safe_mvn(self, mean: torch.Tensor, cov: torch.Tensor):
        L = self._safe_cholesky(cov)
        return torch.distributions.MultivariateNormal(mean, scale_tril=L)