import torch
import torch.nn as nn
import scipy.optimize
import numpy as np

# ----------------------------------------------------------------------
# 0. utilities to read / write skip-connections (shortcut / downsample)
# ----------------------------------------------------------------------
def _get_skip(block):
    """Return the skip-connection module of a residual block."""
    if hasattr(block, "shortcut"):              # custom impl
        return block.shortcut
    return block.downsample                    # torchvision naming

def _set_skip(block, new_skip):
    """Set the skip-connection module of a residual block."""
    if hasattr(block, "shortcut"):
        block.shortcut = new_skip
    else:
        block.downsample = new_skip

# ----------------------------------------------------------------------
# 1. main public helpers
# ----------------------------------------------------------------------
def interpolate_resnet_with_heads(
    src_backbone: nn.Module,
    tgt_backbone: nn.Module,
    src_heads: nn.ModuleList,
    tgt_heads: nn.ModuleList,
    train_loader,
    device,
    alpha=0.5,
    perm_epochs=1,
    bn_epochs=1,
):
    # 1. Add identity shortcuts
    src_backbone = add_junctures(src_backbone, device)
    tgt_backbone = add_junctures(tgt_backbone, device)

    # 2. Channel alignment
    tgt_backbone = permute_network(train_loader, src_backbone, tgt_backbone, device, epochs=perm_epochs)

    # 3. Interpolate backbone weights
    mix_weights(tgt_backbone, alpha, src_backbone, tgt_backbone, device)

    # 4. Interpolate head weights (only for tasks common to both models)
    n_common = min(len(src_heads), len(tgt_heads))
    for i in range(n_common):
        w0 = src_heads[i][0].weight.data.to(device)
        w1 = tgt_heads[i][0].weight.data.to(device)
        tgt_heads[i][0].weight.data.copy_((1 - alpha) * w0 + alpha * w1)

    # 5. Recompute BN statistics
    reset_bn_stats(tgt_backbone, train_loader, device, epochs=bn_epochs)

    # 6. Remove identity shortcuts
    remove_junctures(src_backbone)
    remove_junctures(tgt_backbone)
    return tgt_backbone  # Heads are updated in place

def interpolate_resnet_backbone_only(
    src_backbone: nn.Module,
    tgt_backbone: nn.Module,
    train_loader,
    device,
    *,
    alpha: float = 0.5,
    perm_epochs: int = 1,
    bn_epochs: int = 1,
):
    """
    Interpolate `tgt_backbone` toward `src_backbone` without touching any heads.

    Steps
    -----
    1. Insert identity shortcuts so every block has an explicit path.
    2. Align channel order via correlation-based permutation.
    3. Mix backbone weights with ratio `alpha`.
    4. Re-estimate BatchNorm statistics.
    5. Remove the temporary identity shortcuts.
    """
    # 1) Add identity shortcuts
    src_backbone = add_junctures(src_backbone, device)
    tgt_backbone = add_junctures(tgt_backbone, device)

    # 2) Channel alignment
    tgt_backbone = permute_network(
        train_loader,
        src_backbone,
        tgt_backbone,
        device,
        epochs=perm_epochs,
    )

    # 3) Interpolate backbone parameters
    mix_weights(tgt_backbone, alpha, src_backbone, tgt_backbone, device)

    # 4) Refresh BN running statistics
    reset_bn_stats(tgt_backbone, train_loader, device, epochs=bn_epochs)

    # 5) Clean up: remove identity shortcuts
    remove_junctures(src_backbone)
    remove_junctures(tgt_backbone)

    return tgt_backbone



# ----------------------------------------------------------------------
# 2. low-level building blocks (names kept与原作者一致)
# ----------------------------------------------------------------------
def add_junctures(net, device):
    """
    Insert an identity 1×1 conv into every *empty* skip path.
    Works with both `shortcut` 和 `downsample`.
    """
    for block in get_blocks(net)[1:]:
        skip = _get_skip(block)
        is_empty = (skip is None) or (isinstance(skip, nn.Sequential) and len(skip) == 0)
        if is_empty:
            C = block.bn2.weight.numel()
            identity_conv = nn.Conv2d(C, C, 1, 1, 0, bias=False)
            identity_conv.weight.data[:, :, 0, 0] = torch.eye(C)
            _set_skip(block, identity_conv)
    return net.to(device).eval()


def remove_junctures(net):
    """
    Remove the identity 1×1 convs that were added by `add_junctures`.
    """
    for block in get_blocks(net)[1:]:
        skip = _get_skip(block)
        is_identity = isinstance(skip, nn.Conv2d) and \
            skip.kernel_size == (1, 1) and skip.in_channels == skip.out_channels
        if is_identity:
            # 保持原来字段类型一致
            _set_skip(block, nn.Sequential() if hasattr(block, "shortcut") else None)
    return net


def permute_network(train_aug_loader, source_network, premuted_network, device, epochs=1):
    """
    Reorder channels in `premuted_network` to match `source_network`.
    Only the backbone is touched; classifier weight is permuted at the end.
    """
    blocks0 = get_blocks(source_network)
    blocks1 = get_blocks(premuted_network)

    # -- Stage 1: align conv1 of each block (except block0) ----------
    for k in range(1, len(blocks1)):
        block0 = blocks0[k]
        block1 = blocks1[k]
        subnet0 = nn.Sequential(blocks0[:k], block0.conv1, block0.bn1, nn.ReLU(inplace=True))
        subnet1 = nn.Sequential(blocks1[:k], block1.conv1, block1.bn1, nn.ReLU(inplace=True))
        perm_map = get_layer_perm(subnet0, subnet1, train_aug_loader, device, epochs=epochs)
        permute_output(perm_map, block1.conv1, block1.bn1)
        permute_input(perm_map, block1.conv2)

    # -- Stage 2: share a perm_map within each anchor group ----------
    last_kk = None
    perm_map = None

    for k in range(len(blocks1)):
        kk = get_permk(k)
        if kk != last_kk:
            perm_map = get_layer_perm(
                blocks0[: kk + 1], blocks1[: kk + 1], train_aug_loader, device, epochs=epochs
            )
            last_kk = kk

        # ----- output side -----
        if k > 0:
            permute_output(perm_map, blocks1[k].conv2, blocks1[k].bn2)
            skip = _get_skip(blocks1[k])
            if isinstance(skip, nn.Conv2d):
                permute_output(perm_map, skip)
            elif isinstance(skip, nn.Sequential) and len(skip) > 0:
                permute_output(perm_map, skip[0], skip[1])
        else:  # first block uses stem conv / bn
            permute_output(perm_map, premuted_network.conv1, premuted_network.bn1)

        # ----- input side (for next block) -----
        if k + 1 < len(blocks1):
            permute_input(perm_map, blocks1[k + 1].conv1)
            next_skip = _get_skip(blocks1[k + 1])
            if isinstance(next_skip, nn.Conv2d):
                permute_input(perm_map, next_skip)
            elif isinstance(next_skip, nn.Sequential) and len(next_skip) > 0:
                permute_input(perm_map, next_skip[0])
        else:
            # final: also permute classifier if it exists
            if hasattr(premuted_network, "classifier"):
                premuted_network.classifier.weight.data = \
                    premuted_network.classifier.weight[:, perm_map]

    return premuted_network


# ----------------------------------------------------------------------
# 3. helper functions unchanged from original
# ----------------------------------------------------------------------
def get_blocks(net):
    """Flatten stem + 4 layers into a Sequential list."""
    return nn.Sequential(
        nn.Sequential(net.conv1, net.bn1),
        *net.layer1, *net.layer2, *net.layer3, *net.layer4
    )


def get_layer_perm(net0, net1, train_dataloader, device, epochs=1):
    corr_mtx = run_corr_matrix(net0, net1, train_dataloader, device, epochs=epochs)
    return compute_permutation_matrix(corr_mtx)


def run_corr_matrix(net0, net1, loader, device, epochs=1):
    """
    Compute C×C Pearson correlation between two networks' activations.
    """
    n = epochs * len(loader.dataset)
    mean0 = mean1 = std0 = std1 = None
    outer = None

    def _flatten(feat):
        # (N, C, H, W) -> (N*H*W, C)
        return feat.reshape(feat.size(0), feat.size(1), -1) \
                   .permute(0, 2, 1) \
                   .reshape(-1, feat.size(1)) \
                   .double()

    with torch.no_grad():
        net0.eval(); net1.eval()

        # ---------- pass 1: means & covariance ----------
        for _ in range(epochs):
            for images, _ in loader:
                x = images.float().to(device)
                f0 = _flatten(net0(x))
                f1 = _flatten(net1(x))

                if mean0 is None:
                    C = f0.size(1)
                    mean0 = torch.zeros(C, device=device)
                    mean1 = torch.zeros(C, device=device)
                    outer = torch.zeros(C, C, device=device)

                mean0 += f0.sum(0)
                mean1 += f1.sum(0)
                outer += f0.T @ f1

        mean0 /= n
        mean1 /= n
        outer /= n

        # ---------- pass 2: variances ----------
        for _ in range(epochs):
            for images, _ in loader:
                x = images.float().to(device)
                f0 = _flatten(net0(x))
                f1 = _flatten(net1(x))

                if std0 is None:
                    std0 = torch.zeros_like(mean0)
                    std1 = torch.zeros_like(mean1)

                std0 += ((f0 - mean0) ** 2).sum(0)
                std1 += ((f1 - mean1) ** 2).sum(0)

        std0 = torch.sqrt(std0 / (n - 1))
        std1 = torch.sqrt(std1 / (n - 1))

    cov = outer - torch.outer(mean0, mean1)
    corr = cov / (torch.outer(std0, std1) + 1e-4)
    return corr


def compute_permutation_matrix(corr_mtx):
    corr_np = np.nan_to_num(corr_mtx.cpu().numpy())
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(corr_np, maximize=True)
    assert (row_ind == np.arange(len(corr_np))).all()
    return torch.tensor(col_ind, dtype=torch.long, device=corr_mtx.device)


def get_permk(k):
    if k == 0:
        return 0
    for anchor in (2, 4, 6, 8):
        if k <= anchor:
            return anchor
    raise ValueError(f"Unexpected k={k}")


def permute_input(perm_map, conv):
    conv.weight.data = conv.weight.data[:, perm_map, :, :]


def permute_output(perm_map, conv, bn=None):
    conv.weight.data = conv.weight.data[perm_map]
    if bn is not None:
        for attr in ("weight", "bias", "running_mean", "running_var"):
            getattr(bn, attr).data = getattr(bn, attr).data[perm_map]


def mix_weights(model, alpha, model0, model1, device):
    sd0, sd1 = model0.state_dict(), model1.state_dict()
    blended = {k: (1 - alpha) * sd0[k].to(device) + alpha * sd1[k].to(device) for k in sd0}
    model.load_state_dict(blended)


def reset_bn_stats(model, loader, device, epochs=1):
    # wipe old BN stats
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = None
            m.reset_running_stats()
    # single-epoch forward pass (no grad) to recompute
    model.train()
    with torch.no_grad():
        for _ in range(epochs):
            for images, _ in loader:
                _ = model(images.to(device))





# --------------------------------------------------
# 4. Example usage
# --------------------------------------------------
# if __name__ == "__main__":
#     device = "cuda:0" if torch.cuda.is_available() else "cpu"
#
#     # ---- Build the old model ----
#     S = 64  # bottleneck output dimension
#     old_net = resnet18(S).to(device)
#     old_heads = nn.ModuleList([nn.Sequential(nn.Linear(S, 10, bias=False)).to(device)])
#
#     # ---- Build the new model (add a new task head) ----
#     new_net = resnet18(S).to(device)
#     new_heads = nn.ModuleList([nn.Sequential(nn.Linear(S, 10, bias=False)).to(device)])  # copy old head
#     new_heads.append(nn.Sequential(nn.Linear(S, 5, bias=False)).to(device))              # 5-class new task
#
#     # ---- Fake DataLoader for alignment ----
#     dummy_x = torch.randn(128, 3, 32, 32)
#     dummy_loader = torch.utils.data.DataLoader(
#         torch.utils.data.TensorDataset(dummy_x, torch.zeros(128)),
#         batch_size=64,
#         shuffle=False,
#     )
#
#     # ---- Interpolate ----
#     new_net = interpolate_resnet_with_heads(
#         old_net,
#         new_net,
#         old_heads,
#         new_heads,
#         dummy_loader,
#         device,
#         alpha=0.5,
#     )
#
#     # `new_net` and the updated `new_heads` can now be used as the initialization
#     # for further training on the new task
#     print("Interpolation finished.")


# # build or load the two backbones
# src_backbone = resnet18(64).to(device)
# tgt_backbone = resnet18(64).to(device)

# # a small loader is enough for channel alignment & BN stats
# dummy_loader = torch.utils.data.DataLoader(
#     torch.utils.data.TensorDataset(torch.randn(128, 3, 32, 32),
#                                    torch.zeros(128)),
#     batch_size=64,
#     shuffle=False,
# )

# # interpolate only the backbone
# tgt_backbone = interpolate_resnet_backbone_only(
#     src_backbone,
#     tgt_backbone,
#     dummy_loader,
#     device,
#     alpha=0.5,
# )
