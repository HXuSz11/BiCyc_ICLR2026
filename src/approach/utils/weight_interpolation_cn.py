# Code adapted from https://github.com/KellerJordan/REPAIR
# Modified to work with a ResNet architecture that has an external classification head
# and uses 'downsample' for shortcut connections.

import torch
import torch.nn as nn
import scipy.optimize
import numpy as np

def add_junctures(net: nn.Module, device):
    """给所有空 shortcut 插入恒等 1×1 conv."""
    for block in get_blocks(net)[1:]:
        if isinstance(block.shortcut, nn.Sequential) and len(block.shortcut) == 0:
            C = block.bn2.weight.numel()
            shortcut = nn.Conv2d(C, C, 1, 1, 0, bias=False)
            shortcut.weight.data[:, :, 0, 0] = torch.eye(C)
            block.shortcut = shortcut
    return net.to(device).eval()


def remove_junctures(net: nn.Module):
    for block in get_blocks(net)[1:]:
        conv = block.shortcut
        if isinstance(conv, nn.Conv2d) and conv.kernel_size == (1, 1) and conv.in_channels == conv.out_channels:
            block.shortcut = nn.Sequential()
    return net


def get_blocks(net: nn.Module):
    """把 stem + 四大 layer 展平成 list."""
    return nn.Sequential(nn.Sequential(net.conv1, net.bn1), *net.layer1, *net.layer2, *net.layer3, *net.layer4)


def run_corr_matrix(net0, net1, loader, device, epochs=1):
    """计算两个网络输出激活的 Pearson 相关矩阵 (C×C)."""
    n = epochs * len(loader.dataset)
    mean0 = mean1 = std0 = std1 = cov = None

    with torch.no_grad():
        net0.eval(), net1.eval()
        # 第一次循环：累加均值 & 外积
        for _ in range(epochs):
            for img, _ in loader:
                x = img.float().to(device)
                f0, f1 = net0(x), net1(x)
                f0 = f0.reshape(f0.size(0), f0.size(1), -1).permute(0, 2, 1).reshape(-1, f0.size(1)).double()
                f1 = f1.reshape(f1.size(0), f1.size(1), -1).permute(0, 2, 1).reshape(-1, f1.size(1)).double()

                outer = f0.T @ f1
                if mean0 is None:
                    C = f0.size(1)
                    mean0 = torch.zeros(C, device=device)
                    mean1 = torch.zeros(C, device=device)
                    cov = torch.zeros_like(outer)

                mean0 += f0.sum(0)
                mean1 += f1.sum(0)
                cov += outer

        mean0 /= n
        mean1 /= n
        cov /= n

        # 第二次循环：累加方差
        for _ in range(epochs):
            for img, _ in loader:
                x = img.float().to(device)
                f0, f1 = net0(x), net1(x)
                f0 = f0.reshape(f0.size(0), f0.size(1), -1).permute(0, 2, 1).reshape(-1, f0.size(1)).double()
                f1 = f1.reshape(f1.size(0), f1.size(1), -1).permute(0, 2, 1).reshape(-1, f1.size(1)).double()

                if std0 is None:
                    std0 = torch.zeros_like(mean0)
                    std1 = torch.zeros_like(mean1)

                std0 += ((f0 - mean0).pow(2)).sum(0)
                std1 += ((f1 - mean1).pow(2)).sum(0)

        std0 = torch.sqrt(std0 / (n - 1))
        std1 = torch.sqrt(std1 / (n - 1))

    corr = (cov - torch.outer(mean0, mean1)) / (torch.outer(std0, std1) + 1e-4)
    return corr


def compute_perm(corr_mtx: torch.Tensor):
    """Hungarian 算法求最大相关匹配，返回 perm_map."""
    M = np.nan_to_num(corr_mtx.cpu().numpy())
    rows, cols = scipy.optimize.linear_sum_assignment(M, maximize=True)
    assert np.array_equal(rows, np.arange(len(M)))
    return torch.as_tensor(cols, dtype=torch.long, device=corr_mtx.device)


def permute_input(perm_map: torch.Tensor, conv: nn.Conv2d):
    conv.weight.data = conv.weight.data[:, perm_map, :, :]


def permute_output(perm_map: torch.Tensor, conv: nn.Conv2d, bn: nn.BatchNorm2d = None):
    conv.weight.data = conv.weight.data[perm_map]
    if bn is not None:
        for attr in ("weight", "bias", "running_mean", "running_var"):
            param = getattr(bn, attr)
            param.data = param.data[perm_map]


def get_anchor_idx(k: int):
    """把 block 索引归并到最近偶数 anchor（0,2,4,6,8…）。"""
    if k == 0:
        return 0
    for anchor in (2, 4, 6, 8):
        if k <= anchor:
            return anchor
    raise ValueError(f"Unexpected k={k}")


def permute_network(loader, src_net: nn.Module, tgt_net: nn.Module, device, epochs=1):
    """仅重排主干通道（不再动分类器）。"""
    blocks0, blocks1 = get_blocks(src_net), get_blocks(tgt_net)

    # 第一阶段：逐 block 对齐 conv1/conv2
    for k in range(1, len(blocks1)):
        subnet0 = nn.Sequential(*blocks0[:k], blocks0[k].conv1, blocks0[k].bn1, nn.ReLU(inplace=True))
        subnet1 = nn.Sequential(*blocks1[:k], blocks1[k].conv1, blocks1[k].bn1, nn.ReLU(inplace=True))
        perm = compute_perm(run_corr_matrix(subnet0, subnet1, loader, device, epochs))
        permute_output(perm, blocks1[k].conv1, blocks1[k].bn1)
        permute_input(perm, blocks1[k].conv2)

    # 第二阶段：按 anchor 组共享 perm_map
    last_anchor, perm = None, None
    for k in range(len(blocks1)):
        anchor = get_anchor_idx(k)
        if anchor != last_anchor:
            perm = compute_perm(run_corr_matrix(blocks0[:anchor + 1],
                                                blocks1[:anchor + 1],
                                                loader, device, epochs))
            last_anchor = anchor

        # 输出侧
        if k > 0:
            permute_output(perm, blocks1[k].conv2, blocks1[k].bn2)
            sc = blocks1[k].shortcut
            permute_output(perm, sc if isinstance(sc, nn.Conv2d) else sc[0], None if isinstance(sc, nn.Conv2d) else sc[1])
        else:
            permute_output(perm, tgt_net.conv1, tgt_net.bn1)

        # 输入侧（给下一块）
        if k + 1 < len(blocks1):
            permute_input(perm, blocks1[k + 1].conv1)
            sc = blocks1[k + 1].shortcut
            if isinstance(sc, nn.Conv2d):
                permute_input(perm, sc)
            else:
                permute_input(perm, sc[0])

    # 处理 bottleneck 输入
    permute_input(perm, tgt_net.bottleneck)
    return tgt_net


def mix_weights(model: nn.Module, alpha: float, net0: nn.Module, net1: nn.Module, device):
    sd0, sd1 = net0.state_dict(), net1.state_dict()
    blended = {k: (1 - alpha) * sd0[k].to(device) + alpha * sd1[k].to(device) for k in sd0}
    model.load_state_dict(blended)


def reset_bn_stats(model: nn.Module, loader, device, epochs=1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = None
            m.reset_running_stats()

    model.train()
    with torch.no_grad():
        for _ in range(epochs):
            for img, _ in loader:
                _ = model(img.to(device))


# --------------------------------------------------
# 3. 高层包装：插值主干 + 多头
# --------------------------------------------------
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
    # 1. 加恒等 shortcut
    src_backbone = add_junctures(src_backbone, device)
    tgt_backbone = add_junctures(tgt_backbone, device)

    # 2. 通道对齐
    tgt_backbone = permute_network(train_loader, src_backbone, tgt_backbone,
                                   device, epochs=perm_epochs)

    # 3. 主干权重插值
    mix_weights(tgt_backbone, alpha, src_backbone, tgt_backbone, device)

    # 4. 头部权重插值（仅旧任务共有部分）
    n_common = min(len(src_heads), len(tgt_heads))
    for i in range(n_common):
        w0 = src_heads[i][0].weight.data.to(device)
        w1 = tgt_heads[i][0].weight.data.to(device)
        tgt_heads[i][0].weight.data.copy_((1 - alpha) * w0 + alpha * w1)

    # 5. 重新统计 BN
    reset_bn_stats(tgt_backbone, train_loader, device, epochs=bn_epochs)

    # 6. 移除恒等 shortcut
    remove_junctures(src_backbone)
    remove_junctures(tgt_backbone)
    return tgt_backbone  # heads 已在原地更新


# --------------------------------------------------
# 4. 使用示例
# --------------------------------------------------
# if __name__ == "__main__":
#     device = "cuda:0" if torch.cuda.is_available() else "cpu"

#     # ---- 构造旧模型 ----
#     S = 64                       # bottleneck 输出维度
#     old_net = resnet18(S).to(device)
#     old_heads = nn.ModuleList([nn.Sequential(nn.Linear(S, 10, bias=False)).to(device)])

#     # ---- 构造新模型（添加一个新任务头）----
#     new_net = resnet18(S).to(device)
#     new_heads = nn.ModuleList([nn.Sequential(nn.Linear(S, 10, bias=False)).to(device)])  # 复制旧头
#     new_heads.append(nn.Sequential(nn.Linear(S, 5, bias=False)).to(device))              # 新任务 5 类

#     # ---- 伪造 DataLoader 作为对齐数据 ----
#     dummy_x = torch.randn(128, 3, 32, 32)
#     dummy_loader = torch.utils.data.DataLoader(
#         torch.utils.data.TensorDataset(dummy_x, torch.zeros(128)),
#         batch_size=64,
#         shuffle=False,
#     )

#     # ---- 插值 ----
#     new_net = interpolate_resnet_with_heads(
#         old_net, new_net,
#         old_heads, new_heads,
#         dummy_loader,
#         device,
#         alpha=0.5,
#     )

#     # new_net、新 new_heads 现在可作为新任务初始化
#     print("Interpolation finished.")