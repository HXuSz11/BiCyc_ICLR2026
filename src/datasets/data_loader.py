import os, glob
import random
from itertools import compress
import re

import numpy as np
from torch.utils import data
import torchvision.transforms as transforms
from torchvision.datasets import MNIST as TorchVisionMNIST
from torchvision.datasets import CIFAR100 as TorchVisionCIFAR100
from torchvision.datasets import SVHN as TorchVisionSVHN
from torchvision.datasets import FGVCAircraft

from . import base_dataset as basedat
from .tiny_loader import TinyImageNet200Folder
from . import memory_dataset as memd
from .dataset_config import dataset_config
from PIL import Image


# ----------------------------------------------------------------------
# Helpers for CIFAR100-LT (Ordered)
# ----------------------------------------------------------------------
def _parse_lt_ratio_from_name(name: str, default_ratio: int = 100) -> int:
    """
    Parse imbalance ratio `rNN` from dataset name.
      'cifar100_lt_r100' -> 100
      'cifar100-lt-r50'  -> 50
      'cifar100_lt'      -> default_ratio
    """
    m = re.search(r'lt[_\-]?r(\d+)', name.lower())
    return int(m.group(1)) if m else default_ratio


def _make_cifar100_lt(
    trn_data: dict,
    imb_factor: float = 0.01,   # e.g., r-100 -> 0.01
    seed: int = 0,
    ordered: bool = True,
    base_order=None,            # optional permutation to define head->tail identity
):
    """
    Build a long-tailed subset of CIFAR-100 TRAIN data and the class order.

    Args
    ----
    trn_data: {'x': np.ndarray or list, 'y': list[int]}
    imb_factor: N_min / N_max (e.g., r-100 => 0.01)
    seed: RNG seed (affects per-class sampling and the default base permutation)
    ordered: if True, return class_order sorted by per-class counts desc (head -> tail)
    base_order: an optional list/array of length 100, a permutation of [0..99].
                If provided, the head->tail *identity* follows this permutation
                (i.e., base_order[0] gets the largest quota, base_order[-1] the smallest).

    Returns
    -------
    trn_lt: {'x': ..., 'y': ...}  # long-tailed training subset
    class_order: list[int]        # head->tail class order used to split tasks
    counts: list[int]             # per-class target counts used to sample
    """
    x = trn_data['x']
    y = np.array(trn_data['y'], dtype=np.int64)
    C = int(y.max()) + 1  # should be 100 for CIFAR-100

    # Max available per class (CIFAR-100 train typically 500)
    binc = np.bincount(y, minlength=C)
    img_max = int(binc.max())

    # Exponential schedule across ranks (rank 0 = head)
    ranks = np.arange(C, dtype=np.float64)
    img_num_per_rank = (img_max * (imb_factor ** (ranks / (C - 1)))).round().astype(int)
    img_num_per_rank = np.clip(img_num_per_rank, 1, img_max)

    rng = np.random.RandomState(seed)
    # Decide which classes get which ranks (head->tail identity)
    if base_order is None:
        perm = rng.permutation(C)  # reproducible random identity
    else:
        perm = np.asarray(base_order, dtype=int)
        assert len(perm) == C and set(perm.tolist()) == set(range(C)), \
            "base_order must be a permutation of 0..C-1"

    # Assign largest quota to perm[0], smallest to perm[-1]
    counts = np.zeros(C, dtype=int)
    counts[perm] = img_num_per_rank

    # Sample per class according to 'counts'
    keep_idx = []
    for c in range(C):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        n = int(min(counts[c], len(idx_c)))
        keep_idx.extend(idx_c[:n].tolist())
    keep_idx = np.array(keep_idx, dtype=int)

    # Subset X/Y
    if isinstance(x, np.ndarray):
        x_lt = x[keep_idx]
    else:
        x_lt = [x[i] for i in keep_idx]
    y_lt = y[keep_idx].tolist()
    trn_lt = {'x': x_lt, 'y': y_lt}

    # Head->tail order with stable tie-break (by class id) for reproducibility
    if ordered:
        class_order = sorted(range(C), key=lambda c: (-counts[c], c))
    else:
        class_order = perm.tolist()

    return trn_lt, class_order, counts.tolist()


def get_loaders(datasets, num_tasks, nc_first_task, batch_size, num_workers, pin_memory, validation=.1,
                extra_aug=""):
    """Apply transformations to Datasets and create the DataLoaders for each task"""

    trn_load, val_load, tst_load = [], [], []
    taskcla = []
    dataset_offset = 0
    for idx_dataset, cur_dataset in enumerate(datasets, 0):
        # get configuration for current dataset
        dc = dataset_config[cur_dataset]

        # transformations
        trn_transform, tst_transform = get_transforms(resize=dc['resize'],
                                                      test_resize=dc['test_resize'],
                                                      pad=dc['pad'],
                                                      crop=dc['crop'],
                                                      flip=dc['flip'],
                                                      normalize=dc['normalize'],
                                                      extend_channel=dc['extend_channel'],
                                                      extra_aug=extra_aug, ds_name=cur_dataset)

        # datasets
        trn_dset, val_dset, tst_dset, curtaskcla = get_datasets(cur_dataset, dc['path'], num_tasks, nc_first_task,
                                                                validation=validation,
                                                                trn_transform=trn_transform,
                                                                tst_transform=tst_transform,
                                                                class_order=dc['class_order'])

        # apply offsets in case of multiple datasets
        if idx_dataset > 0:
            for tt in range(num_tasks):
                trn_dset[tt].labels = [elem + dataset_offset for elem in trn_dset[tt].labels]
                val_dset[tt].labels = [elem + dataset_offset for elem in val_dset[tt].labels]
                tst_dset[tt].labels = [elem + dataset_offset for elem in tst_dset[tt].labels]
        dataset_offset = dataset_offset + sum([tc[1] for tc in curtaskcla])

        # reassign class idx for multiple dataset case
        curtaskcla = [(tc[0] + idx_dataset * num_tasks, tc[1]) for tc in curtaskcla]

        # extend final taskcla list
        taskcla.extend(curtaskcla)

        # loaders
        for tt in range(num_tasks):
            trn_load.append(data.DataLoader(trn_dset[tt], batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                            pin_memory=pin_memory))
            val_load.append(data.DataLoader(val_dset[tt], batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                            pin_memory=pin_memory))
            tst_load.append(data.DataLoader(tst_dset[tt], batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                            pin_memory=pin_memory))
    return trn_load, val_load, tst_load, taskcla


def get_datasets(dataset, path, num_tasks, nc_first_task, validation, trn_transform, tst_transform, class_order=None):
    """Extract datasets and create Dataset class"""

    trn_dset, val_dset, tst_dset = [], [], []

    if 'mnist' in dataset:
        tvmnist_trn = TorchVisionMNIST(path, train=True, download=True)
        tvmnist_tst = TorchVisionMNIST(path, train=False, download=True)
        trn_data = {'x': tvmnist_trn.data.numpy(), 'y': tvmnist_trn.targets.tolist()}
        tst_data = {'x': tvmnist_tst.data.numpy(), 'y': tvmnist_tst.targets.tolist()}
        # compute splits
        all_data, taskcla, class_indices = memd.get_data(trn_data, tst_data, validation=validation,
                                                         num_tasks=num_tasks, nc_first_task=nc_first_task,
                                                         shuffle_classes=class_order is None, class_order=class_order)
        # set dataset type
        Dataset = memd.MemoryDataset

    elif 'cifar100' in dataset:
        tvcifar_trn = TorchVisionCIFAR100(path, train=True, download=True)
        tvcifar_tst = TorchVisionCIFAR100(path, train=False, download=True)
        trn_data = {'x': tvcifar_trn.data, 'y': tvcifar_trn.targets}
        tst_data = {'x': tvcifar_tst.data, 'y': tvcifar_tst.targets}

        # ------------------------------
        # Ordered CIFAR100-LT hook:
        # Trigger if dataset name contains "lt".
        # Ratio precedence: ENV(LT_RATIO) > dataset_config[...]['lt']['ratio'] > name 'rNN' > default(100)
        # ------------------------------
        is_lt = ('lt' in dataset.lower())
        if is_lt:
            # 1) read config (if present)
            ds_conf = dataset_config.get(dataset, {})
            lt_conf = ds_conf.get('lt', {}) if isinstance(ds_conf, dict) else {}

            # 2) resolve ratio
            ratio_from_env = os.getenv('LT_RATIO', None)
            ratio_from_cfg = lt_conf.get('ratio', None)
            ratio_from_name = _parse_lt_ratio_from_name(dataset, default_ratio=100)
            if ratio_from_env is not None:
                ratio = int(ratio_from_env)
            elif ratio_from_cfg is not None:
                ratio = int(ratio_from_cfg)
            else:
                ratio = int(ratio_from_name)
            imb_factor = 1.0 / float(ratio)

            # 3) seed and (optional) base order for head->tail identity
            lt_seed = int(os.getenv('LT_SEED', '0'))
            base_order = ds_conf.get('class_order', None)  # if you want iCaRL order to define head->tail, put it here

            trn_data_lt, class_order_lt, _counts = _make_cifar100_lt(
                trn_data, imb_factor=imb_factor, seed=lt_seed, ordered=True, base_order=base_order
            )

            # Fixed class order (head->tail), no shuffling
            all_data, taskcla, class_indices = memd.get_data(
                trn_data_lt, tst_data, validation=validation,
                num_tasks=num_tasks, nc_first_task=nc_first_task,
                shuffle_classes=False,
                class_order=class_order_lt
            )
        else:
            # Original balanced CIFAR-100 path
            all_data, taskcla, class_indices = memd.get_data(
                trn_data, tst_data, validation=validation,
                num_tasks=num_tasks, nc_first_task=nc_first_task,
                shuffle_classes=(class_order is None), class_order=class_order
            )
        # set dataset type
        Dataset = memd.MemoryDataset

    elif dataset == 'svhn':
        tvsvhn_trn = TorchVisionSVHN(path, split='train', download=True)
        tvsvhn_tst = TorchVisionSVHN(path, split='test', download=True)
        trn_data = {'x': tvsvhn_trn.data.transpose(0, 2, 3, 1), 'y': tvsvhn_trn.labels}
        tst_data = {'x': tvsvhn_tst.data.transpose(0, 2, 3, 1), 'y': tvsvhn_tst.labels}
        # compute splits
        all_data, taskcla, class_indices = memd.get_data(trn_data, tst_data, validation=validation,
                                                         num_tasks=num_tasks, nc_first_task=nc_first_task,
                                                         shuffle_classes=class_order is None, class_order=class_order)
        # set dataset type
        Dataset = memd.MemoryDataset

    elif 'imagenet_32' in dataset:
        import pickle
        # load data
        x_trn, y_trn = [], []
        for i in range(1, 11):
            with open(os.path.join(path, 'train_data_batch_{}'.format(i)), 'rb') as f:
                d = pickle.load(f)
            x_trn.append(d['data'])
            y_trn.append(np.array(d['labels']) - 1)  # labels from 0 to 999
        with open(os.path.join(path, 'val_data'), 'rb') as f:
            d = pickle.load(f)
        x_trn.append(d['data'])
        y_tst = np.array(d['labels']) - 1  # labels from 0 to 999
        # reshape data
        for i, d in enumerate(x_trn, 0):
            x_trn[i] = d.reshape(d.shape[0], 3, 32, 32).transpose(0, 2, 3, 1)
        x_tst = x_trn[-1]
        x_trn = np.vstack(x_trn[:-1])
        y_trn = np.concatenate(y_trn)
        trn_data = {'x': x_trn, 'y': y_trn}
        tst_data = {'x': x_tst, 'y': y_tst}
        # compute splits
        all_data, taskcla, class_indices = memd.get_data(trn_data, tst_data, validation=validation,
                                                         num_tasks=num_tasks, nc_first_task=nc_first_task,
                                                         shuffle_classes=class_order is None, class_order=class_order)
        # set dataset type
        Dataset = memd.MemoryDataset

    elif dataset == 'imagenet_subset_kaggle':
        _ensure_imagenet_subset_prepared(path, train_out="imagenet100_train.txt", test_out="imagenet100_val.txt")
        # read data paths and compute splits -- path needs to have a train.txt and a test.txt with image-label pairs
        all_data, taskcla, class_indices = basedat.get_data(path, num_tasks=num_tasks, nc_first_task=nc_first_task,
                                                                validation=validation, shuffle_classes=class_order is None,
                                                                class_order=class_order, train_file='imagenet100_train.txt', test_file='imagenet100_val.txt')
        Dataset = basedat.BaseDataset

    # elif dataset == 'tiny':
    #     _ensure_tiny_prepared(path)
    #     # read data paths and compute splits -- path needs to have a train.txt and a test.txt with image-label pairs
    #     all_data, taskcla, class_indices = basedat.get_data(path, num_tasks=num_tasks, nc_first_task=nc_first_task,
    #                                                             validation=validation, shuffle_classes=class_order is None,
    #                                                             class_order=class_order)
    #     Dataset = basedat.BaseDataset

    elif 'tiny' in dataset:
        # ------------------------------------------------------------------
        # 1) Load the split the same way TorchVision loaders do
        # ------------------------------------------------------------------
        tiny_trn = TinyImageNet200Folder("/data/hx5239", train=True)   # 100 000 images
        tiny_val = TinyImageNet200Folder("/data/hx5239", train=False)  # 10 000 images

        def pil_to_np(p): return np.asarray(Image.open(p).convert("RGB"))

        tiny_trn.data = np.stack([pil_to_np(p) for p, _ in tiny_trn.samples])
        tiny_val.data = np.stack([pil_to_np(p) for p, _ in tiny_val.samples])

        tiny_trn.targets = np.array([lbl for _, lbl in tiny_trn.samples], dtype=np.int64)
        tiny_val.targets = np.array([lbl for _, lbl in tiny_val.samples], dtype=np.int64)

        trn_data = {'x': tiny_trn.data, 'y': tiny_trn.targets}
        tst_data = {'x': tiny_val.data, 'y': tiny_val.targets}

        all_data, taskcla, class_indices = memd.get_data(
            trn_data,
            tst_data,
            validation=validation,
            num_tasks=num_tasks,
            nc_first_task=nc_first_task,
            shuffle_classes=(class_order is None),
            class_order=class_order,
        )
        Dataset = memd.MemoryDataset

    elif dataset == 'cub200':
        _ensure_cub200_subset_prepared(path)
        # read data paths and compute splits -- path needs to have a train.txt and a test.txt with image-label pairs
        all_data, taskcla, class_indices = basedat.get_data(path, num_tasks=num_tasks, nc_first_task=nc_first_task,
                                                                validation=validation, shuffle_classes=(class_order is None),
                                                                class_order=class_order)
        Dataset = basedat.BaseDataset

    elif dataset == 'aircraft':
        _ensure_aircraft_prepared(path)
        # read data paths and compute splits -- path needs to have a train.txt and a test.txt with image-label pairs
        all_data, taskcla, class_indices = basedat.get_data(path, num_tasks=num_tasks, nc_first_task=nc_first_task,
                                                                validation=validation, shuffle_classes=True,
                                                                class_order=None)
        Dataset = basedat.BaseDataset

    elif dataset == 'domainnet':
        _ensure_domainnet_prepared(path, classes_per_domain=nc_first_task, num_tasks=num_tasks)
        all_data, taskcla, class_indices = basedat.get_data(path, num_tasks=num_tasks, nc_first_task=nc_first_task,
                                                                validation=validation, shuffle_classes=False)
        Dataset = basedat.BaseDataset

    # get datasets, apply correct label offsets for each task
    offset = 0
    for task in range(num_tasks):
        all_data[task]['trn']['y'] = [label + offset for label in all_data[task]['trn']['y']]
        all_data[task]['val']['y'] = [label + offset for label in all_data[task]['val']['y']]
        all_data[task]['tst']['y'] = [label + offset for label in all_data[task]['tst']['y']]
        trn_dset.append(Dataset(all_data[task]['trn'], trn_transform, class_indices))
        val_dset.append(Dataset(all_data[task]['val'], tst_transform, class_indices))
        tst_dset.append(Dataset(all_data[task]['tst'], tst_transform, class_indices))
        offset += taskcla[task][1]

    return trn_dset, val_dset, tst_dset, taskcla


def get_transforms(resize, test_resize, pad, crop, flip, normalize, extend_channel, extra_aug="", ds_name=""):
    """Unpack transformations and apply to train or test splits"""

    trn_transform_list = []
    tst_transform_list = []

    # resize
    if resize is not None:
        trn_transform_list.append(transforms.Resize(resize))
        tst_transform_list.append(transforms.Resize(resize))

    # padding
    if pad is not None:
        trn_transform_list.append(transforms.Pad(pad))
        tst_transform_list.append(transforms.Pad(pad))

    # test only resize
    if test_resize is not None:
        tst_transform_list.append(transforms.Resize(test_resize))

    # crop
    if crop is not None:
        if 'cifar' in ds_name.lower():
            trn_transform_list.append(transforms.RandomCrop(crop))
        else:
            trn_transform_list.append(transforms.RandomResizedCrop(crop))
        tst_transform_list.append(transforms.CenterCrop(crop))

    # flips
    if flip:
        trn_transform_list.append(transforms.RandomHorizontalFlip())

    trn_transform_list.append(transforms.ColorJitter(brightness=63 / 255))
    if "cifar" in ds_name.lower():
        trn_transform_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ]
        tst_transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ]

    elif "tiny" in ds_name.lower():
        trn_transform_list = [
             transforms.RandomCrop(64, padding=8),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        tst_transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]

    elif "imagenet_subset" in ds_name.lower():
        trn_transform_list = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        tst_transform_list = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]

    elif "cub200" in ds_name.lower():
        trn_transform_list = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        tst_transform_list = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]

    elif "aircraft" in ds_name.lower():
        trn_transform_list = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        tst_transform_list = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]

    # gray to rgb
    if extend_channel is not None:
        trn_transform_list.append(transforms.Lambda(lambda x: x.repeat(extend_channel, 1, 1)))
        tst_transform_list.append(transforms.Lambda(lambda x: x.repeat(extend_channel, 1, 1)))

    return transforms.Compose(trn_transform_list), transforms.Compose(tst_transform_list)


def _ensure_aircraft_prepared(path):
    train_ds = FGVCAircraft(path.replace("fgvc-aircraft-2013b", ""), split="trainval")
    test_ds = FGVCAircraft(path.replace("fgvc-aircraft-2013b", ""), split="test")

    with open(f"{path}/{'train.txt'}", 'wt') as f:
        for i, image in enumerate(train_ds._image_files):
            f.write(f"{image.split('fgvc-aircraft-2013b/')[1]} {train_ds._labels[i]}\n")

    with open(f"{path}/{'test.txt'}", 'wt') as f:
        for i, image in enumerate(test_ds._image_files):
            f.write(f"{image.split('fgvc-aircraft-2013b/')[1]} {test_ds._labels[i]}\n")


# def _ensure_imagenet_subset_prepared(path):
#     assert os.path.exists(path), f"Please first download and extract dataset from: https://www.kaggle.com/datasets/arjunashok33/imagenet-subset-for-inc-learn to dir: {path}"
#     ds_conf = dataset_config['imagenet_subset_kaggle']
#     clsss2idx = {c:i for i, c in enumerate(ds_conf['lbl_order'])}
#     print(f'Generating train/test splits for ImageNet-Subset directory: {path}')
#     def prepare_split(split='train', outfile='train.txt'):
#         with open(f"{path}/{outfile}", 'wt') as f:
#             for fn in glob.glob(f"{path}/data/{split}/*/*"):
#                 c = fn.split('/')[-2]
#                 lbl = clsss2idx[c]
#                 relative_path = fn.replace(f"{path}/", '')
#                 f.write(f"{relative_path} {lbl}\n")
#     prepare_split()
#     prepare_split('val', outfile='test.txt')

def _ensure_imagenet_subset_prepared(path, train_out="train.txt", test_out="test.txt"):
    assert os.path.exists(path), (
        "Please first download and extract dataset from: "
        "https://www.kaggle.com/datasets/arjunashok33/imagenet-subset-for-inc-learn "
        f"to dir: {path}"
    )

    ds_conf = dataset_config.get('imagenet_subset_kaggle', {})
    lbl_order = ds_conf.get('lbl_order', None)
    num_classes_cfg = ds_conf.get('num_classes', None)

    trn_txt = os.path.join(path, "train.txt")
    tst_txt = os.path.join(path, "test.txt")
    if os.path.exists(trn_txt) and os.path.getsize(trn_txt) > 0 \
       and os.path.exists(tst_txt) and os.path.getsize(tst_txt) > 0:
        print(f"[imagenet_subset] Found non-empty train/test lists under {path}, skip regeneration.")
        return

    print(f'Generating train/test splits for ImageNet-Subset directory: {path}')

    CANDIDATES = {
        "train": ["data/train", "train", "training"],
        "val":   ["data/val", "data/validation", "val", "validation", "valid"],
    }

    def _find_split_root(split_key: str) -> str:
        for rel in CANDIDATES[split_key]:
            abs_root = os.path.join(path, rel)
            if os.path.isdir(abs_root):
                return rel
        tried = ", ".join([os.path.join(path, r) for r in CANDIDATES[split_key]])
        raise FileNotFoundError(
            f"[imagenet_subset] Could not locate a '{split_key}' directory under {path}. "
            f"Tried: {tried}"
        )

    train_root = _find_split_root("train")
    val_root   = _find_split_root("val")

    if not lbl_order:
        classes_txt = os.path.join(path, "classes.txt")
        if os.path.exists(classes_txt) and os.pathsize(classes_txt) > 0:
            with open(classes_txt, "rt") as f:
                lbl_order = [ln.strip() for ln in f if ln.strip()]
            print(f"[imagenet_subset] Loaded class order from {classes_txt} ({len(lbl_order)} classes).")
        else:
            def _list_dirs(rel_root):
                abs_root = os.path.join(path, rel_root)
                return {d for d in os.listdir(abs_root)
                        if os.path.isdir(os.path.join(abs_root, d))}
            train_classes = _list_dirs(train_root)
            val_classes   = _list_dirs(val_root)
            common = sorted(train_classes & val_classes)
            if not common:
                raise RuntimeError(
                    f"[imagenet_subset] No common classes between "
                    f"{os.path.join(path, train_root)} and {os.path.join(path, val_root)}."
                )

            if isinstance(num_classes_cfg, int) and num_classes_cfg > 0:
                if len(common) < num_classes_cfg:
                    raise RuntimeError(
                        f"[imagenet_subset] Found {len(common)} classes < requested {num_classes_cfg}."
                    )
                lbl_order = common[:num_classes_cfg]
            else:
                lbl_order = common

            with open(classes_txt, "wt") as f:
                f.write("\n".join(lbl_order) + "\n")
            print(f"[imagenet_subset] Inferred {len(lbl_order)} classes and saved to {classes_txt}.")

    class2idx = {c: i for i, c in enumerate(lbl_order)}
    allowed = set(lbl_order)

    def _write_split(rel_root: str, outfile: str):
        abs_root = os.path.join(path, rel_root)
        cnt = 0
        unk = set()
        with open(os.path.join(path, outfile), "wt") as f:
            for wnid in os.listdir(abs_root):
                cls_dir = os.path.join(abs_root, wnid)
                if not os.path.isdir(cls_dir):
                    continue
                if wnid not in allowed:
                    unk.add(wnid)
                    continue
                lbl = class2idx[wnid]
                for fn in os.listdir(cls_dir):
                    img_path = os.path.join(cls_dir, fn)
                    if os.path.isfile(img_path):
                        rel = os.path.relpath(img_path, start=path)
                        f.write(f"{rel} {lbl}\n")
                        cnt += 1
        if cnt == 0:
            raise RuntimeError(
                f"[imagenet_subset] No images found under '{abs_root}' for classes in classes.txt/lbl_order."
            )
        if unk:
            print(f"[imagenet_subset] Ignored {len(unk)} unknown class dirs under {abs_root} (not in class order).")
        print(f"[imagenet_subset] Wrote {cnt} samples -> {outfile} from {abs_root}")

    _write_split(train_root, train_out)
    _write_split(val_root,   test_out)


def _ensure_tiny_prepared(path):
    assert os.path.exists(path), f"Please first download and extract dataset from: http://cs231n.stanford.edu/tiny-imagenet-200.zip to dir: {path}"
    classes = os.listdir(path + "/train")
    class2idx = {c: i for i, c in enumerate(classes)}

    with open(f"{path}/train.txt", 'wt') as f:
        for c in classes:
            images = os.listdir(path + f"/train/{c}/images")
            for image in images:
                f.write(f"train/{c}/images/{image} {class2idx[c]}\n")

    with open(f"{path}/val/val_annotations.txt", 'r') as f:
        val_annotations = f.readlines()

    with open(f"{path}/test.txt", 'wt') as f:
        for anno in val_annotations:
            image, label = anno.split("\t")[:2]
            f.write(f"val/images/{image} {class2idx[label]}\n")


def _ensure_cub200_subset_prepared(path):
    assert os.path.exists(path), f"Please first download and extract dataset from: https://www.kaggle.com/datasets/arjunashok33/imagenet-subset-for-inc-learn to dir: {path}"
    image_class_labels = {}
    with open(f"{path}/{'image_class_labels.txt'}", 'r') as f:
        lines = f.readlines()
    for line in lines:
        i, c = line.replace("\n", "").split(" ")
        image_class_labels[i] = c

    images = {}
    with open(f"{path}/{'images.txt'}", 'r') as f:
        lines = f.readlines()
    for line in lines:
        i, c = line.replace("\n", "").split(" ")
        images[i] = c

    train = {}
    test = {}
    with open(f"{path}/{'train_test_split.txt'}", 'r') as f:
        lines = f.readlines()
    for line in lines:
        image_num, is_training = line.replace("\n", "").split(" ")
        if is_training == "1":
            train[images[image_num]] = image_class_labels[image_num]
        else:
            test[images[image_num]] = image_class_labels[image_num]

    with open(f"{path}/{'train.txt'}", 'wt') as f:
        for k, v in train.items():
            f.write(f"images/{k} {int(v) - 1} \n")

    with open(f"{path}/{'test.txt'}", 'wt') as f:
        for k, v in test.items():
            f.write(f"images/{k} {int(v) - 1} \n")


def _ensure_domainnet_prepared(path, classes_per_domain=50, num_tasks=6):
    assert os.path.exists(path), f"Please first download and extract dataset from: http://ai.bu.edu/M3SDA/#dataset into:{path}"
    domains = ["clipart", "infograph", "painting",  "quickdraw", "real", "sketch"] * (num_tasks // 6)
    for set_type in ["train", "test"]:
        samples = []
        for i, domain in enumerate(domains):
            with open(f"{path}/{domain}_{set_type}.txt", 'r') as f:
                lines = list(map(lambda x: x.replace("\n", "").split(" "), f.readlines()))
            paths, classes = zip(*lines)
            classes = np.array(list(map(float, classes)))
            offset = classes_per_domain * i
            for c in range(classes_per_domain):
                is_class = classes == c + ((i // 6) * classes_per_domain)
                class_samples = list(compress(paths, is_class))
                samples.extend([*[f"{row} {c + offset}" for row in class_samples]])
        with open(f"{path}/{set_type}.txt", 'wt') as f:
            for sample in samples:
                f.write(f"{sample}\n")
