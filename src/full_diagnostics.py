"""
full_diagnostics.py
===================

This script provides a complete, end‑to‑end pipeline for running
diagnostic measurements on incremental learning models.  It automates
the process of splitting a dataset into a given number of tasks,
computing per‑task metrics, and aggregating these results into arrays
for further analysis or visualisation.

The key features of this script are:

* **Task indexing:** Compute which class IDs belong to a given task
  based on the total number of classes and the desired number of
  tasks.  This supports datasets like CIFAR‑100 where you might
  divide the 100 classes into 10 tasks of 10 classes each.

* **Dataset loading:** For CIFAR‑100 specifically, the script uses
  torchvision to load the dataset and then filters samples to include
  only those belonging to the classes of a particular task.  You can
  extend this to other datasets by following the same pattern.

* **Model loading:** Placeholder functions are provided to load your
  backbone network, adapter and distiller from a given checkpoint
  directory.  These should be replaced with your actual model
  definitions and checkpoint loading logic.

* **Diagnostic metrics:** A stub implementation of the
  ``run_diagnostics`` function is included.  It shows how you might
  compute cycle consistency losses (``cycle_new`` and ``cycle_old``)
  and singular value decomposition (SVD) statistics for the combined
  adapter/distiller mappings.  You **must** replace the stub with
  your actual diagnostic routine, or import it from your existing
  project.

* **Aggregated metrics:** Functions are provided to run diagnostics
  across a range of tasks and to compare two separate model runs.
  The results are returned as arrays of floats, making them suitable
  for plotting or further analysis.

To use this script in your environment, adapt the placeholder
functions ``load_model_components`` and ``run_diagnostics`` to match
your model architecture and metric definitions.  Then run the
``main`` function or import the provided helpers into your own
scripts.
"""

from __future__ import annotations

import os
import math
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms


def compute_task_class_indices(
    total_classes: int, num_tasks: int, task_idx: int
) -> List[int]:
    """Compute the list of class indices assigned to a task.

    When splitting ``total_classes`` evenly among ``num_tasks`` tasks,
    tasks with lower indices receive one more class if the division
    leaves a remainder.  For example, splitting 100 classes into 10
    tasks results in each task having exactly 10 classes.  Splitting
    101 classes into 10 tasks results in the first task having 11
    classes and the remaining nine tasks having 10 classes each.

    Parameters
    ----------
    total_classes : int
        The total number of distinct class labels in the dataset.
    num_tasks : int
        The number of tasks into which the classes should be split.
    task_idx : int
        The index (0‑based) of the task for which to compute the
        class indices.  Must satisfy 0 ≤ task_idx < num_tasks.

    Returns
    -------
    List[int]
        A list of class IDs corresponding to the requested task.
    """
    if task_idx < 0 or task_idx >= num_tasks:
        raise ValueError(f"task_idx must be in [0, {num_tasks}), got {task_idx}")
    base = total_classes // num_tasks
    remainder = total_classes % num_tasks
    # The first 'remainder' tasks get one extra class
    sizes = [base + (1 if i < remainder else 0) for i in range(num_tasks)]
    start = sum(sizes[:task_idx])
    size = sizes[task_idx]
    return list(range(start, start + size))


class TaskSubsetDataset(Dataset):
    """Dataset wrapper that selects only samples from specified classes.

    This wrapper takes an underlying torchvision dataset (e.g.
    CIFAR100) and filters it to include only those samples whose
    original label is in the provided ``keep_classes`` list.  Labels
    are optionally remapped to a contiguous range starting at 0.

    Parameters
    ----------
    base_dataset : Dataset
        The original torchvision dataset to wrap.
    keep_classes : Iterable[int]
        Iterable of class IDs to keep.  Only samples whose label is in
        this set will be included.
    remap : bool, optional
        If True, remap the kept class labels to a contiguous range
        starting at 0.  If False, keep the original labels.
    """

    def __init__(self, base_dataset: Dataset, keep_classes: Iterable[int], remap: bool = False):
        super().__init__()
        self.base_dataset = base_dataset
        self.keep = set(keep_classes)
        self.indices = []
        self.original_labels = []
        for idx in range(len(self.base_dataset)):
            label = self.base_dataset.targets[idx]
            if label in self.keep:
                self.indices.append(idx)
                self.original_labels.append(label)
        if remap:
            # Create a mapping from original labels to [0..K-1]
            unique = sorted(set(self.original_labels))
            label_map = {c: i for i, c in enumerate(unique)}
            self.labels = [label_map[l] for l in self.original_labels]
        else:
            self.labels = list(self.original_labels)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        base_idx = self.indices[idx]
        img, _ = self.base_dataset[base_idx]
        label = self.labels[idx]
        return img, label


def load_cifar100_task(
    data_root: str,
    task_classes: Iterable[int],
    *,
    train: bool = False,
    batch_size: int = 128,
    num_workers: int = 2,
) -> DataLoader:
    """Load CIFAR‑100 images for a specific task.

    This helper function loads either the training or test portion of
    CIFAR‑100, applies standard normalisation transforms, and filters
    the samples to include only those with labels in ``task_classes``.
    The returned dataloader yields mini‑batches of data suitable for
    evaluating your network.

    Parameters
    ----------
    data_root : str
        Directory where CIFAR‑100 data will be downloaded or is already
        stored.  ``torchvision.datasets.CIFAR100`` will manage
        downloading the files if they are not present.
    task_classes : Iterable[int]
        The class IDs to include in this task.  Samples with other
        labels will be discarded.
    train : bool, optional
        If True, load the training split; otherwise load the test
        split.  Defaults to False (test set).
    batch_size : int, optional
        Batch size for the returned dataloader.  Defaults to 128.
    num_workers : int, optional
        Number of worker processes to use for data loading.  Defaults
        to 2.

    Returns
    -------
    torch.utils.data.DataLoader
        A dataloader yielding batches of (image, label) pairs for
        samples belonging to the specified classes.
    """
    normalize = transforms.Normalize(
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    dataset = torchvision.datasets.CIFAR100(
        root=data_root, train=train, download=True, transform=transform
    )
    task_ds = TaskSubsetDataset(dataset, keep_classes=task_classes, remap=False)
    loader = DataLoader(
        task_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


def load_model_components(model_dir: str):
    """Load the backbone, adapter and distiller from a checkpoint directory.

    This function is a placeholder demonstrating how you might load
    your network and auxiliary modules (adapter and distiller) from
    files on disk.  You should customise it to match your model
    architecture, file naming conventions, and device placement.

    Parameters
    ----------
    model_dir : str
        Path to the directory containing model checkpoints.  The
        function assumes that the backbone weights are stored in a
        file named ``model_X.pth``, the distiller in
        ``distiller_X.pth`` and the adapter in ``adapter_after_X.pth``,
        where ``X`` is the task number.  Adjust these names as
        necessary to reflect your setup.

    Returns
    -------
    Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]
        A tuple containing the backbone model, distiller and adapter.
    """
    # NOTE: Replace the following placeholder model definitions with
    # your actual network classes.  For example, if your backbone is
    # a ResNet18, import and instantiate torchvision.models.resnet18
    # and then load the state dictionary.  Similarly for the
    # adapter/distiller MLPs.
    # Example (pseudo‑code):
    #   backbone = torchvision.models.resnet18(num_classes=...)  # feature extractor
    #   distiller = MyDistillerClass()
    #   adapter = MyAdapterClass()
    #   backbone.load_state_dict(torch.load(os.path.join(model_dir, 'model_1.pth')))
    #   distiller.load_state_dict(torch.load(os.path.join(model_dir, 'distiller_1.pth')))
    #   adapter.load_state_dict(torch.load(os.path.join(model_dir, 'adapter_after_1.pth')))
    #   return backbone, distiller, adapter
    #
    # For this example, we create minimal linear layers so that the
    # script runs without error.  You MUST replace these with your
    # actual models.
    backbone = torch.nn.Identity()
    distiller = torch.nn.Identity()
    adapter = torch.nn.Identity()
    return backbone, distiller, adapter


def run_diagnostics(
    *,
    task_idx: int,
    ds_name: str,
    data_root: str,
    class_indices: Sequence[int],
    split: str,
    batch_size: int,
    num_workers: int,
    model_dir: str,
) -> Dict[str, float]:
    """Run diagnostic metrics for a single task.

    This is a stub implementation demonstrating how you might compute
    the cycle consistency errors and SVD statistics.  In practice you
    should replace the body of this function with calls to your
    existing diagnostic routines.

    Parameters
    ----------
    task_idx : int
        The task index being evaluated.
    ds_name : str
        The name of the dataset.  Only 'cifar100' is supported by
        this stub.
    data_root : str
        Root directory of the dataset.
    class_indices : Sequence[int]
        A list of class IDs belonging to this task.
    split : str
        Which split to use: 'train' or 'test'.  Only 'test' is used in
        this stub.
    batch_size : int
        Batch size for loading data.
    num_workers : int
        Number of worker processes for the dataloader.
    model_dir : str
        Path to the directory containing the model, adapter and
        distiller checkpoints.

    Returns
    -------
    Dict[str, float]
        A dictionary containing the following keys:

        - ``cycle_new``
        - ``cycle_old``
        - ``ad_mean``
        - ``ad_std``
        - ``ad_pct``
        - ``da_mean``
        - ``da_std``
        - ``da_pct``

        The values in this stub are randomised to illustrate the
        structure of the output.  You should replace the body of this
        function with your actual metric computations.
    """
    # Load the relevant data for the task
    if ds_name.lower() == "cifar100":
        loader = load_cifar100_task(
            data_root,
            class_indices,
            train=False,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    else:
        raise NotImplementedError(
            f"Dataset {ds_name} not supported in this stub. Implement your own data loader."
        )

    # Load the backbone, distiller and adapter.  These are dummy
    # identity mappings in this stub.  Replace with your actual models.
    backbone, distiller, adapter = load_model_components(model_dir)
    backbone.eval()
    distiller.eval()
    adapter.eval()

    # Device selection: use CPU in this stub.  Modify as needed.
    device = torch.device("cpu")
    backbone.to(device)
    distiller.to(device)
    adapter.to(device)

    # Placeholder accumulators for the metrics
    cycle_new_errors = []
    cycle_old_errors = []
    # To estimate SVD statistics, accumulate the outputs of the
    # composite mappings over the dataset.  This is a simple
    # approximation: we flatten feature vectors and collect them.  In a
    # real implementation you may be able to extract linear weights
    # directly from your adapter and distiller.
    ad_outputs = []
    da_outputs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            # Forward pass through the backbone to obtain feature
            # representations.  Our stub backbone is identity so
            # features = images.  Replace this with your feature
            # extractor (e.g. backbone(images)).
            features = images.view(images.size(0), -1)
            # Apply distiller then adapter: z_new_hat = A(D(z))
            z_new_hat = adapter(distiller(features))
            # Apply adapter then distiller: z_old_hat = D(A(z))
            z_old_hat = distiller(adapter(features))
            # Compute cycle errors: how well do the compositions
            # reconstruct the original features?
            cycle_new = (z_new_hat - features).pow(2).mean(dim=1)
            cycle_old = (z_old_hat - features).pow(2).mean(dim=1)
            cycle_new_errors.append(cycle_new.cpu())
            cycle_old_errors.append(cycle_old.cpu())
            # Accumulate flattened outputs for SVD estimation
            ad_outputs.append(z_new_hat.cpu())
            da_outputs.append(z_old_hat.cpu())

    # Aggregate cycle errors over all batches
    cycle_new_errors = torch.cat(cycle_new_errors)
    cycle_old_errors = torch.cat(cycle_old_errors)
    cycle_new_value = float(cycle_new_errors.mean().item())
    cycle_old_value = float(cycle_old_errors.mean().item())

    # Stack the outputs and flatten across samples
    ad_matrix = torch.cat(ad_outputs, dim=0)
    da_matrix = torch.cat(da_outputs, dim=0)
    # Compute singular values of the covariance of the outputs as a
    # proxy.  In a real implementation you may want to compute the
    # Jacobians or linear weight matrices instead.
    # We'll compute SVD of the centred outputs.  Note: this is a
    # rough approximation to the behaviour of your adapter/distiller.
    def svd_stats(mat: torch.Tensor) -> Tuple[float, float, float]:
        """Compute statistics of the singular values of a matrix.

        This helper centres the rows of ``mat`` and then performs a
        singular value decomposition (SVD).  It returns the mean,
        standard deviation and percentage of singular values within
        [0.9, 1.1].  A real implementation would likely compute
        singular values of the Jacobian of the adapter/distiller, but
        this serves as a proxy.
        """
        # Center the data per feature dimension
        centred = mat - mat.mean(dim=0, keepdim=True)
        # Compute SVD on CPU for stability
        u, s, v = torch.svd(centred.cpu(), some=False)
        mean = float(s.mean().item())
        std = float(s.std().item())
        # Compute percentage of singular values in the interval [0.9, 1.1]
        pct = float(((s >= 0.9) & (s <= 1.1)).sum().item() / s.numel() * 100.0)
        return mean, std, pct

    ad_mean, ad_std, ad_pct = svd_stats(ad_matrix)
    da_mean, da_std, da_pct = svd_stats(da_matrix)

    return {
        "cycle_new": cycle_new_value,
        "cycle_old": cycle_old_value,
        "ad_mean": ad_mean,
        "ad_std": ad_std,
        "ad_pct": ad_pct,
        "da_mean": da_mean,
        "da_std": da_std,
        "da_pct": da_pct,
    }


def compute_metrics_for_task(
    *,
    model_dir: str,
    ds_name: str,
    data_root: str,
    total_classes: int,
    num_tasks: int,
    task_idx: int,
    split: str = "test",
    batch_size: int = 128,
    num_workers: int = 2,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """Compute diagnostic metrics for a single task.

    This function automatically derives the class indices for the
    specified task and calls ``run_diagnostics`` with those indices.
    It extracts the relevant metrics from the returned dictionary and
    returns them as a tuple in a fixed order.

    Parameters
    ----------
    model_dir : str
        Directory containing the model and auxiliary checkpoints.
    ds_name : str
        Name of the dataset (e.g. 'cifar100').
    data_root : str
        Root directory of the dataset.
    total_classes : int
        Total number of classes in the dataset.
    num_tasks : int
        Number of tasks to split the classes into.
    task_idx : int
        The task index for which to compute metrics.
    split : str, optional
        Which dataset split to evaluate ('train' or 'test').  Defaults
        to 'test'.
    batch_size : int, optional
        Mini‑batch size for data loading.  Defaults to 128.
    num_workers : int, optional
        Number of worker processes for the data loader.  Defaults to
        2.

    Returns
    -------
    tuple
        A tuple of eight floats representing (cycle_new, cycle_old,
        ad_mean, ad_std, ad_pct, da_mean, da_std, da_pct).
    """
    classes = compute_task_class_indices(total_classes, num_tasks, task_idx)
    results = run_diagnostics(
        task_idx=task_idx,
        ds_name=ds_name,
        data_root=data_root,
        class_indices=classes,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        model_dir=model_dir,
    )
    return (
        results["cycle_new"],
        results["cycle_old"],
        results["ad_mean"],
        results["ad_std"],
        results["ad_pct"],
        results["da_mean"],
        results["da_std"],
        results["da_pct"],
    )


def compute_metrics_across_tasks(
    *,
    model_dir: str,
    ds_name: str,
    data_root: str,
    total_classes: int,
    num_tasks: int,
    tasks: Iterable[int],
    split: str = "test",
    batch_size: int = 128,
    num_workers: int = 2,
) -> Dict[str, List[float]]:
    """Compute diagnostic metrics across multiple tasks for a model.

    Iterate over the specified task indices, compute metrics for each
    task, and aggregate the values into lists keyed by the metric
    name.

    Parameters
    ----------
    model_dir : str
        Directory containing the model and auxiliary checkpoints.
    ds_name : str
        Name of the dataset (currently only 'cifar100' is supported by
        the stub diagnostics).  Extend ``run_diagnostics`` for
        additional datasets.
    data_root : str
        Root directory of the dataset.
    total_classes : int
        Total number of classes in the dataset.
    num_tasks : int
        Total number of tasks.
    tasks : Iterable[int]
        Task indices over which to compute metrics.  Should not
        include task 0 if you wish to compare new performance only.
    split : str, optional
        Dataset split to evaluate ('train' or 'test').
    batch_size : int, optional
        Mini‑batch size.
    num_workers : int, optional
        Number of data loader workers.

    Returns
    -------
    Dict[str, List[float]]
        A dictionary whose keys are metric names and whose values are
        lists of metric values in the same order as ``tasks``.
    """
    metrics = {
        "cycle_new": [],
        "cycle_old": [],
        "ad_mean": [],
        "ad_std": [],
        "ad_pct": [],
        "da_mean": [],
        "da_std": [],
        "da_pct": [],
    }
    for t in tasks:
        (
            cycle_new,
            cycle_old,
            ad_mean,
            ad_std,
            ad_pct,
            da_mean,
            da_std,
            da_pct,
        ) = compute_metrics_for_task(
            model_dir=model_dir,
            ds_name=ds_name,
            data_root=data_root,
            total_classes=total_classes,
            num_tasks=num_tasks,
            task_idx=t,
            split=split,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        metrics["cycle_new"].append(cycle_new)
        metrics["cycle_old"].append(cycle_old)
        metrics["ad_mean"].append(ad_mean)
        metrics["ad_std"].append(ad_std)
        metrics["ad_pct"].append(ad_pct)
        metrics["da_mean"].append(da_mean)
        metrics["da_std"].append(da_std)
        metrics["da_pct"].append(da_pct)
    return metrics


def compare_two_paths(
    path_a: str,
    path_b: str,
    *,
    ds_name: str,
    data_root: str,
    total_classes: int,
    num_tasks: int,
    tasks: Iterable[int],
    split: str = "test",
    batch_size: int = 128,
    num_workers: int = 2,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """Compute diagnostic metrics for two model directories and return both.

    Parameters
    ----------
    path_a : str
        The first model directory to compare.
    path_b : str
        The second model directory to compare.
    ds_name : str
        Name of the dataset.
    data_root : str
        Root directory of the dataset.
    total_classes : int
        Total number of classes.
    num_tasks : int
        Total number of tasks.
    tasks : Iterable[int]
        Task indices over which to compute metrics (e.g. 1..9).
    split, batch_size, num_workers : optional
        Passed directly to ``compute_metrics_across_tasks``.

    Returns
    -------
    Tuple[Dict[str, List[float]], Dict[str, List[float]]]
        A pair of metric dictionaries corresponding to path_a and
        path_b respectively.
    """
    metrics_a = compute_metrics_across_tasks(
        model_dir=path_a,
        ds_name=ds_name,
        data_root=data_root,
        total_classes=total_classes,
        num_tasks=num_tasks,
        tasks=tasks,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    metrics_b = compute_metrics_across_tasks(
        model_dir=path_b,
        ds_name=ds_name,
        data_root=data_root,
        total_classes=total_classes,
        num_tasks=num_tasks,
        tasks=tasks,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return metrics_a, metrics_b


def main():
    """Example entry point.

    This demonstrates how to compute metrics for a single model across
    tasks 1..9 (assuming 10 tasks in total) on CIFAR‑100.  It then
    prints the resulting arrays.  Adjust the paths and arguments to
    suit your use case.  If you need to compare two runs, call
    ``compare_two_paths`` instead.
    """
    # Example configuration.  Replace these with your actual paths
    # and settings.
    example_model_path = "/path/to/model/run1"
    dataset_name = "cifar100"
    dataset_root = "/path/to/cifar100"
    num_total_classes = 100
    num_tasks_total = 10
    # Compute metrics for tasks 1 through 9
    task_indices = list(range(1, num_tasks_total))
    metrics = compute_metrics_across_tasks(
        model_dir=example_model_path,
        ds_name=dataset_name,
        data_root=dataset_root,
        total_classes=num_total_classes,
        num_tasks=num_tasks_total,
        tasks=task_indices,
        split="test",
        batch_size=128,
        num_workers=2,
    )
    print("Metrics for model at", example_model_path)
    for key, values in metrics.items():
        print(f"  {key}: {values}")

    # Example of comparing two paths.  Uncomment and customise:
    # path_a = "/path/to/model/run1"
    # path_b = "/path/to/model/run2"
    # metrics_a, metrics_b = compare_two_paths(
    #     path_a, path_b,
    #     ds_name=dataset_name,
    #     data_root=dataset_root,
    #     total_classes=num_total_classes,
    #     num_tasks=num_tasks_total,
    #     tasks=task_indices,
    #     split="test",
    #     batch_size=128,
    #     num_workers=2,
    # )
    # print("Metrics for first model:")
    # for key, values in metrics_a.items():
    #     print(f"  {key}: {values}")
    # print("Metrics for second model:")
    # for key, values in metrics_b.items():
    #     print(f"  {key}: {values}")


if __name__ == "__main__":
    main()