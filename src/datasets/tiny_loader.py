# tiny_imagenet_no_wnid.py
import os
from typing import Callable, Optional, Tuple, List, Any
from PIL import Image
from torchvision.datasets.vision import VisionDataset


class TinyImageNet200Folder(VisionDataset):
    """
    Tiny-ImageNet-200 loader that infers class list from sub-folders,
    so no wnids.txt is required.

    Expected structure (after your re-organisation):

        tiny-imagenet-200/
        ├── train/<wnid>/*.JPEG
        └── val/<wnid>/*.JPEG
    """

    base_folder = "tiny-imagenet-200"

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform, target_transform)
        self.train = train
        split_dir = "train" if train else "val"
        self.split_root = os.path.join(root, self.base_folder, split_dir)

        # ------------------------------------------------------------------
        # 1) Build class list by scanning sub-folders (sorted for stability)
        # ------------------------------------------------------------------
        self.classes: List[str] = sorted(
            d for d in os.listdir(self.split_root)
            if os.path.isdir(os.path.join(self.split_root, d))
        )
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # ------------------------------------------------------------------
        # 2) Gather image paths
        # ------------------------------------------------------------------
        self.samples = self._gather_samples()
        self.targets = [lbl for _, lbl in self.samples]   # CIFAR-style attr

    def _gather_samples(self):
        samples = []
        for wnid in self.classes:
            cls_dir = os.path.join(self.split_root, wnid)
            label = self.class_to_idx[wnid]
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                    samples.append((os.path.join(cls_dir, fname), label))
        return samples

    # ---------------- Dataset API ----------------
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img_path, target = self.samples[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform:         img = self.transform(img)
        if self.target_transform:  target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        return len(self.samples)

    def extra_repr(self) -> str:
        split = "Train" if self.train else "Val"
        return f"Split: {split}, #class: {len(self.classes)}, #images: {len(self)}"
