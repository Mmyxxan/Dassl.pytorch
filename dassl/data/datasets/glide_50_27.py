import os.path as osp

from dassl.utils import listdir_nohidden

from .build import DATASET_REGISTRY
from .base_dataset import Datum, DatasetBase

@DATASET_REGISTRY.register()
class Glide5027(DatasetBase):

    dataset_dir = "deepfake_eval/glide_50_27/images/val"

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        train_x = self._read_data(split="train")
        val = self._read_data(split="val")
        test = self._read_data(split="test")

        super().__init__(train_x=train_x, val=val, test=test)

    def _read_data(self, split="train"):
        items = []

        labels_map = {
            "n01440764": 0,
            "n01443537": 1,
        }

        data_dir = osp.join(self.dataset_dir)
        label_names = listdir_nohidden(data_dir)
        for label_name in label_names:
            label_dir = osp.join(data_dir, label_name)
            label = labels_map[label_name]
            imnames = listdir_nohidden(label_dir)
            for imname in imnames:
                impath = osp.join(label_dir, imname)
                item = Datum(impath=impath, label=label)
                items.append(item)
        
        return items
