import os.path as osp

from dassl.utils import listdir_nohidden

from .build import DATASET_REGISTRY
from .base_dataset import Datum, DatasetBase

@DATASET_REGISTRY.register()
class CNNSpot(DatasetBase):

    dataset_dir = "cnn_spot"

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        train_x = self._read_data(split="train")
        val = self._read_data(split="val")
        test = self._read_data(split="test")

        super().__init__(train_x=train_x, val=val, test=test)

    def _read_data(self, split="train"):
        items = []

        data_dir = osp.join(self.dataset_dir, split)
        if split == "train" or split == "val":
            class_names = listdir_nohidden(data_dir)
            for class_name in class_names:
                class_dir = osp.join(data_dir, class_name)
                label_names = listdir_nohidden(class_dir)
                for label_name in label_names:
                    label_dir = osp.join(class_dir, label_name)
                    label = int(label_name.split("_")[0])
                    imnames = listdir_nohidden(label_dir)
                    for imname in imnames[:9000]:
                        impath = osp.join(label_dir, imname)
                        item = Datum(impath=impath, label=label)
                        items.append(item)
        elif split == "test":
            model_names = listdir_nohidden(data_dir)
            for model_name in model_names:
                model_dir = osp.join(data_dir, model_name)
                if model_name not in ["cyclegan", "progan", "stylegan", "stylegan2"]:
                    label_names = listdir_nohidden(model_dir)
                    for label_name in label_names:
                        label_dir = osp.join(model_dir, label_name)
                        label = int(label_name.split("_")[0])
                        imnames = listdir_nohidden(label_dir)
                        for imname in imnames:
                            impath = osp.join(label_dir, imname)
                            item = Datum(impath=impath, label=label)
                            items.append(item)
                else:
                    class_names = listdir_nohidden(model_dir)
                    for class_name in class_names:
                        class_dir = osp.join(model_dir, class_name)
                        label_names = listdir_nohidden(class_dir)
                        for label_name in label_names:
                            label_dir = osp.join(class_dir, label_name)
                            label = int(label_name.split("_")[0])
                            imnames = listdir_nohidden(label_dir)
                            for imname in imnames:
                                impath = osp.join(label_dir, imname)
                                item = Datum(impath=impath, label=label)
                                items.append(item)
        
        return items
