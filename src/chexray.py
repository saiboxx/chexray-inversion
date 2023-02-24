"""Datasets containing the CheX-ray14 database."""
import ast
import os
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

import pandas as pd
import torch
from PIL import Image
from pandas import read_csv
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from src.iterative import collect_iterative_results
from src.utils import (
    load_image,
    read_file,
    read_pickle, save_pickle,
)

DISEASES = (
    'Atelectasis',
    'Consolidation',
    'Infiltration',
    'Pneumothorax',
    'Edema',
    'Emphysema',
    'Fibrosis',
    'Effusion',
    'Pneumonia',
    'Pleural_Thickening',
    'Cardiomegaly',
    'Nodule',
    'Hernia',
    'Mass',
)


class ChexrayDataset(Dataset):
    """Dataset object for CheX-ray14."""

    def __init__(
        self,
        root: str,
        img_dir_name: str,
        train: bool = True,
        transforms: Any = None,
        rgb_convert: bool = False,
    ) -> None:
        """Initialize CheX-ray14 dataset."""
        self.root = root
        self.img_dir = os.path.join(root, img_dir_name)

        key_file = 'train_val_list.txt' if train else 'test_list.txt'
        self.keys = read_file(os.path.join(self.root, key_file))

        meta_path = os.path.join(self.root, 'meta.pkl')
        if not os.path.exists(meta_path):
            _preprocess_meta(root)
        self.meta = read_pickle(meta_path)

        self.transforms = transforms
        self.do_rgb_convert = rgb_convert

    def __len__(self):
        """Return length of the dataset."""
        return len(self.keys)

    def __getitem__(self, idx: int) -> Dict:
        """Return a data sample."""
        key = self.keys[idx]
        sample_data = self.meta[key]
        img = Image.open(os.path.join(self.img_dir, key))
        if self.do_rgb_convert:
            img = img.convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)

        return {'image': img, 'label': torch.tensor(sample_data['label']).float()}


class Chexray2LatentDataset(ChexrayDataset):
    def __getitem__(self, idx: int) -> Tuple[Tensor, Dict]:  # type: ignore
        key = self.keys[idx]
        sample_data = self.meta[key]

        img = Image.open(os.path.join(self.img_dir, key))
        if self.do_rgb_convert:
            img = img.convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img, {**sample_data, 'idx': idx, 'file': key}

    @staticmethod
    def collate_fn(batch_data: List[Tuple]) -> Tuple[Tensor, List[Dict]]:
        imgs, sample_datas = zip(*batch_data)
        return torch.stack(imgs), list(sample_datas)


class ChexrayEmbedded(Dataset):
    def __init__(
        self,
        z_path: str,
        meta_path: str,
    ) -> None:
        self.meta = pd.read_csv(meta_path)
        self.zs = torch.load(z_path)

    def __len__(self) -> int:
        """Return length of the dataset."""
        return len(self.zs)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return (
            self.zs[idx],
            torch.tensor(ast.literal_eval(self.meta['label'][idx])[8]).float(),
        )


class ChexrayReconstruction(Dataset):
    def __init__(
        self,
        image_root: str,
        z_path: str,
        meta_path: str,
        z_optimized_root: Optional[str] = None,
    ) -> None:
        self.img_root = image_root
        self.meta = pd.read_csv(meta_path)
        self.enc_zs = torch.load(z_path)

        if z_optimized_root is not None:
            self.opt_zs = collect_iterative_results(z_optimized_root)
        else:
            self.opt_zs = self.enc_zs

    def __len__(self) -> int:
        return len(self.opt_zs)

    def __getitem__(self, idx: int) -> Tuple:
        file_id = self.meta['file'][idx]
        img = load_image(os.path.join(self.img_root, file_id))
        enc_z = self.enc_zs[idx]
        opt_z = self.opt_zs[idx]
        return img, enc_z, opt_z

    def get_by_file(self, file_name: str) -> Tuple:
        idx = int(self.meta[self.meta['file'] == file_name]['idx'])
        img = load_image(os.path.join(self.img_root, file_name))
        enc_z = self.enc_zs[idx]
        opt_z = self.opt_zs[idx]
        return img, enc_z, opt_z


def _preprocess_meta(root: str) -> None:
    """
    Preprocess meta data.
    The label is a binary vector, where each entry corresponds to the diagnosis of
    a certain disease. The key is the name of the image file. Further, age & gender,
    as well as patient id & follow up number is saved. The result is a pickled
    byte object `meta.pkl`, which contains a dict for every patient.
    """
    file_name = 'Data_Entry_2017_v2020.csv'
    meta_csv = read_csv(os.path.join(root, file_name))

    result = {}
    print('Preprocessing meta data ...')
    for entry in tqdm(meta_csv):
        diagnosis = entry['Finding Labels']
        label = [0 for _ in range(14)]

        for idx, disease in enumerate(DISEASES):
            if disease in diagnosis:
                label[idx] = 1

        result.update(
            {
                entry['Image Index']: {
                    'label': label,
                    'age': int(entry['Patient Age']),
                    'gender': entry['Patient Gender'],
                    'pat_id': int(entry['Patient ID']),
                    'pat_follow_up': int(entry['Follow-up #']),
                }
            }
        )

    meta_path = os.path.join(root, 'meta.pkl')
    save_pickle(result, meta_path)
    print('Meta data was saved in {}'.format(meta_path))