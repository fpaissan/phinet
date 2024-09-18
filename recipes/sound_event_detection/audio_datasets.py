"""
The file contains all the Dataset instances needed to create the CLAP training set.

Authors:
    - Francesco Paissan 2023, 2024
"""

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import yaml
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets.utils import download_and_extract_archive, download_url
from tqdm import tqdm

duration = 5


def random_crop(tmp, sample_rate):
    crop_len = sample_rate * duration
    if tmp.shape[1] - crop_len > 0:
        start = torch.randint(low=0, high=(tmp.shape[1] - crop_len), size=(1,))
        tmp = tmp[:, start : start + crop_len]

    return tmp


# thanks https://github.com/multitel-ai/urban-sound-classification-and-comparison
class UrbanSound8K(Dataset):
    base_folder = "UrbanSound8K"
    resources = [
        (
            "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz",
            "9aa69802bbf37fb986f71ec1483a196e",
        )
    ]

    def __init__(
        self,
        dataset_folder,
        sample_rate=44100,
        fold=None,
        transform=None,
        download=False,
    ):
        super().__init__()
        self.dataset_folder = dataset_folder
        self.sample_rate = sample_rate
        # os.path.join(dataset_folder, UrbanSound8K.base_folder)

        self.path_to_csv = os.path.join(
            self.dataset_folder, "metadata/UrbanSound8K.csv"
        )
        self.path_to_audio_folder = os.path.join(self.dataset_folder, "audio")
        self.path_to_melTALNet = os.path.join(self.dataset_folder, "melTALNet")

        # Downloading and extracting if needed
        if download:
            self.download()

        # Checking if the dataset exist at specified location
        if not os.path.exists(self.dataset_folder):
            raise RuntimeError(
                "Dataset not found." + " You can use download=True to download it"
            )

        self.raw_annotations = pd.read_csv(self.path_to_csv)
        self.fold = fold

        self.transform = transform
        self.classes = [
            "air_conditioner",
            "car_horn",
            "children_playing",
            "dog_bark",
            "drilling",
            "engine_idling",
            "gun_shot",
            "jackhammer",
            "siren",
            "street_music",
        ]

    def train_validation_split(self):
        """ """
        train_idx = list(
            self.raw_annotations[self.raw_annotations["fold"] != self.fold].index
        )
        val_idx = list(
            self.raw_annotations[self.raw_annotations["fold"] == self.fold].index
        )

        train_set = Subset(self, train_idx)
        val_set = Subset(self, val_idx)
        val_set.transform = None

        return train_set, val_set

    def __len__(self):
        return len(self.raw_annotations)

    def download(self):
        if os.path.exists(os.path.join(self.dataset_folder, "metadata")):
            return

        # Download files
        for url, md5 in self.resources:
            down_root = os.path.dirname(self.dataset_folder)
            download_and_extract_archive(
                url,
                download_root=down_root,
                filename=self.base_folder + ".tar.gz",
                md5=md5,
                remove_finished=True,
            )

    def __getitem__(self, index):
        file_name = self.raw_annotations["slice_file_name"].iloc[index]
        file_path = os.path.join(
            os.path.join(
                self.path_to_audio_folder,
                "fold" + str(self.raw_annotations["fold"].iloc[index]),
            ),
            file_name,
        )

        wav, sr = torchaudio.load(file_path)
        resample = T.Resample(sr, self.sample_rate)

        tmp = resample(wav)

        tmp = random_crop(tmp, self.sample_rate)

        zeros = (duration * self.sample_rate) - tmp.shape[1]
        tmp = F.pad(tmp, (0, zeros))
        tmp = tmp.sum(0, keepdims=True)  # make sure all clips have 1 channel

        idx = torch.tensor(self.raw_annotations["classID"].iloc[index])

        uttid = file_name.split(".")[0]

        return (
            tmp,
            torch.tensor(self.raw_annotations["classID"].iloc[index]),
        )


class ESC50(Dataset):
    base_folder = "ESC-50-master"
    url = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
    filename = "ESC-50-master.zip"
    audio_dir = "audio"
    label_col = "category"
    file_col = "filename"
    meta = {
        "filename": os.path.join("meta", "esc50.csv"),
    }

    def __init__(
        self,
        root,
        sample_rate: int = 44100,
        reading_transformations: Optional[nn.Module] = None,
        download: bool = True,
        train: bool = True,
        cut_off: int = 500,
    ):
        super().__init__()
        self.root = os.path.expanduser(root)
        if download:
            self.download()

        self._load_meta()

        self.targets, self.audio_paths = [], []
        self.pre_transformations = reading_transformations
        print("Loading audio files")

        self.df["category"] = self.df["category"].str.replace("_", " ")

        for _, row in tqdm(self.df.iterrows()):
            file_path = os.path.join(
                self.root, self.base_folder, self.audio_dir, row[self.file_col]
            )
            self.targets.append(row[self.label_col])
            self.audio_paths.append(file_path)

        self.sample_rate = sample_rate

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])

        self.df = pd.read_csv(path)
        self.class_to_idx = {}
        self.classes = [
            x.replace("_", " ") for x in sorted(self.df[self.label_col].unique())
        ]
        for i, category in enumerate(self.classes):

            self.class_to_idx[category] = i

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        file_path, target = self.audio_paths[index], self.targets[index]
        idx = torch.tensor(self.class_to_idx[target])
        tmp, sr = torchaudio.load(file_path)
        resample = T.Resample(sr, self.sample_rate)
        tmp = resample(tmp)

        tmp = random_crop(tmp, self.sample_rate)

        zeros = (duration * self.sample_rate) - tmp.shape[1]
        tmp = F.pad(tmp, (0, zeros))
        tmp = tmp.sum(0, keepdims=True)  # make sure all clips have 1 channel

        utt_id = self.audio_paths[index].split("/")[-1].split(".")[0]

        return tmp, idx

    def __len__(self):
        return len(self.audio_paths)

    def download(self):
        download_url(self.url, self.root, self.filename)

        # extract file
        from zipfile import ZipFile

        with ZipFile(os.path.join(self.root, self.filename), "r") as zip:
            zip.extractall(path=self.root)


def prepare_esc50_loaders(hparams):

    train_data = ESC50(
        hparams.esc50_folder, sample_rate=hparams.sample_rate, train=True
    )
    valid_data = ESC50(
        hparams.esc50_folder, sample_rate=hparams.sample_rate, train=False
    )

    return {
        "train": DataLoader(
            train_data,
            batch_size=hparams.batch_size,
            pin_memory=True,
            persistent_workers=True,
            num_workers=4,
        ),
        "valid": DataLoader(
            valid_data,
            batch_size=hparams.batch_size,
            pin_memory=True,
            persistent_workers=True,
            num_workers=4,
        ),
    }
