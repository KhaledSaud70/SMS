import logging
from dataclasses import dataclass
from multiprocessing import Value
import pandas as pd
import torch
import torchvision.datasets as datasets
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from open_clip import tokenize


class CsvDataset(Dataset):
    def __init__(self,
                 input_filename,
                 transforms,
                 img_key,
                 caption_key,
                 sep="\t",
                 label_key=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()

        num_columns = len(df.columns) - 2

        self.captions_list = []
        for k in range(1, num_columns):
            self.captions_list.append(df[f"{caption_key}_{k}"])

        self.return_label = False
        if label_key is not None:
            self.return_label = True
            self.labels = list(map(int, df[label_key].tolist()))
        self.transforms = transforms
        logging.debug('Done loading data.')

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = tokenize([str(self.captions[idx])])[0]

        if len(self.captions_list) > 0:
            texts_list = [
                tokenize([str(self.captions_list[i][idx])])[0]
                for i in range(len(self.captions_list))
            ]
            texts_list.append(texts)
            texts_list = torch.stack(texts_list, dim=0)
            perm = torch.randperm(texts_list.shape[0])

            texts_list = texts_list[perm, :]

        if self.return_label:
            label = self.labels[idx]
            if len(self.captions_list) > 0:
                return images, texts, texts_list, label
            else:
                return images, texts, label

        if len(self.captions_list) > 0:
            return images, texts, texts_list

        return images, texts


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value
    

@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler,
                                                   DistributedSampler):
            self.sampler.set_epoch(epoch)


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0):
    input_filename = args.ft_data if is_train else args.val_data
    assert input_filename

    if args.get_labeled_csv:
        label_key = args.supervised_label_key

    else:
        label_key = None

    dataset = CsvDataset(input_filename,
                         preprocess_fn,
                         img_key=args.csv_img_key,
                         caption_key=args.csv_caption_key,
                         sep=args.csv_separator,
                         label_key=label_key)
    
    num_samples = len(dataset)

    # sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    sampler = None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
    )

    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)



def get_data(args, preprocess_fns, epoch=0):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.dataset_type == 'auto':
        data["train_ft"] = get_csv_dataset(args, 
                                           preprocess_train, 
                                           is_train=True,
                                           epoch=epoch)
    else:
        raise ValueError(f"Unsupported dataset type: {args.dataset_type}")

    return data