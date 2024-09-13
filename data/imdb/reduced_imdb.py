import os
import random
import shutil
from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import TensorDataset
from tqdm import tqdm
from transformers import BertTokenizer

LABELS_PT = "labels.pt"

ENCODINGS_PT = "train_encodings.pt"


class CustomTensorDataset(TensorDataset):
    """
    This extends the standard PyTorch TensorDataset by adding the option to artificially size the dataset
    """
    tensors: Tuple[Tensor, ...]

    def __init__(self, *tensors: Tensor) -> None:
        super().__init__(*tensors)
        indices = torch.randperm(tensors[0].shape[0])
        shuffled_tensors = []
        for tensor in self.tensors:
            shuffled = tensor[indices]
            shuffled_tensors.append(shuffled)

        self.all_tensors = tuple(shuffled_tensors)

    def set_subrange(self, from_index, to_index):
        new_tensors = []
        for tensor in self.tensors:
            partial_tensor = tensor[from_index:to_index]
            new_tensors.append(partial_tensor)

        self.tensors = tuple(new_tensors)

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)


def load_imdb_data(path):
    texts = []
    labels = []

    for label in ['pos', 'neg']:
        label_path = os.path.join(path, label)
        print(f'Checking directory: {label_path}')
        if os.path.exists(label_path):
            for filename in tqdm(os.listdir(label_path), desc=f'Loading {label} reviews'):
                with open(os.path.join(label_path, filename), 'r', encoding='utf-8') as file:
                    text = file.read()
                    texts.append(text)
                    labels.append(1 if label == 'pos' else 0)
        else:
            print(f'Directory not found: {label_path}')

    return texts, labels


def create_sub_dataset(dataset_path: str, size: int):
    new_dataset_base_path = dataset_path + f'-{size}'
    if not os.path.exists(new_dataset_base_path):
        items_per_class = int(size / 2)
        for label in ['pos', 'neg']:
            files_paths = []
            label_path = os.path.join(dataset_path, label)
            print(f'Checking directory: {label_path}')
            if os.path.exists(label_path):
                for filename in tqdm(os.listdir(label_path), desc=f'Loading {label} reviews'):
                    files_paths.append(os.path.join(label_path, filename))
            sampled_files = random.sample(files_paths, items_per_class)

            dst_path = os.path.join(new_dataset_base_path, label)
            os.makedirs(dst_path)
            for file in sampled_files:
                shutil.copy2(file, dst_path)

    return new_dataset_base_path


def get_imbdb_bert_base_uncased_datasets(data_path):
    base_path = os.path.join(data_path, f"bert-base-uncased-cached")

    if not os.path.isdir(base_path):
        texts, labels = load_imdb_data(data_path)

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Tokenize and encode the dataset
        encodings, labels = _get_encodings_and_labels(tokenizer, texts, labels)

        # save data for future reuse
        os.makedirs(base_path)
        torch.save(encodings, os.path.join(base_path, ENCODINGS_PT))
        torch.save(labels, os.path.join(base_path, LABELS_PT))

    else:
        encodings = torch.load(os.path.join(base_path, ENCODINGS_PT))
        labels = torch.load(os.path.join(base_path, LABELS_PT))

    return CustomTensorDataset(encodings['input_ids'], encodings['attention_mask'], labels)


def _get_encodings_and_labels(tokenizer, texts, labels):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=256, return_tensors='pt')
    labels = torch.tensor(labels)
    return encodings, labels


def _reduced_data(indices, texts, labels):
    reduced_texts, reduced_labels = [], []
    for i in indices:
        reduced_texts.append(texts[i])
        reduced_labels.append(labels[i])
    return reduced_texts, reduced_labels


if __name__ == '__main__':
    create_sub_dataset(
        "/experiments/side_experiments/bert_sentiment/imdb_dummy_data/train", 3)

    print("test")
