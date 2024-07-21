import os
import random

import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
from transformers import BertTokenizer

TEST_LABELS_PT = "test_labels.pt"

TRAIN_LABELS_PT = "train_labels.pt"

TEST_ENCODINGS_PT = "test_encodings.pt"

TRAIN_ENCODINGS_PT = "train_encodings.pt"


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


def get_reduced_imbdb_bert_base_uncased_datasets(root_data_path, train_samples, test_samples):
    base_path = os.path.join(root_data_path, f"bert-base-uncased-{train_samples}-{test_samples}")

    if not os.path.isdir(base_path):
        train_texts, train_labels = load_imdb_data(os.path.join(root_data_path, "train"))
        test_texts, test_labels = load_imdb_data(os.path.join(root_data_path, "test"))

        num_train_samples = len(train_texts)
        num_test_samples = len(test_texts)

        assert num_train_samples >= train_samples
        assert num_test_samples >= test_samples

        train_indices = random.sample(range(num_train_samples), train_samples)
        test_indices = random.sample(range(num_test_samples), test_samples)

        reduced_texts_train, reduced_labels_train = _reduced_data(train_indices, train_texts, train_labels)
        reduced_texts_test, reduced_labels_test = _reduced_data(test_indices, test_texts, test_labels)

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Tokenize and encode the dataset
        train_encodings, train_labels = _get_encodings_and_labels(tokenizer, reduced_texts_train, reduced_labels_train)
        test_encodings, test_labels = _get_encodings_and_labels(tokenizer, reduced_texts_test, reduced_labels_test)

        # save data for future reuse
        os.makedirs(base_path)
        torch.save(train_encodings, os.path.join(base_path, TRAIN_ENCODINGS_PT))
        torch.save(test_encodings, os.path.join(base_path, TEST_ENCODINGS_PT))
        torch.save(train_labels, os.path.join(base_path, TRAIN_LABELS_PT))
        torch.save(test_labels, os.path.join(base_path, TEST_LABELS_PT))

    else:
        train_encodings = torch.load(os.path.join(base_path, TRAIN_ENCODINGS_PT))
        test_encodings = torch.load(os.path.join(base_path, TEST_ENCODINGS_PT))
        train_labels = torch.load(os.path.join(base_path, TRAIN_LABELS_PT))
        test_labels = torch.load(os.path.join(base_path, TEST_LABELS_PT))

    # Create DataLoader object
    train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)

    test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], test_labels)

    return train_dataset, test_dataset


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
    train_data, test_data = get_reduced_imbdb_bert_base_uncased_datasets("./data/aclImdb/", 10, 5)

    print("test")
