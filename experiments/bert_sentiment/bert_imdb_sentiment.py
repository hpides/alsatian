import os
import statistics
import time

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification


# Load IMDb dataset
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


if __name__ == '__main__':
    # Check if GPU is available and set device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("init model ...")
    # Load the pre-trained BERT model for sequence classification
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          num_labels=2)  # 2 for binary classification

    # summary(model, input_size=(2, 128), dtypes=[torch.IntTensor], device=device)

    # sd = model.state_dict()
    # torch.save(sd, './sd.pt')

    # Replace 'path_to_imdb' with the actual path to your IMDb dataset
    imdb_path = '/Users/nils/Desktop/aclImdb-dummy/train'
    texts, labels = load_imdb_data(imdb_path)

    print(f'Number of texts: {len(texts)}')
    print(f'Number of labels: {len(labels)}')

    # Tokenize the text data using the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize and encode the dataset
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=256, return_tensors='pt')
    labels = torch.tensor(labels)

    # Create DataLoader object
    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], labels)

    num_batches = 20
    # for batch_size in [32, 64, 128, 256]: # from 512 on out of memory
    for batch_size in [32, 64, 128, 256]:  # from 512 on out of memory
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        inference_times = []
        inference_cuda_times = []
        print(f'num_batches: {num_batches}')
        print(f'batch_size: {batch_size}')
        # Inference
        model.eval()
        model.to(device)

        with torch.no_grad():
            end_to_end_start = time.time()
            batch_count = 0
            start_time = time.time()
            for batch in tqdm(loader, desc='Inference'):
                input_ids, attention_mask, labels = batch
                print("input_ids: ", input_ids.shape)
                print("attention_mask: ", attention_mask.shape)
                print("labels: ", labels.shape)
                # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                # starter.record()
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                start_time = time.time()
                outputs = model(input_ids, attention_mask=attention_mask)
                # ender.record()
                # torch.cuda.synchronize()  # WAIT FOR GPU SYNC
                # elapsed = starter.elapsed_time(ender)
                end_time = time.time()

                batch_count += 1
                inference_times.append(end_time - start_time)
                # inference_cuda_times.append(elapsed * 10**-3)

                if batch_count == num_batches:
                    break

            end_to_end_end = time.time()
            print(f'end to end: {end_to_end_end - end_to_end_start}')

            input_ids, attention_mask, labels = batch
            print("\nData Type:")
            print(input_ids.dtype)
            print(attention_mask.dtype)
            print("\nShape:")
            print(input_ids.shape)
            print(attention_mask.shape)

            print(f'{batch_size}: {inference_times}')
            print(f'{batch_size}: {inference_cuda_times}')
            print(f'{batch_size}: {statistics.median(inference_times)}')
