# code adopted from: https://luv-bansal.medium.com/fine-tuning-bert-for-text-classification-in-pytorch-503d97342db2

import pandas as pd
import torch
import transformers
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class HGNewsDataset(Dataset):
    def __init__(self, tokenizer, max_length):
        super(HGNewsDataset, self).__init__()
        self.train_csv = pd.read_csv(
            'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv',
            header=None, names=['label', 'title', 'description'])
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_classes = len(self.train_csv['label'].unique())  # Number of unique classes in the dataset

    def __len__(self):
        return len(self.train_csv)

    def __getitem__(self, index):
        title = self.train_csv.iloc[index, 1]
        description = self.train_csv.iloc[index, 2]
        text = title + " " + description

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=self.max_length,
            truncation=True
        )
        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]

        label = self.train_csv.iloc[index, 0] - 1  # Adjust labels to start from 0
        one_hot_label = torch.zeros(self.num_classes)
        one_hot_label[label] = 1

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'target': one_hot_label
        }


class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.bert_model = transformers.BertModel.from_pretrained("bert-base-uncased")
        self.out = nn.Linear(768, 4)

    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.bert_model(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)

        out = self.out(o2)

        return out


def finetune(epochs, dataloader, model, loss_fn, optimizer):
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for batch_idx, dl in enumerate(dataloader):
            ids = dl['ids']
            token_type_ids = dl['token_type_ids']
            mask = dl['mask']
            label = dl['target']

            optimizer.zero_grad()

            output = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output, 1)
            _, _label = torch.max(label, 1)

            correct_predictions += (predicted == _label).sum().item()
            total_predictions += label.size(0)
            print(f"running loss: {running_loss}")

        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader)}, Accuracy: {100. * correct_predictions / total_predictions}%")

    return model


if __name__ == '__main__':
    model_id = 'bert-base-uncased'
    tokenizer = transformers.BertTokenizer.from_pretrained(model_id)

    dataset = HGNewsDataset(tokenizer, max_length=100)

    dataloader = DataLoader(dataset=dataset, batch_size=32)

    model = BERT()

    loss_fn = nn.CrossEntropyLoss()

    # Initialize Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    finetune(10, dataloader, model, loss_fn, optimizer)
