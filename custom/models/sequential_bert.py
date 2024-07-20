import torch
from torch import nn
from transformers import BertModel, BertTokenizer, BertForSequenceClassification


class BertEmbeddings(nn.Module):
    def __init__(self, original_model):
        super(BertEmbeddings, self).__init__()
        self.embeddings = original_model.embeddings

    def forward(self, input):
        input_ids, attention_mask = input
        hidden_states = self.embeddings(input_ids=input_ids)

        return hidden_states, attention_mask

class BertEncoderBlock(nn.Module):
    def __init__(self, original_model, index):
        super(BertEncoderBlock, self).__init__()
        self.encoder = original_model.encoder.layer[index]

    def forward(self, input):
        hidden_states, attention_mask = input

        layer_outputs = self.encoder(hidden_states, attention_mask=attention_mask)
        hidden_states = layer_outputs[0]

        return hidden_states, attention_mask

class BertPooler(nn.Module):
    def __init__(self, original_model):
        super(BertPooler, self).__init__()
        self.pooler = original_model.pooler

    def forward(self, input):
        hidden_states, attention_mask = input

        pooled_output = self.pooler(hidden_states)
        return hidden_states, pooled_output


def get_sequential_bert_model():
    model_name = 'bert-base-uncased'
    original_model = BertModel.from_pretrained(model_name)

    seq_model = torch.nn.Sequential()
    seq_model.append(BertEmbeddings(original_model))

    for i in range(12):
        seq_model.append(BertEncoderBlock(original_model, i))

    seq_model.append(BertPooler(original_model))

    return seq_model

if __name__ == '__main__':
    model_name = 'bert-base-uncased'
    original_model = BertModel.from_pretrained(model_name)
    seq_bert_model = get_sequential_bert_model()

    text = "Hello, how are you?"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    original_model.eval()
    seq_bert_model.eval()
    with torch.no_grad():
        original_output = original_model(input_ids, attention_mask=attention_mask)
        seq_model_output = seq_bert_model((input_ids, attention_mask))

    new_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    print(original_output.pooler_output)
    print(seq_model_output[1])

