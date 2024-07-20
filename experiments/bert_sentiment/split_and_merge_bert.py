import torch.nn
import torch.nn as nn
from transformers import BertModel


class BertStart(nn.Module):
    def __init__(self, original_model):
        super(BertStart, self).__init__()
        self.embeddings = original_model.embeddings
        self.encoder = original_model.encoder.layer[0]

    def forward(self, input):
        input_ids, attention_mask = input
        hidden_states = self.embeddings(input_ids=input_ids)

        layer_outputs = self.encoder(hidden_states, attention_mask=attention_mask)
        hidden_states = layer_outputs[0]

        return hidden_states, attention_mask


class BertMiddle(nn.Module):
    def __init__(self, original_model, index):
        super(BertMiddle, self).__init__()
        self.encoder = original_model.encoder.layer[index]

    def forward(self, input):
        hidden_states, attention_mask = input

        layer_outputs = self.encoder(hidden_states, attention_mask=attention_mask)
        hidden_states = layer_outputs[0]

        return hidden_states, attention_mask


# TODO actually not needed, because we will replace pooler with own FC layer anyways
class BertEnd(nn.Module):
    def __init__(self, original_model):
        super(BertEnd, self).__init__()
        self.encoder = original_model.encoder.layer[-1]
        self.pooler = original_model.pooler

    def forward(self, input):
        hidden_states, attention_mask = input

        layer_outputs = self.encoder(hidden_states, attention_mask=attention_mask)
        hidden_states = layer_outputs[0]

        pooled_output = self.pooler(hidden_states)
        return hidden_states, pooled_output


def get_bert_sequential_model(original_model):
    seq_model = torch.nn.Sequential()
    seq_model.append(BertStart(original_model))
    for i in range(1, 11):
        seq_model.append(BertMiddle(original_model, i))
    seq_model.append(BertEnd(original_model))

    return seq_model


class BertPart(nn.Module):
    def __init__(self, original_model, start_layer, end_layer):
        super(BertPart, self).__init__()
        self.embeddings = original_model.embeddings if start_layer == 0 else None
        self.encoder = nn.ModuleList(original_model.encoder.layer[start_layer:end_layer])
        self.pooler = original_model.pooler if end_layer == len(original_model.encoder.layer) else None

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, hidden_states=None):
        if self.embeddings is not None:
            hidden_states = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)

        for layer_module in self.encoder:
            layer_outputs = layer_module(hidden_states, attention_mask=attention_mask)
            hidden_states = layer_outputs[0]

        if self.pooler is not None:
            pooled_output = self.pooler(hidden_states)
            return hidden_states, pooled_output

        return hidden_states


def split_bert_model(model_name, split_layers):
    """
    Splits a BERT model into multiple parts at the specified layers.

    Parameters:
    model_name (str): The name of the pre-trained BERT model.
    split_layers (list of int): The layers at which to split the model.

    Returns:
    list of BertPart: The parts of the split BERT model.
    """
    # Load pre-trained BERT model
    model = BertModel.from_pretrained(model_name)

    # Ensure the split layers are within the valid range
    num_total_layers = len(model.encoder.layer)
    if any(layer < 0 or layer >= num_total_layers for layer in split_layers):
        raise ValueError(f"Each split_layer must be between 0 and {num_total_layers - 1}")

    split_layers = sorted(set([0] + split_layers + [num_total_layers]))

    parts = []
    for i in range(len(split_layers) - 1):
        part = BertPart(model, split_layers[i], split_layers[i + 1])
        parts.append(part)

    return parts


class BertCombined(nn.Module):
    def __init__(self, parts):
        super(BertCombined, self).__init__()
        self.parts = nn.ModuleList(parts)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        hidden_states = None
        for part in self.parts:
            if part.embeddings is not None:
                hidden_states = part(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            else:
                hidden_states = part(hidden_states=hidden_states, attention_mask=attention_mask)

        return hidden_states


if __name__ == '__main__':
    # # Example usage:
    model_name = 'bert-base-uncased'
    # split_layers = [4, 8, 10]  # Specify the layers at which to split the model
    # parts = split_bert_model(model_name, split_layers)
    #
    # # Join the models back together
    # bert_combined = BertCombined(parts)
    #
    # Example input
    from transformers import BertTokenizer
    #
    text = "Hello, how are you?"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    #
    # # Pass through the combined model
    # with torch.no_grad():
    #     hidden_states, pooled_output = bert_combined(input_ids, attention_mask=attention_mask)
    #
    # print(hidden_states)
    #
    # model = BertModel.from_pretrained(model_name)
    # with torch.no_grad():
    #     out = model(input_ids, attention_mask=attention_mask)
    #     out = out.last_hidden_state
    #
    # print(out)
    # print('test')

    # num_texts = 10
    # max_length = 256
    # vocab_size = 30522  # BERT's vocabulary size for 'bert-base-uncased'
    #
    # # Generate random input IDs
    # input_ids = torch.randint(low=0, high=vocab_size, size=(num_texts, max_length))
    #
    # # Generate random attention masks (values 0 or 1)
    # attention_mask = torch.randint(low=0, high=2, size=(num_texts, max_length))

    # bert_out = bert_combined(input_ids, attention_mask=attention_mask)
    # print(bert_out)

    # merged_out = merged_model(input_ids, attention_mask=attention_mask)
    # print(merged_out)

    model_name = 'bert-base-uncased'

    # model = BertModel.from_pretrained(model_name)

    # parts = []
    # for i in range(20):
    #     part = BertPart(model, i, i + 1)
    #     parts.append(part)
    #
    # print("worked", i)

    original_model = BertModel.from_pretrained(model_name)

    seq_bert = get_bert_sequential_model(original_model)

    prediction = seq_bert((input_ids, attention_mask))

    seq_out = prediction[0]

    out = original_model(input_ids, attention_mask=attention_mask)
    out = out.last_hidden_state

    print("test")
    print(out)
    print(seq_out)

