import torch
from torch.nn import ModuleList
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
from transformers.models.bert.modeling_bert import BertEncoder


def _in_class_list(child, split_classes):
    for cls in split_classes:
        if isinstance(child, cls):
            return True
def list_of_layers(model: torch.nn.Sequential, include_seq=False, split_classes=None):
    result = []
    children = list(model.children())
    for child in children:
        if split_classes and _in_class_list(child, split_classes):
            result += list_of_layers(child, include_seq, split_classes=split_classes)
        elif isinstance(child, torch.nn.Sequential) or (include_seq and len(list(child.children())) > 0):
            result += list_of_layers(child, include_seq, split_classes=split_classes)
        else:
            result += [child]
    return result

if __name__ == '__main__':
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          num_labels=2)  # 2 for binary classification

    layers = list_of_layers(model, split_classes=[BertModel, BertEncoder, ModuleList])

    merged_model = torch.nn.Sequential(*(list(layers)))

    num_texts = 10
    max_length = 256
    vocab_size = 30522  # BERT's vocabulary size for 'bert-base-uncased'

    # Generate random input IDs
    input_ids = torch.randint(low=0, high=vocab_size, size=(num_texts, max_length))

    # Generate random attention masks (values 0 or 1)
    attention_mask = torch.randint(low=0, high=2, size=(num_texts, max_length))

    bert_out = model(input_ids, attention_mask=attention_mask)
    print(bert_out)

    # merged_out = merged_model(input_ids, attention_mask=attention_mask)
    # print(merged_out)

    print("test")

