import torch
from torch import nn
from transformers import BertModel, BertTokenizer, BertForSequenceClassification, BertConfig
from torchinfo import summary
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa


# Simplified/Sequential version of Huggingface bert model

def get_extended_attention_mask(attention_mask, input_shape):
    dtype = torch.float32

    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and the dtype's smallest value for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    return extended_attention_mask


class BertEmbeddings(nn.Module):
    def __init__(self, original_model):
        super(BertEmbeddings, self).__init__()
        self.embeddings = original_model.embeddings

    def forward(self, input):
        input_ids, attention_mask = input

        hidden_states = self.embeddings(input_ids=input_ids)

        input_shape = input_ids.size()

        # TODO either of the two implementations below should work, seems like there are form different versions of transformer library
        # -> TODO check on GPU
        batch_size, seq_length = input_shape
        # version 1
        # attention_mask = _prepare_4d_attention_mask_for_sdpa(
        #     attention_mask, hidden_states.dtype, tgt_len=seq_length
        # )
        # version 2
        attention_mask = get_extended_attention_mask(attention_mask, input_shape)

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


def get_sequential_bert_model(pretrained=True):
    model_name = 'bert-base-uncased'
    if pretrained:
        original_model = BertModel.from_pretrained(model_name)
    else:
        config = BertConfig()
        original_model = BertModel(config)


    seq_model = torch.nn.Sequential()
    seq_model.append(BertEmbeddings(original_model))

    for i in range(12):
        seq_model.append(BertEncoderBlock(original_model, i))

    seq_model.append(BertPooler(original_model))

    return seq_model


if __name__ == '__main__':
    model_name = 'bert-base-uncased'
    original_model = BertModel.from_pretrained(model_name)
    seq_bert_model_pretrained = get_sequential_bert_model(pretrained=True)
    seq_bert_model_pretrained_sd = seq_bert_model_pretrained.state_dict()
    seq_bert_model_random = get_sequential_bert_model(pretrained=False)
    seq_bert_model_random_sd = seq_bert_model_random.state_dict()

    for key, t1, t2 in zip(list(seq_bert_model_pretrained_sd.keys()), list(seq_bert_model_pretrained_sd.values()), list(seq_bert_model_random_sd.values())):
        t1 = t1.to("cpu")
        t2 = t2.to("cpu")
        if not ("running" in key or "tracked" in key):
            print(key, torch.equal(t1, t2))


    sd = seq_bert_model_pretrained.state_dict()
    torch.save(sd, "./test-sd.pt")

    # text = "Hello, how are you?"
    # tokenizer = BertTokenizer.from_pretrained(model_name)
    # inputs = tokenizer(text, return_tensors='pt')
    # input_ids = inputs['input_ids']
    # attention_mask = inputs['attention_mask'].bool()

    texts = ["Hello, how are you?", "This is the second sentence.", "Here is another one."]  # Example batch of texts
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Tokenize the batch of texts
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask'].bool()

    # summary(original_model, input_data=input_ids)

    # original_model.eval()
    # seq_bert_model.eval()
    # with torch.no_grad():
    #     original_output = original_model(input_ids, attention_mask=attention_mask)
    #     seq_model_output = seq_bert_model((input_ids, attention_mask))

    # new_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    #
    # print(original_output.pooler_output)
    # print(seq_model_output[1])

    # input_ids = input_ids.to("cuda")
    # attention_mask = attention_mask.to("cuda")
    # seq_bert_model.to("cuda")
    # original_model.to("cuda")

    seq_bert_model_pretrained.eval()
    with torch.no_grad():
        seq_model_output = seq_bert_model_pretrained((input_ids, attention_mask))
        # original_output = original_model(input_ids, attention_mask=attention_mask)
