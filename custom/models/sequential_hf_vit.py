import torch
from torch import nn
from transformers import AutoModel


# Simplified/Sequential version of Huggingface transformers/models/vit/modeling_vit

class ViTEmbeddings(nn.Module):

    def __init__(self, original_model):
        super(ViTEmbeddings, self).__init__()
        self.embeddings = original_model.embeddings

    def forward(self, input):
        embedding_output = self.embeddings(
            input, bool_masked_pos=None, interpolate_pos_encoding=None
        )

        return embedding_output


class ViTEncoderBlock(nn.Module):
    def __init__(self, original_model, index):
        super(ViTEncoderBlock, self).__init__()
        self.encoder = original_model.encoder.layer[index]

    def forward(self, input):
        hidden_states = input
        layer_head_mask = None
        output_attentions = False
        layer_outputs = self.encoder(hidden_states, layer_head_mask, output_attentions)

        hidden_states = layer_outputs[0]

        return hidden_states


class ViTPooler(nn.Module):
    def __init__(self, original_model):
        super(ViTPooler, self).__init__()
        self.layernorm = original_model.layernorm
        self.pooler = original_model.pooler

    def forward(self, input):
        sequence_output = self.layernorm(input)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return pooled_output


def get_sequential_vit_model(model_id="google/vit-base-patch16-224-in21k", hf_caching_dir=None):
    original_model = AutoModel.from_pretrained(model_id, cache_dir=hf_caching_dir)

    seq_model = torch.nn.Sequential()
    seq_model.append(ViTEmbeddings(original_model))

    for i in range(12):
        seq_model.append(ViTEncoderBlock(original_model, i))

    seq_model.append(ViTPooler(original_model))

    return seq_model


if __name__ == '__main__':
    dummy_image = torch.randn(1, 3, 224, 224)

    hf_model = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k")
    hf_model_sd = hf_model.state_dict()
    hf_model.eval()

    seq_hf_model = get_sequential_vit_model()
    seq_hf_model_sd = seq_hf_model.state_dict()
    seq_hf_model.eval()

    with torch.no_grad():
        outputs = hf_model(dummy_image)
        print(outputs)
        outputs = seq_hf_model(dummy_image)
        print(outputs)
