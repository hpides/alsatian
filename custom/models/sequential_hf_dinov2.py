import torch
from torch import nn
from transformers import AutoModel


# Simplified/Sequential version of Huggingface transformers/models/vit/modeling_vit

class DinoEmbeddings(nn.Module):

    def __init__(self, original_model):
        super(DinoEmbeddings, self).__init__()
        self.embeddings = original_model.embeddings

    def forward(self, input):
        embedding_output = self.embeddings(input, bool_masked_pos=None)

        return embedding_output


class DinoEncoderBlock(nn.Module):
    def __init__(self, original_model, index):
        super(DinoEncoderBlock, self).__init__()
        self.encoder = original_model.encoder.layer[index]

    def forward(self, input):
        hidden_states = input
        layer_head_mask = None
        output_attentions = False
        layer_outputs = self.encoder(hidden_states, layer_head_mask, output_attentions)

        hidden_states = layer_outputs[0]

        return hidden_states


class DinoPooler(nn.Module):
    def __init__(self, original_model):
        super(DinoPooler, self).__init__()
        self.layernorm = original_model.layernorm

    def forward(self, input):
        sequence_output = self.layernorm(input)
        pooled_output = sequence_output[:, 0, :]

        return pooled_output


def get_sequential_dinov2_model(model_id, hf_caching_dir=None):
    original_model = AutoModel.from_pretrained(model_id, cache_dir=hf_caching_dir)

    seq_model = torch.nn.Sequential()
    seq_model.append(DinoEmbeddings(original_model))

    for i in range(len(original_model.encoder.layer)):
        seq_model.append(DinoEncoderBlock(original_model, i))

    seq_model.append(DinoPooler(original_model))

    return seq_model


if __name__ == '__main__':
    dummy_image = torch.randn(1, 3, 224, 224)

    hf_model = AutoModel.from_pretrained("facebook/dinov2-base")
    hf_model_sd = hf_model.state_dict()
    hf_model.eval()

    seq_hf_model = get_sequential_dinov2_model("facebook/dinov2-base")
    seq_hf_model_sd = seq_hf_model.state_dict()
    seq_hf_model.eval()

    with torch.no_grad():
        outputs1 = hf_model(dummy_image)
        print(outputs1)
        outputs2 = seq_hf_model(dummy_image)
        print(outputs2)

        print(torch.equal(outputs1.pooler_output, outputs2))
