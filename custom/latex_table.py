from custom.models.init_models import initialize_model
from global_utils.model_names import *
from global_utils.model_operations import count_parameters, get_model_size

if __name__ == '__main__':
    latex_abb = {
        MOBILE_V2: 'monet',
        RESNET_18: 'res18',
        RESNET_34: 'res34',
        RESNET_50: 'res50',
        RESNET_101: 'res101',
        RESNET_152: 'res152',
        EFF_NET_V2_S: 'eff-s',
        EFF_NET_V2_L: 'eff-l',
        VIT_L_16: 'vit-l-16',
        VIT_L_32: 'vit-l-32',
        VIT_B_16: 'vit-b-16',
        VIT_B_32: 'vit-b-32',
        BERT: 'bert'
    }

    refs = {
        MOBILE_V2: ('Sandler et al.', 'monet'),
        RESNET_18: ('He et al.', 'res-net'),
        RESNET_34: ('He et al.', 'res-net'),
        RESNET_50: ('He et al.', 'res-net'),
        RESNET_101: ('He et al.', 'res-net'),
        RESNET_152: ('He et al.', 'res-net'),
        EFF_NET_V2_S: ('Tan et al.', 'eff-net'),
        EFF_NET_V2_L: ('Tan et al.', 'eff-net'),
        VIT_L_16: ('Dosovitskiy et al.', 'vit'),
        VIT_L_32: ('Dosovitskiy et al.', 'vit'),
        VIT_B_16: ('Dosovitskiy et al.', 'vit'),
        VIT_B_32: ('Dosovitskiy et al.', 'vit'),
        BERT: ('Devlin et al.', 'bert')
    }

    acl_open = '\\acl{'
    close = "}"
    cite_open = '\\cite{'

    for model_name in [MOBILE_V2, RESNET_18, RESNET_34, RESNET_50, RESNET_101, RESNET_152, EFF_NET_V2_S, EFF_NET_V2_L,
                       VIT_B_16, VIT_B_32, VIT_L_16, VIT_L_32, BERT]:
        if model_name in [BERT]:
            model = initialize_model(model_name, features_only=True)
        else:
            model = initialize_model(model_name)
        num_params = count_parameters(model)
        formatted_params = f"{num_params:,}"
        model_size_mb = round(get_model_size(model) / (1000 * 1000), 1)
        formatted_size = f"{model_size_mb:,} MB"
        _str = f'{acl_open}{latex_abb[model_name]}{close} & {formatted_params} & {formatted_size} & {refs[model_name][0]}~{cite_open}{refs[model_name][1]}{close} \\\\'
        print(_str)
