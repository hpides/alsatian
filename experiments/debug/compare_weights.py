import torch
from custom.models.init_models import initialize_model

if __name__ == '__main__':
    image_net_model = initialize_model('resnet18', pretrained=True, sequential_model=True)
    image_net_sd = image_net_model.state_dict()

    trained_model_sd = torch.load("/mount-fs/resnet18-ri-1-id-aqLY-epoch-20.pth")

    for key, t1, t2 in zip(list(image_net_sd.keys()), list(image_net_sd.values()), list(trained_model_sd.values())):
        t1 = t1.to("cpu")
        t2 = t2.to("cpu")
        if not("running" in key or "tracked" in key):
            print(key, torch.equal(t1, t2))

    print("test")
