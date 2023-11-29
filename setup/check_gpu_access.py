import torch


def check_pytorch_version():
    print(f"PyTorch version: {torch.__version__}")


def check_cuda_availability():
    if torch.cuda.is_available():
        print(f"Number of available GPUs: {torch.cuda.device_count()}")
    else:
        print("CUDA is not available. No GPUs detected.")


if __name__ == "__main__":
    check_pytorch_version()
    check_cuda_availability()
