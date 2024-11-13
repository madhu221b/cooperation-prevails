import torch

USE_MPS = True
USE_CUDA = False

if USE_MPS:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = "cpu"
elif USE_CUDA:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = "cpu"
else:
    device = "cpu"
    