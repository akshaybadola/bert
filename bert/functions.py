import torch


def gelu(x):
    return torch.nn.functional.gelu(x)


def swish(x):
    return x * torch.sigmoid(x)


# torch.nn.functional.gelu(x) # Breaks ONNX export
ACT2FN = {"gelu": gelu, "tanh": torch.tanh,  "relu": torch.nn.functional.relu, "swish": swish}
