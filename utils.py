import torch
import diffusers
import numpy as np
import torch.nn as nn
import transformers
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

class GradStore:
    def __init__(self):
        self.grads = []

    def hook_get_grad(self, module, grad_input, grad_output):
        self.grads.append(grad_output.detach().cpu().numpy())

    def get_grad(self, grad):
        self.grads.append(grad.detach().cpu().numpy())

    def dump_grads(self):
        self.grads = []

def hook_vae(module, store):
    if hasattr(module, "conv1"):
        module.conv1.register_backward_hook(store.hook_get_grad)
    if hasattr(module, "conv2"):
        module.conv2.register_backward_hook(store.hook_get_grad)
    if hasattr(module, "conv"):
        module.conv.register_backward_hook(store.hook_get_grad)
    if hasattr(module, "conv_act"):
        module.conv_act.register_backward_hook(store.hook_get_grad)
    if hasattr(module, "nonlinearity"):
        module.nonlinearity.register_backward_hook(store.hook_get_grad)
    if hasattr(module, "conv_out"):
        module.conv_out.register_backward_hook(store.hook_get_grad)

    if hasattr(module, "children"):
        for child in module.children():
            hook_vae(child, store)



class GaussianSmoothing(torch.nn.Module):
    """
    Arguments:
    Apply gaussian smoothing on a 1d, 2d or 3d tensor. Filtering is performed seperately for each channel in the input
    using a depthwise convolution.
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel. sigma (float, sequence): Standard deviation of the
        gaussian kernel. dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    # channels=1, kernel_size=kernel_size, sigma=sigma, dim=2
    def __init__(
            self,
            channels: int = 1,
            kernel_size: int = 3,
            sigma: float = 0.5,
            dim: int = 2,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, float):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels
        self.padding = (size - 1) // 2

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError("Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim))

    def forward(self, input):
        """
        Arguments:
        Apply gaussian filter to input.
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight.to(input.dtype), groups=self.groups, padding=self.padding)


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input_tensor):
    """L2 total variation loss, as in Mahendran et al."""
    input_tensor = F.pad(input_tensor, (0, 1, 0, 1), 'replicate')
    x_diff = input_tensor[..., :-1, 1:] - input_tensor[..., :-1, :-1]
    y_diff = input_tensor[..., 1:, :-1] - input_tensor[..., :-1, :-1]
    return (x_diff ** 2 + y_diff ** 2).mean([1, 2, 3])


def range_loss(input_tensor, low_range=-1, high_range=1):
    return torch.abs(input_tensor - input_tensor.clamp(min=low_range, max=high_range)).mean()

# an attempt to deal with small patches of the image that are very very bright/dark
def squared_range_loss(input_tensor, low_range=-1, high_range=1):
    loss = torch.abs(input_tensor - input_tensor.clamp(min=low_range, max=high_range)).pow(2)
    return loss.mean()


def cosine_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return 1 - (x * y).sum(dim=-1)