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

clip_normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])


def run_args():
    prompt = "A portrait of a man"
    guidance_scale = 200
    num_iterations = 1500
    num_overview_cuts = [12] * 1000
    num_inner_cuts = [4] * 1000
    cut_ic_pow = [1.0] * 1000
    cut_ic_gray = [0.1] * 1000
    cut_ov_gray = [0.1] * 1000
    reference_image = None
    reference_image_scale = 0.0
    tv_scale = 130
    range_scale = 300
    range_high = 0.99
    range_low = -0.99
    squared_range_scale = 6300
    squared_range_high = 0.91
    squared_range_low = -0.91
    height = 512
    width = 512
    clip_model = "openai/clip-vit-large-patch14"
    seed = 0
    set_grads_to_none = True
    do_cutouts = False

    return locals()


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


def run(vae, clip_model, clip_preprocessor, args):
    clip_model = clip_model.cuda()
    clip_model.eval()
    clip_model.requires_grad = False

    vae = vae.cuda()
    vae.eval()
    vae.requires_grad = False

    input_ids = clip_preprocessor(args.prompt, return_tensors="pt", padding="max_length",
                                  truncation=True).input_ids.cuda()
    text_embed = clip_model.get_text_features(input_ids).float().detach()  # .unsqueeze(0)

    torch.manual_seed(args.seed)
    init_latent = torch.randn(1, 4, args.height // 8, args.width // 8).cuda()
    init_latent.requires_grad = True

    optimizer = torch.optim.Adam([init_latent], lr=1)

    all_losses = []

    progress_bar = tqdm(range(args.num_iterations))

    for i in range(args.num_iterations):
        optimizer.zero_grad(set_to_none=args.set_grads_to_none)

        image = vae.decode(init_latent / vae.config.scaling_factor, return_dict=False)[0]
        # get losses
        loss = tv_loss(image) * args.tv_scale
        tv_loss_item = (tv_loss(image) * args.tv_scale).detach().item()

        loss += range_loss(image,high_range=args.range_high,low_range=args.range_low) * args.range_scale
        range_loss_item = (range_loss(image,high_range=args.range_high,low_range=args.range_low) * args.range_scale).detach().item()

        loss += squared_range_loss(image,high_range=args.squared_range_high,low_range=args.squared_range_low) * args.squared_range_scale
        squared_range_loss_item = (squared_range_loss(image,high_range=args.squared_range_high,low_range=args.squared_range_low) * args.squared_range_scale).detach().item()

        image = (image / 2 + 0.5)  # .clamp(0, 1)
        image = clip_normalize(image)
        if args.do_cutouts:
            image = make_cutouts(image, args.num_overview_cuts[i], args.num_inner_cuts[i], args.cut_ic_pow[i],
                                 args.cut_ic_gray[i], args.cut_ov_gray[i])
        else:
            image = torch.nn.functional.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        image_embedding = clip_model.get_image_features(image)

        loss += spherical_dist_loss(image_embedding, text_embed) * args.guidance_scale
        clip_loss_item = (spherical_dist_loss(image_embedding, text_embed) * args.guidance_scale).detach().item()

        # loss += cosine_loss(image_embedding, text_embed) * args.guidance_scale
        # clip_loss_item = (cosine_loss(image_embedding, text_embed) * args.guidance_scale).detach().item()

        all_losses.append(loss.item())

        loss.backward()

        # with torch.no_grad():
        #     gradient = init_latent.grad.cpu().numpy()
        #     gradient = gradient.mean(1).transpose(1,2,0)
        #     plt.imshow(gradient)
        #     plt.savefig(f"gradients/gradient{i}.png")

        optimizer.step()

        logs = {
                "clip_loss": str(clip_loss_item) + "  " + str(clip_loss_item / args.guidance_scale)[:5],
                "tv_loss": str(tv_loss_item) + "  " + str(tv_loss_item / args.tv_scale)[:5],
                "range_loss": str(range_loss_item) + "  " + str(range_loss_item / args.range_scale)[:5],
                "squared_range_loss": str(squared_range_loss_item) + "  " + str(squared_range_loss_item / args.squared_range_scale)[:5]
                }
        progress_bar.set_postfix(**logs)
        progress_bar.update(1)

        if i % 20 == 0:
            with torch.no_grad():
                img = vae.decode(init_latent / vae.config.scaling_factor, return_dict=False)[0]
                img = (img / 2 + 0.5).clamp(0, 1)
                img = img.cpu().permute(0, 2, 3, 1).float().numpy()
                img = Image.fromarray((img[0] * 255).astype(np.uint8))
                img.show()

    with torch.no_grad():
        img = vae.decode(init_latent / vae.config.scaling_factor, return_dict=False)[0]
        img = (img / 2 + 0.5).clamp(0, 1)
        img = img.cpu().permute(0, 2, 3, 1).float().numpy()
        img = Image.fromarray((img[0] * 255).astype(np.uint8))
        plt.plot(all_losses)

        return img