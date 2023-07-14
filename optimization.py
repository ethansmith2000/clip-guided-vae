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

from utils import *

clip_normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
grayscale = T.Grayscale(num_output_channels=3)

def run_args():
    prompt = "A portrait of a man"
    guidance_scale = 30
    num_iterations = 1500
    num_overview_cuts = [12] * 1000
    num_inner_cuts = [4] * 1000
    cut_ic_pow = [1.0] * 1000
    cut_ic_gray = [0.1] * 1000
    cut_ov_gray = [0.1] * 1000
    reference_image = None
    reference_image_scale = 0.0
    tv_scale = 100
    tv_latent_scale = 0.05
    range_scale = 0.001
    range_high = 0.99
    range_low = -0.99
    squared_range_scale = 6500
    squared_range_high = 0.91
    squared_range_low = -0.91
    height = 512
    width = 512
    clip_model = "openai/clip-vit-large-patch14"
    seed = 0
    set_grads_to_none = True
    do_cutouts = False
    rot_angle=60
    gray_latent=True
    gray_prob=0.15
    blur_latent_prob = 0.1
    blur_latent_sigma = 0.5
    lr = 1.0
    warmup_steps=20

    return locals()


def run(vae, clip_model, clip_preprocessor, args, grad_store=None):

    clip_losses = []

    transforms = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=(-args.rot_angle,args.rot_angle)),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        T.RandomPerspective(distortion_scale=0.4, p=0.2),
    ])

    smoother = GaussianSmoothing(4, kernel_size=3, sigma=0.7).to("cuda").to(torch.float16)
    warmup_lrs = torch.linspace(0, args.lr, args.warmup_steps).tolist()

    clip_model = clip_model.cuda()
    clip_model.eval()
    clip_model.requires_grad = False

    vae = vae.cuda()
    vae.eval()
    vae.requires_grad = False

    #
    # if grad_store is not None:
    #     hook_vae(vae.decoder, grad_store)

    input_ids = clip_preprocessor(args.prompt, return_tensors="pt", padding="max_length",
                                  truncation=True).input_ids.cuda()
    text_embed = clip_model.get_text_features(input_ids).float().detach()  # .unsqueeze(0)

    torch.manual_seed(args.seed)
    if args.gray_latent:
        init_latent = torch.zeros(1, 4, args.height // 8, args.width // 8).cuda()
    else:
        init_latent = torch.randn(1, 4, args.height // 8, args.width // 8).cuda()
    init_latent.requires_grad = True

    optimizer = torch.optim.Adam([init_latent], lr=1)

    all_losses = []

    progress_bar = tqdm(range(args.num_iterations))

    #with torch.autocast("cuda"):
    for i in range(args.num_iterations):

        if i < args.warmup_steps:
            for g in optimizer.param_groups:
                g['lr'] = warmup_lrs[i]

        optimizer.zero_grad(set_to_none=args.set_grads_to_none)

        # with torch.no_grad():
        #     if torch.randn([]).item() < args.blur_latent_prob:
        #         init_latent = smoother(init_latent)

        image = vae.decode(init_latent / vae.config.scaling_factor, return_dict=False)[0]
        # get losses
        loss = tv_loss(image) * args.tv_scale
        tv_loss_item = (tv_loss(image) * args.tv_scale).detach().item()

        # loss += tv_loss(init_latent) * args.tv_latent_scale
        # tv_latent_loss_item = (tv_loss(init_latent) * args.tv_latent_scale).detach().item()

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

        image = transforms(image)
        if torch.rand([]).item() < args.gray_prob:
            image = grayscale(image)

        image_embedding = clip_model.get_image_features(image)

        loss += spherical_dist_loss(image_embedding, text_embed) * args.guidance_scale
        clip_loss_item = (spherical_dist_loss(image_embedding, text_embed) * args.guidance_scale).detach().item()

        # loss += cosine_loss(image_embedding, text_embed) * args.guidance_scale
        # clip_loss_item = (cosine_loss(image_embedding, text_embed) * args.guidance_scale).detach().item()

        all_losses.append(loss.item())
        clip_losses.append(clip_loss_item)

        loss.backward()

        if grad_store is not None:
            grad_store.get_grad(init_latent.grad)

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
                "squared_range_loss": str(squared_range_loss_item) + "  " + str(squared_range_loss_item / args.squared_range_scale)[:5],
                #"tv_latent_loss": str(tv_latent_loss_item) + "  " + str(tv_latent_loss_item / args.tv_latent_scale)[:5],
                }
        progress_bar.set_postfix(**logs)
        progress_bar.update(1)

        #
        # if grad_store is not None:
        #     grad_store.dump_grads()

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
        plt.savefig("losses.png")
        plt.plot(clip_losses)
        plt.savefig("clip_losses.png")

        return img, grad_store