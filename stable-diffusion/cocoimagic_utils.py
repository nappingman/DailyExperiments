import os
import sys
import cv2
import torch
import pickle
import argparse
import numpy as np
from PIL import Image
from typing import List
from torch import autocast
from einops import rearrange
import torch.nn.functional as F
from itertools import islice
from torchvision.utils import save_image
from torch.nn.functional import normalize  
from ldm.util import instantiate_from_config
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch  
import torch.nn as nn  
  
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import clip


# sys.path.append("../../cocosnet4imagic/CoCosNet/")
sys.path.append("../../cocov2/CoCosNet-v2/")
# from util.util import masktorgb
# from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel

def gpu_info() -> str:
    info = ''
    for id in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(id)
        info += f'CUDA:{id} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n'
    return info[:-1]


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    return x_image, False

def generate_by_prompt(wm_encoder, c, samplER, model, output_path, start_code, h=256, w=256, ddim_eta=0.0, n_samples=1, scale=7.5,
                       ddim_steps=50):
    batch_size=1
    sample_path = os.path.join(output_path, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(output_path)) - 1
    all_samples = list()
    precision_scope = autocast if True else nullcontext

    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                uc = None
                if scale != 1.0:
                    uc = model.get_learned_conditioning(batch_size * [""])
#                 c = model.get_learned_conditioning([prompt])
                shape = [4, h // 8, w // 8]
                samples_ddim, _ = samplER.sample(S=ddim_steps,
                                                 conditioning=c,
                                                 batch_size=n_samples,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=uc,
                                                 eta=ddim_eta,
                                                 x_T=start_code)
                
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                ret = x_samples_ddim
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
                x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
                # print(f"{x_checked_image_torch.shape}")
                if True:
                    for x_sample in x_checked_image_torch:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8))
                        # img = put_watermark(img, wm_encoder)
                        img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                        base_count += 1
                if False:
                    all_samples.append(x_checked_image_torch)

                if False:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    img = Image.fromarray(grid.astype(np.uint8))
                    img = put_watermark(img, wm_encoder)
                    img.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                    grid_count += 1

                return ret


# def generate_by_prompt_diff(wm_encoder, c, samplER, model, output_path, start_code, h=256, w=256, ddim_eta=0.0, n_samples=1, scale=7.5,
#                        ddim_steps=50):
#     batch_size=1

#     precision_scope = autocast if True else nullcontext

#     with precision_scope("cuda"):
#         with model.ema_scope():
#             uc = None
#             if scale != 1.0:
#                 uc = model.get_learned_conditioning(batch_size * [""])
#             shape = [4, h // 8, w // 8]
#             samples_ddim, _ = samplER.sample(S=ddim_steps,
#                                                 conditioning=c,
#                                                 batch_size=n_samples,
#                                                 shape=shape,
#                                                 verbose=False,
#                                                 unconditional_guidance_scale=scale,
#                                                 unconditional_conditioning=uc,
#                                                 eta=ddim_eta,
#                                                 x_T=start_code)
            
#             x_samples_ddim = model.decode_first_stage(samples_ddim)

#             return x_samples_ddim

def degration(image):
    # transform the image to gray-scale
    r = image[:, 0, :, :]
    g = image[:, 1, :, :]
    b = image[:, 2, :, :]
    n, _, h, w = image.size()
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    image = gray.view(n, 1, h, w).expand(n, 3, h, w)
    return image

def get_cocosnet():
    with open('/home/v-penxiao/workspace/cocosnet4imagic/CoCosNet/coco_opt.pkl', 'rb') as f:
        opt = pickle.load(f)
    opt['checkpoints_dir'] = "/home/v-penxiao/workspace/cocosnet4imagic/CoCosNet/checkpoints/"
    # opt['checkpoints_dir'] = "/home/v-penxiao/workspace/cocov2/CoCosNet-v2/checkpoints/"
    opt = argparse.Namespace(**opt)
    opt.which_epoch = 'latest'
    print(opt.which_epoch)
    opt.name = 'universal_512_colorjitter'
    # opt.name = 'deepfashionHD'
    model = Pix2PixModel(opt)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

def get_cocosnetv2():
    with open('/home/v-penxiao/workspace/cocov2/CoCosNet-v2/cocov2_opt.pkl', 'rb') as f:
        opt = pickle.load(f)
    opt['checkpoints_dir'] = "/home/v-penxiao/workspace/cocov2/CoCosNet-v2/checkpoints/"
    opt = argparse.Namespace(**opt)
    opt.which_epoch = 'latest'
    print(opt.which_epoch)
    opt.name = 'XXX_deepfashionHD'
    # opt.name = 'deepfashionHD'
    model = Pix2PixModel(opt)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def coordinate(img: np.array,
                color_space: str) -> np.array:
    if color_space == "yuv":
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img = img.transpose(2, 0, 1).astype(np.float32)
        img = (img - 127.5) / 127.5
    elif color_space == "gray":
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=0).astype(np.float32)
        img = (img - 127.5) / 127.5
    else:
        img = img[:, :, ::-1].astype(np.float32)
        img = (img.transpose(2, 0, 1) - 127.5) / 127.5

    return img

def totensor(array_list: List) -> torch.Tensor:
    return torch.cuda.FloatTensor(np.array(array_list).astype(np.float32))

def colorized(model, line, color, flag, output_path):
    warped,warped_line = color, line
    data = {'label': line,
            'image': color,
            'path': "",
            'self_ref': torch.ones_like(color),
            'ref': warped,
            'label_ref': warped_line
            }
    out = model(data, mode='inference')
    imgs = out['fake_image'].data
    
    if flag:
        sample_path = os.path.join(output_path, "samples")
        os.makedirs(sample_path, exist_ok=True)
        base_count = len(os.listdir(sample_path))
        imgs = (imgs + 1.) / 2.
        save_image(imgs[0],os.path.join(sample_path, f"coco_{base_count-1:05}.png"))
        # x_sample = 255. * rearrange(imgs[0].cpu().numpy(), 'c h w -> h w c')
        # img = Image.fromarray(x_sample.astype(np.uint8))
        # # img = put_watermark(img, wm_encoder)
        # img.save(os.path.join(sample_path, f"coco_{base_count-1:05}.png"))
        # base_count += 1
    return imgs

# class ContrastiveLoss(nn.Module):  
#     def __init__(self, margin=1.0):  
#         super(ContrastiveLoss, self).__init__()  
#         self.margin = margin  

#     def forward(self, output1, output2, label):  
#         euclidean_distance = nn.functional.pairwise_distance(output1, output2)  
#         loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +  
#                                         (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))  

#         return loss_contrastive  

class CrossEntropyLossWithTemperature(nn.Module):  
    def __init__(self, temperature=1.0):  
        super(CrossEntropyLossWithTemperature, self).__init__()  
        self.temperature = temperature  
  
    def forward(self, anchor, positive, negatives):  
        dot_product_positive = torch.matmul(anchor, positive.t())  
        dot_product_negatives = torch.matmul(anchor, negatives.t())  
        logits = torch.cat((dot_product_positive, dot_product_negatives), dim=1) / self.temperature  
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)  
        loss = nn.functional.cross_entropy(logits, labels, reduction='none')  
        max_loss = -torch.log(torch.tensor(1.0 / (1 + negatives.shape[0]))).to(loss.device)  
        normalized_loss = loss / max_loss  
        return torch.mean(normalized_loss)  


  
def get_clip_loss(model, image, context_prompt):
    func =  Compose([
        Resize(224,interpolation=BICUBIC),
        # CenterCrop(224),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    image = (image + 1.) / 2.
    image = func(image)

    # src_image = (src_image + 1.) / 2.
    # src_image = func(src_image)

    # i_e = model.encode_image(image)
    # src_i_e = model.encode_image(src_image)

    # neg_prompt = "a grayscale photograph"
    #"brownish average colors for objects", "regional color inconsistency", "color bleeding","a faded grayscale picture","a desaturated picture lacking chromaticity", "a picture with scratches"
    texts = [context_prompt, "a grayscale photograph", "regional color inconsistency", "color bleeding"]  
    text_inputs = clip.tokenize(texts).to(image.device)
    # with torch.no_grad():  
    image_features = model.encode_image(image)
    text_features = model.encode_text(text_inputs) 
    criterion = CrossEntropyLossWithTemperature(temperature=1.0)  
    loss = criterion(image_features, text_features[0:1], text_features[1:])  
    return loss  

    # p_t_e = model.encode_text(clip.tokenize(pos_prompt).to(image.device))
    n_t_e = model.encode_text(clip.tokenize(neg_prompt).to(image.device))
    # p_cos_sim = torch.nn.functional.cosine_similarity(i_e, p_t_e).mean()
    n_cos_sim = torch.nn.functional.cosine_similarity(i_e, n_t_e).mean()
    # i_cos_sim = torch.nn.functional.cosine_similarity(i_e, src_i_e).mean()
    # loss = n_cos_sim - p_cos_sim + n_cos_sim ** 2
    # loss = (n_cos_sim ** 2 + (1 - p_cos_sim)**2) / 2
    # loss = (1 - p_cos_sim)**2
    loss = n_cos_sim ** 2
    # loss = - p_cos_sim ** 2
    return loss


def vgg19_loss(inp, rec, vgg19_conv2_1_relu, vgg19_conv3_1_relu):
    ori_conv2_1_feats = vgg19_conv2_1_relu(inp.contiguous())
    rec_conv2_1_feats = vgg19_conv2_1_relu(rec.contiguous())
    ori_conv3_1_feats = vgg19_conv3_1_relu(inp.contiguous())
    rec_conv3_1_feats = vgg19_conv3_1_relu(rec.contiguous())
    p_loss = 0.5 * F.mse_loss(ori_conv2_1_feats.contiguous(), rec_conv2_1_feats.contiguous()) + 0.5 * F.mse_loss(ori_conv3_1_feats.contiguous(), rec_conv3_1_feats.contiguous())
    return p_loss


class MultiNegativeContrastiveLoss(nn.Module):  
    def __init__(self, margin=1.0):  
        super(MultiNegativeContrastiveLoss, self).__init__()  
        self.margin = margin  
  
    def forward(self, positive_output, negative_outputs):  
        positive_distance = torch.norm(positive_output, dim=1)  
        negative_distances = torch.norm(negative_outputs - positive_output.unsqueeze(1), dim=2)  
  
        positive_loss = torch.mean(torch.pow(positive_distance, 2))  
        negative_loss = torch.mean(torch.clamp(self.margin - negative_distances, min=0.0).pow(2).max(dim=1).values)  
  
        return positive_loss + negative_loss  

def get_multi_clip_loss(model, image, pos_prompt):
    func =  Compose([
        Resize(224,interpolation=BICUBIC),
        # CenterCrop(224),
        # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    image = (image + 1.) / 2.
    image = func(image)
    negative_prompts = ["a grayscale photograph"]
                        #, "a damaged photograph", "a flawed photograph", "a photograph of chaotic colors"] 

    # Encode the image  
    with torch.no_grad():  
        image_features = model.encode_image(image)  
    
    # Encode the positive prompt  
    with torch.no_grad():  
        positive_text_features = model.encode_text(clip.tokenize([pos_prompt]).to(image.device))  
    
    # Encode the negative prompts  
    with torch.no_grad():  
        negative_text_features = model.encode_text(clip.tokenize(negative_prompts).to(image.device))
    
    # Normalize the features  
    image_features = normalize(image_features, dim=-1)  
    positive_text_features = normalize(positive_text_features, dim=-1)  
    negative_text_features = normalize(negative_text_features, dim=-1) 
    # Calculate the contrastive loss  
    loss = MultiNegativeContrastiveLoss(margin=1.0)  
    contrastive_loss = loss(image_features, torch.cat([positive_text_features, negative_text_features]).view(1, -1, 512)) 
    return contrastive_loss


def get_clip_loss_2(model, processor, image):
    func =  Compose([
        Resize(224,interpolation=BICUBIC),
        # CenterCrop(224),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    image = (image + 1.) / 2.
    image = func(image)

    text_inputs = processor(text=["a grayscale photograph"], return_tensors="pt", padding=True).to(model.device)
    image_inputs = {"pixel_values": image.unsqueeze(0)}

    with torch.no_grad():  
        outputs = model(**text_inputs, **image_inputs)  
        text_embeddings = outputs.text_embeds  
        image_embeddings = outputs.image_embeds  

    temperature = 1.0  
    logits = (torch.matmul(text_embeddings, image_embeddings.T) / temperature ).to(model.device)
    labels = torch.arange(logits.shape[0]).long().to(model.device)
    loss = torch.nn.CrossEntropyLoss()(logits, labels)  
    return loss
