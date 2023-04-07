import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

# from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

import pdb

from audioclip.utils.transforms import ToTensor1D
import glob
import librosa
from audioclip.model import AudioCLIP   

import random

# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
# safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


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

from audioclip.ignite_trainer import _utils
import torchvision as tv
from typing import Type
import json

def get_dataloader(config_path, config_path_1):
#     config_path = './audioclip/protocols/audioclip-esc50-mtn.json'
    config = json.load(open(config_path))
    config_1 = json.load(open(config_path_1))

    transforms = config['Transforms'] 

    transforms_train = list()
    transforms_test = list()

    for idx, transform in enumerate(transforms):
        use_train = transform.get('train', True)
        use_test = transform.get('test', True)

        transform = _utils.load_class(transform['class'])(**transform['args'])

        if use_train:
            transforms_train.append(transform)
        if use_test:
            transforms_test.append(transform)

        transforms[idx]['train'] = use_train
        transforms[idx]['test'] = use_test

    transforms_train = tv.transforms.Compose(transforms_train)
    transforms_test = tv.transforms.Compose(transforms_test)

    dataset_class = config['Dataset']['class']
    dataset_args = config['Dataset']['args']
    Dataset: Type = _utils.load_class(dataset_class)

    dataset_class_1 = config_1['Dataset']['class']
    dataset_args_1 = config_1['Dataset']['args']
    Dataset_1: Type = _utils.load_class(dataset_class_1)

    batch_train = 1
    batch_test = 1

    workers_train = 1
    workers_test = 1

    train_loader, eval_loader = _utils.get_data_loaders(
        Dataset,
        dataset_args,
        batch_train,
        batch_test,
        workers_train,
        workers_test,
        transforms_train,
        transforms_test
    )

    train_loader_1, eval_loader_1 = _utils.get_data_loaders(
        Dataset_1,
        dataset_args_1,
        batch_train,
        batch_test,
        workers_train,
        workers_test,
        transforms_train,
        transforms_test
    )

    return train_loader, eval_loader, train_loader_1, eval_loader_1

def get_dataloader_one(config_path):
#     config_path = './audioclip/protocols/audioclip-esc50-mtn.json'
    config = json.load(open(config_path))

    transforms = config['Transforms'] 

    transforms_train = list()
    transforms_test = list()

    for idx, transform in enumerate(transforms):
        use_train = transform.get('train', True)
        use_test = transform.get('test', True)

        transform = _utils.load_class(transform['class'])(**transform['args'])

        if use_train:
            transforms_train.append(transform)
        if use_test:
            transforms_test.append(transform)

        transforms[idx]['train'] = use_train
        transforms[idx]['test'] = use_test

    transforms_train = tv.transforms.Compose(transforms_train)
    transforms_test = tv.transforms.Compose(transforms_test)

    dataset_class = config['Dataset']['class']
    dataset_args = config['Dataset']['args']
    Dataset: Type = _utils.load_class(dataset_class)

    batch_train = 1
    batch_test = 1

    workers_train = 1
    workers_test = 1

    train_loader, eval_loader = _utils.get_data_loaders(
        Dataset,
        dataset_args,
        batch_train,
        batch_test,
        workers_train,
        workers_test,
        transforms_train,
        transforms_test
    )


    return train_loader, eval_loader

def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


# def check_safety(x_image):
#     safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
#     x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
#     assert x_checked_image.shape[0] == len(has_nsfw_concept)
#     for i in range(len(has_nsfw_concept)):
#         if has_nsfw_concept[i]:
#             x_checked_image[i] = load_replacement(x_checked_image[i])
#     return x_checked_image, has_nsfw_concept


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--audioclip_ckpt",
        type=str,
        default="models/audioclip/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="full"
    )
    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    
    opt.plms = False
    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "StableDiffusionV1"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    
   
#     if not opt.from_file:
#         prompt = opt.prompt
#         assert prompt is not None
#         data = [batch_size * [prompt]]

#     else:
#         print(f"reading prompts from {opt.from_file}")
#         with open(opt.from_file, "r") as f:
#             data = f.read().splitlines()
#             data = list(chunk(data, batch_size))
    audioclip_path = opt.audioclip_ckpt #'AudioCLIP-Full-Training.pt'
    aclp = AudioCLIP(pretrained=audioclip_path)#AudioCLIP(pretrained=f'audioclip/assets/{MODEL_FILENAME}')
    
#     MODEL_FILENAME = 'AudioCLIP-Full-Training.pt'
#     aclp = AudioCLIP(pretrained=f'audioclip/assets/{MODEL_FILENAME}')
#     paths_to_audio = glob.glob('./audioclip/demo/audio_s2/*.wav')
    
    # ['./audioclip/demo/audio/car_horn_1-24074-A-43.wav', './audioclip/demo/audio/thunder_3-144891-B-19.wav', './audioclip/demo/audio/coughing_1-58792-A-24.wav', './audioclip/demo/audio/cat_3-95694-A-5.wav', './audioclip/demo/audio/alarm_clock_3-120526-B-37.wav']
    
    # ['./audioclip/demo/audio_s1/5-257349-A-15.wav', './audioclip/demo/audio_s1/5-195557-A-19.wav', './audioclip/demo/audio_s1/2-122820-B-36.wav', './audioclip/demo/audio_s1/1-115920-A-22.wav', './audioclip/demo/audio_s1/1-172649-C-40.wav']
    
    batch_size = 1
#     batch_size = len(paths_to_audio) #opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    
#     SAMPLE_RATE = 44100
#     audio = list()
#     for path_to_audio in paths_to_audio:
#         track, _ = librosa.load(path_to_audio, sr=SAMPLE_RATE, dtype=np.float32)

#         # compute spectrograms using trained audio-head (fbsp-layer of ESResNeXt)
#         # thus, the actual time-frequency representation will be visualized
#         spec = aclp.audio.spectrogram(torch.from_numpy(track.reshape(1, 1, -1)))
#         spec = np.ascontiguousarray(spec.detach().numpy()).view(np.complex64)
#         pow_spec = 10 * np.log10(np.abs(spec) ** 2 + 1e-18).squeeze()

#         audio.append((track, pow_spec))
 
#     audio_transforms = ToTensor1D()
#     audio = torch.stack([audio_transforms(track.reshape(1, -1)) for track, _ in audio])
#     data = [audio]
    
#     word_list_unique = ['air conditioner', 'car horn', 'children playing', 'dog bark', 'drilling', 'engine idling', 'gun shot', 'jackhammer', 'siren', 'street music']
    
    word_list_unique = {
         0: 'air conditioner', 
         1: 'car horn', 
         2: 'children playing', 
         3: 'dog bark', 
         4: 'drilling', 
         5: 'engine idling', 
         6: 'gun shot', 
         7: 'jackhammer', 
         8: 'siren', 
         9: 'street music'}
    
    word_list_unique = ['air conditioner', 'car horn', 'children playing', 'dog bark', 'drilling', 'engine idling', 'gun shot', 'jackhammer', 'siren', 'street music']
    
    word_text_list = ['in cartoon style for comics', 'in paining style by Vincent van Gogh', 'in paining style by Picasso', 'in Chinese painting style']
    
    config_path_1 = './audioclip/protocols/audioclip-us8k.json'
    _, test_loader_us8k = get_dataloader_one(config_path_1)
    
    data = test_loader_us8k
    
    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
#                         if isinstance(prompts, tuple):
#                             prompts = list(prompts)
#                         c = model.get_learned_conditioning(prompts)
                        
                        c = model.get_learned_conditioning(prompts[0][:,:,:]) # prompts.to(device)
                        word_txt = random.choice(word_text_list)
                        c_txt = model.get_learned_conditioning(word_txt)
                        
#                         c_in = (c + c_txt) #/ 2
#                         pdb.set_trace()
                        for i_signal in range(3, 8):
                            c_salient = torch.cat((c[:,:i_signal, :], c_txt[:, :i_signal,:]), dim=1)
                            c_back = (c[:,i_signal*2:, :] + c_txt[:, i_signal*2:,:])/2
                            c_in = torch.cat((c_salient, c_back), dim=1)

                            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                            samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                             conditioning=c_in,
                                                             batch_size=batch_size, #opt.n_samples,
                                                             shape=shape,
                                                             verbose=False,
                                                             unconditional_guidance_scale=opt.scale,
                                                             unconditional_conditioning=uc,
                                                             eta=opt.ddim_eta,
                                                             x_T=start_code)

                            x_samples_ddim = model.decode_first_stage(samples_ddim)

                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

    #                         x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
                            x_checked_image = x_samples_ddim
                            x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

    #                         x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
    #                         pdb.set_trace()
                            if not opt.skip_save:
                                idx = 0
                                for x_sample in x_checked_image_torch:
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    img = Image.fromarray(x_sample.astype(np.uint8))
                                    img = put_watermark(img, wm_encoder)
                                    sound_name = prompts[2][idx][:][0].replace(' ', '-') 
#                                     img.save(os.path.join(sample_path, f"{base_count:05}-" + sound_name + '-' + word_txt + ".png"))
                                    word_txt_save = word_txt.replace(' ', '-')
                                    sound_name_save = sound_name.replace(' ', '-')
                                    sound_idx = prompts[2][idx][:][1].split('.')[0]
                                    img.save(os.path.join(sample_path, sound_idx +'-' + sound_name_save+ '-' + word_txt_save +str(i_signal) +  '--' + str(n) + ".png"))
                                    base_count += 1
                                    idx += 1
                            if base_count > 1000:
                                break
                        if not opt.skip_grid:
                            all_samples.append(x_checked_image_torch)

                if not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    img = Image.fromarray(grid.astype(np.uint8))
                    img = put_watermark(img, wm_encoder)
                    img.save(os.path.join(outpath, f'grid-{grid_count:04}' + '.png'))
                    grid_count += 1

                toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
