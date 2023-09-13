from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .guided_diffusion.guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults


class Diffuser:
    def __init__(self, model, diffusion, **kwargs):
        super(Diffuser, self).__init__()

        self._model = model
        self._diffusion = diffusion

        self._image_size = kwargs["image_size"]
        self._real_step = 0
        self._model_kwargs = None
        self._clip_denoised = True

        if kwargs["use_fp16"]:
            model.convert_to_fp16()

    def get_dire(self, img_batch: torch.Tensor) -> torch.Tensor:
        batch_size = img_batch.shape[0]
        reverse_fn = self._diffusion.ddim_reverse_sample_loop

        imgs = preprocess_images(img_batch, self._image_size)

        latent = reverse_fn(
            self._model,
            (batch_size, 3, self._image_size, self._image_size),
            noise=imgs,
            clip_denoised=self._clip_denoised,
            model_kwargs=self._model_kwargs,
            real_step=self._real_step,
            # progress=True,
        )
        sample_fn = self._diffusion.ddim_sample_loop
        recons = sample_fn(
            self._model,
            (batch_size, 3, self._image_size, self._image_size),
            noise=latent,
            clip_denoised=self._clip_denoised,
            model_kwargs=self._model_kwargs,
            real_step=self._real_step,
            # progress=True,
        )

        dire = torch.abs(imgs - recons)

        # Save example images, overwrites each other, meant for debugging
        # from torchvision.utils import save_image
        # dir_out = Path("samples")
        # dir_out.mkdir(exist_ok=True, parents=True)
        # for idx in range(batch_size):
        #     save_image(imgs[idx].cpu(), dir_out.joinpath(f"{idx}_a.png"))
        #     save_image(recons[idx].cpu(), dir_out.joinpath(f"{idx}_b.png"))
        #     save_image(dire[idx].cpu(), dir_out.joinpath(f"{idx}_c.png"))

        return dire

    def to(self, device):
        self._model.to(device)


def setup_diffuser(weights_path: Path):
    settings = model_and_diffusion_defaults()
    hyperparameters = {
        "attention_resolutions": "32,16,8",
        "class_cond": False,
        "diffusion_steps": 1000,
        "dropout": 0.1,
        "image_size": 256,
        "learn_sigma": True,
        "noise_schedule": "linear",
        "num_channels": 256,
        "num_head_channels": 64,
        "num_res_blocks": 2,
        "resblock_updown": True,
        "use_fp16": True,
        "use_scale_shift_norm": True,
        "timestep_respacing": "ddim20",
    }
    settings.update(hyperparameters)
    model, diffusion = create_model_and_diffusion(**settings)
    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return Diffuser(model, diffusion, **settings)


def preprocess_images(img_batch: torch.Tensor, image_size: int) -> torch.Tensor:
    # Replicate the preprocessing of the original code
    imgs = torch.zeros([*img_batch.shape[:2], image_size, image_size],
                       device=img_batch.device)

    for idx, img in enumerate(img_batch):
        img = transforms.ToPILImage()(img)
        img = center_crop_arr(img, image_size)
        img = img.astype(np.float32) / 127.5 - 1
        img = torch.from_numpy(np.transpose(img, [2, 0, 1]))
        imgs[idx] = img

    return imgs


def center_crop_arr(pil_image, image_size):
    # Copied from the original code to avoid irrelevant dependencies:
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y:crop_y + image_size, crop_x:crop_x + image_size]
