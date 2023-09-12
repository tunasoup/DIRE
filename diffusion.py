from pathlib import Path

import torch
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
        imgs = reshape_image(img_batch, self._image_size)

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


def reshape_image(imgs: torch.Tensor, image_size: int) -> torch.Tensor:
    if len(imgs.shape) == 3:
        imgs = imgs.unsqueeze(0)
    if imgs.shape[2] != imgs.shape[3]:
        crop_func = transforms.CenterCrop(image_size)
        imgs = crop_func(imgs)
    if imgs.shape[2] != image_size:
        imgs = torch.nn.functional.interpolate(imgs, size=(image_size, image_size), mode="bicubic")
    return imgs

