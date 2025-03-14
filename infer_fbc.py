import os
import time
import torch
import argparse
from tqdm import tqdm
from torch.hub import download_url_to_file

from misc_utils.train_utils import unit_test_create_model
from pl_trainer.inference.inference import InferenceIP2PVideo
from utils import *

def main(args):
    # Load model config and model
    diffusion_model = unit_test_create_model(args.config_path)
    diffusion_model = diffusion_model.to('cuda').half()

    # Download model if not exists
    os.makedirs('models', exist_ok=True)
    if not os.path.exists(args.model_path):
        download_url_to_file(
            url='https://huggingface.co/aleafy/RelightVid/resolve/main/relvid_mm_sd15_fbc_unet.pth',
            dst=args.model_path
        )

    # Load checkpoint
    ckpt = torch.load(args.model_path, map_location='cpu')
    diffusion_model.load_state_dict(ckpt, strict=False)

    # Setup inference pipeline
    inf_pipe = InferenceIP2PVideo(
        diffusion_model.unet,
        scheduler='ddpm',
        num_ddim_steps=20
    )

    # Prepare input tensors
    fg_tensor = apply_mask_to_video(args.input, args.mask).cuda().unsqueeze(0).to(dtype=torch.float16)
    bg_tensor = load_and_process_video(args.bg_cond).cuda().unsqueeze(0).to(dtype=torch.float16)

    cond_fg_tensor = diffusion_model.encode_image_to_latent(fg_tensor)
    cond_bg_tensor = diffusion_model.encode_image_to_latent(bg_tensor)
    cond_tensor = torch.cat((cond_fg_tensor, cond_bg_tensor), dim=2)

    # Initialize latent tensor
    init_latent = torch.randn_like(cond_fg_tensor)

    # Text conditioning
    text_cond = diffusion_model.encode_text([args.prompt])
    text_uncond = diffusion_model.encode_text([''])

    # Convert to float16
    init_latent, text_cond, text_uncond, cond_tensor = (
        init_latent.to(dtype=torch.float16),
        text_cond.to(dtype=torch.float16),
        text_uncond.to(dtype=torch.float16),
        cond_tensor.to(dtype=torch.float16)
    )
    inf_pipe.unet.to(torch.float16)

    # Inference
    latent_pred = inf_pipe(
        latent=init_latent,
        text_cond=text_cond,
        text_uncond=text_uncond,
        img_cond=cond_tensor,
        text_cfg=7.5,
        img_cfg=1.2,
    )['latent']

    # Decode and save video
    image_pred = diffusion_model.decode_latent_to_image(latent_pred)
    save_video_from_frames(image_pred, args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="/mnt/petrelfs/fangye/nvme/hugface/new_right/RelightVid/assets/input/woman.mp4")
    parser.add_argument("--mask", type=str, default="/mnt/petrelfs/fangye/nvme/hugface/new_right/RelightVid/assets/mask/woman")
    parser.add_argument("--bg_cond", type=str, default="/mnt/petrelfs/fangye/nvme/hugface/new_right/RelightVid/assets/video_bg/universe1.mp4")
    
    parser.add_argument("--model_path", type=str, default="models/relvid_mm_sd15_fbc_unet.pth")
    parser.add_argument("--config_path", type=str, default="configs/instruct_v2v_ic_gradio.yaml")
    parser.add_argument("--output_path", type=str, default="output/test.mp4")
    parser.add_argument("--prompt", type=str, default="change the background")

    args = parser.parse_args()
    main(args)
