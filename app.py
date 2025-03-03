import os
import gradio as gr
import numpy as np
from enum import Enum
import db_examples
import cv2

import spaces

from demo_utils1 import *

from misc_utils.train_utils import unit_test_create_model
from misc_utils.image_utils import save_tensor_to_gif, save_tensor_to_images
import os
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from einops import rearrange
import imageio
import time

from torchvision.transforms import functional as F
from torch.hub import download_url_to_file

import os

# 推理设置
from pl_trainer.inference.inference import InferenceIP2PVideo
from tqdm import tqdm


# if not os.path.exists(filename):
#     original_path = os.getcwd()
#     base_path = './models'
#     os.makedirs(base_path, exist_ok=True)

#     # 直接在代码中写入 Token（注意安全风险）
#     GIT_TOKEN = "955b8ea91095840b76fe38b90a088c200d4c813c"
#     repo_url = f"https://YeFang:{GIT_TOKEN}@code.openxlab.org.cn/YeFang/RIV_models.git"

#     try:
#         if os.system(f'git clone {repo_url} {base_path}') != 0:
#             raise RuntimeError("Git 克隆失败")
#         os.chdir(base_path)
#         if os.system('git lfs pull') != 0:
#             raise RuntimeError("Git LFS 拉取失败")
#     finally:
#         os.chdir(original_path)

def tensor_to_pil_image(x):
    """
    将 4D PyTorch 张量转换为 PIL 图像。
    """
    x = x.float()  # 确保张量类型为 float
    grid_img = torchvision.utils.make_grid(x, nrow=4).permute(1, 2, 0).detach().cpu().numpy()
    grid_img = (grid_img * 255).clip(0, 255).astype("uint8")  # 将 [0, 1] 范围转换为 [0, 255]
    return Image.fromarray(grid_img)

def frame_to_batch(x):
    """
    将帧维度转换为批次维度。
    """
    return rearrange(x, 'b f c h w -> (b f) c h w')

def clip_image(x, min=0., max=1.):
    """
    将图像张量裁剪到指定的最小和最大值。
    """
    return torch.clamp(x, min=min, max=max)

def unnormalize(x):
    """
    将张量范围从 [-1, 1] 转换到 [0, 1]。
    """
    return (x + 1) / 2


# 读取图像文件
def read_images_from_directory(directory, num_frames=16):
    images = []
    for i in range(num_frames):
        img_path = os.path.join(directory, f'{i:04d}.png')
        img = imageio.imread(img_path)
        images.append(torch.tensor(img).permute(2, 0, 1))  # Convert to Tensor (C, H, W)
    return images

def load_and_process_images(folder_path):
    """
    读取文件夹中的所有图片，将它们转换为 [-1, 1] 范围的张量并返回一个 4D 张量。
    """
    processed_images = []
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1)  # 将 [0, 1] 转换为 [-1, 1]
    ])
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            image = Image.open(img_path).convert("RGB")
            processed_image = transform(image)
            processed_images.append(processed_image)
    return torch.stack(processed_images)  # 返回 4D 张量

def load_and_process_video(video_path, num_frames=16, crop_size=512):
    """
    读取视频文件中的前 num_frames 帧，将每一帧转换为 [-1, 1] 范围的张量，
    并进行中心裁剪至 crop_size x crop_size，返回一个 4D 张量。
    """
    processed_frames = []
    transform = transforms.Compose([
        transforms.CenterCrop(crop_size),       # 中心裁剪
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1)  # 将 [0, 1] 转换为 [-1, 1]
    ])
    
    # 使用 OpenCV 读取视频
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    frame_count = 0

    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break  # 视频帧读取完毕或视频帧不足
        
        # 转换为 RGB 格式
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        
        # 应用转换
        processed_frame = transform(image)
        processed_frames.append(processed_frame)
        
        frame_count += 1

    cap.release()  # 释放视频资源

    if len(processed_frames) < num_frames:
        raise ValueError(f"视频帧不足 {num_frames} 帧，仅找到 {len(processed_frames)} 帧。")

    return torch.stack(processed_frames)  # 返回 4D 张量 (帧数, 通道数, 高度, 宽度)


def clear_cache(output_path):
    if os.path.exists(output_path):
        os.remove(output_path)
    return None


#! 加载模型
# 配置路径和加载模型
config_path = 'configs/instruct_v2v_ic_gradio.yaml'
diffusion_model = unit_test_create_model(config_path)
diffusion_model = diffusion_model.to('cuda')

# 加载模型检查点
# ckpt_path = 'models/relvid_mm_sd15_fbc_unet.pth' #! change
# ckpt_path = 'tmp/pytorch_model.bin'
# 下载文件

os.makedirs('models', exist_ok=True)
model_path = "models/relvid_mm_sd15_fbc_unet.pth"

if not os.path.exists(model_path):
    download_url_to_file(url='https://huggingface.co/aleafy/RelightVid/resolve/main/relvid_mm_sd15_fbc_unet.pth', dst=model_path)


ckpt = torch.load(model_path, map_location='cpu')
diffusion_model.load_state_dict(ckpt, strict=False)


# import pdb; pdb.set_trace()

# 更改全局临时目录
new_tmp_dir = "./demo/gradio_bg"
os.makedirs(new_tmp_dir, exist_ok=True)

# import pdb; pdb.set_trace()

def save_video_from_frames(image_pred, save_pth, fps=8):
    """
    将 image_pred 中的帧保存为视频文件。

    参数:
    - image_pred: Tensor，形状为 (1, 16, 3, 512, 512)
    - save_pth: 保存视频的路径，例如 "output_video.mp4"
    - fps: 视频的帧率
    """
    # 视频参数
    num_frames = image_pred.shape[1]
    frame_height, frame_width = 512, 512  # 目标尺寸
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码格式

    # 创建 VideoWriter 对象
    out = cv2.VideoWriter(save_pth, fourcc, fps, (frame_width, frame_height))

    for i in range(num_frames):
        # 反归一化 + 转换为 0-255 范围
        pred_frame = clip_image(unnormalize(image_pred[0][i].unsqueeze(0))) * 255
        pred_frame_resized = pred_frame.squeeze(0).detach().cpu()  # (3, 512, 512)
        pred_frame_resized = pred_frame_resized.permute(1, 2, 0).numpy().astype("uint8")  # (512, 512, 3)

        # Resize 到 256x256
        pred_frame_resized = cv2.resize(pred_frame_resized, (frame_width, frame_height))

        # 将 RGB 转为 BGR（因为 OpenCV 使用 BGR 格式）
        pred_frame_bgr = cv2.cvtColor(pred_frame_resized, cv2.COLOR_RGB2BGR)

        # 写入帧到视频
        out.write(pred_frame_bgr)

    # 释放 VideoWriter 资源
    out.release()
    print(f"视频已保存至 {save_pth}")


# 伪函数占位（生成空白视频）
@spaces.GPU
def dummy_process(input_fg, input_bg):
    # import pdb; pdb.set_trace()
    fg_tensor = load_and_process_video(input_fg).cuda().unsqueeze(0)
    bg_tensor = load_and_process_video(input_bg).cuda().unsqueeze(0) # (1, 16, 4, 64, 64)

    cond_fg_tensor = diffusion_model.encode_image_to_latent(fg_tensor)  # (1, 16, 4, 64, 64)
    cond_bg_tensor = diffusion_model.encode_image_to_latent(bg_tensor)
    cond_tensor = torch.cat((cond_fg_tensor, cond_bg_tensor), dim=2)

    # 初始化潜变量
    init_latent = torch.randn_like(cond_fg_tensor)

    inf_pipe = InferenceIP2PVideo(
        diffusion_model.unet, 
        scheduler='ddpm',
        num_ddim_steps=20
    )

    EDIT_PROMPT = 'change the background'
    VIDEO_CFG = 1.2
    TEXT_CFG = 7.5
    text_cond = diffusion_model.encode_text([EDIT_PROMPT])  # (1, 77, 768)
    text_uncond = diffusion_model.encode_text([''])
    latent_pred = inf_pipe(
        latent=init_latent,
        text_cond=text_cond,
        text_uncond=text_uncond,
        img_cond=cond_tensor,
        text_cfg=TEXT_CFG,
        img_cfg=VIDEO_CFG,
    )['latent']

    image_pred = diffusion_model.decode_latent_to_image(latent_pred) # (1,16,3,512,512)
    output_path = os.path.join(new_tmp_dir, f"output_{int(time.time())}.mp4")
    # clear_cache(output_path)
    
    save_video_from_frames(image_pred, output_path)
    # import pdb; pdb.set_trace()
    # fps = 8
    # frames = []
    # for i in range(16):
    #     pred_frame = clip_image(unnormalize(image_pred[0][i].unsqueeze(0))) * 255
    #     pred_frame_resized = pred_frame.squeeze(0).detach().cpu() #(3,512,512)
    #     pred_frame_resized = pred_frame_resized.permute(1, 2, 0).detach().cpu().numpy().astype("uint8") #(512,512,3) np
    #     Image.fromarray(pred_frame_resized).save(save_pth)

    # # 生成一个简单的黑色视频作为示例
    # output_path = os.path.join(new_tmp_dir, "output.mp4")
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(output_path, fourcc, 20.0, (512, 512))

    # for _ in range(60):  # 生成 3 秒的视频（20fps）
    #     frame = np.zeros((512, 512, 3), dtype=np.uint8)
    #     out.write(frame)
    # out.release()

    return output_path

# 枚举类用于背景选择
class BGSource(Enum):
    UPLOAD = "Use Background Video"
    UPLOAD_FLIP = "Use Flipped Background Video"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"
    GREY = "Ambient"

# Quick prompts 示例
quick_prompts = [
    'beautiful woman',
    'handsome man',
    'beautiful woman, cinematic lighting',
    'handsome man, cinematic lighting',
    'beautiful woman, natural lighting',
    'handsome man, natural lighting',
    'beautiful woman, neo punk lighting, cyberpunk',
    'handsome man, neo punk lighting, cyberpunk',
]
quick_prompts = [[x] for x in quick_prompts]

# Gradio UI 结构
block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## IC-Light (Relighting with Foreground and Background Video Condition)")

    with gr.Row():
        with gr.Column():
            with gr.Row():
                input_fg = gr.Video(label="Foreground Video", height=370, width=370, visible=True)
                input_bg = gr.Video(label="Background Video", height=370, width=370, visible=True)

            prompt = gr.Textbox(label="Prompt")
            bg_source = gr.Radio(choices=[e.value for e in BGSource],
                                 value=BGSource.UPLOAD.value,
                                 label="Background Source", type='value')

            example_prompts = gr.Dataset(samples=quick_prompts, label='Prompt Quick List', components=[prompt])
            bg_gallery = gr.Gallery(height=450, object_fit='contain', label='Background Quick List', value=db_examples.bg_samples, columns=5, allow_preview=False)
            relight_button = gr.Button(value="Relight")

            with gr.Group():
                with gr.Row():
                    num_samples = gr.Slider(label="Videos", minimum=1, maximum=12, value=1, step=1)
                    seed = gr.Number(label="Seed", value=12345, precision=0)
                with gr.Row():
                    video_width = gr.Slider(label="Video Width", minimum=256, maximum=1024, value=512, step=64)
                    video_height = gr.Slider(label="Video Height", minimum=256, maximum=1024, value=640, step=64)

            with gr.Accordion("Advanced options", open=False):
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=7.0, step=0.01)
                highres_scale = gr.Slider(label="Highres Scale", minimum=1.0, maximum=3.0, value=1.5, step=0.01)
                highres_denoise = gr.Slider(label="Highres Denoise", minimum=0.1, maximum=0.9, value=0.5, step=0.01)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality')
                n_prompt = gr.Textbox(label="Negative Prompt", value='lowres, bad anatomy, bad hands, cropped, worst quality')
                normal_button = gr.Button(value="Compute Normal (4x Slower)")

        with gr.Column():
            result_video = gr.Video(label='Output Video', height=600, width=600, visible=True)

    # 输入列表
    # ips = [input_fg, input_bg, prompt, video_width, video_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source]
    ips = [input_fg, input_bg]

    # 按钮绑定处理函数
    # relight_button.click(fn=lambda: None, inputs=[], outputs=[result_video])

    relight_button.click(fn=dummy_process, inputs=ips, outputs=[result_video])
    
    normal_button.click(fn=dummy_process, inputs=ips, outputs=[result_video])

    # 背景库选择
    def bg_gallery_selected(gal, evt: gr.SelectData):
        # import pdb; pdb.set_trace()
        # img_path = gal[evt.index][0]
        img_path = db_examples.bg_samples[evt.index]
        video_path = img_path.replace('frames/0000.png', 'cropped_video.mp4')
        return video_path

    bg_gallery.select(bg_gallery_selected, inputs=bg_gallery, outputs=input_bg)

    # 示例
    # dummy_video_for_outputs = gr.Video(visible=False, label='Result')
    gr.Examples(
        fn=lambda *args: args[-1],
        examples=db_examples.background_conditioned_examples,
        inputs=[input_fg, input_bg, prompt, bg_source, video_width, video_height, seed, result_video],
        outputs=[result_video],
        run_on_click=True, examples_per_page=1024
    )

# 启动 Gradio 应用
# block.launch(server_name='0.0.0.0', server_port=10002, share=True)
block.launch()
