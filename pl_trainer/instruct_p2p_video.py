'''
Use pretrained instruct pix2pix model but add additional channels for reference modification
'''

import torch
from .diffusion import DDIMLDMTextTraining
from einops import rearrange

from modules.video_unet_temporal.resnet import InflatedConv3d
from safetensors.torch import load_file

import torch.nn.functional as F

from torch import nn
import cv2
from torch.hub import download_url_to_file

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(3072, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 4096)
        self.fc4 = nn.Linear(4096, 2304)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)  # 设置Leaky ReLU的负斜率

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
# class CombineMLP(nn.Module):
#     def __init__(self, input_dim=128, output_dim=64, hidden_dim=128):
#         """
#         构造一个 5 层 MLP 网络。
#         :param input_dim: 输入的特征维度，默认 128
#         :param output_dim: 输出的特征维度，默认 64
#         :param hidden_dim: 隐藏层维度，默认 128
#         """
#         super(CombineMLP, self).__init__()
        
#         # 定义 5 层 MLP
#         self.fc1 = nn.Linear(input_dim, hidden_dim) #()
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc4 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc5 = nn.Linear(hidden_dim, output_dim)  # 最后一层映射到 64

#         # 定义激活函数
#         # self.activation = nn.ReLU()
#         self.activation = nn.LeakyReLU(negative_slope=0.01)  # 默认负斜率为 0.01


#     def forward(self, x1, x2):
#         """
#         前向传播，支持两个输入 x1 和 x2
#         :param x1: 第一个输入，形状 (B, 64)
#         :param x2: 第二个输入，形状 (B, 64)
#         :return: 输出特征，形状 (B, 64)
#         """
#         # 将两个输入拼接在一起
#         x = torch.cat([x1, x2], dim=-1)  # 拼接后形状为 (B, 128)

#         # 依次通过 5 层 MLP 和激活函数
#         x = self.activation(self.fc1(x))
#         x = self.activation(self.fc2(x))
#         x = self.activation(self.fc3(x))
#         x = self.activation(self.fc4(x))
#         x = self.fc5(x)  # 最后一层不使用激活函数（根据需求）

#         return x


class CombineMLP(nn.Module):
    def __init__(self, input_dim=4*64*64*2, output_dim=4*64*64, hidden_dim=128, num_layers=5):
        """
        构造一个 5 层 MLP 网络。
        :param input_dim: 输入的特征维度，默认 128
        :param output_dim: 输出的特征维度，默认 64
        :param hidden_dim: 隐藏层维度，默认 128
        """
        super(CombineMLP, self).__init__()
        
        # 创建多个隐藏层
        layers = []
        for i in range(num_layers - 1):  # 生成 num_layers-1 个隐藏层
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # 输出层
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        # 将层组合成一个模块
        self.mlp = nn.Sequential(*layers)


    def forward(self, x1, x2):
        """
        前向传播，支持两个输入 x1 和 x2
        :param x1: 第一个输入，形状 (1,16,4,64,64)
        :param x2: 第二个输入，形状 (1,16,4,64,64)
        :return: 输出特征，形状 (1,16,4,64,64)
        """
        # import pdb; pdb.set_trace()
        # 将两个输入拼接在一起
        x = torch.cat([x1, x2], dim=2)  # 拼接后形状为 (1,16,8,64,64)
        x = torch.flatten(x, start_dim=2)  # Flatten to shape (batch_size, 16, 8*64*64)
        x = self.mlp(x)  # Apply MLP  1,16,16384
        x = x.reshape(x.size(0), x.size(1), 4, 64, 64)  # Reshape back to (1, 16, 4, 64, 64)

        return x



class HDRCtrlModeltmp(nn.Module):
    def __init__(self):
        super(HDRCtrlModel, self).__init__()
        
        # 定义卷积层
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=4, padding=1)
        self.conv_layer2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1)
        
        # 定义 MLP 模型
        self.mlp = MLP()

    def decompose_hdr(self, hdr_latents):
        batch_size, channels, height, width = hdr_latents.shape
        device = hdr_latents.device  # 获取设备信息

        # 生成 4×4 掩码 (batch_size, 1, 4, 4)
        mask_small = torch.rand(batch_size, 1, 4, 4, device=device)  # 从均匀分布生成随机掩码

        # 将掩码调整为与输入相同的大小 (batch_size, 1, height, width)
        mask = torch.nn.functional.interpolate(mask_small, size=(height, width), mode='bilinear', align_corners=False)

        # 保持连续值，不进行二值化 #! 注意此步操作, 注意可视化 random mask的结果...  首先可以可视化mask, 其次可视化
        mask = mask.expand(-1, channels, -1, -1)  # 扩展掩码通道数以匹配 hdr_latents 的形状

        # 应用 mask 生成 L1 和 L2
        hdr_latents_1 = hdr_latents * mask  # L1 = 掩码部分
        hdr_latents_2 = hdr_latents * (1 - mask)  # L2 = 非掩码部分

        return hdr_latents_1, hdr_latents_2

    def forward(self, hdr_latents):
        # import pdb; pdb.set_trace()
        # todo: mask get hdr1, hdr2; input hdr_latents(实际上暂时是ldr)
        # 输入的形状为 (1, 16, 3, 256, 512)，去掉多余的维度
        # import pdb; pdb.set_trace()
        hdr_latents = hdr_latents.squeeze(0)  # 变成 (16, 3, 256, 512)

        batch_size = hdr_latents.shape[0]

        # 转换为 NCHW 形式： (batch, channels, height, width) 输入之前numpy2tensor已经permute过了
        # hdr_latents = hdr_latents.permute(0, 3, 1, 2) #! 注意一下to tensor? (如何进行归一化的) 的时候已经
        # 进行卷积操作
        conv_output = self.conv_layer1(hdr_latents) #! 注意更改此处卷积!!!
        conv_output = self.conv_layer2(conv_output)  # (16, 3, 32, 64)

        # 截取前 32 列，得到最终形状 (16, 3, 32, 32)
        hdr_latents = conv_output[:, :, :, :32]
        # todo: decompose hdr
        hdr_latents_1, hdr_latents_2 = self.decompose_hdr(hdr_latents) # [16, 3, 32, 32], [16, 3, 32, 32]

        # 将输出展平，准备输入到 MLP 中
        hdr_latents = hdr_latents.reshape(hdr_latents.size(0), -1) # [16, 3072]
        hdr_latents_1 = hdr_latents_1.reshape(hdr_latents_1.size(0), -1) # [16, 3072]
        hdr_latents_2 = hdr_latents_2.reshape(hdr_latents_2.size(0), -1)

        # 传递给 MLP
        hdr_latents = self.mlp(hdr_latents) #(16, 2304)  3072 -> 2304
        hdr_latents_1 = self.mlp(hdr_latents_1)
        hdr_latents_2 = self.mlp(hdr_latents_2)

        # 重新调整输出的形状
        hdr_latents = hdr_latents.reshape(batch_size, 3, 768)  # reshape 输出为 (16, 3, 768)
        hdr_latents_1 = hdr_latents_1.reshape(batch_size, 3, 768)
        hdr_latents_2 = hdr_latents_2.reshape(batch_size, 3, 768)

        return hdr_latents, hdr_latents_1, hdr_latents_2

class HDRCtrlModel(nn.Module):
    def __init__(self):
        super(HDRCtrlModel, self).__init__()
        
        # 定义卷积层
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=4, padding=1)
        self.conv_layer2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1)
        
        # 定义 MLP 模型
        self.mlp = MLP()

    def decompose_hdr(self, hdr_latents): # hdr_latents: 16,3,32,32  可以可视化一下这部分代码...
        batch_size, channels, height, width = hdr_latents.shape
        device = hdr_latents.device  # 获取设备信息

        # 生成 4×4 掩码 (batch_size, 1, 4, 4)
        mask_small = torch.rand(batch_size, 1, 4, 4, device=device)  # 从均匀分布生成随机掩码

        threshold = 0.5  # 调节阈值，增加黑色部分比例
        mask_small = (mask_small > threshold).float()
        # 将掩码调整为与输入相同的大小 (batch_size, 1, height, width)  16,1,32,32
        mask = torch.nn.functional.interpolate(mask_small, size=(height, width), mode='bilinear', align_corners=False)
        
        # import pdb; pdb.set_trace()
        # 保持连续值，不进行二值化 #! 注意此步操作, 注意可视化 random mask的结果...  首先可以可视化mask, 其次可视化
        mask = mask.expand(-1, channels, -1, -1)  # 扩展掩码通道数以匹配 hdr_latents 的形状

        # 应用 mask 生成 L1 和 L2
        hdr_latents_1 = hdr_latents * mask  # L1 = 掩码部分
        hdr_latents_2 = hdr_latents * (1 - mask)  # L2 = 非掩码部分

        return hdr_latents_1, hdr_latents_2
    
    def blur_image(self, hdr_latents):
        # 高斯模糊, 输入 (16,3,256,256)
        processed_images = []
        kernel_size = (15, 15)
        sigmaX = 10

        # 对每张图像进行处理
        for i in range(hdr_latents.size(0)):  # 遍历16张图像
            # 获取第i张图像
            image = hdr_latents[i].permute(1, 2, 0).cpu().numpy()  # 将形状变为 (256, 256, 3)
            
            # 进行高斯模糊
            blurred_image = cv2.GaussianBlur(image, kernel_size, sigmaX)
            
            # 将图像缩放到 (32, 32, 3)
            resized_image = cv2.resize(blurred_image, (32, 32), interpolation=cv2.INTER_AREA)
            
            # 将处理后的图像从 numpy 数组转换回 tensor
            resized_image_tensor = torch.tensor(resized_image, dtype=torch.uint8, device=hdr_latents.device).permute(2, 0, 1)  # 转回 (3, 32, 32)
            
            # 将处理后的图像添加到列表中
            processed_images.append(resized_image_tensor)

        # 将列表中的所有图像堆叠成一个 tensor
        processed_images_tensor = torch.stack(processed_images)  # 形状为 (16, 3, 32, 32)

        return processed_images_tensor

    def normalize_hdr(self, img):
        img = img / 255.0
        return img * 2 -1

    def forward(self, hdr_latents):
        # import pdb; pdb.set_trace()
        # todo: mask get hdr1, hdr2; input hdr_latents(实际上暂时是ldr)
        # 输入的形状为 (n, 16, 3, 256, 256)，去掉多余的维度
        # import pdb; pdb.set_trace()
        # hdr_latents = hdr_latents.squeeze(0)  # 变成 (16, 3, 256, 256)
        batch_size_ori = hdr_latents.shape[0]
        # frame_num = hdr_latents.shape[1]

        hdr_latents = rearrange(hdr_latents, 'b f c h w -> (b f) c h w')
        batch_size = hdr_latents.shape[0]
        # batch_size = hdr_latents.shape[0]
        # 转换为 NCHW 形式： (batch, channels, height, width) 输入之前numpy2tensor已经permute过了
        # 高斯模糊
        hdr_latents = self.blur_image(hdr_latents) #(16,3,32,32)  可视化打印一下!
        
        # import pdb; pdb.set_trace()
        # todo: decompose hdr
        hdr_latents_1, hdr_latents_2 = self.decompose_hdr(hdr_latents) # [16, 3, 32, 32], [16, 3, 32, 32]

        # todo: 加一步 normalize  /255 -> -1,1
        hdr_latents, hdr_latents_1, hdr_latents_2 = self.normalize_hdr(hdr_latents), self.normalize_hdr(hdr_latents_1), self.normalize_hdr(hdr_latents_2)

        # import pdb; pdb.set_trace()
        # 将输出展平，准备输入到 MLP 中
        hdr_latents = hdr_latents.reshape(hdr_latents.size(0), -1) # [16, 3072]
        hdr_latents_1 = hdr_latents_1.reshape(hdr_latents_1.size(0), -1) # [16, 3072]
        hdr_latents_2 = hdr_latents_2.reshape(hdr_latents_2.size(0), -1)
        
        # 传递给 MLP
        hdr_latents = self.mlp(hdr_latents) #(16, 2304)  3072 -> 2304
        hdr_latents_1 = self.mlp(hdr_latents_1)
        hdr_latents_2 = self.mlp(hdr_latents_2)

        # 重新调整输出的形状
        hdr_latents = hdr_latents.reshape(batch_size, 3, 768)  # reshape 输出为 (16*n, 3, 768)
        hdr_latents_1 = hdr_latents_1.reshape(batch_size, 3, 768) 
        hdr_latents_2 = hdr_latents_2.reshape(batch_size, 3, 768)
        
        hdr_latents = rearrange(hdr_latents, '(b f) n c -> b f n c', b=batch_size_ori)
        hdr_latents_1 = rearrange(hdr_latents_1, '(b f) n c -> b f n c', b=batch_size_ori)
        hdr_latents_2 = rearrange(hdr_latents_2, '(b f) n c -> b f n c', b=batch_size_ori)

        #! 两个细节: 1. 仅有ldr, 需不需要concat hdr或线性变换 2. mask不同帧不一致 
        return hdr_latents, hdr_latents_1, hdr_latents_2 # 3 x (b,16,3,768)


class InstructP2PVideoTrainer(DDIMLDMTextTraining):
    def __init__(
        self, *args,
        cond_image_dropout=0.1,
        cond_text_dropout=0.1,
        cond_hdr_dropout=0.1,
        prompt_type='output_prompt',
        text_cfg=7.5,
        img_cfg=1.2,
        hdr_cfg=7.5,
        hdr_rate=0.1,
        ic_condition='bg',
        hdr_train=False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.hdr_train = hdr_train
        if self.hdr_train:
            self.hdr_encoder = HDRCtrlModel()
            self.hdr_encoder = self.hdr_encoder.to(self.unet.device)
            self.mlp = CombineMLP()
            self.cond_hdr_dropout = cond_hdr_dropout
            self.hdr_rate = hdr_rate

        self.cond_image_dropout = cond_image_dropout
        self.cond_text_dropout = cond_text_dropout

        assert ic_condition in ['fg', 'bg']
        assert prompt_type in ['output_prompt', 'edit_prompt', 'mixed_prompt']
        self.prompt_type = prompt_type
        self.ic_condition = ic_condition

        self.text_cfg = text_cfg
        self.img_cfg = img_cfg
        self.hdr_cfg = hdr_cfg

        #! 开启xformers训练设置
        # self.unet.enable_xformers_memory_efficient_attention()
        # self.unet.enable_gradient_checkpointing()

    def encode_text(self, text):
        with torch.cuda.amp.autocast(dtype=torch.float16):
            encoded_text = super().encode_text(text)
        return encoded_text

    def encode_image_to_latent(self, image):
        # with torch.cuda.amp.autocast(dtype=torch.float16):
        latent = super().encode_image_to_latent(image)
        return latent

    # @torch.cuda.amp.autocast(dtype=torch.float16)
    @torch.no_grad()
    def get_prompt(self, batch, mode):
        # if mode == 'train':
        #     if self.prompt_type == 'output_prompt':
        #         prompt = batch['output_prompt']
        #     elif self.prompt_type == 'edit_prompt': # training的时候是edit prompt
        #         prompt = batch['edit_prompt']
        #     elif self.prompt_type == 'mixed_prompt':
        #         if int(torch.rand(1)) > 0.5:
        #             prompt = batch['output_prompt']
        #         else:
        #             prompt = batch['edit_prompt']
        # else:
        #     prompt = batch['output_prompt']
        prompt = batch['text_prompt']
        if not self.hdr_train: #! 如果hdr后续加进来text了, 还是需要?
            if torch.rand(1).item() < self.cond_text_dropout:
                prompt = 'change the background'
        cond_text = self.encode_text(prompt)
        if mode == 'train':
            if torch.rand(1).item() < self.cond_text_dropout:
                cond_text = torch.zeros_like(cond_text)
        # import pdb; pdb.set_trace()
        return cond_text

    # @torch.cuda.amp.autocast(dtype=torch.float16)
    @torch.no_grad()
    def encode_image_to_latent(self, image):
        b, f, c, h, w = image.shape
        image = rearrange(image, 'b f c h w -> (b f) c h w')
        latent = super().encode_image_to_latent(image)
        latent = rearrange(latent, '(b f) c h w -> b f c h w', b=b)
        return latent

    # @torch.cuda.amp.autocast(dtype=torch.float16)
    @torch.no_grad()
    def decode_latent_to_image(self, latent):
        b, f, c, h, w = latent.shape
        latent = rearrange(latent, 'b f c h w -> (b f) c h w')

        image = []
        for latent_ in latent:
            image_ = super().decode_latent_to_image(latent_[None])
            image.append(image_.sample) #! 注意一下这里 之前没报过错吗; -> 之前不是一个类
        image = torch.cat(image, dim=0)
        # image = super().decode_latent_to_image(latent)
        image = rearrange(image, '(b f) c h w -> b f c h w', b=b)
        return image

    @torch.no_grad()
    def get_cond_image(self, batch, mode):
        # import pdb; pdb.set_trace()
        cond_fg_image = batch['fg_video'] # 这边condition 就是 input_video了, 估计是concat或者ctrlnet
        cond_fg_image = self.encode_image_to_latent(cond_fg_image)
        if self.ic_condition == 'bg':
            cond_bg_image = batch['bg_video']
            if torch.all(cond_bg_image == 0):
                cond_bg_image = torch.zeros_like(cond_fg_image) #! 背景一定概率为0, 置为0.3
            else:
                cond_bg_image = self.encode_image_to_latent(cond_bg_image)
            cond_image = torch.cat((cond_fg_image, cond_bg_image), dim=2) #(1,16,8,64,64)
        else:
            cond_image = cond_fg_image
        # test code: 可视化代码
        # from PIL import Image
        # Image.fromarray(((batch['input_video'] + 1) / 2 * 255).byte()[0,0].permute(1,2,0).cpu().numpy()).save('img1.png')

        # ip2p does not scale cond image, so we unscale the cond image
        # cond_image = self.encode_image_to_latent(cond_image) / self.scale_factor # 额 就是一个vae encode，没有缩放；这边不进行缩放吗? 啥意思呢

        if mode == 'train':
            # if int(torch.rand(1)) < self.cond_image_dropout: # 0.1的概率随机初始化, 应该是为了保障一个鲁棒性 难怪有的时候是全0, 不是代码的bug   #! 艹 bug, 这么久才发现....
            if torch.rand(1).item() < self.cond_image_dropout:
                cond_image = torch.zeros_like(cond_image)
        return cond_image

    @torch.no_grad()
    def get_diffused_image(self, batch, mode):
        # import pdb; pdb.set_trace()
        x = batch['tgt_video'] # 这边编辑的时候, 具体加噪和去噪的gt, 整个这套流程都是以编辑后, 即edited video作为输入
        # from PIL import Image
        # Image.fromarray(((batch['edited_video'] + 1) / 2 * 255).byte()[0,0].permute(1,2,0).cpu().numpy()).save('img2.png')
        b, *_ = x.shape
        x = self.encode_image_to_latent(x) # (1, 16, 4, 32, 32), 经过了vae encode
        eps = torch.randn_like(x)

        if mode == 'train':
            t = torch.randint(0, self.num_timesteps, (b,), device=x.device).long()
        else:
            t = torch.full((b,), self.num_timesteps-1, device=x.device, dtype=torch.long)
        x_t = self.add_noise(x, t, eps) # 加噪t步长  eps表示高斯噪声, 和scheduler的加噪

        if self.prediction_type == 'epsilon':
            return x_t, eps, t 
        else:
            return x_t, x, t


    @torch.no_grad()
    def get_hdr_image(self, batch, mode):
        x = batch['ldr_video'] # todo (16,3,256,512), float, tensor, device -> (1,16,3,256,256) 注意此时仅有ldr
        # import pdb; pdb.set_trace()
        hdr_latents, hdr_latents_1, hdr_latents_2 = self.hdr_encoder(x)
        if mode == 'train': #! 考虑一下这个开不开, 因为后面要拉consistency loss
            if torch.rand(1).item() < self.cond_hdr_dropout:
                hdr_latents = torch.zeros_like(hdr_latents)
                hdr_latents_1 = torch.zeros_like(hdr_latents_1)
                hdr_latents_2 = torch.zeros_like(hdr_latents_2)
        return hdr_latents, hdr_latents_1, hdr_latents_2

    @torch.no_grad() # batch中需要加载mask
    def get_mask(self, batch, mode, target):
        # (1,16,1,512,512)
        # import pdb; pdb.set_trace()
        mask = batch['fg_mask'] # todo 返回mask  (n,16,1,512,512)
        bs = mask.shape[0]
        target_height, target_width = target.shape[-2:] #(n,16,3,64,64)

        mask = rearrange(mask, 'b f c h w -> (b f) c h w')
        resized_mask = F.interpolate(mask, size=(target_height, target_width), mode='bilinear', align_corners=False)
        # resized_mask = resized_mask.unsqueeze(0)
        resized_mask = rearrange(resized_mask, '(b f) c h w -> b f c h w', b=bs)
        if target.shape[2] != resized_mask.shape[2]:
            resized_mask = resized_mask.expand(-1, -1, target.shape[2], -1, -1)  # 匹配目标通道数
        
        return resized_mask

    @torch.no_grad()
    def process_batch(self, batch, mode): #! 可视化这边的image, 查看问题出在哪了。。。 √, 应该是randn_drop的事
        # import pdb; pdb.set_trace()
        cond_image = self.get_cond_image(batch, mode) # 把输入的src image进行一个编码, 这边只有vae的encode, 且没有乘缩放的系数(ip2p本身没乘...)
        diffused_image, target, t = self.get_diffused_image(batch, mode) # diffused_image: 经过了vae encode, 和scheduler的加噪，标准的降噪输入
        # target: 这边是epsilon目标, 因此还是拉成epsilon的损失；t: 训练阶段是随机的一个数值, 推理阶段一般都是1000
        prompt = self.get_prompt(batch, mode)
        model_kwargs = {
            'encoder_hidden_states': prompt
        }
        # import pdb; pdb.set_trace()
        if self.hdr_train:
            hdr_image, hdr_image_1, hdr_image_2 = self.get_hdr_image(batch, mode) #(16,3,768)
            fg_mask = self.get_mask(batch, mode, target) # 把原图像前景mask resize到target大小
        
            model_kwargs = {
                'encoder_hidden_states': {'hdr_latents': hdr_image, 'encoder_hidden_states': prompt, 'hdr_latents_1': hdr_image_1, 'hdr_latents_2': hdr_image_2, 'fg_mask': fg_mask}
            }
        

        return {
            'diffused_input': diffused_image, # (1, 16, 4, 64, 64), 经过了vae encode, 和scheduler的加噪 
            'condition': cond_image, # 把输入的src image进行一个编码, 这边只有vae的encode, 且没有乘缩放的系数  (1,16,8,64,64)
            'target': target, # 这个是加到tgt video的高斯噪声
            't': t, # 0~1000的一个时刻
            'model_kwargs': model_kwargs, # 这边就是一个text_hidden_states
        }

    def training_step(self, batch, batch_idx): #! 注意一下仅仅训motion layer
        # import pdb; pdb.set_trace()
        processed_batch = self.process_batch(batch, mode='train') #(1,16,3,256,256), 读取的序列化图片, 仅仅做了一个归一化操作
        diffused_input = processed_batch['diffused_input'] # (1,16,4,64,64), edit images, 经过了vae encode, 和scheduler的加噪
        condition = processed_batch['condition'] # (1,16,8,64,64) 把输入的src images进行一个编码, 这边只有vae的encode, 且没有乘缩放的系数
        target = processed_batch['target'] # (1,16,4,64,64), target是加入的高斯噪声
        t = processed_batch['t'] # [257], 一个0~1000的随机时刻

        model_kwargs = processed_batch['model_kwargs'] # dict, 仅包含一项: encoder_hidden_states, [1, 77, 768] text_hidden_states
    
        model_input = torch.cat([diffused_input, condition], dim=2) # b, f, c, h, w [1,16,8,32,32] 这边是做的concat, 很多edit文章经典操作, 把两个东西concat起来
        #! 半精度
        # model_input = model_input.float()
        # model_kwargs['encoder_hidden_states'] = model_kwargs['encoder_hidden_states'].half()
        model_input = rearrange(model_input, 'b f c h w -> b c f h w') # [1,8,16,32,32]

        pred = self.unet(model_input, t, **model_kwargs).sample # (1,4,16,64,64) #!
        pred = rearrange(pred, 'b c f h w -> b f c h w') # (1,16,4,64,64) #!
        
        if not self.hdr_train:
            loss = self.get_loss(pred, target, t) # 0.320
        else:
            fg_mask = model_kwargs['encoder_hidden_states']['fg_mask']
            loss = self.get_hdr_loss(fg_mask, pred, target)
        ### add consistency loss ###
        # todo: 三个相同的model_input, 不同的model_kwargs (注意stack到一起, attn里面的逻辑也得改...)
        # if self.hdr_train:
            # fg_mask = model_kwargs['encoder_hidden_states']['fg_mask']
            # hdr_latents = model_kwargs['encoder_hidden_states']['hdr_latents_1']
            # hdr_latents_1 = model_kwargs['encoder_hidden_states']['hdr_latents_1']
            # hdr_latents_2 = model_kwargs['encoder_hidden_states']['hdr_latents_2']
            
            # model_input = torch.cat([diffused_input, condition], dim=2)
            # model_input = rearrange(model_input, 'b f c h w -> b c f h w')
            # model_input_1 = model_input.clone()
            # model_input_2 = model_input.clone()
            # model_input_all = torch.cat([model_input, model_input_1, model_input_2], dim=0)
            
            # prompt = model_kwargs['encoder_hidden_states']['encoder_hidden_states'] #(1*n,77,768)
            # prompt_all = torch.cat([prompt, prompt, prompt], dim=0) #(3*n,77,768)
            # # import pdb; pdb.set_trace()
            # model_kwargs['encoder_hidden_states']['encoder_hidden_states'] = prompt_all
            
            # # import pdb; pdb.set_trace()
            # hdr_latents_all = torch.cat([hdr_latents, hdr_latents_1, hdr_latents_2], dim=0) #(3*n,16,77,768)
            # model_kwargs['encoder_hidden_states']['hdr_latents']=hdr_latents_all
            # pred_all = self.unet(model_input_all, t, **model_kwargs).sample # (1,4,16,64,64) 
            # pred_all = rearrange(pred_all, 'b c f h w -> b f c h w')

            # pred, pred1, pred2 = pred_all.chunk(3, dim=0)
            # loss_ori = self.get_hdr_loss(fg_mask, pred, target)

            # # 假设获得了L1, L2
            # # hdr_latents_1 = mask(hdr_latents) # 随机构造一个mask + 逻辑矫正
            # # model_kwargs['encoder_hidden_states']['hdr_latents']=hdr_latents_1
            # # pred1 = self.unet(model_input, t, **model_kwargs).sample # get L1下的预测值 (1,16,4,64,64)
            # # pred1 = rearrange(pred1, 'b c f h w -> b f c h w') 

            # # model_input = torch.cat([diffused_input, condition], dim=2)
            # # model_input = rearrange(model_input, 'b f c h w -> b c f h w')
            # # # hdr_latents_2 = 1-mask(hdr_latents)
            # # model_kwargs['encoder_hidden_states']['hdr_latents']=hdr_latents_2
            # # pred2 = self.unet(model_input, t, **model_kwargs).sample # get L2下的预测值
            # # pred2 = rearrange(pred2, 'b c f h w -> b f c h w')
            # # import pdb; pdb.set_trace()
            # pred_combine = self.mlp(pred1, pred2) #! todo: 构造mlp loss 错了!! 搞对一下, 应该需要展平....
            # loss_c = self.get_hdr_loss(fg_mask, pred, pred_combine)
            # # loss_c = MSELoss(mask*pred, mask*pred_conbine) # todo: change to函数, 逻辑矫正
            
            # loss = loss_ori + self.hdr_rate * loss_c # 设一个系数, 好控制变化
        ### end ###
        self.log('train_loss', loss, sync_dist=True)

        latent_pred = self.predict_x_0_from_x_t(pred, t, diffused_input) # (1,16,4,32,32)
        image_pred = self.decode_latent_to_image(latent_pred) # 这边相当于是pred_x0了, (1,16,3,256,256)
        drop_out = torch.all(condition == 0).item()

        res_dict = {
            'loss': loss,
            'pred': image_pred,
            'drop_out': drop_out,
            'time': t[0].item()
        }
        return res_dict

    @torch.no_grad()
    @torch.cuda.amp.autocast(dtype=torch.bfloat16)
    def validation_step(self, batch, batch_idx): # 没写好 可以先pass
        # pass
        # import pdb; pdb.set_trace()
        if not self.hdr_train:
            from .inference.inference import InferenceIP2PVideo
            inf_pipe = InferenceIP2PVideo(
                self.unet, 
                beta_start=self.scheduler.config.beta_start,
                beta_end=self.scheduler.config.beta_end,
                beta_schedule=self.scheduler.config.beta_schedule,
                num_ddim_steps=20
            )
            # import pdb; pdb.set_trace()
            processed_batch = self.process_batch(batch, mode='val')
            diffused_input = torch.randn_like(processed_batch['diffused_input']) #(1,16,4,64,64)

            condition = processed_batch['condition'] # 这边其实留有一个接口给condition   (1,16,8,64,64)
            img_cond = condition
            text_cond = processed_batch['model_kwargs']['encoder_hidden_states']
            # import pdb; pdb.set_trace()
            res = inf_pipe(
                latent = diffused_input,
                text_cond = text_cond,
                text_uncond = self.encode_text(['']),
                img_cond = img_cond,
                text_cfg = self.text_cfg,
                img_cfg = self.img_cfg,
                hdr_cfg = self.hdr_cfg
            )

            latent_pred = res['latent']
            image_pred = self.decode_latent_to_image(latent_pred)
            res_dict = {
                'pred': image_pred,
            }
        else:
            from .inference.inference import InferenceIP2PVideoHDR
            inf_pipe = InferenceIP2PVideoHDR(
                self.unet, 
                beta_start=self.scheduler.config.beta_start,
                beta_end=self.scheduler.config.beta_end,
                beta_schedule=self.scheduler.config.beta_schedule,
                num_ddim_steps=20
            )
            # import pdb; pdb.set_trace()
            processed_batch = self.process_batch(batch, mode='val')
            diffused_input = torch.randn_like(processed_batch['diffused_input']) #(1,16,4,64,64)

            condition = processed_batch['condition'] # 这边其实留有一个接口给condition   (1,16,8,64,64)
            model_kwargs = processed_batch['model_kwargs']
            img_cond = condition
            text_cond = model_kwargs['encoder_hidden_states']['encoder_hidden_states']
            hdr_cond = model_kwargs['encoder_hidden_states']['hdr_latents']

            # import pdb; pdb.set_trace()
            res = inf_pipe(
                latent = diffused_input,
                text_cond = text_cond,
                text_uncond = self.encode_text(['']),
                hdr_cond = hdr_cond,
                img_cond = img_cond,
                text_cfg = self.text_cfg,
                img_cfg = self.img_cfg,
            )

            latent_pred = res['latent']
            image_pred = self.decode_latent_to_image(latent_pred)
            res_dict = {
                'pred': image_pred,
            }
        return res_dict

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.unet.parameters(), lr=self.optim_args['lr'])
        import bitsandbytes as bnb
        params = []
        for name, p in self.unet.named_parameters():
            if ('transformer_in' in name) or ('temp_' in name):
                # p.requires_grad = True
                params.append(p)
            else:
                pass
                # p.requires_grad = False
        optimizer = bnb.optim.Adam8bit(params, lr=self.optim_args['lr'], betas=(0.9, 0.999))
        return optimizer

    def initialize_unet(self, unet_init_weights):
        if unet_init_weights is not None:
            print(f'INFO: initialize denoising UNet from {unet_init_weights}')
            sd = torch.load(unet_init_weights, map_location='cpu')
            model_sd = self.unet.state_dict()
            # fit input conv size
            for k in model_sd.keys():
                if k in sd.keys():
                    pass
                else:
                # handling temporal layers
                    if (('temp_' in k) or ('transformer_in' in k)) and 'proj_out' in k:
                        # print(f'INFO: initialize {k} from {model_sd[k].shape} to zeros')
                        sd[k] = torch.zeros_like(model_sd[k])
                    else:
                        # print(f'INFO: initialize {k} from {model_sd[k].shape} to random')
                        sd[k] = model_sd[k]
            self.unet.load_state_dict(sd)

class InstructP2PVideoTrainerTemporal(InstructP2PVideoTrainer):
    def initialize_unet(self, unet_init_weights): # 这边对比上一级来说, 新加的部分在于 rewrite了unet的load函数
        if unet_init_weights is not None:
            print(f'INFO: initialize denoising UNet from {unet_init_weights}')
            sd_init_weights, motion_module_init_weights, iclight_init_weights = unet_init_weights
            os.makedirs(sd_init_weights, exist_ok=True)
            sd_init_weights, motion_module_init_weights, iclight_init_weights = f'models/{sd_init_weights}', f'models/{motion_module_init_weights}', f'models/{iclight_init_weights}'

            if not os.path.exists(sd_init_weights):
                url = 'https://huggingface.co/stablediffusionapi/realistic-vision-v51/resolve/main/unet/diffusion_pytorch_model.safetensors'
                download_url_to_file(url=url, dst=sd_init_weights)
            if not os.path.exists(motion_module_init_weights):
                url = 'https://huggingface.co/aleafy/RelightVid/resolve/main/relvid_mm_sd15_fbc.pth'
                download_url_to_file(url=url, dst=motion_module_init_weights)
            if not os.path.exists(iclight_init_weights):
                url = 'https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fbc.safetensors'
                download_url_to_file(url=url, dst=iclight_init_weights)

            sd = load_file(sd_init_weights) #! 关于加载iclight的unet, 后面再加到yaml里面... 我甚至觉得只要改unet, vae和text其实都差不多
    
            # sd = torch.load(sd_init_weights, map_location='cpu') # 注意debug看看这是啥 + 打印一下原有和加载的keys
            if self.unet.use_motion_module:
                motion_sd = torch.load(motion_module_init_weights, map_location='cpu')
                assert len(sd) + len(motion_sd) == len(self.unet.state_dict()), f'Improper state dict length, got {len(sd) + len(motion_sd)} expected {len(self.unet.state_dict())}' #! 注意一下这行保证了加载的key至少在数量上是对应的; 这行的目的是self.unet是自己定义的 而这两个加载的是别的地方训练的(可能是diffusers中的)
                sd.update(motion_sd)
            
                for k, v in self.unet.state_dict().items():
                    if 'pos_encoder.pe' in k: # 这边是原来iv2v的代码 temporal_position_encoding_max_len, 设置为 32
                        sd[k] = v # the size of pe may change, 主要是temporal layer的size会发生改变... √ 由于输入的max_len变了
                    # if 'conv_in.weight' in k: #! tmp, 这里是test一下
                    #     sd[k] = v
            else:
                assert len(sd) == len(self.unet.state_dict())

            self.unet.load_state_dict(sd) # 为什么这里可以完美适配? √
            # todo: 更改sd的conv_in.weight的shape到12; 更改函数forward, 支持多个输入cond; iclight的sd_offset加载进去; 
            unet = self.unet # saVe一下
            # 这里是更改conv_in的shape; #! 这边注意一下要改成3D版本的unet
            with torch.no_grad():
                # new_conv_in = torch.nn.Conv2d(12, unet.conv_in.out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding)
                new_conv_in = InflatedConv3d(12, unet.conv_in.out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding)
                new_conv_in.weight.zero_()
                new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
                new_conv_in.bias = unet.conv_in.bias
                unet.conv_in = new_conv_in
            
            ###### -- 更改 forward函数 --- #####

            # 这里是更改forward函数。  具体调用的部分在main后面，那里也得改
            # unet_original_forward = unet.forward
            # def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
            #     c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample) # (1,8,67,120)  
            #     c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0) # (2,8,67,120) 应该是复制一份，用于cfg
            #     new_sample = torch.cat([sample, c_concat], dim=2) #(2,12,67,120) 这边还是在通道维度上进行的concat #! change 在第二维cat  (2,1,12,67,120)
            #     # todo 这边中间可以加一个f的通道 b,c,f,h,w  ; 另一种方式: 对于数据进行改变, 那么上述concat的代码也需要进行变换了...
            #     # new_sample = new_sample.unsqueeze(2) # (2,12,1,67,120)  #! 这里需要change, 要在一输入之前就要更改他的维度, 因此前面concat也需要稍微改一下    不要在forward中增加f维度 (因为要依赖输入)

            #     new_sample = rearrange(new_sample, 'b f c h w -> b c f h w')
            #     kwargs['cross_attention_kwargs'] = {}
            #     # return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)
            #     result = unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)
            #     # return (result[0].squeeze(2),) #! tmp
            #     return (rearrange(result[0], 'b c f h w -> b f c h w'),)
            # unet.forward = hooked_unet_forward
            
            ##### -- 更改 forward函数 --- #####

            # model_path = '/home/fy/Code/instruct-video-to-video/IC-Light/models/iclight_sd15_fbc.safetensors'
            # 这里是加载iclight的lora weight
            sd_offset = load_file(iclight_init_weights)
            sd_origin = unet.state_dict()
            keys = sd_origin.keys()
            for k in sd_offset.keys():
                sd_origin[k] = sd_origin[k] + sd_offset[k]
            # sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
            self.unet.load_state_dict(sd_origin, strict=True)
            del sd_offset, sd_origin, unet, keys

            # print(1)    
            # todo 试写一下iclight unet的加载方式
            # sd = load_file('/home/fy/Code/IC-Light/cache_models/models--stablediffusionapi--realistic-vision-v51/snapshots/19e3643d7d963c156d01537188ec08f0b79a514a/unet/diffusion_pytorch_model.safetensors')

            # debug: print参数
            # with open('logs/sd_keys.txt', 'w') as f:
            #     f.write("SD Keys:\n")
            #     for key in sd_ori.keys():
            #         f.write(f"{key}\n")

            # unet_state_dict = self.unet.state_dict()
            # with open('logs/unet_state_dict_keys.txt', 'w') as f:
            #         f.write("UNet State Dict Keys:\n")
            #         for key in unet_state_dict.keys():
            #             f.write(f"{key}\n")
        else:
            with torch.no_grad():
                new_conv_in = InflatedConv3d(12, self.unet.conv_in.out_channels, self.unet.conv_in.kernel_size, self.unet.conv_in.stride, self.unet.conv_in.padding)
                self.unet.conv_in = new_conv_in

    def configure_optimizers(self): # 决定了仅仅训练motion_module的参数  注意一下pl.Trainer独有的函数
        import bitsandbytes as bnb
        motion_params = []
        remaining_params = []
        train_names = [] # for debug
        for name, p in self.unet.named_parameters():
            if ('motion' in name): #! 哦哦 这里决定了哪些参数用于训练... 这里实际训练的只有motion相关参数
                motion_params.append(p)
                train_names.append(name)
            elif ('attentions' in name):
                motion_params.append(p)
                train_names.append(name)
            else:
                remaining_params.append(p)
        # import pdb; pdb.set_trace()
        optimizer = bnb.optim.Adam8bit([
            {'params': motion_params, 'lr': self.optim_args['lr']},
        ], betas=(0.9, 0.999))
        return optimizer


class InstructP2PVideoTrainerTemporalText(InstructP2PVideoTrainerTemporal):
    def initialize_unet(self, unet_init_weights): # 这边对比上一级来说, 新加的部分在于 rewrite了unet的load函数
        if unet_init_weights is not None:
            print(f'INFO: initialize denoising UNet from {unet_init_weights}')
            sd_init_weights, motion_module_init_weights, iclight_init_weights = unet_init_weights
            if self.base_path:
                sd_init_weights = f"{self.base_path}/{sd_init_weights}"
            if '.safetensors' in sd_init_weights: # .safetensors的加载方式
                sd = load_file(sd_init_weights) #! 关于加载iclight的unet, 后面再加到yaml里面... 我甚至觉得只要改unet, vae和text其实都差不多
            else: #'.ckpt'场景 
                sd = torch.load(sd_init_weights, map_location='cpu') 

            # sd = torch.load(sd_init_weights, map_location='cpu') # 注意debug看看这是啥 + 打印一下原有和加载的keys
            if self.unet.use_motion_module:
                motion_sd = torch.load(motion_module_init_weights, map_location='cpu')
                assert len(sd) + len(motion_sd) == len(self.unet.state_dict()), f'Improper state dict length, got {len(sd) + len(motion_sd)} expected {len(self.unet.state_dict())}' #! 注意一下这行保证了加载的key至少在数量上是对应的; 这行的目的是self.unet是自己定义的 而这两个加载的是别的地方训练的(可能是diffusers中的)
                sd.update(motion_sd)
            
                for k, v in self.unet.state_dict().items():
                    if 'pos_encoder.pe' in k: # 这边是原来iv2v的代码 temporal_position_encoding_max_len, 设置为 32
                        sd[k] = v # the size of pe may change, 主要是temporal layer的size会发生改变... √ 由于输入的max_len变了
                    # if 'conv_in.weight' in k: #! tmp, 这里是test一下
                    #     sd[k] = v
            else:
                assert len(sd) == len(self.unet.state_dict())

            self.unet.load_state_dict(sd) # 为什么这里可以完美适配? √
            # todo: 更改sd的conv_in.weight的shape到12; 更改函数forward, 支持多个输入cond; iclight的sd_offset加载进去; 
            unet = self.unet # saVe一下
            # 这里是更改conv_in的shape; #! 这边注意一下要改成3D版本的unet
            with torch.no_grad():
                # new_conv_in = torch.nn.Conv2d(12, unet.conv_in.out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding)
                new_conv_in = InflatedConv3d(8, unet.conv_in.out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding)
                new_conv_in.weight.zero_()
                new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
                new_conv_in.bias = unet.conv_in.bias
                unet.conv_in = new_conv_in
            

            # model_path = '/home/fy/Code/instruct-video-to-video/IC-Light/models/iclight_sd15_fbc.safetensors'
            # 这里是加载iclight的lora weight
            sd_offset = load_file(iclight_init_weights)
            sd_origin = unet.state_dict()
            keys = sd_origin.keys()
            for k in sd_offset.keys():
                sd_origin[k] = sd_origin[k] + sd_offset[k]
            # sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
            self.unet.load_state_dict(sd_origin, strict=True)
            del sd_offset, sd_origin, unet, keys

        else:
            with torch.no_grad():
                new_conv_in = InflatedConv3d(8, self.unet.conv_in.out_channels, self.unet.conv_in.kernel_size, self.unet.conv_in.stride, self.unet.conv_in.padding)
                self.unet.conv_in = new_conv_in
