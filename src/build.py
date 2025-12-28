from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from src import (ContentEncoder, 
                 StyleEncoder, 
                 UNet,
                 SCR)




# def build_unet(args):
#     # 使用传入的参数来构建一个UNet模型实例
#     # 这里的参数可能是从命令行、配置文件或其他方式获取的
#     # 创建一个UNet模型实例
#     unet = UNet(
#         # 输入图像的分辨率（通常是宽和高）
#         sample_size=args.resolution,
#         # 输入图像的通道数，对于RGB图像通常是3
#         in_channels=3,
#         # 输出图像的通道数，对于RGB图像也是3（尽管在某些应用中可能不同）
#         out_channels=3,
#         # 一个布尔值，决定是否将正弦位置编码转换为余弦位置编码
#         flip_sin_to_cos=True,
#         # 频率偏移参数（可能用于位置编码的某种变换）
#         freq_shift=0,
#         # 定义下采样块类型的列表。这里使用了四种不同的下采样块
#         down_block_types=(
#             'DownBlock2D',  # 普通的下采样块
#             'MCADownBlock2D',  # 可能具有某种多通道注意力机制的下采样块
#             'MCADownBlock2D',  # 同上
#             'DownBlock2D'  # 普通的下采样块
#         ),
#         # 定义上采样块类型的列表。这里也使用了四种不同的上采样块
#         up_block_types=(
#             'UpBlock2D',  # 普通的上采样块
#             'StyleRSIUpBlock2D',  # 可能具有某种风格残差注入机制的上采样块
#             'StyleRSIUpBlock2D',  # 同上
#             'UpBlock2D'  # 普通的上采样块
#         ),
#         # 每个块（下采样或上采样）中的通道数
#         block_out_channels=args.unet_channels,
#         # 每个块中的层数
#         layers_per_block=2,
#         # 下采样时的填充大小
#         downsample_padding=1,
#         # 中间块的缩放因子（可能是为了调整中间层的特征图大小）
#         mid_block_scale_factor=1,
#         # 激活函数，这里使用的是'silu'（也被称为Swish激活函数）
#         act_fn='silu',
#         # 归一化层中的组数（用于组归一化）
#         norm_num_groups=32,
#         # 归一化层中的epsilon值（用于避免除以零）
#         norm_eps=1e-05,
#         # 交叉注意力机制的维度（可能与风格编码相关）
#         cross_attention_dim=args.style_start_channel * 16,
#         # 注意力机制的头部维度
#         attention_head_dim=1,
#         # 是否使用通道注意力机制
#         channel_attn=args.channel_attn,
#         # 内容编码器的下采样大小（可能是为了匹配某种风格编码器的尺寸）
#         content_encoder_downsample_size=args.content_encoder_downsample_size,
#         # 内容编码器开始时的通道数
#         content_start_channel=args.content_start_channel,
#         # 某个操作的缩减因子（可能是为了降低维度或复杂度）
#         reduction=32
#     )
#     # 返回构建的UNet模型实例
#     return unet

def build_unet(args):
    # 创建一个UNet模型实例，该实例的配置通过参数args来指定
    unet = UNet(
        # 指定输入图像的大小
        sample_size=args.resolution,
        # 输入通道数，对于RGB图像为3
        in_channels=3,
        # 输出通道数，这里也设置为3，可能用于多通道输出或特定任务（如图像重建）
        out_channels=3,
        # 一个布尔值，控制是否将正弦位置编码翻转为正弦和余弦的组合，可能用于增强位置信息的表示
        flip_sin_to_cos=True,
        # 频率偏移量，用于调整位置编码的频率，这里设置为0表示不偏移
        freq_shift=0,
        # 定义下采样块（DownBlock）的类型序列，这里混合使用了普通的DownBlock和MCADownBlock
        down_block_types=('DownBlock2D',
                          'MCADownBlock2D',
                          'MCADownBlock2D',
                          'DownBlock2D'),
        # 定义上采样块（UpBlock）的类型序列，这里混合使用了普通的UpBlock和StyleRSIUpBlock
        up_block_types=('UpBlock2D',
                        'StyleRSIUpBlock2D',
                        'StyleRSIUpBlock2D',
                        'UpBlock2D'),
        # 每个块的输出通道数，通过args.unet_channels指定
        block_out_channels=args.unet_channels,
        # 每个块中的层数，这里设置为2
        layers_per_block=2,
        # 下采样时使用的填充量，这里设置为1
        downsample_padding=1,
        # 中间块的缩放因子，这里设置为1，表示不进行缩放
        mid_block_scale_factor=1,
        # 激活函数类型，这里使用'silu'，即Sigmoid Linear Unit，一种平滑的ReLU变体
        act_fn='silu',
        # 归一化层的组数，用于组归一化（Group Normalization），这里设置为32
        norm_num_groups=32,
        # 归一化层的epsilon值，用于防止除以零的错误，这里设置为1e-05
        norm_eps=1e-05,
        # 交叉注意力机制的维度，这里设置为args.style_start_channel * 16
        cross_attention_dim=args.style_start_channel * 16,
        # 注意力机制的头部维度，这里设置为1
        attention_head_dim=1,
        # 是否使用通道注意力机制，通过args.channel_attn指定
        channel_attn=args.channel_attn,
        # 内容编码器下采样的目标大小，通过args.content_encoder_downsample_size指定
        content_encoder_downsample_size=args.content_encoder_downsample_size,
        # 内容编码器的起始通道数，通过args.content_start_channel指定
        content_start_channel=args.content_start_channel,
        # 用于减少特征维度的因子，这里设置为32
        reduction=32
    )
    # 返回构建好的UNet模型实例
    return unet

# def build_unet(args):
#     unet = UNet(
#         sample_size=args.resolution,
#         in_channels=3,
#         out_channels=3,
#         flip_sin_to_cos=True,
#         freq_shift=0,
#         down_block_types=('DownBlock2D',
#                           'MCADownBlock2D',
#                           'MCADownBlock2D',
#                           'DownBlock2D'),
#         up_block_types=('UpBlock2D',
#                         'StyleRSIUpBlock2D',
#                         'StyleRSIUpBlock2D',
#                         'UpBlock2D'),
#         block_out_channels=args.unet_channels,
#         layers_per_block=2,
#         downsample_padding=1,
#         mid_block_scale_factor=1,
#         act_fn='silu',
#         norm_num_groups=32,
#         norm_eps=1e-05,
#         cross_attention_dim=args.style_start_channel * 16,
#         attention_head_dim=1,
#         channel_attn=args.channel_attn,
#         content_encoder_downsample_size=args.content_encoder_downsample_size,
#         content_start_channel=args.content_start_channel,
#         reduction=32)
#
#     return unet



def build_style_encoder(args):
    # 定义一个函数，用于构建风格编码器，该函数接受一个参数args，通常是一个包含各种配置选项的对象
    # 创建一个StyleEncoder的实例，这是一个用于编码风格图像的神经网络模型
    # G_ch参数指定了生成器网络中初始的通道数（channels），这通常与模型的复杂度和图像大小有关
    # resolution参数指定了输入图像的高度（或宽度，假设输入是正方形），这里使用args.style_image_size[0]来获取
    style_image_encoder = StyleEncoder(
        G_ch=args.style_start_channel,
        resolution=args.style_image_size[0])
    # 打印一条消息，表明已经获取了CG-GAN的风格编码器
    # 这有助于在调试或运行代码时了解模型的构建进度
    print("Get CG-GAN Style Encoder!")
    # 返回构建好的风格编码器实例，以便后续在代码中使用
    return style_image_encoder


def build_content_encoder(args):
    content_image_encoder = ContentEncoder(
        G_ch=args.content_start_channel,
        resolution=args.content_image_size[0])
    print("Get CG-GAN Content Encoder!")
    return content_image_encoder


def build_scr(args):
    scr = SCR(
        temperature=args.temperature,
        mode=args.mode,
        image_size=args.scr_image_size)
    print("Loaded SCR module for supervision successfully!")
    return scr


def build_ddpm_scheduler(args):
    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,      # 用于控制噪声添加过程的β值的起始和结束值
        beta_schedule=args.beta_scheduler,
        trained_betas=None,
        variance_type="fixed_small",
        clip_sample=True)
    return ddpm_scheduler

# 参数解释：
# num_train_timesteps=1000：这个参数指定了训练过程中使用的总时间步数（timesteps）。在DDPM中，数据通过逐步添加噪声（前向过程）和逐步去噪（反向过程）来学习，num_train_timesteps 定义了反向过程中每一步的迭代次数。
# beta_start=0.0001 和 beta_end=0.02：这两个参数定义了用于控制噪声添加过程的β值的起始和结束值。β值决定了每一步添加的噪声量，β值越大，添加的噪声越多。在训练开始时，噪声添加量较小，随着训练的进行，噪声添加量逐渐增加。
# beta_schedule=args.beta_scheduler：这个参数指定了β值从beta_start变化到beta_end的方式。args.beta_scheduler可能是一个字符串，指示了具体的调度策略（如线性、余弦等），或者是直接提供了一个预定义的β值序列。
# trained_betas=None：这个参数在当前的函数实现中被设置为None，意味着不使用预训练的β值。在某些情况下，如果已经有了通过其他方式训练得到的β值，可以通过这个参数传入。
# variance_type="fixed_small"：这个参数定义了反向过程中用于预测噪声的方差类型。"fixed_small" 表示使用固定的较小方差，这有助于模型的稳定性。
# clip_sample=True：这个参数指定了在采样过程中是否对预测值进行裁剪。裁剪可以帮助确保预测值在合理的范围内，有助于提高生成样本的质量。
# 函数功能：
# 函数build_ddpm_scheduler根据提供的参数构建了一个DDPMScheduler对象，该对象封装了DDPM训练过程中的β值调度逻辑、方差类型等关键配置。这个调度器是DDPM训练过程中的重要组成部分，它决定了训练过程中噪声的添加和去除策略，从而影响了模型的性能。