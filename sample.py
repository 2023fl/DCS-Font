import os
import cv2
import time
import random
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
from accelerate.utils import set_seed

from src import (FontDiffuserDPMPipeline,
                 FontDiffuserModelDPM,
                 build_ddpm_scheduler,
                 build_unet,
                 build_content_encoder,
                 build_style_encoder)
from utils import (ttf2im,
                   load_ttf,
                   is_char_in_font,
                   save_args_to_yaml,
                   save_single_image,
                   save_image_with_content_style)


def arg_parse():     # 主要目的是设置并解析命令行参数
    from configs.fontdiffuser import get_parser

    parser = get_parser()  # 创建和修改参数解析器
    # 代码为解析器添加了一些新的参数
    # *`--ckpt_dir`: 一个字符串，默认为 `None`。
    # *`--demo`: 一个标志（`action = "store_true" `），表示是否启用演示模式。
    # *`--controlnet`: 一个布尔值，默认为 `False`，当在演示模式下，控制网（controlnet）可以被添加。
    # *`--character_input`: 一个标志，表示是否使用字符输入。
    # *`--content_character`: 一个字符串，默认为 `None`，可能用于指定内容字符。
    # *`--content_image_path`: 一个字符串，默认为 `None`，表示内容图像的路径。
    # *`--style_image_path`: 一个字符串，默认为`None`，表示样式图像的路径。
    # *`--save_image`: 一个标志，表示是否保存图像。
    # *`--save_image_dir`: 一个字符串，默认为 `None`，表示保存图像的目录。
    # *`--device`: 一个字符串，默认为 `"cuda:0"`，表示使用的设备（如GPU）。
    # *`--ttf_path`: 一个字符串，默认为 `"ttf/KaiXinSongA.ttf"`，可能表示字体文件的路径。
    # 原
    # parser.add_argument("--ckpt_dir", type=str, default=None)
    # parser.add_argument("--demo", action="store_true")
    # parser.add_argument("--controlnet", type=bool, default=False,
    #                     help="If in demo mode, the controlnet can be added.")
    # parser.add_argument("--character_input", action="store_true")
    # parser.add_argument("--content_character", type=str, default=None)
    # parser.add_argument("--content_image_path", type=str, default=None)
    # parser.add_argument("--style_image_path", type=str, default=None)
    # parser.add_argument("--save_image", action="store_true")
    # parser.add_argument("--save_image_dir", type=str, default=None,
    #                     help="The saving directory.")
    # parser.add_argument("--device", type=str, default="cuda:0")
    # parser.add_argument("--ttf_path", type=str, default="ttf/KaiXinSongA.ttf")
    # 改
    parser.add_argument("--ckpt_dir", type=str, default='phase_1_ckpt/global_step_400000')
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--controlnet", type=bool, default=False,
                        help="If in demo mode, the controlnet can be added.")
    parser.add_argument("--character_input", action="store_true", default=True)    #  default=True/False是两种不同的加载图片的方式，控制的是271行if args.character_input:
    parser.add_argument("--content_character", type=str, default='仙')
    parser.add_argument("--content_image_path", type=str, default= r'data_examples/sampling/example_content.jpg')     #  D:/FILES/Project/pytorch/Github/FontDiffuser/data_examples/sampling/example_content.jpg
    parser.add_argument("--style_image_path", type=str, default= 'data_examples/train/TargetImage/Coca-ColaCareFontKaiTi/Coca-ColaCareFontKaiTi+仙.jpg')     #  D:/FILES/Project/pytorch/Github/FontDiffuser/data_examples/sampling/example_style.jpg
    parser.add_argument("--save_image", action="store_true",default=True)
    # 原parser.add_argument("--save_image", action="store_true")
    parser.add_argument("--save_image_dir", type=str, default='D:/FILES/Project/pytorch/Github/FontDiffuser/save_image_dir' ,
                        help="The saving directory.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ttf_path", type=str, default="ttf/KaiXinSongA.ttf")

    # 解析参数：这行代码会解析命令行参数并返回一个 Namespace 对象，其中包含了所有解析后的参数值。
    args = parser.parse_args()
    # 处理图像大小参数
    style_image_size = args.style_image_size
    content_image_size = args.content_image_size
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)

    return args

# 根据给定的参数处理内容图像和样式图像
def image_process(args, content_image=None, style_image=None):
    if not args.demo:
        # Read content image and style image读取内容图像和样式图像
        if args.character_input:
            assert args.content_character is not None, "The content_character should not be None."
            if not is_char_in_font(font_path=args.ttf_path, char=args.content_character):
                return None, None
            font = load_ttf(ttf_path=args.ttf_path)     # 如果字符存在于字体中，则加载该字体文件（使用 load_ttf 函数）并生成该字符的图像（使用 ttf2im 函数）
            content_image = ttf2im(font=font, char=args.content_character)
            content_image_pil = content_image.copy()
        else:
            content_image = Image.open(args.content_image_path).convert('RGB')
            # # 自己加的
            # content_image = Image.open('D:/FILES/Project/pytorch/Github/FontDiffuser/data_examples/sampling/example_content.jpg').convert('RGB')

            content_image_pil = None

        style_image = Image.open(args.style_image_path).convert('RGB')
        # # 自己加的
        # style_image = Image.open('D:/FILES/Project/pytorch/Github/FontDiffuser/data_examples/sampling/example_style.jpg').convert('RGB')

    else:
        assert style_image is not None, "The style image should not be None."
        if args.character_input:
            assert args.content_character is not None, "The content_character should not be None."
            if not is_char_in_font(font_path=args.ttf_path, char=args.content_character):
                return None, None
            font = load_ttf(ttf_path=args.ttf_path)
            content_image = ttf2im(font=font, char=args.content_character)
        else:
            assert content_image is not None, "The content image should not be None."
        content_image_pil = None
        
    # Dataset transform!!!!!!!!!!!!对数据集进行预处理的过程，特别是针对内容图像和样式图像的变换
    # content_inference_transforms = transforms.Compose(
    #     [transforms.Resize(args.content_image_size, \
    #                         interpolation=transforms.InterpolationMode.BILINEAR),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.5], [0.5])])       # 将图像数据缩放到[-1, 1]范围
    # style_inference_transforms = transforms.Compose(
    #     [transforms.Resize(args.style_image_size, \
    #                        interpolation=transforms.InterpolationMode.BILINEAR),
    #      transforms.ToTensor(),
    #      transforms.Normalize([0.5], [0.5])])
    # content_image = content_inference_transforms(content_image)[None, :]
    # style_image = style_inference_transforms(style_image)[None, :]
    #
    # return content_image, style_image, content_image_pil


    # 定义一个组合的图像预处理流程，用于内容图像的推理
    content_inference_transforms = transforms.Compose(
        # 使用transforms.Compose将多个预处理步骤组合在一起
        [
            # 使用transforms.Resize将图像的大小调整为args.content_image_size指定的尺寸
            # interpolation参数指定了插值方法，这里使用双线性插值（BILINEAR）
            transforms.Resize(args.content_image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            # 使用transforms.ToTensor将PIL图像或numpy.ndarray转换为PyTorch张量
            # 该转换会将像素值从[0, 255]的整数范围缩放到[0.0, 1.0]的浮点数范围
            transforms.ToTensor(),
            # 使用transforms.Normalize对图像张量进行标准化
            # 这里通过减去均值[0.5]并除以标准差[0.5]，将值从[0.0, 1.0]范围缩放到[-1, 1]范围
            transforms.Normalize([0.5], [0.5])  # 将图像数据缩放到[-1, 1]范围
        ]
    )
    # 定义一个组合的图像预处理流程，用于风格图像的推理
    style_inference_transforms = transforms.Compose(
        [
            # 使用transforms.Resize将图像的大小调整为args.style_image_size指定的尺寸
            # 同样使用双线性插值（BILINEAR）
            transforms.Resize(args.style_image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            # 使用transforms.ToTensor将PIL图像或numpy.ndarray转换为PyTorch张量
            transforms.ToTensor(),
            # 使用transforms.Normalize对图像张量进行标准化
            # 同样将值从[0.0, 1.0]范围缩放到[-1, 1]范围
            transforms.Normalize([0.5], [0.5])
        ]
    )
    # 对内容图像应用前面定义的预处理流程
    # 并将处理后的张量增加一个维度以匹配PyTorch的batch维度（即在张量前增加一个维度，表示batch size为1）
    content_image = content_inference_transforms(content_image)[None, :]
    # 对风格图像应用前面定义的预处理流程
    # 同样增加一个维度以匹配PyTorch的batch维度
    style_image = style_inference_transforms(style_image)[None, :]
    # 返回处理后的内容图像张量、风格图像张量以及原始的content_image_pil PIL图像对象
    # content_image_pil可能是用于后续显示或保存的原始图像
    return content_image, style_image, content_image_pil


# 核心
def load_fontdiffuser_pipeline(args):

    # Load the model state_dict
    # 加载模型权重：
    # 使用 build_unet 函数（可能是自定义的）构建了一个U - Net 结构的模型，并命名为unet。
    # 使用 torch.load从指定路径加载U - Net的权重，并使用 load_state_dict方法将其加载到unet模型中。
    # 类似地，加载了风格编码器（style_encoder）和内容编码器（content_encoder）的权重。
    # 将这三个组件组合成一个 FontDiffuserModelDPM类型的模型，命名为model。
    # 使用 model.to(args.device)将模型移动到指定的设备（如CPU或 GPU）上。
    # unet = build_unet(args=args)
    # unet.load_state_dict(torch.load(os.path.join('D:/FILES/Project/pytorch/Github/FontDiffuser/ckpt_dir', 'unet.pth')))      #  原代吗：unet.load_state_dict(torch.load(f"{args.ckpt_dir}/unet.pth"))
    # style_encoder = build_style_encoder(args=args)
    # style_encoder.load_state_dict(torch.load(f"{'D:/FILES/Project/pytorch/Github/FontDiffuser/ckpt_dir'}/style_encoder.pth"))   # 原 style_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/style_encoder.pth"))
    # content_encoder = build_content_encoder(args=args)
    # content_encoder.load_state_dict(torch.load(f"{'D:/FILES/Project/pytorch/Github/FontDiffuser/ckpt_dir'}/content_encoder.pth"))    # content_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/content_encoder.pth"))
    unet = build_unet(args=args)
    unet.load_state_dict(torch.load(f"{args.ckpt_dir}/unet.pth"))
    style_encoder = build_style_encoder(args=args)
    style_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/style_encoder.pth"))
    content_encoder = build_content_encoder(args=args)
    content_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/content_encoder.pth"))

    model = FontDiffuserModelDPM(
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder)
    model.to(args.device)   # 使用 model.to(args.device)将模型移动到指定的设备（如CPU或 GPU）上。
    print("Loaded the model state_dict successfully!")

    # Load the training ddpm_scheduler.加载训练调度器
    train_scheduler = build_ddpm_scheduler(args=args)
    print("Loaded training DDPM scheduler sucessfully!")

    # Load the DPM_Solver to generate the sample.加载扩散概率模型（DPM）管道
    pipe = FontDiffuserDPMPipeline(
        model=model,
        ddpm_train_scheduler=train_scheduler,
        model_type=args.model_type,
        guidance_type=args.guidance_type,
        guidance_scale=args.guidance_scale,
    )
    print("Loaded dpm_solver pipeline sucessfully!")

    return pipe

# 核心！！！！！！
def sampling(args, pipe, content_image=None, style_image=None):
    # print(args.demo)
    if not args.demo:
        # args.save_image_dir = 'D:/FILES/Project/pytorch/Github/FontDiffuser/save_image_dir'  # 自己加的
        os.makedirs(args.save_image_dir, exist_ok=True)
        # saving sampling config保存采样配置
        save_args_to_yaml(args=args, output_file=f"{args.save_image_dir}/sampling_config.yaml")  # 创建一个保存图像的目录，并将配置信息保存为一个 YAML 文件

    if args.seed:
        set_seed(seed=args.seed)  # 设置随机种子: 如果提供了种子（args.seed），则设置随机种子以确保结果的可重复性
    # 处理图像: 使用 image_process 函数处理内容图像和风格图像
    content_image, style_image, content_image_pil = image_process(args=args,
                                                                  content_image=content_image,
                                                                  style_image=style_image)
    # 自己改的
    # content_image, style_image, content_image_pil = image_process(args=args,
    #                                                               content_image='D:/FILES/Project/pytorch/Github/FontDiffuser/data_examples/sampling/example_content.jpg',
    #                                                               style_image='D:/FILES/Project/pytorch/Github/FontDiffuser/data_examples/sampling/example_style.jpg')
    if content_image == None:
        print(f"The content_character you provided is not in the ttf. \
                Please change the content_character or you can change the ttf.")
        return None

    with torch.no_grad():
        # 将图像移动到指定设备
        content_image = content_image.to(args.device)
        style_image = style_image.to(args.device)
        print(f"Sampling by DPM-Solver++ ......")   # 打印一条消息，表明正在使用 "DPM-Solver++" 方法进行采样。这是一个特定的风格迁移或图像生成算法
        start = time.time()   # 记录开始时间
        # 图像生成
        # 调用生成函数：images = pipe.generate(...)调用 pipe  对象的generate 方法来生成图像。
        # 这个方法接收很多参数，包括内容图像、风格图像、批量大小、算法的各种配置选项等。
        images = pipe.generate(
            content_images=content_image,
            style_images=style_image,
            batch_size=1,
            order=args.order,
            num_inference_step=args.num_inference_steps,
            content_encoder_downsample_size=args.content_encoder_downsample_size,
            t_start=args.t_start,
            t_end=args.t_end,
            dm_size=args.content_image_size,
            algorithm_type=args.algorithm_type,
            skip_type=args.skip_type,
            method=args.method,
            correcting_x0_fn=args.correcting_x0_fn)
        end = time.time()   # 记录结束时间

        if args.save_image:
            print(f"Saving the image ......")
            save_single_image(save_dir=args.save_image_dir, image=images[0])    # 保存图像到指定的目录（args.save_image_dir）

            if args.character_input:
                # 保存带有内容和样式的图像（字符输入模式）   # 从这里跳入找到load_ttf、ttf2im修改fsize可调整内容图片输出的大小
                save_image_with_content_style(save_dir=args.save_image_dir,
                                            image=images[0],
                                            content_image_pil=content_image_pil,
                                            content_image_path=None,
                                            style_image_path=args.style_image_path,
                                            resolution=args.resolution)
            else:
                save_image_with_content_style(save_dir=args.save_image_dir,
                                            image=images[0],
                                            content_image_pil=None,
                                            content_image_path=args.content_image_path,
                                            style_image_path=args.style_image_path,
                                            resolution=args.resolution)
            print(f"Finish the sampling process, costing time {end - start}s")   # 用户采样过程已完成，并显示所花费的时间
        return images[0]   # 返回第一个图像


# def load_controlnet_pipeline(args,
#                              config_path="lllyasviel/sd-controlnet-canny",
#                              ckpt_path="runwayml/stable-diffusion-v1-5"):
#     from diffusers import ControlNetModel, AutoencoderKL
#     # load controlnet model and pipeline           # 导入必要的库和模型
#     from diffusers import StableDiffusionControlNetPipeline, UniPCMultistepScheduler
#     # 加载ControlNet模型
#     controlnet = ControlNetModel.from_pretrained(config_path,
#                                                  torch_dtype=torch.float16,
#                                                  cache_dir=f"{args.ckpt_dir}/controlnet")
#     print(f"Loaded ControlNet Model Successfully!")
#     # 加载Stable Diffusion ControlNet Pipeline
#     pipe = StableDiffusionControlNetPipeline.from_pretrained(ckpt_path,
#                                                              controlnet=controlnet,
#                                                              torch_dtype=torch.float16,
#                                                              cache_dir=f"{args.ckpt_dir}/controlnet_pipeline")
#     # faster优化和启用CPU卸载，会使生成过程更快或更稳定
#     pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
#     pipe.enable_model_cpu_offload()
#     print(f"Loaded ControlNet Pipeline Successfully!")
#
#     return pipe




def load_controlnet_pipeline(args,
                             config_path="lllyasviel/sd-controlnet-canny",
                             ckpt_path="runwayml/stable-diffusion-v1-5"):
    # 导入必要的库和模型，包括ControlNetModel和AutoencoderKL
    from diffusers import ControlNetModel, AutoencoderKL
    # 导入StableDiffusionControlNetPipeline和UniPCMultistepScheduler，用于加载和配置ControlNet Pipeline
    from diffusers import StableDiffusionControlNetPipeline, UniPCMultistepScheduler
    # 加载ControlNet模型，通过指定配置文件路径（config_path）、数据类型（torch.float16）和缓存目录（args.ckpt_dir/controlnet）
    controlnet = ControlNetModel.from_pretrained(config_path,
                                                 torch_dtype=torch.float16,
                                                 cache_dir=f"{args.ckpt_dir}/controlnet")
    # 打印成功加载ControlNet模型的提示信息
    print(f"Loaded ControlNet Model Successfully!")
    # 加载Stable Diffusion ControlNet Pipeline，通过指定预训练模型路径（ckpt_path）、已加载的ControlNet模型（controlnet）、数据类型（torch.float16）和缓存目录（args.ckpt_dir/controlnet_pipeline）
    pipe = StableDiffusionControlNetPipeline.from_pretrained(ckpt_path,
                                                             controlnet=controlnet,
                                                             torch_dtype=torch.float16,
                                                             cache_dir=f"{args.ckpt_dir}/controlnet_pipeline")
    # 配置pipeline的scheduler为UniPCMultistepScheduler，用于更快或更稳定的生成过程
    # 这里通过从scheduler的配置中创建新的UniPCMultistepScheduler实例来实现
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # 启用模型CPU卸载，当GPU内存不足时，部分模型可以卸载到CPU上，以节省GPU内存
    pipe.enable_model_cpu_offload()
    # 打印成功加载ControlNet Pipeline的提示信息
    print(f"Loaded ControlNet Pipeline Successfully!")
    # 返回加载好的ControlNet Pipeline
    return pipe

# 旨在使用ControlNet和Stable Diffusion模型来生成一个根据文本提示和给定图像边缘（使用Canny边缘检测）控制的图像
def controlnet(text_prompt, 
               pil_image,
               pipe):
    image = np.array(pil_image)
    # get canny image
    image = cv2.Canny(image=image, threshold1=100, threshold2=200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    
    seed = random.randint(0, 10000)
    generator = torch.manual_seed(seed)
    image = pipe(text_prompt, 
                 num_inference_steps=50, 
                 generator=generator, 
                 image=canny_image,
                 output_type='pil').images[0]
    return image

# 加载预训练的StableDiffusionInstructPix2PixPipeline模型
def load_instructpix2pix_pipeline(args,
                                  ckpt_path="timbrooks/instruct-pix2pix"):
    from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(ckpt_path, 
                                                                  torch_dtype=torch.float16)
    pipe.to(args.device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    return pipe
# 使用给定的文本提示和输入图像，通过StableDiffusionInstructPix2PixPipeline模型生成新的图像
def instructpix2pix(pil_image, text_prompt, pipe):
    image = pil_image.resize((512, 512))
    seed = random.randint(0, 10000)
    generator = torch.manual_seed(seed)
    image = pipe(prompt=text_prompt, image=image, generator=generator, 
                 num_inference_steps=20, image_guidance_scale=1.1).images[0]

    return image




if __name__=="__main__":
    args = arg_parse()
    # args.ckpt_dir = "checkpoint"
    # args.ckpt_dir = "ckpt_dir"
    # print(args.ckpt_dir)
    print(os.getcwd())
    abs_file = os.path.abspath(__file__)  # 获取model.py文件的绝对路径
    # 找到绝对路径的同级目录
    abs_dir = abs_file[:abs_file.rfind('\\')] if os.name == 'nt' else abs_file[:abs_file.rfind(r'/')]
    # 构造模型文件的绝对路径
    model_dir = os.path.join(abs_dir, 'checkpoints/unet.pth')

    # unet_path = 'D:/FILES/Project/pytorch/Github/FontDiffuser/checkpoints/unet.pth'   # 自己加的
    # result = f"FontDiffuser: {FontDiffuser}"
    # load fontdiffuser pipeline
    pipe = load_fontdiffuser_pipeline(args=args)
    out_image = sampling(args=args, pipe=pipe)
