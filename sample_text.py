# #  修改了samply代码和utils代码  版本1可用  batch_size = 8  # 每次处理的最大字符数量，可以根据显存调整
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
from utils_fyy import (ttf2im,
                       load_ttf,
                       is_char_in_font,
                       save_args_to_yaml,
                       save_single_image,
                       save_image_with_content_style)


def arg_parse():
    from configs.fontdiffuser import get_parser
    parser = get_parser()
    parser.add_argument("--ckpt_dir", type=str, default='ckpt_dir\safm_ckpt\global_step_450000')
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--controlnet", type=bool, default=False)
    parser.add_argument("--character_input", action="store_true", default=True)
    parser.add_argument("--content_character_file", type=str, default='content_characters.txt',
                        help="Path to the text file containing content characters")
    parser.add_argument("--content_image_path", type=str, default='data_examples/sampling/example_content.jpg')
    parser.add_argument("--style_image_dir", type=str,
                        default=r'E:\Files\Project-E\MMFont-01\data_examples\train\TargetImage',
                        help="Base directory containing all style font directories")
    parser.add_argument("--save_image", action="store_true", default=True)
    parser.add_argument("--save_image_dir", type=str, default='E:\Files\Project-E\index_text\output\save_safm_45\save_image_dir')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ttf_path", type=str, default="ttf/KaiXinSongA.ttf")

    args = parser.parse_args()

    # 从文本文件中读取字符列表
    if os.path.exists(args.content_character_file):
        with open(args.content_character_file, 'r', encoding='utf-8') as file:
            args.content_character = file.read().strip()
    else:
        raise FileNotFoundError(f"Character file {args.content_character_file} not found.")

    style_image_size = args.style_image_size
    content_image_size = args.content_image_size
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)

    return args


def image_process(args, content_image=None, style_image=None):
    content_images = []
    style_images = []
    content_image_pils = []

    if not args.demo:
        # 处理每个字符
        for char in args.content_character:
            if args.character_input:
                if not is_char_in_font(font_path=args.ttf_path, char=char):
                    continue
                font = load_ttf(ttf_path=args.ttf_path)
                content_image = ttf2im(font=font, char=char)
                content_image_pil = content_image.copy()

                # 构建对应的style图片路径
                style_font_name = os.path.basename(args.style_image_dir.rstrip('/'))
                style_path = os.path.join(args.style_image_dir, f"{style_font_name}+{char}.jpg")
                if not os.path.exists(style_path):
                    continue
                style_image = Image.open(style_path).convert('RGB')

                # 应用transforms
                content_inference_transforms = transforms.Compose([
                    transforms.Resize(args.content_image_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5])
                ])

                style_inference_transforms = transforms.Compose([
                    transforms.Resize(args.style_image_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5])
                ])

                content_image = content_inference_transforms(content_image)
                style_image = style_inference_transforms(style_image)

                content_images.append(content_image)
                style_images.append(style_image)
                content_image_pils.append(content_image_pil)

    # 将列表转换为batch
    content_images = torch.stack(content_images)
    style_images = torch.stack(style_images)

    return content_images, style_images, content_image_pils


def load_fontdiffuser_pipeline(args):
    unet = build_unet(args=args)
    unet.load_state_dict(torch.load(f"{args.ckpt_dir}/unet.pth"))
    style_encoder = build_style_encoder(args=args)
    # style_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/style_encoder.pth"), strict=False)
    style_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/style_encoder.pth"))
    content_encoder = build_content_encoder(args=args)
    content_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/content_encoder.pth"))
    # content_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/content_encoder.pth"), strict=False)

    model = FontDiffuserModelDPM(
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder)
    model.to(args.device)
    print("Loaded the model state_dict successfully!")

    train_scheduler = build_ddpm_scheduler(args=args)
    print("Loaded training DDPM scheduler successfully!")

    pipe = FontDiffuserDPMPipeline(
        model=model,
        ddpm_train_scheduler=train_scheduler,
        model_type=args.model_type,
        guidance_type=args.guidance_type,
        guidance_scale=args.guidance_scale,
    )
    print("Loaded dpm_solver pipeline successfully!")

    return pipe


def sampling(args, pipe, content_image=None, style_image=None):
    if not args.demo:
        # 从style_image_dir路径中提取字体名称
        style_font_name = os.path.basename(args.style_image_dir.rstrip('/'))

        # 在save_image_dir下创建以字体名称命名的目录
        font_save_dir = os.path.join(args.save_image_dir, style_font_name)
        os.makedirs(font_save_dir, exist_ok=True)

        # 创建生成图片和参考图片的集中目录
        gen_dir = os.path.join(font_save_dir, f"gen-{style_font_name}")
        ref_dir = os.path.join(font_save_dir, f"ref-{style_font_name}")
        os.makedirs(gen_dir, exist_ok=True)
        os.makedirs(ref_dir, exist_ok=True)

        # 保存配置文件到字体目录下
        save_args_to_yaml(args=args, output_file=f"{font_save_dir}/sampling_config.yaml")

    if args.seed:
        set_seed(seed=args.seed)

    # 分批处理字符数据
    batch_size = 8  # 每次处理的最大字符数量，可以根据显存调整
    content_characters = args.content_character
    total_batches = (len(content_characters) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(content_characters))
        batch_characters = content_characters[start_idx:end_idx]

        print(f"Processing batch {batch_idx + 1}/{total_batches} with characters: {batch_characters}")

        # 临时替换 args.content_character 以处理当前批次
        args.content_character = batch_characters
        content_images, style_images, content_image_pils = image_process(args=args,
                                                                         content_image=content_image,
                                                                         style_image=style_image)

        if content_images is None or style_images is None:
            print(f"No valid images in batch {batch_idx + 1}. Skipping.")
            continue

        with torch.no_grad():
            content_images = content_images.to(args.device)
            style_images = style_images.to(args.device)

            print(f"Sampling by DPM-Solver++ for batch {batch_idx + 1} ......")
            start = time.time()

            images = pipe.generate(
                content_images=content_images,
                style_images=style_images,
                batch_size=len(batch_characters),
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

            end = time.time()

            if args.save_image:
                print(f"Saving the images for batch {batch_idx + 1} ......")
                for idx, (image, char) in enumerate(zip(images, batch_characters)):
                    # 在字体目录下为每个字符创建子目录
                    char_save_dir = os.path.join(font_save_dir, char)
                    os.makedirs(char_save_dir, exist_ok=True)

                    # 生成文件名
                    gen_filename = f"gen-{style_font_name}-{char}.png"
                    ref_filename = f"ref-{style_font_name}-{char}.png"

                    # 1. 保存生成的图像到字符目录
                    save_single_image(save_dir=char_save_dir,
                                      image=image,
                                      filename=gen_filename)

                    # 同时保存到生成图片集中目录
                    save_single_image(save_dir=gen_dir,
                                      image=image,
                                      filename=gen_filename)

                    # 2. 复制参考样式图片
                    style_src = os.path.join(args.style_image_dir, f"{style_font_name}+{char}.jpg")
                    if os.path.exists(style_src):
                        # 保存到字符目录
                        save_single_image(save_dir=char_save_dir,
                                          image=Image.open(style_src).convert('RGB'),
                                          filename=ref_filename)

                        # 同时保存到参考图片集中目录
                        save_single_image(save_dir=ref_dir,
                                          image=Image.open(style_src).convert('RGB'),
                                          filename=ref_filename)

                    # 3. 保存组合图片
                    if args.character_input:
                        save_image_with_content_style(
                            save_dir=char_save_dir,
                            image=image,
                            content_image_pil=content_image_pils[idx],
                            content_image_path=None,
                            style_image_path=style_src,
                            resolution=args.resolution,
                            filename=f"out_with_cs-{char}.jpg"
                        )

                print(f"Finished processing batch {batch_idx + 1}/{total_batches}, costing time {end - start}s.")

    # 恢复 args.content_character
    args.content_character = content_characters



def get_style_fonts(base_dir):
    """获取所有风格字体目录"""
    return [d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))]


def process_single_style(args, pipe, style_font_name, base_style_dir):
    """处理单个风格"""
    print(f"\nProcessing style font: {style_font_name}")

    # 更新当前风格的图片目录
    args.style_image_dir = os.path.join(base_style_dir, style_font_name)

    try:
        # 对当前风格进行采样
        images = sampling(args=args, pipe=pipe)
        print(f"Successfully processed style: {style_font_name}")
        return images
    except Exception as e:
        print(f"Error processing style {style_font_name}: {str(e)}")
        return None


def main():
    # 1. 基础设置
    args = arg_parse()

    # 保存原始的style_image_dir作为基础目录
    base_style_dir = args.style_image_dir

    # 2. 加载模型和pipeline
    pipe = load_fontdiffuser_pipeline(args=args)

    # 3. 获取所有风格字体
    style_fonts = get_style_fonts(base_style_dir)
    print(f"Found {len(style_fonts)} style fonts: {style_fonts}")

    # 4. 处理每个风格
    for style_font_name in style_fonts:
        process_single_style(args, pipe, style_font_name, base_style_dir)


if __name__ == "__main__":
    main()
