import os
import cv2
import yaml
import copy
import pygame
import numpy as np
from PIL import Image
from fontTools.ttLib import TTFont

import torch
import torchvision.transforms as transforms

def save_args_to_yaml(args, output_file):
    # Convert args namespace to a dictionary
    args_dict = vars(args)

    # Write the dictionary to a YAML file
    with open(output_file, 'w') as yaml_file:
        yaml.dump(args_dict, yaml_file, default_flow_style=False)


def save_single_image(save_dir, image):

    save_path = f"{save_dir}/out_single.png"
    image.save(save_path)


def save_image_with_content_style(save_dir, image, content_image_pil, content_image_path, style_image_path, resolution):
    
    new_image = Image.new('RGB', (resolution*3, resolution))
    if content_image_pil is not None:
        content_image = content_image_pil
    else:
        content_image = Image.open(content_image_path).convert("RGB").resize((resolution, resolution), Image.BILINEAR)
    style_image = Image.open(style_image_path).convert("RGB").resize((resolution, resolution), Image.BILINEAR)

    new_image.paste(content_image, (0, 0))
    new_image.paste(style_image, (resolution, 0))
    new_image.paste(image, (resolution*2, 0))

    save_path = f"{save_dir}/out_with_cs.jpg"
    new_image.save(save_path)


def x0_from_epsilon(scheduler, noise_pred, x_t, timesteps):
    """Return the x_0 from epsilon
    """
    batch_size = noise_pred.shape[0]
    for i in range(batch_size):
        noise_pred_i = noise_pred[i]
        noise_pred_i = noise_pred_i[None, :]
        t = timesteps[i]
        x_t_i = x_t[i]
        x_t_i = x_t_i[None, :]

        pred_original_sample_i = scheduler.step(
            model_output=noise_pred_i,
            timestep=t,
            sample=x_t_i,
            # predict_epsilon=True,
            generator=None,
            return_dict=True,
        ).pred_original_sample
        if i == 0:
            pred_original_sample = pred_original_sample_i
        else:
            pred_original_sample = torch.cat((pred_original_sample, pred_original_sample_i), dim=0)

    return pred_original_sample


def reNormalize_img(pred_original_sample):
    pred_original_sample = (pred_original_sample / 2 + 0.5).clamp(0, 1)
    
    return pred_original_sample


def normalize_mean_std(image):
    transforms_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = transforms_norm(image)

    return image


def is_char_in_font(font_path, char):
    TTFont_font = TTFont(font_path)
    cmap = TTFont_font['cmap']
    for subtable in cmap.tables:
        if ord(char) in subtable.cmap:
            return True
    return False


def load_ttf(ttf_path, fsize=85):       # fsize=100：这是 load_ttf 函数的默认字体大小参数。如果您在调用该函数时没有指定字体大小，它将默认为 100。
    pygame.init()

    font = pygame.freetype.Font(ttf_path, size=fsize)
    return font


# def ttf2im(font, char, fsize=100):     #   将一个字符渲染到图像上，定义一个函数，它接受一个字体对象font，一个字符char，和一个可选的字体大小fsize
#
#     try:
#         surface, _ = font.render(char)
#     except:
#         print("No glyph for char {}".format(char))
#         return
#     bg = np.full((fsize, fsize), 255)    #   创建背景图像 bg 创建了一个 fsize x fsize 的白色（因为255是白色在8位RGB中的值）背景图像。
#
#     imo = pygame.surfarray.pixels_alpha(surface).transpose(1, 0)     # 处理像素数据这行代码试图获取表面的alpha通道（透明度）
#     imo = 255 - np.array(Image.fromarray(imo))
#     im = copy.deepcopy(bg)
#
#     h, w = imo.shape[:2]
#     if h > fsize:
#         h, w = fsize, round(w*fsize/h)
#         imo = cv2.resize(imo, (w, h))
#     if w > fsize:
#         h, w = round(h*fsize/w), fsize
#         imo = cv2.resize(imo, (w, h))
#     x, y = round((fsize-w)/2), round((fsize-h)/2)
#     im[y:h+y, x:x+w] = imo
#     pil_im = Image.fromarray(im.astype('uint8')).convert('RGB')
#
#     return pil_im




def ttf2im(font, char, fsize=100):  # 将一个字符使用指定的字体和大小渲染到图像上.画布的大小
    try:
        # 尝试使用给定的字体和字符渲染文本，但注意这里假设font对象有render方法
        # 这在pygame的标准font模块中是成立的，但在freetype模块中则不同
        surface, _ = font.render(char)
    except:
        # 如果渲染失败（例如，字符在字体中不存在），则打印错误信息并返回
        print("No glyph for char {}".format(char))
        return
        # 创建一个fsize x fsize大小的白色背景图像（用255填充，代表白色）
    bg = np.full((fsize, fsize), 255)  # 创建背景图像 bg 时
    # 尝试从渲染的表面中获取alpha通道（透明度）数据，并对其进行转置
    # 注意：这里存在一个问题，因为render通常返回的是RGBA图像，而不是单独的alpha通道
    imo = pygame.surfarray.pixels_alpha(surface).transpose(1, 0)
    # 将alpha通道数据转换为PIL图像，再将其转换为NumPy数组，并取反（这步逻辑可能不正确）
    # 因为alpha通道数据通常是用于透明度的，而不是颜色值
    imo = 255 - np.array(Image.fromarray(imo))
    # 复制背景图像到新的变量im中（这里其实不需要deepcopy，因为bg是numpy数组，直接赋值即可）
    im = copy.deepcopy(bg)
    # 获取处理后的图像（imo）的高度和宽度
    h, w = imo.shape[:2]
    # 如果图像高度大于指定的大小fsize，则重新计算宽度并缩放图像
    if h > fsize:
        h, w = fsize, round(w * fsize / h)
        imo = cv2.resize(imo, (w, h))
        # 如果图像宽度大于指定的大小fsize，则重新计算高度并缩放图像
    if w > fsize:
        h, w = round(h * fsize / w), fsize
        imo = cv2.resize(imo, (w, h))
        # 计算图像在背景图像中的位置（居中）
    x, y = round((fsize - w) / 2), round((fsize - h) / 2)
    # 将处理后的图像（imo）放置在背景图像（im）的中心位置
    im[y:h + y, x:x + w] = imo
    # 将处理后的背景图像（im）转换为PIL图像，并转换为RGB模式（这里可能不需要，因为im已经是RGB了）
    pil_im = Image.fromarray(im.astype('uint8')).convert('RGB')
    # 返回PIL图像对象
    return pil_im
