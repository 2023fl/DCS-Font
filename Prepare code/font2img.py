# from PIL import Image,ImageDraw,ImageFont
# import matplotlib.pyplot as plt
# import os
# import numpy as np
# import pathlib
# import argparse
# from fontTools.ttLib import TTFont
#
#
# parser = argparse.ArgumentParser(description='Obtaining characters from .ttf')#从.ttf获取字符
# # parser.add_argument('--ttf_path', type=str, default='ttf_folder/方正新楷体_GBK(完整).TTF',help='ttf directory')#ttf 目录
# parser.add_argument('--ttf_path', type=str, default=r'D:\FILES\Project\pytorch\py_text01\FontDiffuser_text\ttf\ttf',help='ttf directory')#ttf 目录
# parser.add_argument('--chara', type=str, default='content_characters.txt',help='characters')#字符
# parser.add_argument('--save_path', type=str, default=r'D:\FILES\Project\pytorch\py_text01\FontDiffuser_text\ttf\img',help='images directory')#images 目录
# parser.add_argument('--img_size', type=int, default=128,help='The size of generated images')#生成图像的大小
# parser.add_argument('--chara_size', type=int,default=96, help='The size of generated characters')#生成字符的大小
# args = parser.parse_args()
#
# file_object = open(args.chara,encoding='utf-8')
# try:
# 	characters = file_object.read()
# finally:
#     file_object.close()
#
# def draw_single_char(ch, font, canvas_size, x_offset, y_offset):
#     img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
#     draw = ImageDraw.Draw(img)
#     draw.text((x_offset, y_offset), ch, (0, 0, 0), font=font)
#     return img
#
# def draw_example(ch, src_font, canvas_size, x_offset, y_offset):
#     src_img = draw_single_char(ch, src_font, canvas_size, x_offset, y_offset)
#     example_img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
#     example_img.paste(src_img, (0, 0))
#     return example_img
#
# data_dir = args.ttf_path
# data_root = pathlib.Path(data_dir)
# print(data_root)
#
# all_image_paths = list(data_root.glob('*.*'))  # *.ttf TTF
# all_image_paths = [str(path) for path in all_image_paths]
# total_num = len(all_image_paths)
# print(total_num)
#
# seq = list()
#
# if not os.path.exists(args.save_path):
#     os.mkdir(args.save_path)
#
# def get_char_list_from_ttf(font_file):
#     f_obj = TTFont(font_file)
#     m_dict = f_obj.getBestCmap()
#
#     unicode_list = []
#     for key, uni in m_dict.items():
#         unicode_list.append(key)
#
#     char_list = [chr(ch_unicode) for ch_unicode in unicode_list]
#     return char_list
#
# for idx, (label, item) in enumerate(zip(range(len(all_image_paths)),all_image_paths)):
#     print("{} / {} ".format(idx, total_num), item)
#     src_font = ImageFont.truetype(item, size=args.chara_size)
#     # font_name = item.split('/')[-1].split('.')[0]
#     # 只获取字体文件名，不包括路径
#     font_name = os.path.basename(item).split('.')[0]
#     chars = get_char_list_from_ttf(item)  #
#     img_cnt = 0
#     filter_cnt = 0
#
#     # 使用字体样式作为文件夹名称
#     font_style_folder = os.path.join(args.save_path, font_name)  # 字体样式文件夹
#     if not os.path.exists(font_style_folder):
#         os.mkdir(font_style_folder)
#
#     for (chara, cnt) in zip(characters, range(len(characters))):
#         img = draw_example(chara, src_font, args.img_size, (args.img_size-args.chara_size)/2, (args.img_size-args.chara_size)/2)
#         # path_full = os.path.join(args.save_path, 'id_%d'%(label))
#         # if not os.path.exists(path_full):
#         #     os.mkdir(path_full)
#         if args.img_size * args.img_size * 3 - np.sum(np.array(img) / 255.) < 100:
#             filter_cnt += 1
#         else:
#             img_cnt += 1
#             # img.save(os.path.join(path_full, "%05d.png" % (cnt)))
#             # 使用字体样式和字符内容生成文件名
#             filename = f"{font_name}+{chara}.jpg"  # 新的文件名格式
#             # filename = f"{chara}.jpg"  # 新的文件名格式
#             img.save(os.path.join(font_style_folder, filename))  # 保存到字体样式文件夹
#
#     print(filter_cnt,' characters are missing in this font')#此字体中缺少字符
#
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os
import numpy as np
import pathlib
import argparse
from fontTools.ttLib import TTFont

parser = argparse.ArgumentParser(description='Obtaining characters from .ttf')  # 从.ttf获取字符
parser.add_argument('--ttf_path', type=str, default=r'D:\FILES\Project\pytorch\py_text01\FontDiffuser_text\ttf\ttf', help='ttf directory')  # ttf 目录
parser.add_argument('--chara', type=str, default='content_characters.txt', help='characters')  # 字符
parser.add_argument('--save_path', type=str, default=r'D:\FILES\Project\pytorch\py_text01\FontDiffuser_text\ttf\img', help='images directory')  # images 目录
parser.add_argument('--img_size', type=int, default=128, help='The size of generated images')  # 生成图像的大小
parser.add_argument('--chara_size', type=int, default=96, help='The size of generated characters')  # 生成字符的大小
args = parser.parse_args()

file_object = open(args.chara, encoding='utf-8')
try:
    characters = file_object.read()
finally:
    file_object.close()

def draw_single_char(ch, font, canvas_size, x_offset, y_offset):
    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch, (0, 0, 0), font=font)
    return img

def draw_example(ch, src_font, canvas_size, x_offset, y_offset):
    src_img = draw_single_char(ch, src_font, canvas_size, x_offset, y_offset)
    example_img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    example_img.paste(src_img, (0, 0))
    return example_img

data_dir = args.ttf_path
data_root = pathlib.Path(data_dir)
print(data_root)

# 修改：只获取 .ttf 和 .TTF 文件
ttf_files = list(data_root.glob('*.ttf')) + list(data_root.glob('*.TTF'))
all_image_paths = [str(path) for path in ttf_files]
total_num = len(all_image_paths)
print(f"Total number of TTF files: {total_num}")

seq = list()

if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)

def get_char_list_from_ttf(font_file):
    f_obj = TTFont(font_file)
    m_dict = f_obj.getBestCmap()
    unicode_list = []
    for key, uni in m_dict.items():
        unicode_list.append(key)
    char_list = [chr(ch_unicode) for ch_unicode in unicode_list]
    return char_list

for idx, (label, item) in enumerate(zip(range(len(all_image_paths)), all_image_paths)):
    print(f"{idx + 1} / {total_num} ", item)
    src_font = ImageFont.truetype(item, size=args.chara_size)
    font_name = os.path.basename(item).split('.')[0]
    chars = get_char_list_from_ttf(item)
    img_cnt = 0
    filter_cnt = 0

    font_style_folder = os.path.join(args.save_path, font_name)
    if not os.path.exists(font_style_folder):
        os.mkdir(font_style_folder)

    for (chara, cnt) in zip(characters, range(len(characters))):
        img = draw_example(chara, src_font, args.img_size, (args.img_size - args.chara_size) / 2, (args.img_size - args.chara_size) / 2)
        if args.img_size * args.img_size * 3 - np.sum(np.array(img) / 255.) < 100:
            filter_cnt += 1
        else:
            img_cnt += 1
            # filename = f"{font_name}+{chara}.jpg"
            filename = f"{chara}.jpg"
            img.save(os.path.join(font_style_folder, filename))

    print(f"{filter_cnt} characters are missing in this font")  # 此字体中缺少字符