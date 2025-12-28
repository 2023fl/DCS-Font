import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def get_nonorm_transform(resolution):
    nonorm_transform =  transforms.Compose(
            [transforms.Resize((resolution, resolution), 
                               interpolation=transforms.InterpolationMode.BILINEAR), 
             transforms.ToTensor()])
    return nonorm_transform


class FontDataset(Dataset):
    """The dataset of font generation
    """

    # 定义FontDataset类，继承自Dataset基类，用于生成字体数据集
    def __init__(self, args, phase, transforms=None, scr=False):
        super().__init__()  # 调用基类的构造函数
        self.root = args.data_root  # 从参数中获取数据根目录
        self.phase = phase  # 从参数中获取数据集的阶段（如训练或测试）
        self.scr = scr  # 是否为SCR（Style Content Relevance）模式
        if self.scr:  # 如果是SCR模式
            self.num_neg = args.num_neg  # 获取负样本的数量

        # 获取数据路径
        self.get_path()
        self.transforms = transforms  # 传入的变换操作
        self.nonorm_transforms = get_nonorm_transform(args.resolution)  # 获取不进行归一化变换的操作

    def get_path(self):
        # 定义获取数据路径的函数
        self.target_images = []  # 初始化目标图像列表
        # 初始化一个字典，用于存储每种风格的相关图像列表
        self.style_to_images = {}

        # 定义目标图像的目录
        target_image_dir = 'D:/FILES/Project/pytorch/Github/FontDiffuser/data_examples/train/TargetImage'  # 硬编码的路径
        # 遍历目标图像目录中的所有风格
        for style in os.listdir(target_image_dir):
            images_related_style = []  # 存储当前风格的相关图像路径
            # 遍历当前风格目录中的所有图像
            for img in os.listdir(f"{target_image_dir}/{style}"):
                img_path = f"{target_image_dir}/{style}/{img}"  # 获取图像的完整路径
                self.target_images.append(img_path)  # 添加到目标图像列表
                images_related_style.append(img_path)  # 添加到当前风格的图像列表
            # 将当前风格的图像列表添加到字典中
            self.style_to_images[style] = images_related_style

    def __getitem__(self, index):
        # 定义获取单个数据样本的函数
        target_image_path = self.target_images[index]  # 根据索引获取目标图像的路径
        target_image_name = target_image_path.split('/')[-1]  # 获取目标图像的文件名
        style, content = target_image_name.split('.')[0].split('+')  # 分离风格和内容

        # 读取内容图像
        content_image_path = f"D:/FILES/Project/pytorch/Github/FontDiffuser/data_examples/train/ContentImage/{content}.jpg"  # 硬编码的路径
        content_image = Image.open(content_image_path).convert('RGB')  # 打开并转换为RGB图像

        # 随机选择用于风格的图像
        images_related_style = self.style_to_images[style].copy()  # 获取当前风格的图像列表的副本
        images_related_style.remove(target_image_path)  # 从列表中移除目标图像的路径
        style_image_path = random.choice(images_related_style)  # 随机选择一个图像作为风格图像
        style_image = Image.open(style_image_path).convert("RGB")  # 打开并转换为RGB图像

        # 读取目标图像
        target_image = Image.open(target_image_path).convert("RGB")  # 打开并转换为RGB图像
        nonorm_target_image = self.nonorm_transforms(target_image)  # 应用不进行归一化的变换

        # 如果传入了变换操作，则应用它们
        if self.transforms is not None:
            content_image = self.transforms[0](content_image)
            style_image = self.transforms[1](style_image)
            target_image = self.transforms[2](target_image)

        # 构建样本字典
        sample = {
            "content_image": content_image,
            "style_image": style_image,
            "target_image": target_image,
            "target_image_path": target_image_path,
            "nonorm_target_image": nonorm_target_image}

        # 如果是SCR模式
        if self.scr:
            # 获取不同风格的相同内容的负样本图像路径
            style_list = list(self.style_to_images.keys())
            style_index = style_list.index(style)
            style_list.pop(style_index)
            choose_neg_names = []
            for i in range(self.num_neg):
                choose_style = random.choice(style_list)  # 随机选择一个风格
                choose_index = style_list.index(choose_style)
                style_list.pop(choose_index)
                choose_neg_name = f"{self.root}/train/TargetImage/{choose_style}/{choose_style}+{content}.jpg"  # 构建负样本图像的路径
                choose_neg_names.append(choose_neg_name)

            # 加载负样本图像
            for i, neg_name in enumerate(choose_neg_names):
                neg_image = Image.open(neg_name).convert("RGB")  # 打开并转换为RGB图像
                if self.transforms is not None:
                    neg_image = self.transforms[2](neg_image)  # 应用变换
                # 将负样本图像添加到样本字典中
                if i == 0:
                    neg_images = neg_image[None, :, :, :]
                else:
                    neg_images = torch.cat([neg_images, neg_image[None, :, :, :]], dim=0)
            sample["neg_images"] = neg_images

        # 返回构建的样本
        return sample

    def __len__(self):
        # 定义获取数据集大小的函数
        return len(self.target_images)  # 返回目标图像列表的长度
# class FontDataset(Dataset):
#     """The dataset of font generation
#     """
#     def __init__(self, args, phase, transforms=None, scr=False):
#         super().__init__()
#         self.root = args.data_root
#         self.phase = phase
#         self.scr = scr
#         if self.scr:
#             self.num_neg = args.num_neg
#
#         # Get Data path
#         self.get_path()
#         self.transforms = transforms
#         self.nonorm_transforms = get_nonorm_transform(args.resolution)
#
#     def get_path(self):
#         self.target_images = []
#         # images with related style
#         self.style_to_images = {}
#         target_image_dir = 'D:/FILES/Project/pytorch/Github/FontDiffuser/data_examples/train/TargetImage'   # 自己加的
#         # target_image_dir = f"{self.root}/{self.phase}/TargetImage"
#         for style in os.listdir(target_image_dir):
#             images_related_style = []
#             for img in os.listdir(f"{target_image_dir}/{style}"):
#                 img_path = f"{target_image_dir}/{style}/{img}"
#                 self.target_images.append(img_path)
#                 images_related_style.append(img_path)
#             self.style_to_images[style] = images_related_style
#
#     def __getitem__(self, index):
#         target_image_path = self.target_images[index]
#         target_image_name = target_image_path.split('/')[-1]
#         style, content = target_image_name.split('.')[0].split('+')
#
#         # Read content image
#         # 自己加的
#         content_image_path = f"D:/FILES/Project/pytorch/Github/FontDiffuser/data_examples/train/ContentImage/{content}.jpg"
#         # 原代码 content_image_path = f"{self.root}/{self.phase}/ContentImage/{content}.jpg"
#         content_image = Image.open(content_image_path).convert('RGB')
#
#         # Random sample used for style image
#         images_related_style = self.style_to_images[style].copy()
#         images_related_style.remove(target_image_path)
#         style_image_path = random.choice(images_related_style)
#         style_image = Image.open(style_image_path).convert("RGB")
#
#         # Read target image
#         target_image = Image.open(target_image_path).convert("RGB")
#         nonorm_target_image = self.nonorm_transforms(target_image)
#
#         if self.transforms is not None:
#             content_image = self.transforms[0](content_image)
#             style_image = self.transforms[1](style_image)
#             target_image = self.transforms[2](target_image)
#
#         sample = {
#             "content_image": content_image,
#             "style_image": style_image,
#             "target_image": target_image,
#             "target_image_path": target_image_path,
#             "nonorm_target_image": nonorm_target_image}
#
#         if self.scr:
#             # Get neg image from the different style of the same content
#             style_list = list(self.style_to_images.keys())
#             style_index = style_list.index(style)
#             style_list.pop(style_index)
#             choose_neg_names = []
#             for i in range(self.num_neg):
#                 choose_style = random.choice(style_list)
#                 choose_index = style_list.index(choose_style)
#                 style_list.pop(choose_index)
#                 choose_neg_name = f"{self.root}/train/TargetImage/{choose_style}/{choose_style}+{content}.jpg"
#                 choose_neg_names.append(choose_neg_name)
#
#             # Load neg_images
#             for i, neg_name in enumerate(choose_neg_names):
#                 neg_image = Image.open(neg_name).convert("RGB")
#                 if self.transforms is not None:
#                     neg_image = self.transforms[2](neg_image)
#                 if i == 0:
#                     neg_images = neg_image[None, :, :, :]
#                 else:
#                     neg_images = torch.cat([neg_images, neg_image[None, :, :, :]], dim=0)
#             sample["neg_images"] = neg_images
#
#         return sample
#
#     def __len__(self):
#         return len(self.target_images)
