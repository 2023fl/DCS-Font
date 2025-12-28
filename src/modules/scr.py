import torch
import torch.nn as nn
import src.modules.scr_modules as SCRModules  # 导入自定义模块 SCRModules    原


from info_nce import InfoNCE  # 导入 InfoNCE 损失函数
import kornia.augmentation as K  # 导入 Kornia 库中的数据增强模块


class SCR(nn.Module):  # 定义 SCR 类，继承自 nn.Module

    def __init__(self,
                 temperature,  # 设定 InfoNCE 损失函数的温度参数
                 mode='training',  # 模式，默认为 'training' 模式
                 image_size=96):  # 图像大小，默认为 96
        super().__init__()  # 调用父类的构造函数

        # 初始化风格 VGG 网络
        style_vgg = SCRModules.vgg  # 从 SCRModules 模块中获取 VGG 模型
        style_vgg = nn.Sequential(*list(style_vgg.children()))  # 将 VGG 模型拆分为子模块后重组为顺序模型

        # 初始化风格特征提取器，传入 VGG 编码器
        self.StyleFeatExtractor = SCRModules.StyleExtractor(encoder=style_vgg)
        # 初始化风格特征投影器
        self.StyleFeatProjector = SCRModules.Projector()

        # 根据模式设置梯度计算
        if mode == 'training':  # 如果模式是 'training'
            self.StyleFeatExtractor.requires_grad_(True)  # 设置风格特征提取器参数需要计算梯度
            self.StyleFeatProjector.requires_grad_(True)  # 设置风格特征投影器参数需要计算梯度
        else:  # 如果模式不是 'training'
            self.StyleFeatExtractor.requires_grad_(False)  # 设置风格特征提取器参数不需要计算梯度
            self.StyleFeatProjector.requires_grad_(False)  # 设置风格特征投影器参数不需要计算梯度

        # 初始化 InfoNCE 损失函数
        self.nce_loss = InfoNCE(
            temperature=temperature,  # 传入温度参数
            negative_mode='paired',  # 设置负样本模式为 'paired'
        )

        # 初始化随机裁剪增强器
        self.patch_sampler = K.RandomResizedCrop(
            (image_size, image_size),  # 设置裁剪后的图像大小
            scale=(0.8, 1.0),  # 设置裁剪的比例范围
            ratio=(0.75, 1.33)  # 设置裁剪的宽高比范围
        )

    def forward(self, sample_imgs, pos_imgs, neg_imgs, nce_layers='0,1,2,3,4,5'):

        # 获取生成图像的风格嵌入
        sample_style_embeddings = self.StyleFeatProjector(
            self.StyleFeatExtractor(
                sample_imgs,  # 输入生成的样本图像
                nce_layers),  # 指定要提取的层
            nce_layers)  # 输出风格嵌入，形状为 N * C(2048)

        # 对正样本图像进行随机裁剪
        pos_imgs = self.patch_sampler(pos_imgs)  # 使用随机裁剪增强器对正样本图像进行裁剪
        # 获取正样本图像的风格嵌入
        pos_style_embeddings = self.StyleFeatProjector(
            self.StyleFeatExtractor(
                pos_imgs,  # 输入正样本图像
                nce_layers),  # 指定要提取的层
            nce_layers)  # 输出风格嵌入

        # 获取负样本图像的风格嵌入
        _, num_neg, _, _, _ = neg_imgs.shape  # 获取负样本图像的数量
        for i in range(num_neg):
            neg_imgs_once = neg_imgs[:, i, :, :]  # 获取单个负样本图像
            neg_style_embeddings_once = self.StyleFeatProjector(
                self.StyleFeatExtractor(
                    neg_imgs_once,  # 输入单个负样本图像
                    nce_layers),  # 指定要提取的层
                nce_layers)  # 输出风格嵌入
            for j, layer_out in enumerate(neg_style_embeddings_once):
                if j == 0:
                    neg_style_embeddings_mid = layer_out[None, :, :]  # 初始化中间变量
                else:
                    neg_style_embeddings_mid = torch.cat(
                        [neg_style_embeddings_mid, layer_out[None, :, :]],  # 连接负样本风格嵌入
                        dim=0)
            if i == 0:
                neg_style_embeddings = neg_style_embeddings_mid[:, :, None, :]  # 初始化负样本嵌入
            else:
                neg_style_embeddings = torch.cat(
                    [neg_style_embeddings, neg_style_embeddings_mid[:, :, None, :]],  # 连接负样本嵌入
                    dim=2)

        return sample_style_embeddings, pos_style_embeddings, neg_style_embeddings  # 返回样本、正样本和负样本的风格嵌入

    def calculate_nce_loss(self, sample_s, pos_s, neg_s):

        num_layer = neg_s.shape[0]  # 获取层数
        neg_s_list = []
        for i in range(num_layer):
            neg_s_list.append(neg_s[i])  # 将每一层的负样本嵌入添加到列表中

        total_scm_loss = 0.  # 初始化总损失
        for layer, (sample, pos, neg) in enumerate(zip(sample_s, pos_s, neg_s_list)):
            scm_loss = self.nce_loss(sample, pos, neg)  # 计算每层的 NCE 损失
            total_scm_loss += scm_loss  # 累加损失

        return total_scm_loss / num_layer  # 返回平均损失

# import torch
#
# import torch.nn as nn
# import src.modules.scr_modules as SCRModules
#
# from info_nce import InfoNCE
# import kornia.augmentation as K
#
# class SCR(nn.Module):
#
#     def __init__(self,
#                  temperature,
#                  mode='training',
#                  image_size=96):
#         super().__init__()
#         style_vgg = SCRModules.vgg
#         style_vgg = nn.Sequential(*list(style_vgg.children()))
#         self.StyleFeatExtractor = SCRModules.StyleExtractor(
#             encoder=style_vgg)
#         self.StyleFeatProjector = SCRModules.Projector()
#
#         if mode == 'training':
#             self.StyleFeatExtractor.requires_grad_(True)
#             self.StyleFeatProjector.requires_grad_(True)
#         else:
#             self.StyleFeatExtractor.requires_grad_(False)
#             self.StyleFeatProjector.requires_grad_(False)
#
#         # NCE Loss
#         self.nce_loss = InfoNCE(
#             temperature=temperature,
#             negative_mode='paired',
#         )
#
#         # Pos Image random resize and crop
#         self.patch_sampler = K.RandomResizedCrop(
#             (image_size, image_size),
#             scale=(0.8,1.0),
#             ratio=(0.75,1.33))
#
#     def forward(self, sample_imgs, pos_imgs, neg_imgs, nce_layers='0,1,2,3,4,5'):
#
#         # Get generated image style embedding
#         sample_style_embeddings = self.StyleFeatProjector(
#             self.StyleFeatExtractor(
#                 sample_imgs,
#                 nce_layers),
#             nce_layers) # out: N * C(2048)
#
#         # Random resize and crop for positive images
#         pos_imgs = self.patch_sampler(pos_imgs)
#         # Get positive image style embedding
#         pos_style_embeddings = self.StyleFeatProjector(
#             self.StyleFeatExtractor(
#                 pos_imgs,
#                 nce_layers),
#             nce_layers)
#
#         # Get negative image style embedding
#         _, num_neg, _, _, _ = neg_imgs.shape
#         for i in range(num_neg):
#             neg_imgs_once = neg_imgs[:, i, :, :]
#             neg_style_embeddings_once = self.StyleFeatProjector(
#                 self.StyleFeatExtractor(
#                     neg_imgs_once,
#                     nce_layers),
#                 nce_layers)
#             for j, layer_out in enumerate(neg_style_embeddings_once):
#                 if j == 0:
#                     neg_style_embeddings_mid = layer_out[None, :, :]
#                 else:
#                     neg_style_embeddings_mid = torch.cat(
#                         [neg_style_embeddings_mid, layer_out[None, :, :]],
#                         dim=0)
#             if i == 0:
#                 neg_style_embeddings = neg_style_embeddings_mid[:, :, None, :]
#             else:
#                 neg_style_embeddings = torch.cat(
#                     [neg_style_embeddings, neg_style_embeddings_mid[:, :, None, :]],
#                     dim=2)
#
#         return sample_style_embeddings, pos_style_embeddings, neg_style_embeddings
#
#     def calculate_nce_loss(self, sample_s, pos_s, neg_s):
#
#         num_layer = neg_s.shape[0]
#         neg_s_list = []
#         for i in range(num_layer):
#             neg_s_list.append(neg_s[i])
#
#         total_scm_loss = 0.
#         for layer, (sample, pos, neg) in enumerate(zip(sample_s, pos_s, neg_s_list)):
#             scm_loss = self.nce_loss(sample, pos, neg)
#             total_scm_loss += scm_loss
#
#         return total_scm_loss / num_layer
