import torch
import torch.nn as nn
import torchvision 


# class VGG16(nn.Module):
#     def __init__(self):
#         super(VGG16, self).__init__()
#         vgg16 = torchvision.models.vgg16(pretrained=True)
#
#         self.enc_1 = nn.Sequential(*vgg16.features[:5])
#         self.enc_2 = nn.Sequential(*vgg16.features[5:10])
#         self.enc_3 = nn.Sequential(*vgg16.features[10:17])
#
#         for i in range(3):
#             for param in getattr(self, f'enc_{i+1:d}').parameters():
#                 param.requires_grad = False
#
#     def forward(self, image):
#         results = [image]
#         for i in range(3):
#             func = getattr(self, f'enc_{i+1:d}')
#             results.append(func(results[-1]))
#         return results[1:]
#
#
# class ContentPerceptualLoss(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#         self.VGG = VGG16()
#
#     def calculate_loss(self, generated_images, target_images, device):
#         self.VGG = self.VGG.to(device)
#
#         generated_features = self.VGG(generated_images)
#         target_features = self.VGG(target_images)
#
#         perceptual_loss = 0
#         perceptual_loss += torch.mean((target_features[0] - generated_features[0]) ** 2)
#         perceptual_loss += torch.mean((target_features[1] - generated_features[1]) ** 2)
#         perceptual_loss += torch.mean((target_features[2] - generated_features[2]) ** 2)
#         perceptual_loss /= 3
#         return perceptual_loss


# 定义VGG16类，继承自nn.Module
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()  # 调用父类nn.Module的构造函数
        vgg16 = torchvision.models.vgg16(pretrained=True)  # 加载预训练的VGG16模型

        # 截取VGG16模型的前几层作为特征提取器
        self.enc_1 = nn.Sequential(*vgg16.features[:5])  # 第一部分.这行代码截取了vgg16.features中的前5个层（索引从0到4），并将它们封装成一个新的nn.Sequential模块，即enc_1。这些层通常包括卷积层、ReLU激活层和可能的最大池化层，具体取决于VGG16的实现细节。
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])  # 第二部分.这行代码截取了vgg16.features中从第6个层到第10个层（索引从5到9），并将它们封装成enc_2。
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])  # 第三部分.这行代码截取了vgg16.features中从第11个层到第17个层（索引从10到16），并将它们封装成enc_3。

        # 将截取的部分的参数设置为不需要梯度更新
        for i in range(3):
            for param in getattr(self, f'enc_{i + 1:d}').parameters():
                param.requires_grad = False

    def forward(self, image):
        # 存储每一层的输出
        results = [image]  # 初始包含原始图像
        for i in range(3):
            # 动态获取编码器层并执行前向传播
            func = getattr(self, f'enc_{i + 1:d}')
            results.append(func(results[-1]))  # 将当前层的输出添加到列表中
        # 返回除原始图像外的所有层的输出
        return results[1:]

    # 定义ContentPerceptualLoss类，继承自nn.Module


class ContentPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()  # 调用父类nn.Module的构造函数
        self.VGG = VGG16()  # 实例化VGG16特征提取器

    def calculate_loss(self, generated_images, target_images, device):
        # 将VGG模型移动到指定的设备上
        self.VGG = self.VGG.to(device)

        # 使用VGG模型提取生成图像和目标图像的特征
        generated_features = self.VGG(generated_images)  # 假设生成图像和目标图像都是batch_size x channels x height x width
        target_features = self.VGG(target_images)

        # 初始化感知损失
        perceptual_loss = 0
        # 分别计算三层特征图的均方误差(MSE)，并累加到感知损失中
        perceptual_loss += torch.mean((target_features[0] - generated_features[0]) ** 2)
        # 第一层特征损失.
        # (target_features[0] - generated_features[0]) ** 2 计算了目标图像和生成图像在第一层特征图上的逐元素差的平方。这个操作会生成一个新的张量（Tensor），其形状与特征图相同，每个元素都是对应位置差的平方。
        # torch.mean(...) 计算了上一步生成的张量中所有元素的平均值，即第一层特征图上的平均均方误差（MSE）。这个值表示了目标图像和生成图像在第一层特征表示上的差异程度。
        # 最后，这个平均MSE被加到perceptual_loss变量上，累积为总的感知损失的一部分。
        perceptual_loss += torch.mean((target_features[1] - generated_features[1]) ** 2)  # 第二层特征损失
        perceptual_loss += torch.mean((target_features[2] - generated_features[2]) ** 2)  # 第三层特征损失
        # 将累计的感知损失除以3，得到平均感知损失（虽然这一步在数学上是多余的，因为只是将总和重新缩放，但保持原样）
        perceptual_loss /= 3
        return perceptual_loss

    # 注意：此代码假设generated_images和target_images的维度与VGG16的输入要求相匹配，
# 并且它们的batch_size是相同的。此外，它仅计算了三层特征的损失。