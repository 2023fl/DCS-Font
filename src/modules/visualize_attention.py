import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# ======================
# 1️⃣ ContentSEAttention 模块定义
# ======================
class ContentSEAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ContentSEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ======================
# 2️⃣ ContentEncoder 定义（带通道注意力输出）
# ======================
class DBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ContentEncoder(nn.Module):
    def __init__(self):
        super(ContentEncoder, self).__init__()
        self.blocks = nn.ModuleList([
            nn.ModuleList([DBlock(3, 64)]),
            nn.ModuleList([DBlock(64, 128)]),
            nn.ModuleList([DBlock(128, 256)]),
            nn.ModuleList([DBlock(256, 512)])
        ])

        self.save_features = [0, 1, 2, 3]
        self.se_attentions = nn.ModuleList([
            ContentSEAttention(64),
            ContentSEAttention(128),
            ContentSEAttention(256),
            ContentSEAttention(512)
        ])

    def forward(self, x, return_attn=False):
        h = x
        residual_features = []
        attention_maps = []

        residual_features.append(h)
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
            if index in self.save_features[:-1]:
                residual_features.append(h)

            se_module = self.se_attentions[index]
            h = se_module(h)

            if return_attn:
                # 提取每层通道注意力
                b, c, _, _ = h.size()
                attn = se_module.fc(se_module.avg_pool(h).view(b, c)).detach().cpu().numpy()
                attention_maps.append(attn)

        if return_attn:
            return h, residual_features, attention_maps
        else:
            return h, residual_features


# ======================
# 3️⃣ 可视化通道注意力函数
# ======================
def visualize_attention(attn_maps):
    for i, attn in enumerate(attn_maps):
        attn = attn.squeeze()
        plt.figure(figsize=(8, 3))
        plt.bar(np.arange(len(attn)), attn)
        plt.title(f"ContentSEAttention Layer {i+1} Channel Attention")
        plt.xlabel("Channel Index")
        plt.ylabel("Attention Weight")
        plt.tight_layout()
        plt.show()


# ======================
# 4️⃣ 主程序入口
# ======================
if __name__ == "__main__":
    # 初始化模型
    encoder = ContentEncoder()
    encoder.eval()

    # 生成一张随机测试图片（假设输入为 128x128 字体图像）
    x = torch.randn(1, 3, 128, 128)

    # 前向传播，获取注意力分布
    _, _, attn_maps = encoder(x, return_attn=True)

    # 绘制通道注意力分布
    visualize_attention(attn_maps)

    print("✅ 通道注意力分布可视化完成！")
