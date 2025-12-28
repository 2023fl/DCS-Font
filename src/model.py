import math
import torch
import torch.nn as nn

from diffusers import ModelMixin
from diffusers.configuration_utils import (ConfigMixin, 
                                           register_to_config)




#  重点
class FontDiffuserModel(ModelMixin, ConfigMixin):
    """Forward function for FontDiffuer with content encoder \
        style encoder and unet.
    """

    @register_to_config
    def __init__(
        self,
        unet,
        style_encoder,
        content_encoder,
    ):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder

    def forward(
        self,
        x_t,
        timesteps,
        style_images,
        content_images,
        content_encoder_downsample_size,
    ):
        # print('x_t:',x_t.shape)       # x_t: torch.Size([2, 3, 96, 96])
        # 从风格图像中提取特征
        # style_img_feature, _, _ = self.style_encoder(style_images)
        # print('style_images:',style_images.shape)    # style_images: torch.Size([2, 3, 96, 96])
        style_img_feature, _, _ = self.config.style_encoder(style_images)   # # 使用风格编码器提取风格图像的特征

        batch_size, channel, height, width = style_img_feature.shape    # 获取风格图像特征的形状
        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(batch_size, height*width, channel)

        # Get the content feature从内容图像中提取特征
        # content_img_feature, content_residual_features = self.content_encoder(content_images)
        content_img_feature, content_residual_features = self.config.content_encoder(content_images)     # 自己加的
        content_residual_features.append(content_img_feature)
        # Get the content feature from reference image从风格图像中提取内容特征
        # style_content_feature, style_content_res_features = self.content_encoder(style_images)
        style_content_feature, style_content_res_features = self.config.content_encoder(style_images)    # 自己加的
        style_content_res_features.append(style_content_feature)

        input_hidden_states = [style_img_feature, content_residual_features, \
                               style_hidden_states, style_content_res_features]


        out = self.config.unet(
            x_t,
            timesteps,
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )
        noise_pred = out[0]
        offset_out_sum = out[1]

        return noise_pred, offset_out_sum


class FontDiffuserModelDPM(ModelMixin, ConfigMixin):
    """DPM Forward function for FontDiffuer with content encoder \
        style encoder and unet.
    """
    @register_to_config
    def __init__(
        self, 
        unet, 
        style_encoder,
        content_encoder,
    ):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder
    
    def forward(
        self, 
        x_t, 
        timesteps, 
        cond,
        content_encoder_downsample_size,
        version,
    ):
        content_images = cond[0]
        style_images = cond[1]

        # style_img_feature, _, style_residual_features = self.style_encoder(style_images)
        style_img_feature, _, style_residual_features = self.config.style_encoder(style_images)   # 自己加的
        
        batch_size, channel, height, width = style_img_feature.shape
        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(batch_size, height*width, channel)
        
      # 获取内容特性
        # content_img_feture, content_residual_features = self.content_encoder(content_images)
        content_img_feture, content_residual_features = self.config.content_encoder(content_images)   # 自己加的
        content_residual_features.append(content_img_feture)
        # Get the content feature from reference image
        # style_content_feature, style_content_res_features = self.content_encoder(style_images)
        style_content_feature, style_content_res_features = self.config.content_encoder(style_images)   # 自己加的
        style_content_res_features.append(style_content_feature)

        input_hidden_states = [style_img_feature, content_residual_features, style_hidden_states, style_content_res_features]

        # out = self.unet(
        #     x_t,
        #     timesteps,
        #     encoder_hidden_states=input_hidden_states,
        #     content_encoder_downsample_size=content_encoder_downsample_size,
        # )
        out = self.config.unet(
            x_t,
            timesteps,
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )
        noise_pred = out[0]
        
        return noise_pred
