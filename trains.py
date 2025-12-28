import os
import math
import time
import logging
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler

from dataset.font_dataset import FontDataset
from dataset.collate_fn import CollateFN
from configs.fontdiffuser import get_parser
# from configs.fontdiffuser_train import get_parser
from src import (FontDiffuserModel,
                 ContentPerceptualLoss,
                 build_unet,
                 build_style_encoder,
                 build_content_encoder,
                 build_ddpm_scheduler,
                 build_scr)
from utils import (save_args_to_yaml,
                   x0_from_epsilon,
                   reNormalize_img,
                   normalize_mean_std)

logger = get_logger(__name__)


def get_args():
    parser = get_parser()  # 获取get_parser()里ArgumentParser变量
    args = parser.parse_args()  # 解析命令行参数,结果储存到args
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))  # 从环境变量中获取"LOCAL_RANK"的值，并将其转换为整数。如果环境变量不存在，则返回-1
    if env_local_rank != -1 and env_local_rank != args.local_rank:  # 环境变量中的local_rank不等于-1，且不同于解析的命令行参数中的local_rank
        args.local_rank = env_local_rank  # 将命令行参数中的local_rank设置为环境变量中的local_rank值
    style_image_size = args.style_image_size  # 在args中获取表示风格图像的尺寸（假定为正方形）
    content_image_size = args.content_image_size  # 获取内容图像的尺寸96
    args.style_image_size = (style_image_size, style_image_size)  # 将风格图像的尺寸更新为一个元组，表示图像的宽度和高度（相等）
    args.content_image_size = (content_image_size, content_image_size)

    return args


def load_checkpoint(args, model, optimizer, lr_scheduler):
    """
    从断点文件加载训练状态。
    """
    checkpoint_path = os.path.join(args.output_dir, "latest_checkpoint.pth")
    if os.path.isfile(checkpoint_path):
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # 获取当前的 global_step
        global_step = checkpoint['global_step']
        save_dir = os.path.join(args.output_dir, f"global_step_{global_step}")

        # 检查模型权重文件是否存在
        unet_path = os.path.join(save_dir, "unet.pth")
        style_encoder_path = os.path.join(save_dir, "style_encoder.pth")
        content_encoder_path = os.path.join(save_dir, "content_encoder.pth")

        if os.path.isfile(unet_path) and os.path.isfile(style_encoder_path) and os.path.isfile(content_encoder_path):
            model.unet.load_state_dict(torch.load(unet_path))
            model.style_encoder.load_state_dict(torch.load(style_encoder_path))
            model.content_encoder.load_state_dict(torch.load(content_encoder_path))
            logging.info(f"Loaded model weights from {save_dir}")
        else:
            logging.error(f"Model weight files not found in {save_dir}")
            return 0, 0

        # 恢复优化器和学习率调度器状态
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

        # 恢复全局训练状态
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
        logging.info(f"Resumed training from epoch {start_epoch}, global step {global_step}")
        print(f"加载成功, 从 epoch {start_epoch}, global step {global_step} 开始继续训练")
        return start_epoch, global_step
    else:
        logging.info("No checkpoint found. Starting from scratch.")
        return 0, 0


def main():
    args = get_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)  # 使用args对象中的output_dir和logging_dir属性构造日志目录的完整路径

    accelerator = Accelerator(  # 初始化一个Accelerator对象，用于管理梯度累积、混合精度、日志记录和项目目录
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # 在执行向后/更新传递之前累积的更新步骤数。
        mixed_precision=args.mixed_precision,  # 选择是否使用混合精度
        log_with=args.report_to,  # 选择用于日志记录的工具，比如TensorBoard
        project_dir=logging_dir  # 模型预测和检查点的输出目录与日志目录
    )

    if accelerator.is_main_process:  # 检查是否为主进程
        os.makedirs(args.output_dir, exist_ok=True)  # 创建输出目录

    logging.basicConfig(  # 配置日志记录的基本设置，包括日志文件名、日期格式和日志级别  日志将被写入到args.output_dir目录下的fontdiffuser_training.log文件中
        filename=os.path.join(args.output_dir, "fontdiffuser_training.log"),
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    # 设置训练种子
    if args.seed is not None:
        set_seed(args.seed)

    # 加载模型和噪声调度器
    unet = build_unet(args=args)  # 构建一个U-net模型实例
    style_encoder = build_style_encoder(args=args)  # 构建风格编码器实例
    content_encoder = build_content_encoder(args=args)  # 构建内容编码器实例
    noise_scheduler = build_ddpm_scheduler(args)  # 构建一个噪声调度器实例
    if args.phase_2:  # 是否指定第二阶段训练
        unet.load_state_dict(torch.load(os.path.join(args.phase_1_ckpt_dir, "unet.pth")))
        style_encoder.load_state_dict(torch.load(os.path.join(args.phase_1_ckpt_dir, "style_encoder.pth")))
        content_encoder.load_state_dict(torch.load(os.path.join(args.phase_1_ckpt_dir, "content_encoder.pth")))

    model = FontDiffuserModel(  # 使用构建的实例组合成一个FontDiffuserModel模型，字体风格迁移模型
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder
    )

    # 构建内容感知损失
    perceptual_loss = ContentPerceptualLoss()

    # 加载 SCR 模块进行监督
    if args.phase_2:  # 是否为第二阶段训练
        scr = build_scr(args=args)  # 构建SCR
        scr.load_state_dict(torch.load(args.scr_ckpt_path))  # 加载预训练的SCR模型权重
        scr.requires_grad_(False)  # 不更新SCR的梯度

    # 加载数据集
    content_transforms = transforms.Compose([  # 内容图像转换
        transforms.Resize(args.content_image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        # 使用双线性插值调整图像大小
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # 标准化处理
    ])

    style_transforms = transforms.Compose([  # 样式图像转换
        transforms.Resize(args.style_image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    target_transforms = transforms.Compose([  # 目标图像转换
        transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_font_dataset = FontDataset(  # 训练用 FontDataset 数据集
        args=args,
        phase='train',
        transforms=[content_transforms, style_transforms, target_transforms],
        scr=args.phase_2
    )
    train_dataloader = torch.utils.data.DataLoader(  # 封装 train_font_dataset
        train_font_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        collate_fn=CollateFN()
    )

    # 构建优化器和学习率调度器
    if args.scale_lr:  # 按 GPU 数量、梯度累积步骤和批量大小缩放学习率
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # 加载断点
    start_epoch, global_step = load_checkpoint(args, model, optimizer, lr_scheduler)

    # Accelerate 准备工作
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    if args.phase_2:  # 判断是否为第二阶段训练
        scr = scr.to(accelerator.device)

    # 在主进程上初始化跟踪器，并保存命令行参数
    if accelerator.is_main_process:
        accelerator.init_trackers(args.experience_name)
        save_args_to_yaml(args=args, output_file=os.path.join(args.output_dir, f"{args.experience_name}_config.yaml"))

    # 只在本地主进程显示进度条
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # 转换为训练周期
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # 如果从断点续训，调整 global_step 的初始值
    if global_step > 0:
        progress_bar.update(global_step)

    # 训练循环
    for epoch in range(start_epoch, num_train_epochs):  # 从断点开始的 epoch
        train_loss = 0.0  # 训练损失
        for step, samples in enumerate(train_dataloader):  # 遍历训练数据加载器中的每个批次
            model.train()  # 训练模式
            content_images = samples["content_image"]  # 内容图像
            style_images = samples["style_image"]  # 风格图像
            target_images = samples["target_image"]  # 目标图像
            nonorm_target_images = samples["nonorm_target_image"]  # 未标准化的目标图像

            with accelerator.accumulate(model):  # 使用加速器来累积梯度
                # 采样并添加噪声
                noise = torch.randn_like(target_images)
                bsz = target_images.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,),
                                          device=target_images.device)
                timesteps = timesteps.long()

                noisy_target_images = noise_scheduler.add_noise(target_images, noise, timesteps)

                # 无分类器训练策略，通过随机丢弃部分信息增强泛化能力
                context_mask = torch.bernoulli(torch.zeros(bsz) + args.drop_prob).to(target_images.device)
                mask_indices = (context_mask == 1)
                content_images[mask_indices] = 1
                style_images[mask_indices] = 1

                # 预测噪声残差并计算损失
                noise_pred, offset_out_sum = model(
                    x_t=noisy_target_images,
                    timesteps=timesteps,
                    style_images=style_images,
                    content_images=content_images,
                    content_encoder_downsample_size=args.content_encoder_downsample_size
                )
                diff_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                offset_loss = offset_out_sum / 2

                # 输出处理以计算内容感知损失
                pred_original_sample_norm = x0_from_epsilon(
                    scheduler=noise_scheduler,
                    noise_pred=noise_pred,
                    x_t=noisy_target_images,
                    timesteps=timesteps
                )
                pred_original_sample = reNormalize_img(pred_original_sample_norm)
                norm_pred_ori = normalize_mean_std(pred_original_sample)
                norm_target_ori = normalize_mean_std(nonorm_target_images)
                percep_loss = perceptual_loss.calculate_loss(
                    generated_images=norm_pred_ori,
                    target_images=norm_target_ori,
                    device=target_images.device
                )

                loss = diff_loss + \
                       args.perceptual_coefficient * percep_loss + \
                       args.offset_coefficient * offset_loss

                if args.phase_2:  # 二阶段训练
                    neg_images = samples["neg_images"]  # 负图像
                    # 计算风格对比损失
                    sample_style_embeddings, pos_style_embeddings, neg_style_embeddings = scr(
                        pred_original_sample_norm,
                        target_images,
                        neg_images,
                        nce_layers=args.nce_layers
                    )
                    sc_loss = scr.calculate_nce_loss(
                        sample_s=sample_style_embeddings,
                        pos_s=pos_style_embeddings,
                        neg_s=neg_style_embeddings
                    )
                    loss += args.sc_coefficient * sc_loss

                # 收集所有进程的损失以进行日志记录（如果使用分布式训练）
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # 反向传播
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # 检查是否需要同步梯度
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)  # 记录训练损失
                train_loss = 0.0

                if accelerator.is_main_process:
                    if global_step % args.ckpt_interval == 0:
                        save_dir = os.path.join(args.output_dir, f"global_step_{global_step}")
                        os.makedirs(save_dir, exist_ok=True)
                        torch.save(model.unet.state_dict(), os.path.join(save_dir, "unet.pth"))
                        torch.save(model.style_encoder.state_dict(), os.path.join(save_dir, "style_encoder.pth"))
                        torch.save(model.content_encoder.state_dict(), os.path.join(save_dir, "content_encoder.pth"))
                        torch.save(model, os.path.join(save_dir, "total_model.pth"))

                        # 保存断点信息到根目录
                        checkpoint = {
                            'epoch': epoch,
                            'global_step': global_step,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                            'args': vars(args)  # 保存训练参数配置
                        }
                        torch.save(checkpoint, os.path.join(args.output_dir, "latest_checkpoint.pth"))
                        logging.info(f"在步骤:{global_step}中保存的 Checkpoint ")
                        print(f"在全局步骤上保存检查点 {global_step}")  # 记录日志

            # 日志记录
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if global_step % args.log_interval == 0:
                logging.info(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}] 全局步骤 {global_step} => train_loss = {loss}")
            progress_bar.set_postfix(**logs)

            # 检查是否达到最大训练步骤
            if global_step >= args.max_train_steps:
                break

    accelerator.end_training()


if __name__ == "__main__":
    main()
