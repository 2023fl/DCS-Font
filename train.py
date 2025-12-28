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
# 主要用于从命令行参数或环境变量中获取配置
def get_args():
    parser = get_parser()
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    style_image_size = args.style_image_size                # 处理图像大小参数
    content_image_size = args.content_image_size
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)

    return args


def main():

    # 获取命令行参数
    args = get_args()
    # args.output_dir = 'D:/FILES/Project/pytorch/Github/FontDiffuser/myfile_mane'    # 自己加的
    # 设置日志目录
    logging_dir = f"{args.output_dir}/{args.logging_dir}"
    # 初始化加速器
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir)
    # 检查当前进程是否是主进程
    if accelerator.is_main_process:
        # 如果当前进程是主进程，创建输出目录（如果不存在）
        os.makedirs(args.output_dir, exist_ok=True)      # os.makedirs(args.output_dir, exist_ok=True)
    # 配置日志、日志将写入到output_dir目录下的fontdiffuser_training.log文件中
    logging.basicConfig(
        filename=f"{args.output_dir}/fontdiffuser_training.log",       # filename=f"{args.output_dir}/fontdiffuser_training.log",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    # Ser training seed设置训练种子，调用set_seed函数来设置随机种子，确保实验的可重复性
    if args.seed is not None:
        set_seed(args.seed)

    # Load model and noise_scheduler加载模型和噪声调度器：通过调用不同的构建函数，来加载或初始化U-Net、风格编码器、内容编码器和噪声调度器。
    unet = build_unet(args=args)
    # print("unet:",unet)
    style_encoder = build_style_encoder(args=args)
    content_encoder = build_content_encoder(args=args)
    noise_scheduler = build_ddpm_scheduler(args)
    # 加载第一阶段的模型权重：从指定的目录phase_1_ckpt_dir中加载U-Net、风格编码器和内容编码器的权重
    if args.phase_2:
        unet.load_state_dict(torch.load(f"{args.phase_1_ckpt_dir}/unet.pth"))
        style_encoder.load_state_dict(torch.load(f"{args.phase_1_ckpt_dir}/style_encoder.pth"))
        content_encoder.load_state_dict(torch.load(f"{args.phase_1_ckpt_dir}/content_encoder.pth"))
    # 构建完整的FontDiffuser模型        重点！！！！！
    model = FontDiffuserModel(
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder)

    # Build content perceptaual Loss构建内容感知损失函数：该损失函数可能用于在训练过程中评估生成的字体与参考字体之间的内容相似性
    perceptual_loss = ContentPerceptualLoss()

    # 加载 SCR 模块进行监控
    if args.phase_2:
        scr = build_scr(args=args)
        scr.load_state_dict(torch.load(args.scr_ckpt_path))
        scr.requires_grad_(False)

    # 加载数据集
    '''
    定义数据转换（Transforms）：
content_transforms：用于内容图像的数据转换。首先，将图像调整为args.content_image_size大小，然后使用双线性插值进行缩放。
                    接着，将图像转换为PyTorch张量，并对其进行归一化，使得像素值范围从[0, 1]转换到[-1, 1]。
style_transforms：用于风格图像的数据转换。与content_transforms类似，但图像大小调整为args.style_image_size。
target_transforms：用于目标图像的数据转换。图像大小被调整为(args.resolution, args.resolution)，并同样进行归一化。
    '''
    content_transforms = transforms.Compose(
        [transforms.Resize(args.content_image_size, 
                           interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    style_transforms = transforms.Compose(
        [transforms.Resize(args.style_image_size, 
                           interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    target_transforms = transforms.Compose(
        [transforms.Resize((args.resolution, args.resolution), 
                           interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])

    # 创建数据加载器（DataLoader）：
    # 使用PyTorch的torch.utils.data.DataLoader类创建了一个名为train_dataloader的数据加载器。
    # 这个加载器会从train_font_dataset中加载数据，并在每个epoch开始时对数据进行随机打乱（shuffle = True）。
    # 它使用args.train_batch_size作为批处理大小，并使用一个名为CollateFN()的自定义collate_fn函数来合并样本（这通常用于处理非标准的数据结构或执行额外的数据预处理）
    train_font_dataset = FontDataset(
        args=args,
        phase='train', 
        transforms=[
            content_transforms, 
            style_transforms, 
            target_transforms],
        scr=args.phase_2)
    train_dataloader = torch.utils.data.DataLoader(
        train_font_dataset, shuffle=True, batch_size=args.train_batch_size, collate_fn=CollateFN())
    
    # Build optimizer and learning rate
    # 设置学习率和优化器：
    # 根据args.scale_lr选项，代码可能会调整学习率（args.learning_rate）。如果args.scale_lr为True，则学习率会根据梯度累积步数（args.gradient_accumulation_steps）、批处理大小（args.train_batch_size）和加速器的进程数（accelerator.num_processes）进行调整。
    # 使用torch.optim.AdamW创建了一个AdamW优化器，这是一种带有权重衰减的Adam优化器变种。优化器用于更新模型的参数。
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon)
    # 设置学习率调度器：
    # 使用get_scheduler函数获取一个学习率调度器（lr_scheduler）。调度器用于在训练过程中调整学习率。
    # 预热步数（args.lr_warmup_steps）和总训练步数（args.max_train_steps）。
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,)

    # Accelerate preparation
    # 模型、优化器、数据加载器和调度器的准备：
    # 使用accelerator.prepare方法对模型、优化器、数据加载器和调度器进行准备。
    # 这通常涉及将模型、优化器等移动到正确的设备（如GPU或TPU），并可能包含分布式训练的准备工作。
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler)

    #   move scr module to the target deivces将SCR模块移动到目标设备
    if args.phase_2:
        scr = scr.to(accelerator.device)

    # The trackers initializes automatically on the main process.
    # 初始化跟踪器和保存配置：
    # 如果当前进程是主进程，则使用accelerator.init_trackers初始化跟踪器，并使用save_args_to_yaml函数将命令行参数保存到YAML文件中
    if accelerator.is_main_process:
        accelerator.init_trackers(args.experience_name)
        save_args_to_yaml(args=args, output_file=f"{args.output_dir}/{args.experience_name}_config.yaml")

    # Only show the progress bar once on each machine.创建进度条：使用tqdm库创建一个进度条，用于在训练过程中显示进度
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # Convert to the training epoch
    # 计算训练周期数和更新步数：
    # 根据数据加载器的长度、梯度累积步数和最大训练步数，计算出每个训练周期的更新步数和总训练周期数。这有助于在训练过程中跟踪进度
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # 以下为自己加的检查点恢复
    # 初始化全局步数
    # start_global_step = 0

    # # 检查是否需要从检查点恢复训练
    # if args.resume_from_checkpoint is not None:
    #     try:
    #         # 加载检查点
    #         checkpoint = torch.load(args.resume_from_checkpoint)
    #
    #         # 如果是保存的整个模型
    #         if isinstance(checkpoint, FontDiffuserModel):
    #             model = checkpoint
    #         else:
    #             # 恢复模型状态
    #             model.unet.load_state_dict(torch.load(f"{os.path.dirname(args.resume_from_checkpoint)}/unet.pth"))
    #             model.style_encoder.load_state_dict(
    #                 torch.load(f"{os.path.dirname(args.resume_from_checkpoint)}/style_encoder.pth"))
    #             model.content_encoder.load_state_dict(
    #                 torch.load(f"{os.path.dirname(args.resume_from_checkpoint)}/content_encoder.pth"))
    #
    #         # 推荐：在保存检查点时同时保存优化器、学习率调度器和全局步数
    #         optimizer_path = f"{os.path.dirname(args.resume_from_checkpoint)}/optimizer.pth"
    #         lr_scheduler_path = f"{os.path.dirname(args.resume_from_checkpoint)}/lr_scheduler.pth"
    #
    #         if os.path.exists(optimizer_path):
    #             optimizer.load_state_dict(torch.load(optimizer_path))
    #
    #         if os.path.exists(lr_scheduler_path):
    #             lr_scheduler.load_state_dict(torch.load(lr_scheduler_path))
    #
    #         # 恢复全局步数
    #         if args.resume_global_step is not None:
    #             start_global_step = args.resume_global_step
    #         else:
    #             # 从文件名中提取全局步数
    #             start_global_step = int(os.path.basename(os.path.dirname(args.resume_from_checkpoint)).split('_')[-1])
    #
    #         logging.info(f"Resumed training from checkpoint {args.resume_from_checkpoint} at step {start_global_step}")
    #
    #     except Exception as e:
    #         logging.error(f"Failed to load checkpoint: {e}")
    #         logging.info("Starting training from scratch")
    #         start_global_step = 0

    global_step = 0     # 初始化全局步长
    # 训练循环
    # global_step = start_global_step    # 自己加的
    # 训练循环：外层循环遍历整个训练数据集num_train_epochs次
    for epoch in range(num_train_epochs):
        train_loss = 0.0  # 计算训练损失
        for step, samples in enumerate(train_dataloader):   # 数据加载循环:内层循环遍历数据加载器中的每个批次
            model.train()   # 确保模型处于训练模式
            # 处理输入数据：从数据批次中提取所需的内容图像、风格图像、目标图像和非归一化的目标图像
            content_images = samples["content_image"]
            style_images = samples["style_image"]
            target_images = samples["target_image"]
            nonorm_target_images = samples["nonorm_target_image"]
            # 使用如分布式训练来累积梯度。这意味着在每个with块内的操作会累积梯度，而不是立即进行反向传播和更新
            with accelerator.accumulate(model):
                # 添加噪声和选择时间步
                # Sample noise that we'll add to the samples
                noise = torch.randn_like(target_images)
                bsz = target_images.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=target_images.device)
                timesteps = timesteps.long()

                # Add noise to the target_images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                # 根据每个时间步长的噪声幅度向target_images添加噪声
                # （这是前向扩散过程）
                noisy_target_images = noise_scheduler.add_noise(target_images, noise, timesteps)

                # Classifier-free training strategy无分类器训练策略
                # 生成掩码:这行代码生成一个伯努利分布的掩码，用于决定哪些图像的内容和风格将被设置为全1。bsz 是批次大小，args.drop_prob 是决定伯努利分布的参数，用于生成接近 0 或 1 的值。
                context_mask = torch.bernoulli(torch.zeros(bsz) + args.drop_prob)
                # 设置内容和风格图像
                for i, mask_value in enumerate(context_mask):
                    if mask_value==1:
                        content_images[i, :, :, :] = 1
                        style_images[i, :, :, :] = 1

                # Predict the noise residual and compute loss
                # 预测噪声残差并计算损失
                noise_pred, offset_out_sum = model(
                    x_t=noisy_target_images, 
                    timesteps=timesteps, 
                    style_images=style_images,
                    content_images=content_images,
                    content_encoder_downsample_size=args.content_encoder_downsample_size)
                diff_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")   # MSE扩散损耗
                offset_loss = offset_out_sum / 2   # 偏移损失
                
                # output processing for content perceptual loss
                # 内容感知损失
                pred_original_sample_norm = x0_from_epsilon(
                    scheduler=noise_scheduler,
                    noise_pred=noise_pred,
                    x_t=noisy_target_images,
                    timesteps=timesteps)
                pred_original_sample = reNormalize_img(pred_original_sample_norm)   # 重新归一化处理，以确保其像素值在合适的范围内
                norm_pred_ori = normalize_mean_std(pred_original_sample)     # 标准化处理
                norm_target_ori = normalize_mean_std(nonorm_target_images)    # 标准化处理
                # 这行代码调用 perceptual_loss 对象的 calculate_loss 方法来计算感知损失。它比较经过标准化处理的生成图像 norm_pred_ori 和目标图像 norm_target_ori 之间的差异。device 参数指定了用于计算的设备
                percep_loss = perceptual_loss.calculate_loss(
                    generated_images=norm_pred_ori,
                    target_images=norm_target_ori,
                    device=target_images.device)
                # 总损失:总损失是噪声预测损失、内容感知损失和偏移损失的加权和。
                #      `args.perceptual_coefficient`和`args.offset_coefficient`是这些不同损失的权重/系数。
                loss = diff_loss + \
                        args.perceptual_coefficient * percep_loss + \
                            args.offset_coefficient * offset_loss

                # 第二阶段训练:
                # 从samples字典中获取neg_images，这些可能是用于对比学习的负样本图像。
                # 使用风格对比（SCR）的方法scr来计算正样本（目标图像）和负样本（neg_images）的风格嵌入。
                # 计算风格对比损失（sc_loss），这是通过比较正样本和负样本的风格嵌入来完成的。
                # 将风格对比损失乘以一个系数（args.sc_coefficient）并加到总损失loss上。
                if args.phase_2:
                    neg_images = samples["neg_images"]  # 负样本图像
                    # sc loss
                    sample_style_embeddings, pos_style_embeddings, neg_style_embeddings = scr(
                        pred_original_sample_norm, 
                        target_images, 
                        neg_images, 
                        nce_layers=args.nce_layers)
                    sc_loss = scr.calculate_nce_loss(
                        sample_s=sample_style_embeddings,
                        pos_s=pos_style_embeddings,
                        neg_s=neg_style_embeddings)
                    loss += args.sc_coefficient * sc_loss

                # Gather the losses across all processes for logging (if we use distributed training).
                # 分布式训练的损失聚合:
                # 如果使用了分布式训练，则需要将所有进程中的损失进行聚合以便记录。这里使用了一个名为accelerator的对象，它负责跨进程聚合损失。
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()  # 这一行将每个进程中的损失聚合在一起，并计算平均值。这是因为在分布式训练中，每个进程可能只处理批次的一部分数据
                train_loss += avg_loss.item() / args.gradient_accumulation_steps   # 将聚合后的平均损失添加到train_loss中以供后续记录或监控

                # Backpropagate反向传播和优化步骤
                accelerator.backward(loss) # 使用accelerator.backward(loss)进行反向传播，该操作会计算损失函数关于模型参数的梯度
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)     #  如果启用了梯度同步（accelerator.sync_gradients），则使用accelerator.clip_grad_norm_来裁剪梯度，防止梯度爆炸。
                optimizer.step()      # 使用优化器（optimizer）执行一步参数更新。
                lr_scheduler.step()   # 使用学习率调度器（lr_scheduler）更新学习率。
                optimizer.zero_grad()  # 使用optimizer.zero_grad()清除之前计算的梯度，为下一次迭代做准备


            # Checks if the accelerator has performed an optimization step behind the scenes
            # 检查梯度同步并更新进度条:
            # 使用progress_bar.update(1)更新进度条，表示完成了一个训练步骤。
            # 增加全局步骤计数器global_step。
            # 使用accelerator.log记录训练损失train_loss。
            # 重置train_loss为0.0，以便在下一个训练周期中重新累积。
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0


                # 保存模型检查点:
                # 创建一个保存目录，格式为{args.output_dir} / global_step_{global_step}。
                # 使用torch.save保存模型的各个部分的状态字典（state_dict），包括unet、style_encoder、content_encoder，以及整个模型的状态。
                # 使用logging.info和print记录保存检查点的信息。
                if accelerator.is_main_process:
                    if global_step % args.ckpt_interval == 0:
                        save_dir = f"{args.output_dir}/global_step_{global_step}"
                        os.makedirs(save_dir, exist_ok=True)

                        torch.save(model.unet.state_dict(), f"{save_dir}/unet.pth")
                        torch.save(model.style_encoder.state_dict(), f"{save_dir}/style_encoder.pth")
                        torch.save(model.content_encoder.state_dict(), f"{save_dir}/content_encoder.pth")
                        torch.save(model, f"{save_dir}/total_model.pth")
                        logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}] Save the checkpoint on global step {global_step}")
                        print("Save the checkpoint on global step {}".format(global_step))
            # 记录日志:
            # 创建一个包含当前步骤损失（loss.detach().item()）和学习率（lr_scheduler.get_last_lr()[0]）的字典logs。
            # 如果global_step是日志记录间隔（args.log_interval）的倍数，则使用logging.info记录全局步骤和训练损失。
            # 使用progress_bar.set_postfix(**logs)更新进度条的后缀，显示当前步骤的损失和学习率。
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if global_step % args.log_interval == 0:
                logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}] Global Step {global_step} => train_loss = {loss}")
            progress_bar.set_postfix(**logs)
            
            # Quit
            # 退出条件:
            # 如果global_step达到了最大训练步骤数（args.max_train_steps），则退出训练循环。
            if global_step >= args.max_train_steps:
                break
    # 结束训练:
    # 在训练循环结束后，调用accelerator.end_training()来执行任何必要的清理或资源释放操作。这通常是特定于加速库（如PyTorchLightning）的调用。
    accelerator.end_training()

if __name__ == "__main__":
    main()