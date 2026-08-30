import os
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import lpips
from pytorch_fid import fid_score
from skimage.metrics import structural_similarity as ssim
import shutil
from tqdm import tqdm

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义图像预处理转换
transform_real = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor()
])

transform_generated = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor()
])

def get_style_fonts(genimgs_dir):
    """获取genimgs_dir下所有风格目录"""
    return [d for d in os.listdir(genimgs_dir) if os.path.isdir(os.path.join(genimgs_dir, d))]

def calculate_fid(real_dir, generated_dir):
    fid_value = fid_score.calculate_fid_given_paths([real_dir, generated_dir],
                                                    batch_size=128,
                                                    device=device,
                                                    dims=2048)
    return fid_value

def calculate_ssim(img1_path, img2_path):
    try:
        img1 = Image.open(img1_path).convert('L')
        img2 = Image.open(img2_path).convert('L')
    except Exception as e:
        raise ValueError(f"无法读取图像: {e}")

    img1 = img1.resize((96, 96))
    img2 = img2.resize((96, 96))

    img1_np = np.array(img1)
    img2_np = np.array(img2)

    ssim_value, _ = ssim(img1_np, img2_np, full=True)
    return ssim_value

def calculate_lpips(img1, img2, loss_fn):
    if img1.size() != img2.size():
        img2 = torch.nn.functional.interpolate(img2.unsqueeze(0), size=(96, 96), mode='bilinear',
                                               align_corners=False).squeeze(0)
    img1 = img1.to(device)
    img2 = img2.to(device)
    lpips_value = loss_fn(img1, img2).item()
    return lpips_value

def calculate_l1(img1, img2):
    img1_np = img1.cpu().numpy().transpose(1, 2, 0)
    img2_np = img2.cpu().numpy().transpose(1, 2, 0)
    l1 = np.mean(np.abs(img1_np - img2_np))
    return l1

def evaluate_style(genimgs_dir, gtimgs_dir, style_name, results_file):
    """评估单个风格的指标"""
    gt_dir = os.path.join(gtimgs_dir, style_name)
    gen_dir = os.path.join(genimgs_dir, style_name)

    if not (os.path.exists(gt_dir) and os.path.exists(gen_dir)):
        print(f"目录不存在: {gt_dir} 或 {gen_dir}")
        return

    # 初始化LPIPS
    loss_fn = lpips.LPIPS(net='alex').to(device)

    metrics = {
        'fid': None,
        'ssim': [],
        'lpips': [],
        'l1': []
    }

    # 1. 计算FID
    try:
        fid = calculate_fid(gt_dir, gen_dir)
        if isinstance(fid, complex):
            print(f"警告: FID为复数值 {fid}，使用其实部")
            fid = fid.real
        metrics['fid'] = fid
    except Exception as e:
        print(f"计算FID失败: {str(e)}")
        metrics['fid'] = None

    # 2. 计算其他指标
    try:
        gt_images = sorted([f for f in os.listdir(gt_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        gen_images = sorted([f for f in os.listdir(gen_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

        for gt_img, gen_img in zip(gt_images, gen_images):
            gt_path = os.path.join(gt_dir, gt_img)
            gen_path = os.path.join(gen_dir, gen_img)

            try:
                ssim_value = calculate_ssim(gt_path, gen_path)
                metrics['ssim'].append(ssim_value)
            except Exception as e:
                print(f"计算SSIM失败 ({gt_img}): {str(e)}")

            try:
                img1 = transform_real(Image.open(gt_path).convert('RGB'))
                img2 = transform_generated(Image.open(gen_path).convert('RGB'))

                lpips_value = calculate_lpips(img1, img2, loss_fn)
                metrics['lpips'].append(lpips_value)

                l1_value = calculate_l1(img1, img2)
                metrics['l1'].append(l1_value)
            except Exception as e:
                print(f"计算LPIPS/L1失败 ({gt_img}): {str(e)}")

    except Exception as e:
        print(f"处理图像时出错: {str(e)}")

    with open(results_file, 'a', encoding='utf-8') as f:
        f.write(f"\n=== Style: {style_name} ===\n")

        if metrics['fid'] is not None:
            f.write(f"FID: {metrics['fid']:.4f}\n")
        else:
            f.write("FID: 计算失败\n")

        if metrics['ssim']:
            f.write(f"SSIM (avg): {np.mean(metrics['ssim']):.4f}\n")
            f.write(f"SSIM (std): {np.std(metrics['ssim']):.4f}\n")
        else:
            f.write("SSIM: 计算失败\n")

        if metrics['lpips']:
            f.write(f"LPIPS (avg): {np.mean(metrics['lpips']):.4f}\n")
            f.write(f"LPIPS (std): {np.std(metrics['lpips']):.4f}\n")
        else:
            f.write("LPIPS: 计算失败\n")

        if metrics['l1']:
            f.write(f"L1 (avg): {np.mean(metrics['l1']):.4f}\n")
            f.write(f"L1 (std): {np.std(metrics['l1']):.4f}\n")
        else:
            f.write("L1: 计算失败\n")

        f.write("========================\n")

    print(f"完成风格 {style_name} 的评估")

def calculate_all_styles_fid(genimgs_dir, gtimgs_dir):
    style_fonts = get_style_fonts(genimgs_dir)
    
    # 创建临时目录
    temp_real_dir = os.path.join(os.path.dirname(genimgs_dir), 'temp_real')
    temp_gen_dir = os.path.join(os.path.dirname(genimgs_dir), 'temp_gen')
    
    os.makedirs(temp_real_dir, exist_ok=True)
    os.makedirs(temp_gen_dir, exist_ok=True)
    
    # 复制所有风格的真实和生成图像到临时目录，并重命名避免覆盖
    for style_idx, style_name in enumerate(style_fonts):
        gt_dir = os.path.join(gtimgs_dir, style_name)
        gen_dir = os.path.join(genimgs_dir, style_name)
        
        if not (os.path.exists(gt_dir) and os.path.exists(gen_dir)):
            print(f"目录不存在: {gt_dir} 或 {gen_dir}")
            continue
        
        gt_images = sorted([f for f in os.listdir(gt_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        gen_images = sorted([f for f in os.listdir(gen_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # 确保图像数量匹配
        min_len = min(len(gt_images), len(gen_images))
        gt_images = gt_images[:min_len]
        gen_images = gen_images[:min_len]
        
        for img_idx, (gt_img, gen_img) in enumerate(zip(gt_images, gen_images)):
            # 生成唯一文件名: 风格索引_风格名_图像索引_原文件名
            gt_ext = os.path.splitext(gt_img)[1]
            gen_ext = os.path.splitext(gen_img)[1]
            new_gt_name = f"style_{style_idx}_{style_name}_img_{img_idx}{gt_ext}"
            new_gen_name = f"style_{style_idx}_{style_name}_img_{img_idx}{gen_ext}"
            
            shutil.copy(os.path.join(gt_dir, gt_img), os.path.join(temp_real_dir, new_gt_name))
            shutil.copy(os.path.join(gen_dir, gen_img), os.path.join(temp_gen_dir, new_gen_name))
    
    # 计算FID
    try:
        fid = calculate_fid(temp_real_dir, temp_gen_dir)
        if isinstance(fid, complex):
            print(f"警告: FID为复数值 {fid}，使用其实部")
            fid = fid.real
        return fid
    except Exception as e:
        print(f"计算FID失败: {str(e)}")
        return None
    finally:
        # 删除临时目录
        shutil.rmtree(temp_real_dir, ignore_errors=True)
        shutil.rmtree(temp_gen_dir, ignore_errors=True)

def evaluate_all_styles(genimgs_dir, gtimgs_dir, results_file):
    """评估所有风格的总体指标"""
    metrics = {
        'fid': None,
        'ssim': [],
        'lpips': [],
        'l1': []
    }

    # 1. 动态计算FID
    try:
        fid = calculate_all_styles_fid(genimgs_dir, gtimgs_dir)
        if isinstance(fid, complex):
            print(f"警告: FID为复数值 {fid}，使用其实部")
            fid = fid.real
        metrics['fid'] = fid
    except Exception as e:
        print(f"计算FID失败: {str(e)}")
        metrics['fid'] = None

    # 2. 计算其他指标
    style_fonts = get_style_fonts(genimgs_dir)
    loss_fn = lpips.LPIPS(net='alex').to(device)

    for style_name in style_fonts:
        gt_dir = os.path.join(gtimgs_dir, style_name)
        gen_dir = os.path.join(genimgs_dir, style_name)

        if not (os.path.exists(gt_dir) and os.path.exists(gen_dir)):
            print(f"目录不存在: {gt_dir} 或 {gen_dir}")
            continue

        gt_images = sorted([f for f in os.listdir(gt_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        gen_images = sorted([f for f in os.listdir(gen_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

        for gt_img, gen_img in zip(gt_images, gen_images):
            gt_path = os.path.join(gt_dir, gt_img)
            gen_path = os.path.join(gen_dir, gen_img)

            try:
                ssim_value = calculate_ssim(gt_path, gen_path)
                metrics['ssim'].append(ssim_value)
            except Exception as e:
                print(f"计算SSIM失败 ({gt_img}): {str(e)}")

            try:
                img1 = transform_real(Image.open(gt_path).convert('RGB'))
                img2 = transform_generated(Image.open(gen_path).convert('RGB'))

                lpips_value = calculate_lpips(img1, img2, loss_fn)
                metrics['lpips'].append(lpips_value)

                l1_value = calculate_l1(img1, img2)
                metrics['l1'].append(l1_value)
            except Exception as e:
                print(f"计算LPIPS/L1失败 ({gt_img}): {str(e)}")

    with open(results_file, 'a', encoding='utf-8') as f:
        f.write("\n=== All Styles Combined ===\n")

        if metrics['fid'] is not None:
            f.write(f"FID: {metrics['fid']:.4f}\n")
        else:
            f.write("FID: 计算失败\n")

        if metrics['ssim']:
            f.write(f"SSIM (avg): {np.mean(metrics['ssim']):.4f}\n")
            f.write(f"SSIM (std): {np.std(metrics['ssim']):.4f}\n")
        else:
            f.write("SSIM: 计算失败\n")

        if metrics['lpips']:
            f.write(f"LPIPS (avg): {np.mean(metrics['lpips']):.4f}\n")
            f.write(f"LPIPS (std): {np.std(metrics['lpips']):.4f}\n")
        else:
            f.write("LPIPS: 计算失败\n")

        if metrics['l1']:
            f.write(f"L1 (avg): {np.mean(metrics['l1']):.4f}\n")
            f.write(f"L1 (std): {np.std(metrics['l1']):.4f}\n")
        else:
            f.write("L1: 计算失败\n")

        f.write("========================\n")

    print("完成所有风格的总体评估")

def main(test_individual_styles=True):
    results_file = 'work2_sfuc_6-1.txt'

    with open(results_file, 'w') as f:
        f.write("DSFont Evaluation Results\n")
        f.write("============================\n")

    # base_dir = "samplefyy/UFUC"
    genimgs_dir = "samplefyy/SFUC"
    gtimgs_dir = "/sampling/xhp/SFUC/"

    style_fonts = get_style_fonts(genimgs_dir)
    print(f"Found {len(style_fonts)} styles: {style_fonts}")

    if test_individual_styles:
        for style_name in style_fonts:
            evaluate_style(genimgs_dir, gtimgs_dir, style_name, results_file)
        evaluate_all_styles(genimgs_dir, gtimgs_dir, results_file)
    else:
        evaluate_all_styles(genimgs_dir, gtimgs_dir, results_file)

    print(f"所有评估结果已保存到 {results_file}")

if __name__ == "__main__":
    main(test_individual_styles=False)
