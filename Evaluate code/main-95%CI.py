import re
import os
import numpy as np
from scipy import stats

# =========================
# 输入/输出文件夹
# =========================
input_dir = "txt"
output_dir = "results"

# 自动创建 results 文件夹
os.makedirs(output_dir, exist_ok=True)

# =========================
# 计算 95% CI
# =========================
def compute_ci(data, confidence=0.95):
    data = np.array(data)

    if len(data) < 2:
        return np.mean(data), 0.0

    mean = np.mean(data)
    std = np.std(data, ddof=1)
    n = len(data)

    # t 分布临界值
    t_value = stats.t.ppf((1 + confidence) / 2, df=n - 1)

    margin = t_value * std / np.sqrt(n)

    return mean, margin


# =========================
# 遍历 txt 文件夹
# =========================
txt_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]

if len(txt_files) == 0:
    print("txt 文件夹中没有找到 txt 文件！")

for txt_file in txt_files:

    txt_path = os.path.join(input_dir, txt_file)

    print(f"\n正在处理: {txt_file}")

    # =========================
    # 读取 txt 文件
    # =========================
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    # =========================
    # 正则提取数据
    # =========================
    fid_list = []
    ssim_list = []
    lpips_list = []
    l1_list = []

    # 按 style 分块
    blocks = content.split("========================")

    for block in blocks:

        if "FID:" not in block:
            continue

        # 提取指标
        fid_match = re.search(r"FID:\s*([0-9.]+)", block)
        ssim_match = re.search(r"SSIM \(avg\):\s*([0-9.]+)", block)
        lpips_match = re.search(r"LPIPS \(avg\):\s*([0-9.]+)", block)
        l1_match = re.search(r"L1 \(avg\):\s*([0-9.]+)", block)

        if fid_match:
            fid_list.append(float(fid_match.group(1)))

        if ssim_match:
            ssim_list.append(float(ssim_match.group(1)))

        if lpips_match:
            lpips_list.append(float(lpips_match.group(1)))

        if l1_match:
            l1_list.append(float(l1_match.group(1)))

    # =========================
    # 指标字典
    # =========================
    metrics = {
        "FID": fid_list,
        "SSIM": ssim_list,
        "LPIPS": lpips_list,
        "L1": l1_list
    }

    # =========================
    # 输出文件路径
    # =========================
    base_name = os.path.splitext(txt_file)[0]

    output_path = os.path.join(
        output_dir,
        f"{base_name}_95%CI.txt"
    )

    # =========================
    # 保存结果
    # =========================
    with open(output_path, "w", encoding="utf-8") as f:

        header = "===== 95% Confidence Interval Results =====\n\n"

        print(header)
        f.write(header)

        for name, values in metrics.items():

            if len(values) == 0:

                result_text = (
                    f"{name}:\n"
                    f"No valid data found.\n\n"
                )

            else:

                mean, ci = compute_ci(values)

                result_text = (
                    f"{name}:\n"
                    f"Mean = {mean:.4f}\n"
                    f"95% CI = ±{ci:.4f}\n"
                    f"Final Report: {mean:.4f} ± {ci:.4f}\n\n"
                )

            print(result_text)
            f.write(result_text)

    print(f"结果已保存到: {output_path}")

print("\n全部 txt 文件处理完成！")