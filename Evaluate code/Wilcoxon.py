import re
import os
from scipy.stats import wilcoxon

# =========================
# 输入文件
# =========================
dcs_path = "funit_ufuc_youhua.txt"
msd_path = "results_msd_ufuc.txt"

# =========================
# 解析 txt 文件
# =========================
def parse_metrics(txt_path):

    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.split("========================")

    results = {}

    for block in blocks:

        style_match = re.search(r"=== Style:\s*(.*?)\s*===", block)

        if not style_match:
            continue

        style = style_match.group(1).strip()

        fid_match = re.search(r"FID:\s*([0-9.]+)", block)
        ssim_match = re.search(r"SSIM \(avg\):\s*([0-9.]+)", block)
        lpips_match = re.search(r"LPIPS \(avg\):\s*([0-9.]+)", block)
        l1_match = re.search(r"L1 \(avg\):\s*([0-9.]+)", block)

        results[style] = {
            "FID": float(fid_match.group(1)),
            "SSIM": float(ssim_match.group(1)),
            "LPIPS": float(lpips_match.group(1)),
            "L1": float(l1_match.group(1)),
        }

    return results

# =========================
# 读取结果
# =========================
dcs_results = parse_metrics(dcs_path)
msd_results = parse_metrics(msd_path)

# =========================
# 获取共同 style
# =========================
common_styles = sorted(
    set(dcs_results.keys()) & set(msd_results.keys())
)

print(f"Number of paired styles: {len(common_styles)}")

if len(common_styles) == 0:
    raise ValueError("No common styles found!")

# =========================
# Wilcoxon Test
# =========================
metrics = ["FID", "SSIM", "LPIPS", "L1"]

output_lines = []

header = (
    "===== Wilcoxon Signed-Rank Test Results =====\n\n"
    f"Number of paired styles: {len(common_styles)}\n\n"
)

print(header)
output_lines.append(header)

for metric in metrics:

    dcs_values = []
    msd_values = []

    for style in common_styles:

        dcs_values.append(dcs_results[style][metric])
        msd_values.append(msd_results[style][metric])

    # Wilcoxon signed-rank test
    stat, p_value = wilcoxon(dcs_values, msd_values)

    # 显著性判断
    if p_value < 0.001:
        significance = "*** (p < 0.001)"
    elif p_value < 0.01:
        significance = "** (p < 0.01)"
    elif p_value < 0.05:
        significance = "* (p < 0.05)"
    else:
        significance = "Not Significant"

    result_text = (
        f"{metric}:\n"
        f"Wilcoxon statistic = {stat:.4f}\n"
        f"p-value = {p_value:.6f}\n"
        f"Significance = {significance}\n\n"
    )

    print(result_text)
    output_lines.append(result_text)

# =========================
# 输出 txt 文件
# =========================
output_path = "wilcoxon_signed_rank_results.txt"

with open(output_path, "w", encoding="utf-8") as f:
    f.writelines(output_lines)

print(f"Results saved to: {output_path}")