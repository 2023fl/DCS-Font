import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors  # 新增：用于最近邻计算
from scipy.spatial.distance import pdist  # 新增：用于距离计算
from tqdm import tqdm


# 1. t-SNE专用数据加载器（保持数据顺序固定）
class TSNE_Dataset(Dataset):
    """专门用于t-SNE可视化的字体风格数据集加载器（固定样本顺序）"""

    def __init__(self, data_root, resolution=128, transform=None):
        self.data_root = data_root
        self.resolution = resolution
        self.transform = transform

        self.image_paths = []  # 按固定顺序存储图像路径
        self.style_labels = []  # 对应风格标签（与图像路径顺序一致）
        self.style_names = []  # 所有独特风格名称（按字母顺序固定）

        # 按字母顺序遍历风格文件夹（确保每次加载顺序一致）
        for style in sorted(os.listdir(data_root)):
            style_dir = os.path.join(data_root, style)
            if not os.path.isdir(style_dir):
                continue

            self.style_names.append(style)

            # 按字母顺序遍历图像文件
            for img_name in sorted(os.listdir(style_dir)):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(style_dir, img_name)
                    self.image_paths.append(img_path)
                    self.style_labels.append(style)

        if len(self.image_paths) == 0:
            raise ValueError(f"在 {data_root} 下未找到任何图像文件，请检查目录结构")

        print(f"成功加载数据集：共 {len(self.style_names)} 种风格，{len(self.image_paths)} 张图像")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"无法打开图像 {img_path}：{str(e)}")

        style_label = self.style_labels[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, style_label


# 2. 可视化+指标计算函数
def visualize_style_features(model, dataloader, dataset,
                             device, num_samples=8000,
                             feature_type='style_emd', save_path='tsne_visualization.png',
                             perplexity=30, random_state=42,
                             # 新增：指标计算参数
                             k_neighbors=5):  # 计算类间混淆率时的近邻数量
    """包含类内平均距离和类间混淆率计算"""
    model.eval()
    features = []
    labels = []
    total_collected = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="提取风格特征"):
            images, batch_labels = batch
            images = images.to(device)

            # 模型前向传播
            style_emd, global_feat, _ = model(images)
            if feature_type == 'global':
                feat = global_feat
            else:
                feat = nn.functional.adaptive_avg_pool2d(style_emd, (1, 1)).view(style_emd.size(0), -1)

            # 收集样本（固定数量）
            remaining = num_samples - total_collected
            if remaining <= 0:
                break
            feat_batch = feat.cpu().numpy()[:remaining]
            labels_batch = batch_labels[:remaining]
            features.append(feat_batch)
            labels.extend(labels_batch)
            total_collected += len(feat_batch)

    features = np.concatenate(features, axis=0)
    labels = labels[:num_samples]

    # 固定风格到ID的映射
    unique_styles = dataset.style_names
    style_to_id = {style: i for i, style in enumerate(unique_styles)}
    labels_id = np.array([style_to_id[style] for style in labels])
    num_classes = len(unique_styles)

    # --------------------------
    # 新增：计算类内平均距离
    # --------------------------
    print("\n计算类内平均距离...")
    intra_distances = []  # 存储每个类别的平均距离
    for class_id in range(num_classes):
        # 提取当前类别的所有特征
        class_features = features[labels_id == class_id]
        if len(class_features) < 2:
            # 样本数不足2，无法计算距离，跳过
            intra_distances.append(0.0)
            continue
        # 计算该类别内所有样本间的欧氏距离，取平均值
        distances = pdist(class_features, metric='euclidean')  #  pairwise距离
        intra_avg = np.mean(distances)
        intra_distances.append(intra_avg)
    # 所有类别的平均类内距离
    overall_intra_avg = np.mean(intra_distances)

    # --------------------------
    # 新增：计算类间混淆率
    # --------------------------
    print("计算类间混淆率...")
    # 构建最近邻模型
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1,  # +1是因为会包含样本自身
                            metric='euclidean',
                            n_jobs=-1).fit(features)
    # 找到每个样本的k+1个最近邻（第0个是自身）
    distances, indices = nbrs.kneighbors(features)
    # 统计每个样本的邻居中不同类别的比例
    confusion_ratios = []
    for i in range(len(features)):
        # 当前样本的类别
        current_class = labels_id[i]
        # 排除自身后的k个邻居
        neighbor_classes = labels_id[indices[i][1:]]  # indices[i][0]是自身
        # 计算不同类别的数量占比
        non_self_count = np.sum(neighbor_classes != current_class)
        confusion_ratio = non_self_count / k_neighbors
        confusion_ratios.append(confusion_ratio)
    # 整体类间混淆率（所有样本的平均）
    overall_confusion_rate = np.mean(confusion_ratios)

    # --------------------------
    # 输出指标结果
    # --------------------------
    print("\n===== 特征评估指标 =====")
    print(f"1. 类内平均距离: {overall_intra_avg:.4f}")
    print(f"   （每个类别内部样本的平均距离，值越小越好）")
    print(f"2. 类间混淆率（k={k_neighbors}）: {overall_confusion_rate:.4f}")
    print(f"   （每个样本的近邻中不同类别的比例，值越小越好）")
    print("=======================\n")

    # t-SNE降维（保持不变）
    print("正在进行t-SNE降维...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        n_jobs=-1
    )
    features_tsne = tsne.fit_transform(features)

    # --------------------------
    # 样式参数调整区（所有可调参数集中在此）
    # --------------------------
    # 1. 字体大小参数（根据需要修改数值）
    font_sizes = {
        'title': 20,          # 图表标题字号
        'axis_label': 18,     # 坐标轴标签字号
        'tick_label': 16,     # 刻度标签字号
        'legend_title': 16,   # 图例标题字号
        'legend_label': 10    # 图例标签字号
    }

    # 2. 刻度符号参数（根据需要修改数值）
    tick_params = {
        'major_size': 4,      # 主刻度符号长度
        'minor_size': 4,      # 副刻度符号长度（若显示）
        'width': 4            # 刻度线粗细
    }

    # 3. 边框参数（根据需要修改数值）
    border_width = 2        # 图表边框粗细

    # --------------------------
    # 绘图逻辑（使用上述参数）
    # --------------------------
    plt.figure(figsize=(12, 10))
    ax = plt.gca()  # 获取当前轴对象，用于调整边框和刻度

    # 设置颜色映射
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_styles)))
    cmap = ListedColormap(colors)

    # 绘制散点图
    scatter = ax.scatter(
        features_tsne[:, 0],
        features_tsne[:, 1],
        c=labels_id,
        cmap=cmap,
        alpha=0.7,
        s=50
    )

    # 设置图表标题（使用字体参数）
    ax.set_title(
        f't-SNE visualizes the style feature distribution',
        fontsize=font_sizes['title']
    )

    # 设置坐标轴标签（使用字体参数）
    ax.set_xlabel('t-SNE dimension 1', fontsize=font_sizes['axis_label'])
    ax.set_ylabel('t-SNE dimension 2', fontsize=font_sizes['axis_label'])

    # 设置刻度样式（字体大小和符号大小）
    ax.tick_params(
        axis='both',          # 同时设置x和y轴
        which='major',        # 应用于主刻度
        labelsize=font_sizes['tick_label'],  # 刻度标签字号
        length=tick_params['major_size'],    # 刻度符号长度
        width=tick_params['width']           # 刻度线粗细
    )
    ax.tick_params(
        axis='both',
        which='minor',        # 应用于副刻度（默认不显示，若需显示可开启）
        length=tick_params['minor_size'],
        width=tick_params['width']
    )

    # 设置边框粗细
    for spine in ax.spines.values():  # 遍历所有边框（上、下、左、右）
        spine.set_linewidth(border_width)

    # 设置图例（使用字体参数）
    handles, _ = scatter.legend_elements()
    ax.legend(
        handles,
        unique_styles,
        title="Font style",
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=font_sizes['legend_label'],    # 图例标签字号
        title_fontsize=font_sizes['legend_title']  # 图例标题字号
    )

    plt.tight_layout()

    # 保存图像
    dir_name = os.path.dirname(save_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"可视化结果已保存至: {save_path}")
    plt.close()

    return features_tsne, overall_intra_avg, overall_confusion_rate  # 返回指标


# 3. 主函数
if __name__ == "__main__":
    # 1. 固定全局随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 2. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 3. 初始化风格编码器
    model = StyleEncoder(  # 假设已定义StyleEncoder
        G_ch=64,
        resolution=128,
        G_activation=nn.ReLU(inplace=False)
    ).to(device)

    # 4. 配置数据参数
    data_root = r"E:\Files\Project-E\DSFont-CS\data_examples\t-SNE\text2"
    resolution = 128

    # 5. 定义图像变换
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 6. 实例化数据集
    dataset = TSNE_Dataset(
        data_root=data_root,
        resolution=resolution,
        transform=transform
    )

    # 7. 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    # 8. 验证数据加载
    print(f"数据集总样本数: {len(dataset)}")
    for batch in dataloader:
        images, labels = batch
        print(f"批次图像形状: {images.shape}")
        print(f"批次风格标签示例: {labels[:5]}")
        break

    # 9. 可视化并计算指标（可调整k_neighbors参数）
    visualize_style_features(
        model=model,
        dataloader=dataloader,
        dataset=dataset,
        device=device,
        num_samples=8000,
        feature_type='global',
        save_path='tsne_global_features.png',
        perplexity=30,
        random_state=42,
        k_neighbors=5  # 近邻数量，可根据需求调整（如3、5、10）
    )

    visualize_style_features(
        model=model,
        dataloader=dataloader,
        dataset=dataset,
        device=device,
        num_samples=8000,
        feature_type='style_emd',
        save_path='visualizations/tsne_style_emd_features.png',
        perplexity=30,
        random_state=42,
        k_neighbors=5
    )

