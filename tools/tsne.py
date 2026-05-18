import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# 读取特征文件
# data = torch.load('/home/wenbo/data/projects/Mask2Former_ConOcc/feature_map_no.pth')
data = torch.load('/home/wenbo/data/projects/Mask2Former_ConOcc/feature_map.pth')

features = torch.cat([f.view(1, -1) for f in data['features']], dim=0).cpu().numpy()
labels = np.array(data['occlusion_label'])

# 随机抽取部分样本用于可视化
num_samples = 200
if features.shape[0] > num_samples:
    idx = np.random.choice(features.shape[0], num_samples, replace=False)
    features = features[idx]
    labels = labels[idx]

# 标准化 + t-SNE 降维
features = StandardScaler().fit_transform(features)
tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200, max_iter=1000)
features_2d = tsne.fit_transform(features)

# 绘图
plt.figure(figsize=(6, 6))
color_map = {'low': '#2ca02c', 'mid': '#ff7f0e', 'high': '#1f77b4'}
labels_name = {'low': 'Low Occlusion', 'mid': 'Mid Occlusion', 'high': 'High Occlusion'}

for key in ['low', 'mid', 'high']:
    idx = labels == key
    if np.sum(idx) > 0:
        plt.scatter(
            features_2d[idx, 0],
            features_2d[idx, 1],
            s=12,
            color=color_map[key],
            label=labels_name[key],
            alpha=0.65,
            edgecolors='none'
        )

# ===== 图例美化 =====
legend = plt.legend(
    # title="Occlusion Level",
    fontsize=11,
    title_fontsize=12,
    loc='lower right',           # 位置：右下角（常见论文风格）
    frameon=True,                # 开启边框
    framealpha=0.9,              # 轻微透明
    facecolor='white',           # 白底
    edgecolor='gray',            # 灰色边框
    handlelength=1.6,            # 控制小圆点大小
    handletextpad=0.4,           # 圆点与文字间距
    borderpad=0.8                # 图例内部间距
)
legend.get_title().set_weight('bold')  # 图例标题加粗

# 其他图形样式
# plt.title("t-SNE Visualisation of Baseline Features", fontsize=13, pad=10)
plt.title("t-SNE Visualisation of Contrastive Features", fontsize=13, pad=10)

plt.xticks([]); plt.yticks([])
plt.grid(False)
plt.tight_layout()

# save_path = 'tsne_no.pdf'
save_path = 'tsne.pdf'

plt.savefig(save_path, dpi=600, bbox_inches='tight', transparent=True)
plt.show()
