"""
详细可视化邻接矩阵 - 每个图单独绘制，信息丰富
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from model import RGCN_v1

# 设置绘图风格
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 100
sns.set_style("whitegrid")


def load_data_and_model():
    """加载数据和训练后的模型"""
    data_dir = os.path.join(os.path.dirname(current_dir), 'data_processing', 'data')
    data = np.load(os.path.join(data_dir, 'prepped_colorado.npz'), allow_pickle=True)

    adj_original = data['dist_matrix']
    basin_names = data['basin_names']

    # 加载模型
    device = torch.device('cpu')
    model = RGCN_v1(
        input_dim=len(data['x_vars']),
        hidden_dim=20,
        adj_matrix=adj_original,
        device=device,
        use_learnable_adj=True
    )

    weight_file = os.path.join(current_dir, 'output', 'finetuned_weights.pth')
    if os.path.exists(weight_file):
        model.load_state_dict(torch.load(weight_file, map_location=device))
        print(f"✓ 成功加载训练后的权重")
    else:
        print(f"✗ 未找到权重文件")
        return None, None, None, None, None

    model.eval()

    with torch.no_grad():
        adaptive_A = model.get_adaptive_adjacency().numpy()
        alpha = torch.sigmoid(model.adj_fusion_weight).item()
        learned_A = torch.sigmoid(model.learnable_A).numpy()

    return adj_original, adaptive_A, learned_A, basin_names, alpha


def plot_matrix_with_details(matrix, basin_names, title, filename,
                             cmap='Blues', description='', vmin=None, vmax=None):
    """
    绘制详细的邻接矩阵图
    """
    n_basins = len(basin_names)

    # 创建大图
    fig = plt.figure(figsize=(16, 14))

    # 主热图
    ax_main = plt.subplot2grid((4, 4), (0, 0), colspan=3, rowspan=3)

    # 使用统一的颜色映射，确保可比性
    if vmin is None:
        vmin = 0  # 权重不会是负数
    if vmax is None:
        vmax = matrix.max()

    im = ax_main.imshow(matrix, cmap=cmap, aspect='auto', interpolation='nearest',
                       vmin=vmin, vmax=vmax)

    # 设置刻度
    ax_main.set_xticks(range(n_basins))
    ax_main.set_yticks(range(n_basins))
    ax_main.set_xticklabels(basin_names, rotation=90, ha='center', fontsize=10)
    ax_main.set_yticklabels(basin_names, fontsize=10)

    # 添加网格
    ax_main.set_xticks(np.arange(n_basins) - 0.5, minor=True)
    ax_main.set_yticks(np.arange(n_basins) - 0.5, minor=True)
    ax_main.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    # 标题
    ax_main.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax_main.set_xlabel('Basin (To)', fontsize=13, fontweight='bold')
    ax_main.set_ylabel('Basin (From)', fontsize=13, fontweight='bold')

    # 颜色条
    cbar = plt.colorbar(im, ax=ax_main, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Connection Weight', rotation=270, labelpad=20, fontsize=12)

    # 根据主热图的colormap选择条形图颜色
    if cmap == 'Blues':
        bar_color = 'steelblue'
        text_highlight = 'darkblue'
    elif cmap == 'Greens':
        bar_color = 'seagreen'
        text_highlight = 'darkgreen'
    else:
        bar_color = 'steelblue'
        text_highlight = 'darkblue'

    # 右侧：行求和（每个流域的出度）
    ax_row = plt.subplot2grid((4, 4), (0, 3), rowspan=3)
    row_sums = matrix.sum(axis=1)
    ax_row.barh(range(n_basins), row_sums, color=bar_color, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax_row.set_ylim(-0.5, n_basins - 0.5)
    ax_row.invert_yaxis()
    ax_row.set_xlabel('Row Sum\n(Out-degree)', fontsize=11, fontweight='bold')
    ax_row.set_yticks([])
    ax_row.grid(axis='x', alpha=0.3)

    # 标注最大/最小值
    max_idx = row_sums.argmax()
    min_idx = row_sums.argmin()
    ax_row.text(row_sums[max_idx], max_idx, f' {basin_names[max_idx]}\n {row_sums[max_idx]:.3f}',
                va='center', fontsize=9, color='red', fontweight='bold')
    ax_row.text(row_sums[min_idx], min_idx, f' {basin_names[min_idx]}\n {row_sums[min_idx]:.3f}',
                va='center', fontsize=9, color=text_highlight)

    # 底部：列求和（每个流域的入度）
    ax_col = plt.subplot2grid((4, 4), (3, 0), colspan=3)
    col_sums = matrix.sum(axis=0)
    ax_col.bar(range(n_basins), col_sums, color=bar_color, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax_col.set_xlim(-0.5, n_basins - 0.5)
    ax_col.set_ylabel('Column Sum\n(In-degree)', fontsize=11, fontweight='bold')
    ax_col.set_xticks([])
    ax_col.grid(axis='y', alpha=0.3)

    # 标注最大/最小值
    max_idx = col_sums.argmax()
    min_idx = col_sums.argmin()
    ax_col.text(max_idx, col_sums[max_idx], f'{basin_names[max_idx]}\n{col_sums[max_idx]:.3f}',
                ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')
    ax_col.text(min_idx, col_sums[min_idx], f'{basin_names[min_idx]}\n{col_sums[min_idx]:.3f}',
                ha='center', va='bottom', fontsize=9, color=text_highlight)

    # 统计信息文本框
    stats_text = f"""Statistical Summary:

Mean:           {matrix.mean():.6f}
Std Dev:        {matrix.std():.6f}
Min:            {matrix.min():.6f}
Max:            {matrix.max():.6f}
Median:         {np.median(matrix):.6f}

Diagonal Mean:  {np.diag(matrix).mean():.6f}
Off-diag Mean:  {matrix[~np.eye(n_basins, dtype=bool)].mean():.6f}

Sparsity:       {(matrix < 0.001).sum() / matrix.size * 100:.1f}%
"""

    # 添加统计信息
    ax_stats = plt.subplot2grid((4, 4), (3, 3))
    ax_stats.axis('off')
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                 fontsize=10, verticalalignment='top', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 添加描述
    if description:
        fig.text(0.5, 0.02, description, ha='center', fontsize=11,
                style='italic', wrap=True)

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    # 保存
    output_file = os.path.join(current_dir, 'output', 'detailed_plots', filename)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ 保存: {filename}")
    plt.close()


def plot_difference_matrix(adj_original, adaptive_A, basin_names):
    """
    绘制差异矩阵的详细图
    """
    diff = adaptive_A - adj_original
    n_basins = len(basin_names)

    fig = plt.figure(figsize=(18, 14))

    # 主热图
    ax_main = plt.subplot2grid((4, 5), (0, 0), colspan=3, rowspan=3)

    # 使用红蓝配色
    max_abs = max(abs(diff.min()), abs(diff.max()))
    im = ax_main.imshow(diff, cmap='RdBu_r', aspect='auto',
                        vmin=-max_abs, vmax=max_abs, interpolation='nearest')

    # 设置刻度
    ax_main.set_xticks(range(n_basins))
    ax_main.set_yticks(range(n_basins))
    ax_main.set_xticklabels(basin_names, rotation=90, ha='center', fontsize=10)
    ax_main.set_yticklabels(basin_names, fontsize=10)

    # 添加网格
    ax_main.set_xticks(np.arange(n_basins) - 0.5, minor=True)
    ax_main.set_yticks(np.arange(n_basins) - 0.5, minor=True)
    ax_main.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    # 标注显著变化的点
    diff_abs = np.abs(diff)
    np.fill_diagonal(diff_abs, 0)
    threshold = np.percentile(diff_abs, 95)  # Top 5%

    for i in range(n_basins):
        for j in range(n_basins):
            if diff_abs[i, j] >= threshold:
                ax_main.plot(j, i, 'k*', markersize=8, alpha=0.7)

    ax_main.set_title('Weight Changes: Adaptive - Original\n★ marks top 5% changes',
                     fontsize=16, fontweight='bold', pad=20)
    ax_main.set_xlabel('Basin (To)', fontsize=13, fontweight='bold')
    ax_main.set_ylabel('Basin (From)', fontsize=13, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax_main, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Weight Change (Blue=Increased, Red=Decreased)',
                      rotation=270, labelpad=25, fontsize=11)

    # 右侧：行变化总和
    ax_row = plt.subplot2grid((4, 5), (0, 3), rowspan=3)
    row_changes = diff.sum(axis=1)
    colors = ['green' if x > 0 else 'red' for x in row_changes]
    ax_row.barh(range(n_basins), row_changes, color=colors, alpha=0.7)
    ax_row.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax_row.set_ylim(-0.5, n_basins - 0.5)
    ax_row.invert_yaxis()
    ax_row.set_xlabel('Total Change\n(Out-degree)', fontsize=11, fontweight='bold')
    ax_row.set_yticks([])
    ax_row.grid(axis='x', alpha=0.3)

    # 标注极值
    max_idx = row_changes.argmax()
    min_idx = row_changes.argmin()
    if abs(row_changes[max_idx]) > 0.001:
        ax_row.text(row_changes[max_idx], max_idx,
                   f' {basin_names[max_idx]} +{row_changes[max_idx]:.4f}',
                   va='center', fontsize=9, color='darkgreen', fontweight='bold')
    if abs(row_changes[min_idx]) > 0.001:
        ax_row.text(row_changes[min_idx], min_idx,
                   f'{basin_names[min_idx]} {row_changes[min_idx]:.4f} ',
                   va='center', ha='right', fontsize=9, color='darkred', fontweight='bold')

    # 底部：列变化总和
    ax_col = plt.subplot2grid((4, 5), (3, 0), colspan=3)
    col_changes = diff.sum(axis=0)
    colors = ['green' if x > 0 else 'red' for x in col_changes]
    ax_col.bar(range(n_basins), col_changes, color=colors, alpha=0.7)
    ax_col.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax_col.set_xlim(-0.5, n_basins - 0.5)
    ax_col.set_ylabel('Total Change\n(In-degree)', fontsize=11, fontweight='bold')
    ax_col.set_xticks([])
    ax_col.grid(axis='y', alpha=0.3)

    # 直方图 - 变化分布
    ax_hist = plt.subplot2grid((4, 5), (0, 4), rowspan=2)
    diff_off_diag = diff[~np.eye(n_basins, dtype=bool)]
    ax_hist.hist(diff_off_diag, bins=40, orientation='horizontal',
                color='steelblue', alpha=0.7, edgecolor='black')
    ax_hist.axhline(y=0, color='red', linestyle='--', linewidth=2, label='No change')
    ax_hist.set_ylabel('Weight Change', fontsize=11, fontweight='bold')
    ax_hist.set_xlabel('Count', fontsize=11, fontweight='bold')
    ax_hist.legend(loc='upper right')
    ax_hist.grid(axis='x', alpha=0.3)

    # 统计信息
    stats_text = f"""Change Statistics:

Increased:  {(diff_off_diag > 0).sum()} ({(diff_off_diag > 0).sum()/len(diff_off_diag)*100:.1f}%)
Decreased:  {(diff_off_diag < 0).sum()} ({(diff_off_diag < 0).sum()/len(diff_off_diag)*100:.1f}%)
Unchanged:  {(diff_off_diag == 0).sum()}

Mean Δ:     {diff_off_diag.mean():.6f}
Std Δ:      {diff_off_diag.std():.6f}
Max +Δ:     {diff_off_diag.max():.6f}
Max -Δ:     {diff_off_diag.min():.6f}

Mean |Δ|:   {np.abs(diff_off_diag).mean():.6f}
Median |Δ|: {np.median(np.abs(diff_off_diag)):.6f}
"""

    ax_stats = plt.subplot2grid((4, 5), (2, 4), rowspan=2)
    ax_stats.axis('off')
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                 fontsize=10, verticalalignment='top', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()

    output_file = os.path.join(current_dir, 'output', 'detailed_plots',
                              'fig4_difference_matrix.png')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ 保存: fig4_difference_matrix.png")
    plt.close()


def plot_top_changes_detailed(adj_original, adaptive_A, basin_names):
    """
    绘制Top变化的详细分析图
    """
    diff = adaptive_A - adj_original
    diff_abs = np.abs(diff)
    np.fill_diagonal(diff_abs, 0)

    # 找出top 30
    top_k = 30
    top_indices = np.argpartition(diff_abs.ravel(), -top_k)[-top_k:]
    top_indices = np.unravel_index(top_indices, diff_abs.shape)

    changes = []
    for i in range(top_k):
        row, col = top_indices[0][i], top_indices[1][i]
        changes.append({
            'from': basin_names[row],
            'to': basin_names[col],
            'from_idx': row,
            'to_idx': col,
            'original': adj_original[row, col],
            'adaptive': adaptive_A[row, col],
            'change': diff[row, col],
            'change_abs': diff_abs[row, col],
            'change_pct': (diff[row, col] / adj_original[row, col] * 100)
                         if adj_original[row, col] > 1e-6 else float('inf')
        })

    changes.sort(key=lambda x: x['change_abs'], reverse=True)

    # 创建图
    fig, axes = plt.subplots(2, 1, figsize=(18, 14))

    # 上图：条形图
    ax1 = axes[0]

    x_pos = np.arange(top_k)
    colors = ['green' if c['change'] > 0 else 'red' for c in changes]

    bars = ax1.bar(x_pos, [c['change_abs'] for c in changes],
                   color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # 添加数值标签
    for i, (bar, change) in enumerate(zip(bars, changes)):
        height = bar.get_height()
        sign = '+' if change['change'] > 0 else ''
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{sign}{change["change"]:.4f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"{c['from']}→{c['to']}" for c in changes],
                        rotation=45, ha='right', fontsize=10)
    ax1.set_ylabel('Absolute Weight Change', fontsize=13, fontweight='bold')
    ax1.set_title(f'Top {top_k} Basin Pairs by Weight Change\n(Green=Increased, Red=Decreased)',
                 fontsize=15, fontweight='bold', pad=15)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=0, color='black', linewidth=1)

    # 下图：详细表格
    ax2 = axes[1]
    ax2.axis('off')

    # 准备表格数据
    table_data = [['Rank', 'From', 'To', 'Original', 'Adaptive', 'Change', 'Change %', 'Type']]

    for i, c in enumerate(changes, 1):
        change_type = '↑ Increased' if c['change'] > 0 else '↓ Decreased'
        change_pct_str = f"{c['change_pct']:+.0f}%" if abs(c['change_pct']) < 10000 else "Very Large"

        table_data.append([
            str(i),
            c['from'],
            c['to'],
            f"{c['original']:.5f}",
            f"{c['adaptive']:.5f}",
            f"{c['change']:+.5f}",
            change_pct_str,
            change_type
        ])

    # 创建表格
    table = ax2.table(cellText=table_data, cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # 样式设置
    for i in range(8):
        cell = table[(0, i)]
        cell.set_facecolor('#2E7D32')
        cell.set_text_props(weight='bold', color='white', fontsize=10)

    # 交替行颜色 + 高亮LCR相关
    for i in range(1, len(table_data)):
        for j in range(8):
            cell = table[(i, j)]

            # 检查是否包含LCR
            if 'LCR' in table_data[i][1] or 'LCR' in table_data[i][2]:
                cell.set_facecolor('#FFF9C4')  # 浅黄色高亮
                cell.set_text_props(fontweight='bold')
            elif i % 2 == 0:
                cell.set_facecolor('#f0f0f0')

            # 变化类型列添加颜色
            if j == 7:
                if '↑' in table_data[i][7]:
                    cell.set_text_props(color='green', fontweight='bold')
                else:
                    cell.set_text_props(color='red', fontweight='bold')

    ax2.set_title('Detailed Change Table (Yellow highlighted = LCR-related)',
                 fontsize=13, fontweight='bold', pad=10)

    plt.tight_layout()

    output_file = os.path.join(current_dir, 'output', 'detailed_plots',
                              'fig5_top_changes.png')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ 保存: fig5_top_changes.png")
    plt.close()


def plot_fusion_analysis(adj_original, learned_A, adaptive_A, alpha):
    """
    绘制融合过程分析图
    """
    fig = plt.figure(figsize=(20, 10))

    # 确定统一的颜色范围
    sample_max = max(adj_original[:10, :10].max(), adaptive_A[:10, :10].max())

    # 创建示意图
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(adj_original[:10, :10], cmap='Blues', aspect='auto',
                     vmin=0, vmax=sample_max)
    ax1.set_title('Original Matrix\n(Geographic Distance)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('First 10 basins (sample)')
    ax1.set_ylabel('First 10 basins (sample)')
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(learned_A[:10, :10], cmap='Greens', aspect='auto',
                     vmin=0, vmax=1.0)
    ax2.set_title('Learned Matrix (sigmoid)\n(Data-Driven)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('First 10 basins (sample)')
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    ax3 = plt.subplot(2, 3, 3)
    im3 = ax3.imshow(adaptive_A[:10, :10], cmap='Blues', aspect='auto',
                     vmin=0, vmax=sample_max)
    ax3.set_title(f'Adaptive Matrix\n(Fused Result)', fontsize=13, fontweight='bold')
    ax3.set_xlabel('First 10 basins (sample)')
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    # 融合公式可视化
    ax4 = plt.subplot(2, 3, 4)
    ax4.axis('off')

    formula_text = f"""
Fusion Formula:

Adaptive = α × Original + (1-α) × Learned

Where:
  α = sigmoid(fusion_weight)
  α = {alpha:.4f}

This means:
  {alpha*100:.1f}% from Geographic Distance
  {(1-alpha)*100:.1f}% from Data-Driven Learning

Interpretation:
  → Model relies heavily on learned patterns
  → Geographic distance is weak prior
  → Actual hydrological connectivity differs
    from geographic proximity
"""

    ax4.text(0.1, 0.9, formula_text, transform=ax4.transAxes,
            fontsize=12, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))

    # Alpha值的演变（如果有训练日志的话，这里简化展示）
    ax5 = plt.subplot(2, 3, 5)

    # 模拟alpha的可能变化趋势
    epochs = np.arange(0, 100)
    # 初始值是0.5，训练后是current alpha
    alpha_trajectory = 0.5 + (alpha - 0.5) * (1 - np.exp(-epochs/20))

    ax5.plot(epochs, alpha_trajectory, 'b-', linewidth=2, label='α (fusion weight)')
    ax5.axhline(y=0.5, color='gray', linestyle='--', label='Initial value')
    ax5.axhline(y=alpha, color='red', linestyle='--', label=f'Final value ({alpha:.4f})')
    ax5.fill_between(epochs, 0, alpha_trajectory, alpha=0.3, color='blue',
                     label=f'Original matrix weight ({alpha*100:.1f}%)')
    ax5.fill_between(epochs, alpha_trajectory, 1, alpha=0.3, color='green',
                     label=f'Learned matrix weight ({(1-alpha)*100:.1f}%)')

    ax5.set_xlabel('Training Epoch (simulated)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Fusion Weight α', fontsize=12, fontweight='bold')
    ax5.set_title('Evolution of Fusion Weight during Training\n(Conceptual)',
                 fontsize=13, fontweight='bold')
    ax5.legend(loc='right', fontsize=10)
    ax5.grid(alpha=0.3)
    ax5.set_ylim(0, 1)

    # 比较三个矩阵的统计分布 - 只显示统计表格
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    # 获取非对角线元素
    n = adj_original.shape[0]
    mask = ~np.eye(n, dtype=bool)

    orig_vals = adj_original[mask]
    learn_vals = learned_A[mask]
    adapt_vals = adaptive_A[mask]

    # 顺序：Original -> Learned -> Adaptive（与上方热图顺序一致）
    data_to_plot = [orig_vals, learn_vals, adapt_vals]
    labels = ['Original', 'Learned', 'Adaptive']
    colors = ['steelblue', 'seagreen', 'darkblue']

    # 添加标题
    ax6.text(0.5, 0.95, 'Distribution Comparison\n(Statistical Summary)',
             transform=ax6.transAxes, fontsize=13, fontweight='bold',
             ha='center', va='top')

    # 准备统计信息表格
    stats_data = []
    for label, vals in zip(labels, data_to_plot):
        stats_data.append([
            label,
            f"{vals.min():.4f}",
            f"{np.percentile(vals, 25):.4f}",
            f"{np.median(vals):.4f}",
            f"{np.percentile(vals, 75):.4f}",
            f"{vals.max():.4f}",
            f"{vals.mean():.4f}",
            f"{vals.std():.4f}"
        ])

    # 创建表格（占据整个子图空间）
    table_headers = ['Matrix', 'Min', 'Q25', 'Median', 'Q75', 'Max', 'Mean', 'Std']
    table = ax6.table(cellText=stats_data, colLabels=table_headers,
                     cellLoc='center', loc='center',
                     bbox=[0.05, 0.1, 0.9, 0.8])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # 设置列宽：第一列(Matrix)更宽，其他列均匀分配
    col_widths = [0.18, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12]  # Matrix列18%，其他各12%
    for i in range(len(table_headers)):
        for j in range(len(stats_data) + 1):  # +1 for header
            table[(j, i)].set_width(col_widths[i])

    # 设置表头样式
    for i in range(len(table_headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white', fontsize=11)

    # 设置数据行样式
    for i in range(1, len(stats_data) + 1):
        # 第一列（标签）加粗并着色
        table[(i, 0)].set_text_props(weight='bold', fontsize=10)
        table[(i, 0)].set_facecolor(colors[i-1])
        table[(i, 0)].set_text_props(color='white')

        # 其他列交替背景色
        for j in range(1, len(table_headers)):
            table[(i, j)].set_text_props(fontsize=10)
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

            # 为关键统计量添加边框
            table[(i, j)].set_edgecolor('#cccccc')
            table[(i, j)].set_linewidth(0.5)

    plt.tight_layout()

    output_file = os.path.join(current_dir, 'output', 'detailed_plots',
                              'fig6_fusion_analysis.png')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ 保存: fig6_fusion_analysis.png")
    plt.close()


def main():
    """主函数"""
    print("="*80)
    print("生成详细的可视化图表（每个单独一张大图）")
    print("="*80)

    # 加载数据
    result = load_data_and_model()
    if result[0] is None:
        return

    adj_original, adaptive_A, learned_A, basin_names, alpha = result

    print(f"\n开始生成图表...")
    print(f"Alpha = {alpha:.4f} ({alpha*100:.1f}% original + {(1-alpha)*100:.1f}% learned)")

    # 确定统一的颜色范围（用于直接对比）
    # 调整vmax让颜色对比更明显：0.3以上就显示为深蓝
    global_vmax = 0.3  # 原来是 max(adj_original.max(), adaptive_A.max()) ≈ 0.665

    print(f"\n颜色映射范围: [0, {global_vmax:.4f}] (调整后，增强对比度)")
    print(f"  - 原始矩阵最大值: {adj_original.max():.4f}")
    print(f"  - 自适应矩阵最大值: {adaptive_A.max():.4f}")
    print(f"  - 使用统一配色方案: Blues (浅蓝→深蓝)")
    print(f"  - 注意: 超过0.3的值将显示为最深蓝色")

    # 图1: 原始距离矩阵
    plot_matrix_with_details(
        adj_original, basin_names,
        title='Figure 1: Original Distance Matrix (Geographic Prior)',
        filename='fig1_original_matrix.png',
        cmap='Blues',
        vmin=0,
        vmax=global_vmax,
        description='Matrix derived from Euclidean distance between basin coordinates. '
                   'Darker blue = stronger connection. Color scale: 0.3+ = darkest blue for better contrast.'
    )

    # 图2: 可学习矩阵（这个保持不同配色以示区别，因为它的值域完全不同）
    plot_matrix_with_details(
        learned_A, basin_names,
        title='Figure 2: Learned Matrix (Data-Driven Patterns, after sigmoid)',
        filename='fig2_learned_matrix.png',
        cmap='Greens',  # 使用绿色系区分
        vmin=0,
        vmax=1.0,  # sigmoid后的值域是0-1
        description='Matrix learned from streamflow data (after sigmoid transformation). '
                   'Values in [0,1]. Darker green = stronger learned relationship. '
                   'Uses different scale than Fig 1&3 due to sigmoid transformation.'
    )

    # 图3: 自适应邻接矩阵
    plot_matrix_with_details(
        adaptive_A, basin_names,
        title=f'Figure 3: Adaptive Adjacency Matrix ({alpha:.1%} Geographic + {1-alpha:.1%} Learned)',
        filename='fig3_adaptive_matrix.png',
        cmap='Blues',
        vmin=0,
        vmax=global_vmax,
        description=f'Final matrix used in RGCN model. Darker blue = stronger connection. '
                   f'Fusion weight α={alpha:.4f} learned during training. '
                   f'Color scale: 0.3+ = darkest blue for better contrast with Fig 1.'
    )

    # 图4: 差异矩阵
    plot_difference_matrix(adj_original, adaptive_A, basin_names)

    # 图5: Top变化详细分析
    plot_top_changes_detailed(adj_original, adaptive_A, basin_names)

    # 图6: 融合分析
    plot_fusion_analysis(adj_original, learned_A, adaptive_A, alpha)

    print("\n" + "="*80)
    print("✓ 所有图表生成完成！")
    print(f"保存位置: {current_dir}/output/detailed_plots/")
    print("="*80)
    print("\n生成的图表:")
    print("  - fig1_original_matrix.png    : 原始地理距离矩阵")
    print("  - fig2_learned_matrix.png     : 可学习矩阵")
    print("  - fig3_adaptive_matrix.png    : 最终自适应邻接矩阵")
    print("  - fig4_difference_matrix.png  : 变化差异矩阵")
    print("  - fig5_top_changes.png        : Top变化详细分析")
    print("  - fig6_fusion_analysis.png    : 融合过程分析")
    print("="*80)


if __name__ == "__main__":
    main()
