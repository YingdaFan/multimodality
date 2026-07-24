"""
计算真正的水文响应矩阵：basin j的输入对预测basin i的输出的影响

方法：梯度归因（Gradient Attribution）
计算 ∂Y_i / ∂X_j，即basin j的输入变化对basin i的预测的影响
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

current_dir = os.path.dirname(os.path.abspath(__file__))
imputation_dir = os.path.dirname(current_dir)
data_processing_dir = os.path.join(imputation_dir, 'data_processing')
sys.path.insert(0, data_processing_dir)

from model import RGCN_v1 as Model


def compute_cross_basin_influence_matrix(model, data, device='cpu', num_samples=100):
    """
    计算跨basin影响矩阵：basin j的输入对预测basin i的输出的梯度贡献

    返回：
    - influence_matrix[i, j]: basin j的输入对预测basin i的影响强度
    """
    model.eval()

    # 使用验证集
    x_val = torch.from_numpy(data['x_val']).float().to(device)
    ids_val = data['ids_val']

    unique_basins = np.unique(ids_val[:, 0, 0])
    n_basins = len(unique_basins)

    print(f"计算 {n_basins}x{n_basins} 的跨basin影响矩阵...")
    print(f"方法：梯度归因（∂Y_i/∂X_j）")

    # 创建basin索引映射
    basin_to_idx = {basin_id: idx for idx, basin_id in enumerate(unique_basins)}
    sample_to_basin = np.array([basin_to_idx[ids_val[i, 0, 0]] for i in range(len(ids_val))])

    # 影响矩阵：influence[i, j] = basin j的输入对预测basin i的平均梯度
    influence_matrix = np.zeros((n_basins, n_basins))
    count_matrix = np.zeros((n_basins, n_basins))

    # 限制样本数以节省计算
    num_samples = min(num_samples, len(x_val))

    print(f"使用 {num_samples} 个样本计算梯度...")

    for sample_idx in range(num_samples):
        if sample_idx % 10 == 0:
            print(f"  处理样本 {sample_idx}/{num_samples}...")

        # 获取单个样本（batch=1）
        x_single = x_val[sample_idx:sample_idx+1].clone().requires_grad_(True)

        # 前向传播
        output = model(x_single)  # (1, seq_len, 1)

        # 对每个时间步计算梯度
        seq_len = output.shape[1]

        # 计算这个样本属于哪个basin
        target_basin_idx = sample_to_basin[sample_idx]

        # 对输出的每个时间步求和，然后反向传播
        loss = output.sum()

        # 反向传播计算梯度
        model.zero_grad()
        if x_single.grad is not None:
            x_single.grad.zero_()

        loss.backward()

        # 获取梯度：grad shape = (1, seq_len, n_features)
        grad = x_single.grad.detach().cpu().numpy()  # (1, seq_len, n_features)

        # 计算梯度的L2范数作为影响强度
        # 注意：x_single对应的是当前样本，但在RGCN中，由于batch包含所有basins，
        # 实际上我们需要重新设计这个实验

        # 梯度范数：对时间和特征维度求平均
        grad_norm = np.linalg.norm(grad, axis=(1, 2))[0]  # 标量

        # 这个梯度是"当前basin的输入"对"当前basin的输出"的影响
        # 但我们想要的是跨basin的影响...

        # 问题：在当前的batch=1设置下，无法计算跨basin影响
        # 因为每个样本只包含一个basin的数据

        # 我们需要改变策略！
        influence_matrix[target_basin_idx, target_basin_idx] += grad_norm
        count_matrix[target_basin_idx, target_basin_idx] += 1

    # 归一化
    influence_matrix = np.divide(influence_matrix, count_matrix,
                                  where=count_matrix>0, out=np.zeros_like(influence_matrix))

    return influence_matrix, unique_basins


def compute_cross_basin_influence_via_gcn_weights(model, data, device='cpu'):
    """
    方法2：通过GCN的权重结构分析跨basin影响

    理论：
    在RGCN中，basin j对basin i的影响路径是：
    X_j → h_j → q_j = tanh(h_j @ W_q) → spatial_context_i = A[i,j] * q_j → affects Y_i

    影响强度可以近似为：
    Influence[i,j] ∝ A[i,j] * ||W_q|| * ||W_output||

    但这只是粗略近似，真正的影响需要考虑LSTM的动态
    """
    print("\n方法2：基于模型权重的近似影响矩阵")

    with torch.no_grad():
        # 获取关键权重
        A_adaptive = model.get_adaptive_adjacency().cpu().numpy()
        W_q = model.weight_q.cpu().numpy()
        W_output = model.dense.weight.cpu().numpy()

        # 简化的影响估计：A[i,j] 已经编码了basin j对basin i的贡献
        # 这是在hidden state空间的贡献
        influence_approx = A_adaptive.copy()

        print(f"  邻接矩阵统计:")
        print(f"    均值: {A_adaptive.mean():.4f}")
        print(f"    标准差: {A_adaptive.std():.4f}")
        print(f"    非对角线均值: {A_adaptive[~np.eye(A_adaptive.shape[0], dtype=bool)].mean():.4f}")

    return influence_approx


def compute_cross_basin_influence_batch_mode(model, data, device='cpu', num_batches=20):
    """
    方法3：批量模式下的跨basin影响计算

    关键：在RGCN中，所有basins在同一个batch中处理
    我们可以计算basin j的特征对basin i的预测的梯度
    """
    print("\n方法3：批量模式梯度归因（最准确）")

    model.eval()

    x_val = data['x_val']
    y_val = data['y_obs_val']
    ids_val = data['ids_val']

    unique_basins = np.unique(ids_val[:, 0, 0])
    n_basins = len(unique_basins)

    # 创建basin索引映射
    basin_to_idx = {basin_id: idx for idx, basin_id in enumerate(unique_basins)}

    # 按batch分组（每个batch包含所有basins的样本）
    # 在你的数据中，batch_size=29意味着每个batch包含29个basins的样本

    # 找到完整的batches（每个batch应该包含所有29个basins）
    batch_size = n_basins
    num_complete_batches = len(x_val) // batch_size

    print(f"  总样本数: {len(x_val)}")
    print(f"  Basin数: {n_basins}")
    print(f"  完整batch数: {num_complete_batches}")

    influence_matrix = np.zeros((n_basins, n_basins))

    num_batches = min(num_batches, num_complete_batches)

    for batch_idx in range(num_batches):
        if batch_idx % 5 == 0:
            print(f"  处理batch {batch_idx}/{num_batches}...")

        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size

        # 获取这个batch（包含所有basins）
        x_batch = torch.from_numpy(x_val[start_idx:end_idx]).float().to(device)
        x_batch.requires_grad_(True)

        # 获取这个batch中每个样本对应的basin索引
        batch_basin_indices = [basin_to_idx[ids_val[i, 0, 0]]
                               for i in range(start_idx, end_idx)]

        # 前向传播
        output = model(x_batch)  # (batch_size, seq_len, 1)

        # 对每个target basin i，计算所有source basins j的贡献
        for target_idx in range(batch_size):
            target_basin = batch_basin_indices[target_idx]

            # 只对这个target basin的输出计算梯度
            target_output = output[target_idx].sum()  # 对seq_len求和

            # 反向传播
            model.zero_grad()
            if x_batch.grad is not None:
                x_batch.grad.zero_()

            target_output.backward(retain_graph=True)

            # 获取梯度
            grads = x_batch.grad.detach().cpu().numpy()  # (batch_size, seq_len, n_features)

            # 对每个source basin j，计算其输入的梯度范数
            for source_idx in range(batch_size):
                source_basin = batch_basin_indices[source_idx]

                # 计算source basin的梯度范数
                source_grad = grads[source_idx]  # (seq_len, n_features)
                grad_norm = np.linalg.norm(source_grad)

                # 累积到影响矩阵
                influence_matrix[target_basin, source_basin] += grad_norm

    # 归一化
    influence_matrix /= num_batches

    print(f"\n  影响矩阵统计:")
    print(f"    均值: {influence_matrix.mean():.4f}")
    print(f"    标准差: {influence_matrix.std():.4f}")
    print(f"    对角线均值: {np.diag(influence_matrix).mean():.4f}")
    print(f"    非对角线均值: {influence_matrix[~np.eye(n_basins, dtype=bool)].mean():.4f}")

    return influence_matrix, unique_basins


def main():
    """主函数"""
    print("="*80)
    print("计算真正的跨basin水文响应矩阵")
    print("="*80)

    # 加载数据和模型
    data = np.load(os.path.join(data_processing_dir, 'data', 'prepped.npz'),
                   allow_pickle=True)

    adj_mx = data['dist_matrix']
    in_dim = len(data['x_vars'])
    device = torch.device('cpu')

    model = Model(
        input_dim=in_dim,
        hidden_dim=20,
        adj_matrix=adj_mx,
        device=device,
        seed=42,
        use_learnable_adj=True,
        use_pseudo_labels=True,
        use_domain_adapt=True,
        use_spatial_attention=True
    )

    weights_path = os.path.join(current_dir, 'output', 'finetuned_weights.pth')
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # 方法1：基于权重的近似（最快）
    print("\n" + "="*80)
    influence_approx = compute_cross_basin_influence_via_gcn_weights(model, data, device)

    # 方法2：批量模式梯度归因（最准确）
    print("\n" + "="*80)
    influence_gradient, basins = compute_cross_basin_influence_batch_mode(
        model, data, device, num_batches=20
    )

    # 可视化对比
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 1. 邻接矩阵（基于权重的近似）
    with torch.no_grad():
        A_adaptive = model.get_adaptive_adjacency().cpu().numpy()

    im1 = axes[0].imshow(A_adaptive, cmap='RdYlBu_r', aspect='auto')
    axes[0].set_title('Learned Adjacency Matrix\n(Structure in Hidden State Space)',
                      fontweight='bold', fontsize=14)
    axes[0].set_xlabel('Source Basin j', fontweight='bold')
    axes[0].set_ylabel('Target Basin i', fontweight='bold')
    plt.colorbar(im1, ax=axes[0])

    # 2. 梯度归因矩阵（真正的X→Y影响）
    im2 = axes[1].imshow(influence_gradient, cmap='RdYlBu_r', aspect='auto')
    axes[1].set_title('Cross-Basin Influence Matrix\n(∂Y_i/∂X_j via Gradient Attribution)',
                      fontweight='bold', fontsize=14)
    axes[1].set_xlabel('Source Basin j (Input)', fontweight='bold')
    axes[1].set_ylabel('Target Basin i (Output)', fontweight='bold')
    plt.colorbar(im2, ax=axes[1])

    # 3. 两者的差异
    # 归一化到相同尺度以便比较
    A_norm = (A_adaptive - A_adaptive.min()) / (A_adaptive.max() - A_adaptive.min())
    I_norm = (influence_gradient - influence_gradient.min()) / (influence_gradient.max() - influence_gradient.min())
    diff = I_norm - A_norm

    im3 = axes[2].imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
    axes[2].set_title('Difference:\nGradient Influence - Adjacency Matrix',
                      fontweight='bold', fontsize=14)
    axes[2].set_xlabel('Source Basin j', fontweight='bold')
    axes[2].set_ylabel('Target Basin i', fontweight='bold')
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()

    output_dir = os.path.join(current_dir, 'output', 'cross_basin_analysis')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'cross_basin_influence.png'),
                dpi=300, bbox_inches='tight')
    print(f"\n可视化已保存到: {output_dir}/cross_basin_influence.png")

    # 计算相关性
    from scipy.stats import spearmanr

    mask = ~np.eye(len(A_adaptive), dtype=bool)
    corr, p = spearmanr(A_adaptive[mask], influence_gradient[mask])

    print("\n" + "="*80)
    print("邻接矩阵 vs 梯度影响矩阵")
    print("="*80)
    print(f"Spearman相关性: {corr:.4f} (p={p:.4e})")

    if corr > 0.5:
        print("✓ 强正相关！邻接矩阵确实反映了跨basin的输入-输出影响")
    elif corr > 0.3:
        print("→ 中等相关，邻接矩阵部分反映了跨basin影响")
    else:
        print("~ 弱相关，邻接矩阵与直接的X→Y梯度影响不同")
        print("  这说明邻接矩阵在hidden state空间工作，不是直接的输入-输出映射")

    # 保存结果
    np.save(os.path.join(output_dir, 'adjacency_matrix.npy'), A_adaptive)
    np.save(os.path.join(output_dir, 'influence_gradient.npy'), influence_gradient)

    print(f"\n矩阵已保存到: {output_dir}/")
    print("  - adjacency_matrix.npy")
    print("  - influence_gradient.npy")

    return influence_gradient, A_adaptive, basins


if __name__ == "__main__":
    influence_gradient, A_adaptive, basins = main()
