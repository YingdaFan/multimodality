import os
import sys
import yaml
import numpy as np
import torch
import torch.optim as optim
from torch_utils import train_torch, iterative_self_training

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
imputation_dir = os.path.dirname(current_dir)
data_processing_dir = os.path.join(imputation_dir, 'data_processing')
sys.path.insert(0, data_processing_dir)

# 加载配置
config_file_path = os.path.join(current_dir, 'config.yml')
with open(config_file_path, 'r') as f:
    config = yaml.safe_load(f)

from model import RGCN_v1 as Model
from predict import predict_from_io_data

# CUDA device configuration (use environment variable or default to cuda:0)
cuda_device = 'cuda'


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def identify_missing_basins(y_obs_trn, ids_trn, threshold=0.95):
    """
    识别缺失流域

    Args:
        y_obs_trn: 训练观测数据
        ids_trn: 流域ID
        threshold: 缺失比例阈值，超过此比例被认为是缺失流域

    Returns:
        missing_basin_ids: 缺失流域的ID列表
        missing_basin_mask: 布尔掩码，标记哪些样本属于缺失流域
    """
    missing_basin_ids = []
    missing_basin_mask = np.zeros(len(y_obs_trn), dtype=bool)

    # 获取唯一的流域ID
    unique_basins = np.unique(ids_trn[:, 0, 0])

    for basin in unique_basins:
        # 找到属于该流域的样本
        basin_indices = np.where(ids_trn[:, 0, 0] == basin)[0]

        if len(basin_indices) > 0:
            # 计算该流域的缺失比例
            basin_data = y_obs_trn[basin_indices]
            missing_ratio = np.isnan(basin_data).mean()

            if missing_ratio > threshold:
                missing_basin_ids.append(basin)
                missing_basin_mask[basin_indices] = True
                print(f"Basin {basin}: {missing_ratio*100:.1f}% missing (marked as missing basin)")

    print(f"\nIdentified {len(missing_basin_ids)} missing basins out of {len(unique_basins)}")

    return missing_basin_ids, missing_basin_mask


def split_data_by_missing_status(data, missing_basin_mask):
    """
    根据缺失状态分割数据

    Returns:
        data_observed: 有观测值的流域数据
        data_missing: 缺失流域数据
    """
    observed_indices = np.where(~missing_basin_mask)[0]
    missing_indices = np.where(missing_basin_mask)[0]

    data_observed = {}
    data_missing = {}

    for key in ['x_trn', 'y_obs_trn', 'ids_trn', 'times_trn']:
        if key in data:
            data_observed[key] = data[key][observed_indices]
            data_missing[key] = data[key][missing_indices]

    return data_observed, data_missing


def main():
    """主训练流程"""

    outdir = os.path.join(current_dir, 'output')
    ensure_dir(f"{outdir}/")

    # 1. 加载数据
    data = np.load(os.path.join(data_processing_dir, 'data', 'prepped.npz'), allow_pickle=True)
    num_segs = len(np.unique(data['ids_trn']))
    adj_mx = data['dist_matrix']
    in_dim = len(data['x_vars'])
    device = torch.device(cuda_device if torch.cuda.is_available() else 'cpu')
    model = Model(
        input_dim=in_dim,
        hidden_dim=config['hidden_size'],
        adj_matrix=adj_mx,
        device=device,
        seed=42,
        use_learnable_adj=True,
        use_pseudo_labels=True,
        use_domain_adapt=True,
        use_spatial_attention=True
    )

    # 使用 DataParallel 进行并行训练


    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)



    # 2. 识别缺失流域
    print("\nIdentifying missing basins...")
    missing_basin_ids, missing_basin_mask = identify_missing_basins(
        data['y_obs_trn'], data['ids_trn'], threshold=0.95
    )

    # 3. 分割数据
    print("\nSplitting data by missing status...")
    data_observed, data_missing = split_data_by_missing_status(
        {key: data[key] for key in data.files},
        missing_basin_mask
    )

    print(f"Observed basins samples: {len(data_observed['x_trn'])}")
    print(f"Missing basins samples: {len(data_missing['x_trn'])}")



    # 4. 第一阶段：在有观测值的流域上预训练
    print("\n=== Phase 1: Pre-training on observed basins ===")

    # 只使用有观测值的流域进行初始训练
    # if len(data_observed['x_trn']) > 0:
    # 预训练阶段不需要basin mask（只有observed数据）
    model = train_torch(
        model=model,
        optimizer=optimizer,
        #x_train=data_observed['x_trn'],
        #y_train=data_observed['y_obs_trn'],
        x_train=data['x_trn'],
        y_train=data['y_obs_trn'],
        x_val=data['x_val'],  # 使用完整验证集
        y_val=data['y_obs_val'],
        x_tst=data['x_tst'],
        y_tst=data['y_obs_tst'],
        #batch_size=min(num_segs, len(data_observed['x_trn'])),
        batch_size=min(num_segs, len(data['x_trn'])),
        max_epochs=config['ft_epochs'] // 2,  # 预训练阶段用一半的epochs
        early_stopping_patience=config['early_stopping'],
        weights_file=f"{outdir}/pretrained_weights.pth",
        log_file=f"{outdir}/pretrain_log.csv",
        device=device,
        missing_basin_mask=None,  # 预训练时不需要mask
        lambda_pseudo=0.0,  # 预训练时不使用伪标签
        lambda_domain=0.0,  # 预训练时不使用域适应
        warmup_epochs=5
    )

    print("Pre-training completed!")

    # 5. 第二阶段：使用伪标签和域适应进行微调
    print("\n=== Phase 2: Fine-tuning with pseudo-labels and domain adaptation ===")

    # 重新初始化优化器（较小的学习率）
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # 结合所有数据进行训练（包括缺失流域）
    model = train_torch(
        model=model,
        optimizer=optimizer,
        x_train=data['x_trn'],  # 使用完整训练集
        y_train=data['y_obs_trn'],
        x_val=data['x_val'],
        y_val=data['y_obs_val'],
        x_tst=data['x_tst'],
        y_tst=data['y_obs_tst'],
        batch_size=num_segs,
        max_epochs=config['ft_epochs'],
        early_stopping_patience=config['early_stopping'],
        weights_file=f"{outdir}/finetuned_weights.pth",
        log_file=f"{outdir}/finetune_log.csv",
        device=device,
        missing_basin_mask=missing_basin_mask,  # 传递预先计算的mask
        lambda_pseudo=0.2,  # 伪标签权重
        lambda_domain=0.1,  # 域适应权重
        curriculum_learning=True,  # 使用课程学习
        warmup_epochs=10
    )

    print("Fine-tuning completed!")

    # 6. （可选）第三阶段：迭代自训练
    if len(data_missing['x_trn']) > 0 and config.get('use_self_training', False):
        print("\n=== Phase 3: Iterative self-training ===")

        # model已经包含第二阶段的权重，无需重新加载
        # 只需要重新初始化优化器，使用更小的学习率
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

        model = iterative_self_training(
            model=model,
            optimizer=optimizer,
            x_labeled=data_observed['x_trn'],
            y_labeled=data_observed['y_obs_trn'],
            x_unlabeled=data_missing['x_trn'],
            iterations=3,
            # 训练参数
            x_val=data['x_val'],
            y_val=data['y_obs_val'],
            batch_size=num_segs,
            max_epochs=20,  # 每次迭代的epochs数
            device=device,
            weights_file=f"{outdir}/finetuned_weights.pth",
            log_file=f"{outdir}/self_train_log.csv"
        )

        print("Self-training completed!")

    # 7. 预测
    print("\n=== Making predictions ===")

    # 加载最佳模型
    model.load_state_dict(torch.load(f"{outdir}/finetuned_weights.pth"))
    model.eval()
    partitions = ['trn', 'val', 'tst']

    for partition in partitions:
        outfile = f"{outdir}/preds/{partition}.npy"
        ensure_dir(outfile)

        predict_from_io_data(model=model,
                            io_data=os.path.join(data_processing_dir, 'data', 'prepped.npz'),
                            partition=partition,
                            outfile=outfile,
                            log_vars=False,
                            trn_offset=1,
                            tst_val_offset=1,
                            spatial_idx_name="spatial_idx_name",
                            time_idx_name="date")



if __name__ == "__main__":
    main()