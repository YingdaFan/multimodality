import numpy as np
import torch
import torch.utils.data
import pandas as pd
from tqdm import tqdm


def rmse_masked(y_true, y_pred):
    """计算有观测值位置的RMSE"""
    num_y_true = torch.count_nonzero(~torch.isnan(y_true))
    if num_y_true > 0:
        zero_or_error = torch.where(
            torch.isnan(y_true), torch.zeros_like(y_true), y_pred - y_true
        )
        sum_squared_errors = torch.sum(torch.square(zero_or_error))
        rmse_loss = torch.sqrt(sum_squared_errors / num_y_true)
    else:
        rmse_loss = torch.tensor(0.0, device=y_true.device, dtype=y_true.dtype)
    return rmse_loss


def pseudo_label_loss(pseudo_preds, confidences, threshold=0.5):
    """
    伪标签损失函数
    只对置信度高于阈值的预测计算损失
    """
    if pseudo_preds is None or confidences is None:
        return torch.tensor(0.0)

    # 选择置信度高的伪标签
    high_conf_mask = confidences > threshold

    if high_conf_mask.sum() > 0:
        # 时间一致性损失 - 相邻时间步的预测应该平滑
        temporal_diff = torch.diff(pseudo_preds, dim=1)
        temporal_loss = torch.mean(torch.square(temporal_diff[high_conf_mask[:, :-1]]))

        # 值范围正则化 - 防止预测值过大或过小
        value_loss = torch.mean(torch.relu(torch.abs(pseudo_preds[high_conf_mask]) - 3.0))

        return temporal_loss + 0.1 * value_loss
    else:
        return torch.tensor(0.0, device=pseudo_preds.device)


def domain_adaptation_loss(domain_features, basin_mask):
    """
    域适应损失 - 使缺失流域和有观测流域的特征分布更接近
    """
    if domain_features is None or basin_mask is None:
        return torch.tensor(0.0)

    # 分离有观测和缺失流域的特征
    observed_features = domain_features[~basin_mask]
    missing_features = domain_features[basin_mask]

    if len(observed_features) > 0 and len(missing_features) > 0:
        # 计算MMD (Maximum Mean Discrepancy)
        observed_mean = torch.mean(observed_features, dim=0)
        missing_mean = torch.mean(missing_features, dim=0)
        mmd_loss = torch.mean(torch.square(observed_mean - missing_mean))

        return mmd_loss
    else:
        return torch.tensor(0.0, device=domain_features.device)


def train_loop(epoch_index, dataloader, model, optimizer, device='cpu',
                       lambda_pseudo=0.1, lambda_domain=0.1):
    """
    改进的训练循环，支持伪标签和域适应

    Args:
        epoch_index: 当前epoch索引
        dataloader: 数据加载器，包含(x, y, basin_mask_batch)
        model: 模型
        optimizer: 优化器
        device: 设备
        lambda_pseudo: 伪标签损失权重
        lambda_domain: 域适应损失权重
    """
    train_loss = []
    pseudo_losses = []
    domain_losses = []

    with tqdm(dataloader, ncols=100, desc=f"Epoch {epoch_index + 1}", unit="batch") as tepoch:
        for batch_data in tepoch:
            # 解包数据：x, y, 以及可选的basin_mask
            if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                x, y = batch_data[0], batch_data[1]
                # 第三个元素是每个批次对应的basin_mask
                basin_mask = batch_data[2] if len(batch_data) > 2 else None
            else:
                x, y = batch_data, None
                basin_mask = None

            trainx = x.to(device)
            trainy = y.to(device) if y is not None else None

            # 如果提供了basin_mask，确保它在正确的设备上
            if basin_mask is not None:
                basin_mask = basin_mask.to(device)

            # 如果所有标签都是NaN则跳过
            if torch.isnan(trainy).all():
                continue

            optimizer.zero_grad()

            # 前向传播，获取预测和额外信息
            if hasattr(model, 'use_pseudo_labels') and model.use_pseudo_labels:
                output, extras = model(trainx, basin_mask=basin_mask, return_extras=True)
            else:
                output = model(trainx)
                extras = {}

            # 主损失 - 只在有观测值的位置计算
            main_loss = rmse_masked(trainy, output)

            # 伪标签损失
            p_loss = torch.tensor(0.0).to(device)
            if 'pseudo_labels' in extras and extras['pseudo_labels'] is not None:
                p_loss = pseudo_label_loss(extras['pseudo_labels'], extras['confidences'])
                pseudo_losses.append(p_loss.item())

            # 域适应损失
            d_loss = torch.tensor(0.0).to(device)
            if 'domain_features' in extras and extras['domain_features'] is not None:
                d_loss = domain_adaptation_loss(extras['domain_features'], basin_mask)
                domain_losses.append(d_loss.item())

            # 总损失
            total_loss = main_loss + lambda_pseudo * p_loss + lambda_domain * d_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()

            train_loss.append(main_loss.item())
            tepoch.set_postfix(
                loss=main_loss.item(),
                p_loss=p_loss.item() if p_loss > 0 else 0,
                d_loss=d_loss.item() if d_loss > 0 else 0
            )

    mean_loss = np.mean(train_loss)
    mean_pseudo = np.mean(pseudo_losses) if pseudo_losses else 0
    mean_domain = np.mean(domain_losses) if domain_losses else 0

    return mean_loss, mean_pseudo, mean_domain


def val_loop(dataloader, model, device='cpu'):
    """验证循环"""
    val_loss = []

    with torch.no_grad():
        for x, y in dataloader:
            testx = x.to(device)
            testy = y.to(device)

            # 验证时不使用额外功能
            output = model(testx)
            loss = rmse_masked(testy, output)
            val_loss.append(loss.item() if loss > 0 else np.nan)

    mval_loss = np.nanmean(val_loss)
    print(f"Valid/Test loss: {mval_loss:.2f}")
    return mval_loss


def train_torch(model,
                         optimizer,
                         x_train,
                         y_train,
                         batch_size,
                         max_epochs,
                         early_stopping_patience=False,
                         x_val=None,
                         y_val=None,
                         x_tst=None,
                         y_tst=None,
                         shuffle=False,
                         weights_file=None,
                         log_file=None,
                         device='cpu',
                         missing_basin_mask=None,
                         lambda_pseudo=0.1,
                         lambda_domain=0.1,
                         curriculum_learning=True,
                         warmup_epochs=10):
    """
    改进的训练函数，支持伪标签、域适应和课程学习

    @param curriculum_learning: 是否使用课程学习（逐步增加伪标签权重）
    @param warmup_epochs: 预热周期数
    @param lambda_pseudo: 伪标签损失权重
    @param lambda_domain: 域适应损失权重
    @param missing_basin_mask: 预先计算的缺失流域掩码（布尔数组，标记哪些样本属于缺失流域）
    """

    print(f"Training on {device}")
    print("Starting improved training with pseudo-labels and domain adaptation...")

    if not early_stopping_patience:
        early_stopping_patience = max_epochs

    epochs_since_best = 0
    best_loss = 1000

    # 准备训练数据
    train_data = []
    for i in range(len(x_train)):
        data_item = [
            torch.from_numpy(x_train[i]).float(),
            torch.from_numpy(y_train[i]).float()
        ]
        # 如果提供了missing_basin_mask，为每个样本添加对应的掩码
        if missing_basin_mask is not None:
            data_item.append(torch.tensor(missing_basin_mask[i], dtype=torch.bool))
        train_data.append(data_item)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=shuffle, pin_memory=True
    )

    # 准备验证数据
    if x_val is not None:
        val_data = []
        for i in range(len(x_val)):
            val_data.append([
                torch.from_numpy(x_val[i]).float(),
                torch.from_numpy(y_val[i]).float()
            ])
        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=batch_size, shuffle=False, pin_memory=True
        )

    # 准备测试数据
    if x_tst is not None:
        tst_data = []
        for i in range(len(x_tst)):
            tst_data.append([
                torch.from_numpy(x_tst[i]).float(),
                torch.from_numpy(y_tst[i]).float()
            ])
        tst_loader = torch.utils.data.DataLoader(
            tst_data, batch_size=batch_size, shuffle=False, pin_memory=True
        )

    model.to(device)

    # 训练日志
    log_cols = ['epoch', 'loss', 'pseudo_loss', 'domain_loss', 'val_loss', 'tst_loss']
    train_log = pd.DataFrame(columns=log_cols)

    for epoch in range(max_epochs):
        # 课程学习：逐步增加伪标签和域适应的权重
        if curriculum_learning:
            current_lambda_pseudo = lambda_pseudo * min(1.0, epoch / warmup_epochs)
            current_lambda_domain = lambda_domain * min(1.0, epoch / warmup_epochs)
        else:
            current_lambda_pseudo = lambda_pseudo
            current_lambda_domain = lambda_domain

        model.train()

        # 训练
        epoch_loss, epoch_pseudo, epoch_domain = train_loop(
            epoch, train_loader, model, optimizer, device,
            current_lambda_pseudo, current_lambda_domain
        )

        # 记录到日志
        log_entry = {
            'epoch': epoch,
            'loss': epoch_loss,
            'pseudo_loss': epoch_pseudo,
            'domain_loss': epoch_domain,
            'val_loss': np.nan,
            'tst_loss': np.nan
        }

        # 验证
        if x_val is not None:
            model.eval()
            epoch_val_loss = val_loop(val_loader, model, device)
            log_entry['val_loss'] = epoch_val_loss

            # 早停机制
            if epoch_val_loss < best_loss:
                torch.save(model.state_dict(), weights_file)
                best_loss = epoch_val_loss
                epochs_since_best = 0
                print(f"  New best validation loss: {best_loss:.4f}")
            else:
                epochs_since_best += 1

            if epochs_since_best > early_stopping_patience:
                print(f"Early Stopping at Epoch {epoch}")
                break

        # 测试
        if x_tst is not None:
            model.eval()
            epoch_tst_loss = val_loop(tst_loader, model, device)
            log_entry['tst_loss'] = epoch_tst_loss

        # 添加到日志
        train_log = pd.concat([
            train_log,
            pd.DataFrame([log_entry], index=[epoch])
        ])

        # 打印进度
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss={epoch_loss:.4f}, "
                  f"Pseudo={epoch_pseudo:.4f}, Domain={epoch_domain:.4f}")

    # 保存日志
    if log_file:
        train_log.to_csv(log_file)

    # 如果没有验证集，保存最后的模型
    if x_val is None and weights_file:
        torch.save(model.state_dict(), weights_file)

    print("Training completed!")
    print(f"Best validation loss: {best_loss:.4f}")

    return model


def self_training_iteration(model, x_unlabeled, confidence_threshold=0.7,
                           device='cpu', batch_size=32):
    """
    自训练迭代：使用模型对无标签数据生成伪标签
    """
    model.eval()
    pseudo_labels = []
    high_confidence_mask = []

    # 准备数据
    unlabeled_data = [torch.from_numpy(x).float() for x in x_unlabeled]
    unlabeled_loader = torch.utils.data.DataLoader(
        unlabeled_data, batch_size=batch_size, shuffle=False
    )

    with torch.no_grad():
        for x in unlabeled_loader:
            x = x.to(device)

            # 获取预测和置信度
            if hasattr(model, 'use_pseudo_labels') and model.use_pseudo_labels:
                output, extras = model(x, return_extras=True)

                if 'confidences' in extras and extras['confidences'] is not None:
                    conf = extras['confidences'].squeeze()

                    # 筛选高置信度的样本
                    high_conf = conf > confidence_threshold
                    pseudo_labels.append(output)
                    high_confidence_mask.append(high_conf)
            else:
                output = model(x)
                pseudo_labels.append(output)
                # 如果没有置信度估计，使用预测的标准差作为不确定性度量
                high_confidence_mask.append(torch.ones_like(output[:, :, 0], dtype=torch.bool))

    # 合并所有批次
    pseudo_labels = torch.cat(pseudo_labels, dim=0)
    high_confidence_mask = torch.cat(high_confidence_mask, dim=0)

    return pseudo_labels.cpu().numpy(), high_confidence_mask.cpu().numpy()


def iterative_self_training(model, optimizer, x_labeled, y_labeled,
                           x_unlabeled, iterations=5, **train_kwargs):
    """
    迭代自训练过程
    """
    for iteration in range(iterations):
        print(f"\n--- Self-training iteration {iteration + 1}/{iterations} ---")

        # 生成伪标签
        pseudo_labels, confidence_mask = self_training_iteration(
            model, x_unlabeled,
            confidence_threshold=0.7 - iteration * 0.05,  # 逐步降低阈值
            device=train_kwargs.get('device', 'cpu')
        )

        # 选择高置信度的样本加入训练集
        high_conf_indices = np.where(confidence_mask.any(axis=(1, 2)))[0]

        if len(high_conf_indices) > 0:
            print(f"Adding {len(high_conf_indices)} high-confidence samples")

            # 扩充训练集
            x_augmented = np.concatenate([x_labeled, x_unlabeled[high_conf_indices]], axis=0)

            # 混合真实标签和伪标签
            y_pseudo = pseudo_labels[high_conf_indices]
            y_augmented = np.concatenate([y_labeled, y_pseudo], axis=0)

            # 重新训练
            model = train_torch(
                model, optimizer,
                x_augmented, y_augmented,
                **train_kwargs
            )

            # 更新标记和未标记数据集
            x_labeled = x_augmented
            y_labeled = y_augmented

            # 从未标记集中移除已使用的样本
            remaining_indices = np.setdiff1d(
                np.arange(len(x_unlabeled)),
                high_conf_indices
            )
            x_unlabeled = x_unlabeled[remaining_indices]
        else:
            print("No high-confidence samples found in this iteration")

    return model




def predict_torch(x_data, model, batch_size):
    """
    Make predictions using trained model
    
    @param x_data: input data for prediction
    @param model: [object] trained torch model
    @param batch_size: [int] batch size for prediction
    @return: [tensor] predicted values
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    data = []
    for i in range(len(x_data)):
        data.append(torch.from_numpy(x_data[i]).float())

    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, pin_memory=True)
    model.eval()
    predicted = []
    
    with torch.no_grad():
        for x in dataloader:
            trainx = x.to(device)
            output = model(trainx).detach().cpu()
            predicted.append(output)
            
    predicted = torch.cat(predicted, dim=0)
    return predicted