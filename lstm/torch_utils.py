import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
import pandas as pd
import time
from tqdm import tqdm
import math as m


## Generic PyTorch Training Routine
def train_loop(epoch_index, dataloader, model, loss_function, optimizer, device='cpu',
               smap_provider=None, smap_split='trn'):
    """
    @param epoch_index: [int] Epoch number
    @param dataloader: [object] torch dataloader with train and val data
    @param model: [object] initialized torch model
    @param loss_function: loss function
    @param optimizer: [object] Chosen optimizer
    @param device: [str] cpu or gpu
    @return: [float] epoch loss
    """
    train_loss = []
    for batch in dataloader:
        x, y, idx = batch if len(batch) == 3 else (*batch, None)
        trainx = x.to(device)
        trainy = y.to(device)
        if torch.isnan(trainy).all():
            continue
        optimizer.zero_grad()
        if smap_provider is not None:
            smap_args = smap_provider.batch(smap_split, idx, device)
            output = model(trainx, *smap_args)
        else:
            output = model(trainx)
        loss = loss_function(trainy, output)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
        optimizer.step()
        train_loss.append(loss.item())
    mean_loss = np.mean(train_loss)
    return mean_loss


def val_loop(dataloader, model, loss_function, device='cpu',
             smap_provider=None, smap_split='val'):
    """
    @param dataloader: [object] torch dataloader with train and val data
    @param model: [object] initialized torch model
    @param loss_function: loss function
    @param device: [str] cpu or gpu
    @return: [float] epoch validation loss
    """
    val_loss = []
    with torch.no_grad():
        for iter, batch in enumerate(dataloader):
            x, y, idx = batch if len(batch) == 3 else (*batch, None)
            testx = x.to(device)
            testy = y.to(device)

            if smap_provider is not None:
                smap_args = smap_provider.batch(smap_split, idx, device)
                output = model(testx, *smap_args)
            else:
                output = model(testx)
            loss = loss_function(testy, output)
            val_loss.append(np.nan if type(loss)==float else loss.item())
    mval_loss = np.nanmean(val_loss)
    print(f"Valid/Test loss: {mval_loss:.2f}")
    return mval_loss


def train_torch(model,
                loss_function,
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
                keep_portion=None,
                use_scheduler=False,
                scheduler_milestones=[40, 50],
                scheduler_gamma=0.3,
                smap_provider=None):
    """
    @param model: [object] initialized torch model
    @param loss_function: loss function
    @param optimizer: [object] chosen optimizer
    @param x_train: training input data
    @param y_train: training target data
    @param batch_size: [int]
    @param max_epochs: [int] maximum number of epochs to run for
    @param early_stopping_patience: [int] number of epochs without improvement in validation loss to run before stopping training
    @param x_val: validation input data
    @param y_val: validation target data
    @param x_tst: test input data
    @param y_tst: test target data
    @param shuffle: [bool] Shuffle training batches
    @param weights_file: [str] path save trained model weights
    @param log_file: [str] path to save training log to
    @param device: [str] cpu or gpu
    @param keep_portion: [float/int] portion of data to keep for training
    @param use_scheduler: [bool] whether to use learning rate scheduler
    @param scheduler_milestones: [list] milestones for scheduler
    @param scheduler_gamma: [float] gamma for scheduler
    @return: [object] trained model
    """

    print(f"Training on {device}")
    print("start training...", flush=True)

    if not early_stopping_patience:
        early_stopping_patience = max_epochs

    epochs_since_best = 0
    best_loss = 1000

    if keep_portion is not None:
        if keep_portion > 1:
            period = int(keep_portion)
        else:
            period = int(keep_portion * y_train.shape[1])
        y_train[:, :-period, ...] = np.nan
        if y_val is not None:
            y_val[:, :-period, ...] = np.nan

    # Prepare training data (sample index rides along for the SMAP provider)
    train_data = []
    for i in range(len(x_train)):
        train_data.append([torch.from_numpy(x_train[i]).float(),
                           torch.from_numpy(y_train[i]).float(), i])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


    # Prepare validation data
    if x_val is not None:
        val_data = []
        for i in range(len(x_val)):
            val_data.append([torch.from_numpy(x_val[i]).float(),
                             torch.from_numpy(y_val[i]).float(), i])
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

    # Prepare test data
    if x_tst is not None:
        tst_data = []
        for i in range(len(x_tst)):
            tst_data.append([torch.from_numpy(x_tst[i]).float(),
                             torch.from_numpy(y_tst[i]).float(), i])
        tst_loader = torch.utils.data.DataLoader(tst_data, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

    val_time = []
    train_time = []

    model.to(device)
    
    # Optional scheduler setup
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_milestones, gamma=scheduler_gamma)

    log_cols = ['epoch', 'loss', 'val_loss', 'tst_loss', 'train_time', 'val_time']
    train_log = pd.DataFrame(columns=log_cols)

    for i in range(max_epochs):
        t1 = time.time()

        # Step scheduler if using
        if use_scheduler:
            scheduler.step()

        model.train()
        epoch_loss = train_loop(i, train_loader, model, loss_function, optimizer, device, smap_provider, "trn")
        epoch_time = time.time() - t1
        train_time.append(epoch_time)
        print(f"Epoch {i+1}: train_loss={epoch_loss:.3f}, time={epoch_time:.1f}s")
        train_log = pd.concat(
        [train_log, pd.DataFrame([[i, epoch_loss, np.nan, np.nan, epoch_time, np.nan]], columns=log_cols, index=[i])])


        #val
        if x_val is not None:
            s1 = time.time()
            model.eval()
            epoch_val_loss = val_loop(val_loader, model, loss_function, device, smap_provider, "val")


            if epoch_val_loss < best_loss:
                torch.save(model.state_dict(), weights_file)
                best_loss = epoch_val_loss
                epochs_since_best = 0
            else:
                epochs_since_best += 1
                
            if epochs_since_best > early_stopping_patience:
                print(f"Early Stopping at Epoch {i}")
                break
                
            train_log.loc[train_log.epoch == i, "val_loss"] = epoch_val_loss
        
        #tst
        if x_tst is not None:
            model.eval()
            epoch_tst_loss = val_loop(tst_loader, model, loss_function, device, smap_provider, "tst")
            train_log.loc[train_log.epoch == i, "tst_loss"] = epoch_tst_loss

    train_log.to_csv(log_file)
    if x_val is None:
        torch.save(model.state_dict(), weights_file)
        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    else:
        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        print("Average Validation (Inference) Time: {:.4f} secs/epoch".format(np.mean(val_time)))
    return model


def rmse_masked(y_true, y_pred):

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


def predict_torch(x_data, model, batch_size, smap_provider=None, smap_split=None):
    """
    Make predictions using trained model

    @param x_data: input data for prediction
    @param model: [object] trained torch model
    @param batch_size: [int] batch size for prediction
    @param smap_provider: [SMAPProvider] optional SMAP batch provider
    @param smap_split: [str] provider split name ('trn'/'val'/'tst')
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
        offset = 0
        for iter, x in enumerate(dataloader):
            trainx = x.to(device)
            if smap_provider is not None:
                idx = range(offset, offset + len(x))
                smap_args = smap_provider.batch(smap_split, idx, device)
                output = model(trainx, *smap_args).detach().cpu()
            else:
                output = model(trainx).detach().cpu()
            offset += len(x)
            predicted.append(output)

    predicted = torch.cat(predicted, dim=0)
    return predicted