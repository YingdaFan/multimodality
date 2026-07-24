import pandas as pd
import numpy as np
import xarray as xr
import datetime
import torch
from numpy.lib.npyio import NpzFile
from torch_utils import predict_torch
from pathlib import Path





def get_data_if_file(d):
    """
    rudimentary check if data .npz file is already loaded. if not, load it
    :param d:
    :return:
    """
    if isinstance(d, NpzFile) or isinstance(d, dict):
        return d
    else:
        return np.load(d, allow_pickle=True)


def unscale_output(y_scl, y_std, y_mean, y_vars, log_vars=None):
    """
    unscale output data given a standard deviation and a mean value for the
    outputs
    :param y_scl: [pd dataframe] scaled output data (predicted or observed)
    :param y_std:[numpy array] array of standard deviation of variables_to_log [n_out]
    :param y_mean:[numpy array] array of variable means [n_out]
    :param y_vars: [list-like] y_dataset variable names
    :param log_vars: [list-like] which variables_to_log (if any) were logged in data
    prep
    :return: unscaled data
    """
    y_unscaled = y_scl.copy()
    # I'm replacing just the variable columns. I have to specify because, at
    # least in some cases, there are other columns (e.g., "seg_id_nat" and
    # date")
    y_unscaled[y_vars] = (y_scl[y_vars] * y_std) + y_mean
    if log_vars:
        y_unscaled[log_vars] = np.exp(y_unscaled[log_vars])
    return y_unscaled


def predict_from_io_data(
        model,
        io_data,
        partition,
        outfile,
        log_vars=False,
        trn_offset=1.0,
        tst_val_offset=1.0,
        spatial_idx_name="seg_id_nat",
        time_idx_name="date",
        trn_latest_time=None,
        val_latest_time=None,
        tst_latest_time=None
):
    """
    make predictions from trained model
    :param io_data: [str] directory to prepped data file
    :param partition: [str] must be 'trn' or 'tst'; whether you want to predict
    for the train or the dev period
    :param outfile: [str] the file where the output data should be stored
    :param log_vars: [list-like] which variables_to_log (if any) were logged in data
    prep
    :param trn_offset: [str] value for the training offset
    :param tst_val_offset: [str] value for the testing and validation offset
    :param trn_latest_time: [str] when specified, the training partition preds will
    be trimmed to use trn_latest_time as the last date
    :param val_latest_time: [str] when specified, the validation partition preds will
    be trimmed to use val_latest_time as the last date
    :param tst_latest_time: [str] when specified, the test partition preds will
    be trimmed to use tst_latest_time as the last date
    :return: [pd dataframe] predictions
    """
    io_data = get_data_if_file(io_data)
    if partition == "trn":
        keep_portion = trn_offset
        if trn_latest_time:
            latest_time = trn_latest_time
        else:
            latest_time = None
    elif partition == "val":
        keep_portion = tst_val_offset
        if val_latest_time:
            latest_time = val_latest_time
        else:
            latest_time = None
    elif partition == "tst":
        keep_portion = tst_val_offset
        if tst_latest_time:
            latest_time = tst_latest_time
        else:
            latest_time = None

    preds = predict(
        model,
        io_data[f"x_{partition}"],
        io_data[f"ids_{partition}"],
        io_data[f"times_{partition}"],
        io_data["y_std"],
        io_data["y_mean"],
        io_data["y_obs_vars"],
        keep_last_portion=keep_portion,
        outfile=outfile,
        log_vars=log_vars,
        spatial_idx_name=spatial_idx_name,
        time_idx_name=time_idx_name,
        latest_time=latest_time,
        #pad_mask=io_data[f"padded_{partition}"]
        pad_mask = io_data.get(f"padded_{partition}", None)
    )
    return preds


def predict(
        model,
        x_data,
        pred_ids,
        pred_dates,
        y_stds,
        y_means,
        y_vars,
        keep_last_portion=1,
        outfile=None,
        log_vars=False,
        spatial_idx_name="spatial_idx_name",
        time_idx_name="date",
        latest_time=None,
        pad_mask=None
):
    """
    use trained model to make predictions
    :param model: [tf model] trained TF model to use for predictions
    :param x_data: [np array] numpy array of scaled and centered x_data
    :param pred_ids: [np array] the ids of the segments (same shape as x_data)
    :param pred_dates: [np array] the dates of the segments (same shape as
    x_data)
    :param keep_last_portion: [float] fraction of the predictions to keep starting
    from the *end* of the predictions (0-1). (1 means you keep all of the
    predictions, .75 means you keep the final three quarters of the predictions). Alternatively, if
    keep_last_portion is > 1 it's taken as an absolute number of predictions to retain from the end of the
    prediction sequence.
    :param y_stds:[np array] the standard deviation of the y_dataset data
    :param y_means:[np array] the means of the y_dataset data
    :param y_vars:[np array] the variable names of the y_dataset data
    :param outfile: [str] the file where the output data should be stored
    :param log_vars: [list-like] which variables_to_log (if any) were logged in data
    :param latest_time: [str] when provided, the latest time that should be included
    in the returned dataframe
    :param pad_mask: [np array] bool array with True for padded data and False
    otherwise
    :return: out predictions
    """

    num_segs = len(np.unique(pred_ids))
    if issubclass(type(model), torch.nn.Module):
        if len(x_data.shape) > 3:  # Catch for dealing with different GraphWaveNet vs RGCN output, consider changing to bool argument

            y_pred = predict_torch(x_data, model, batch_size=5)
            y_pred = y_pred.transpose(1, 3)
            if pad_mask is not None:
                pad_mask = np.transpose(pad_mask, (0, 3, 2, 1))
            pred_ids = np.transpose(pred_ids, (0, 3, 2, 1))
            pred_dates = np.transpose(pred_dates, (0, 3, 2, 1))
        else:

            y_pred = predict_torch(x_data, model, batch_size=num_segs)

    else:
        raise TypeError("Model must be a torch.nn.Module or tf.Keras.Model")

    # keep only specified part of predictions
    if keep_last_portion > 1:
        frac_seq_len = int(keep_last_portion)
    else:
        frac_seq_len = round(pred_ids.shape[1] * (keep_last_portion))

    y_pred = y_pred[:, -frac_seq_len:, ...]
    print(y_pred.shape)
    #y_pred.to_feather(outfile)
    if outfile:
        outpath = Path(outfile).with_suffix('.npy')
        np.save(outpath, y_pred)
        print(f"Predictions saved to {outpath}")
    return y_pred




