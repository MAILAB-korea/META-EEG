import os
import numpy as np
import torch
import pandas as pd
import json


def LOO_BCI(n_ch=3, n_time=1125, val_sbj=None, test_sbj=None, train=False, val=False, adapt=False):
    all = np.arange(1, 10)
    if train:
        tmp = np.setdiff1d(all, [test_sbj])
        if not adapt:
            sources = np.setdiff1d(tmp, [val_sbj])
        if adapt:
            sources = tmp
        all_srcdat = np.empty(shape=(0, 1, n_ch, n_time), dtype=np.float32)
        all_srclbl = np.empty(shape=(0), dtype=np.int32)

        for i in sources:
            print("=" * 30, f"> Load Sub. {i}'s data")

            trdat, tsdat, trlbl, tslbl = load_time(i, use_2b=True)

            srcdat = np.concatenate((trdat, tsdat), axis=0)
            srclbl = np.concatenate((trlbl, tslbl), axis=0)
            all_srcdat = np.concatenate((all_srcdat, srcdat), axis=0)
            all_srclbl = np.concatenate((all_srclbl, srclbl), axis=0)

        # Load numpy dataset
        data_ = torch.from_numpy(all_srcdat).float()
        label_ = torch.from_numpy(all_srclbl).long()
        print("Train dataset shape = {}".format(data_.shape))

    if not train:
        if val:
            print("=" * 30, f"> Load Sub. {val_sbj}'s data For Validation Dataset")
            trdat_z, tsdat_z, trlbl_z, tslbl_z = load_time(val_sbj, use_2b=True)
        elif not val:
            print("=" * 30, f"> Load Sub. {test_sbj}'s data For Test Dataset")
            trdat_z, tsdat_z, trlbl_z, tslbl_z = load_time(test_sbj, use_2b=True)
        all_zrdat = np.concatenate((trdat_z, tsdat_z), axis=0)
        all_zrlbl = np.concatenate((trlbl_z, tslbl_z), axis=0)

        data_ = torch.from_numpy(all_zrdat).float()
        label_ = torch.from_numpy(all_zrlbl).long()
        if val:
            print("Validation dataset shape = {}".format(data_.shape))
        elif not val:
            print("Test dataset shape = {}".format(data_.shape))

    return data_, label_


def load_time(sbj, cbcic=False, use_2b=False):
    path = 'datapath'
    tr = np.load(os.path.join(path, f"S{sbj:02}_train_X.npy"))  # session1
    ts = np.load(os.path.join(path, f"S{sbj:02}_test_X.npy"))  # session2
    trl = np.load(os.path.join(path, f"S{sbj:02}_train_y.npy"))  # session1 label
    tsl = np.load(os.path.join(path, f"S{sbj:02}_test_y.npy"))  # session2 label

    trl = trl.reshape(trl.shape[0])
    tsl = tsl.reshape(tsl.shape[0])
    return tr, ts, trl, tsl


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass

    return path


def dataexport(args, config, subject, epoch_list, loss_list, acc_list, save_path, mode):
    arg_info = {}
    arg_info.update(vars(args))
    arg_info.update(config)

    csv_filename = None
    txt_filename = None

    if mode == 'train':
        csv_filename = '/train_subject0' + str(subject) + '.csv'
        txt_filename = '/train_subject0' + str(subject) + '_argparse.txt'
    elif mode == 'val':
        csv_filename = '/val_subject0' + str(subject) + '.csv'
        txt_filename = '/val_subject0' + str(subject) + '_argparse.txt'
    elif mode == 'test':
        csv_filename = '/test_subject0' + str(subject) + '.csv'
        txt_filename = '/test_subject0' + str(subject) + '_argparse.txt'

    df = pd.DataFrame({'Epoch': epoch_list,
                       'Loss': loss_list,
                       'ACC': acc_list
                       })

    df.to_csv(save_path + csv_filename, index=False)
    with open(save_path + txt_filename, 'w') as file:
        file.write(json.dumps(arg_info))


def get_accuracy(logits, targets):
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(targets).float())

