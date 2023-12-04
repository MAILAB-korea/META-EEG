# --config=./train.yaml

import os
import time
import random
import argparse
from datetime import datetime
import torch.nn.functional as F

import yaml
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
import higher
import copy

import utils as ut
from model import get_model
from data_generator import task_generator


#  ------------------------------------------------------------------------------------------


def main(config, model, iteration, device, dataset, mini_task_size, Train=False):
    # Model
    model.to(device)
    model.train()

    # Create tasks
    srcdat, srclbl, zrdat, zrlbl = dataset[0], dataset[1], dataset[2], dataset[3]

    if Train:
        tasks_data, tasks_labels = task_generator(config, srcdat, srclbl, iteration, mini_task_size)
    else:
        tasks_data, tasks_labels = list([list([srcdat, zrdat])]), list([list([srclbl, zrlbl])])

    # To control update parameter
    head_params = [p for name, p in model.named_parameters() if 'classifier' in name]
    body_params = [p for name, p in model.named_parameters() if 'classifier' not in name]

    # outer optimizer
    meta_optimizer = torch.optim.Adam([{'params': body_params, 'lr': config['train']['meta_lr']},
                                       {'params': head_params, 'lr': config['train']['meta_lr']
                                       if iteration != 0 and (iteration + 1) % config['train'][
                                           'freeze_epoch'] == 0 else 0}])

    inner_optimizer = torch.optim.Adam([{'params': body_params, 'lr': config['train']['task_lr']},
                                        {'params': head_params, 'lr': config['train']['task_lr']
                                        if iteration != 0 and (iteration + 1) % config['train'][
                                            'freeze_epoch'] == 0 else 0}])

    meta_optimizer.zero_grad()
    inner_optimizer.zero_grad()

    total_loss = torch.tensor(0., device=device)
    accuracy = torch.tensor(0., device=device)

    for task_idx, (task_data, task_label) in enumerate(zip(tasks_data, tasks_labels)):
        with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=False) as (fnet, diffopt):

            outer_loss = torch.tensor(0., device=device)
            spt_data, qry_data = task_data[0], task_data[1]
            spt_label, qry_label = task_label[0], task_label[1]

            src_tensor = TensorDataset(spt_data, spt_label)
            src_loader = DataLoader(src_tensor,
                                    batch_size=config['train']['inner_batch_size'] if Train else config['train'][
                                        'test_batch_size'], shuffle=True, drop_last=False)

            if Train:
                for batch_idx, (inputs, labels) in enumerate(src_loader):
                    inputs, labels = inputs.to(device), labels.to(device)

                    spt_logits = fnet(inputs)
                    spt_loss = F.cross_entropy(spt_logits, labels)
                    diffopt.step(spt_loss)

                inputs, labels = qry_data.to(device), qry_label.to(device)
                query_logit = fnet(inputs)
                outer_loss += F.cross_entropy(query_logit, labels)

                with torch.no_grad():
                    accuracy += ut.get_accuracy(query_logit, labels)
                    total_loss += outer_loss

                if task_idx != 0 and task_idx % config['train']['meta_batch_size'] == 0 or (task_idx + 1) == len(
                        tasks_data):
                    total_loss.div_(config['train']['meta_batch_size'])
                    total_loss.backward()
                    meta_optimizer.step()

            else:
                for batch_idx, (inputs, labels) in enumerate(src_loader):
                    inner_optimizer.zero_grad()
                    inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    logit = model(inputs)
                    loss = criterion(logit, labels)
                    loss.backward()
                    inner_optimizer.step()

                inputs, labels = qry_data.to(device), qry_label.to(device)
                query_logit = model(inputs)
                outer_loss += F.cross_entropy(query_logit, labels)

                with torch.no_grad():
                    accuracy += ut.get_accuracy(query_logit, labels)
                    total_loss += outer_loss

    if Train:
        accuracy.div_(task_idx + 1)
        total_loss.div_(task_idx + 1)

    return total_loss.detach().cpu().numpy(), accuracy.detach().cpu().numpy()


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


if __name__ == '__main__':
    now = datetime.now()
    parser = argparse.ArgumentParser()

    """ GPU """
    parser.add_argument('--gpu', default='1', type=str, help='insert one or more than one int ex: 1, 2, 3')

    """ Date, Time """
    parser.add_argument('--date', default=now.strftime('%Y-%m-%d'), help="Please do not enter any value.")
    parser.add_argument('--time', default=now.strftime('%H:%M:%S'), help="Please do not enter any value.")

    parser.add_argument('--config', help='configuration file')
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r', encoding='UTF8'), Loader=yaml.FullLoader)

    """ Set GPU """
    print('set gpu:', args.gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    current_path = str(os.getcwd())

    for ti in test_subjects:

        default_path = config['save_path']

        if config['backbone'] == 'EEGNet':
            save_path = f'{current_path}/{default_path}/EEGNet/'
        elif config['backbone'] == 'DeepConvNet':
            save_path = f'{current_path}/{default_path}/DeepConvNet/'

        save_result_path = '{}/subject0{}/accuracy/'.format(save_path, str(ti))
        save_trainmodel_path = '{}/subject0{}/trained_model/'.format(save_path, str(ti))

        print("Save Path =", save_result_path)
        ut.create_dir(save_result_path)
        ut.create_dir(save_trainmodel_path)

        criterion = nn.CrossEntropyLoss()
        model = get_model(args, device, config, ti, current_path)

        srcdat_, srclbl_ = torch.rand((4032, 1, 22, 1125)), torch.randint(4, size=(4032,))
        zrdat_, zrlbl_ = torch.rand((576, 1, 22, 1125)), torch.randint(4, size=(576,))

        all_dataset = [srcdat_, srclbl_, zrdat_, zrlbl_]

        train_iter = []
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []

        for meta_iteration in range(config['train']['batch_iter']):
            iter_time = time.time()
            print('Meta iteration =', meta_iteration + 1)
            if meta_iteration != 0 and (meta_iteration + 1) % config['train']['freeze_epoch'] == 0:
                print("Update head parameters")

            mini_task_size = config['train']['mini_task_size']
            meta_train_loss, meta_train_acc = main(config, model, meta_iteration, device, all_dataset, mini_task_size,
                                                   Train=True)

            train_iter.append(meta_iteration + 1)
            train_loss.append(meta_train_loss)
            train_acc.append(meta_train_acc)

            copy_model = copy.deepcopy(model)
            meta_val_loss, meta_val_acc = main(config, copy_model, meta_iteration, device, all_dataset,
                                                 mini_task_size, Train=False)

            val_loss.append(meta_val_loss)
            val_acc.append(meta_val_acc)

            ut.dataexport(args, config, ti, train_iter, train_loss, train_acc, save_result_path, mode='train')
            ut.dataexport(args, config, ti, train_iter, val_loss, val_acc, save_result_path, mode='val')

            print("Train Iter Loss = {:.4f}".format(meta_train_loss))
            print("Train Iter Acc = {:.4f}".format(meta_train_acc))
            print("Validation Iter Loss = {:.4f}".format(meta_val_loss))
            print("Validation Iter Acc = {:.4f}".format(meta_val_acc))

            m, s = divmod(time.time() - iter_time, 60)
            h, m = divmod(m, 60)

            print("Iteration total_time = {} mins {:.6} secs".format(m, s))
            print("=" * 30)

        filename = os.path.join(
            '{}/epoch{}_subject0{}_model_state_dict.pt'.format(save_trainmodel_path, str(meta_iteration + 1),
                                                               str(ti)))
        with open(filename, 'wb') as f:
            state_dict = model.state_dict()
            torch.save(state_dict, f)
