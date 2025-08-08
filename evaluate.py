from models import models_vit
import sys
import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from models.Generate_Model import GenerateModel
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import itertools
import datetime
from dataloader.video_dataloader import train_data_loader, test_data_loader
import tqdm
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore", category=UserWarning)
import random

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='DFEW')
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--temporal-layers', type=int, default=1)
    parser.add_argument('--img-size', type=int, default=224)

    parser.add_argument('--workers', type=int, default=8)

    parser.add_argument('--fold', type=int, default=1)
    args = parser.parse_args()
    return args


def main(set, args):
    data_set = set

    # 回归任务无需指定类别数，注释或移除相关代码

    print("*********** Dataset Fold  " + str(data_set) + " ***********")
    test_annotation_file_path = "./annotation/test_labels.txt"


    model = GenerateModel(args=args)
    model = torch.nn.DataParallel(model).cuda()

    test_data = test_data_loader(
        list_file=test_annotation_file_path,
        num_segments=16,
        duration=1,
        image_size=args.img_size
    )

    val_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    rmse, mae = compute_rmse_mae(val_loader, model, args.checkpoint, data_set)

    return rmse, mae


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", log_txt_path=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.log_txt_path = log_txt_path

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        with open(self.log_txt_path, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def compute_rmse_mae(val_loader, model, checkpoint_path, data_set):
    """计算回归任务的RMSE和MAE指标"""
    # 加载模型权重
    pre_trained_dict = torch.load(checkpoint_path)['state_dict']
    model.load_state_dict(pre_trained_dict)
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, target, audio in tqdm.tqdm(val_loader):
            # 数据移至GPU
            images = images.cuda()
            target = target.cuda()  # 目标分数（整数）
            audio = audio.cuda()

            # 模型预测
            output = model(images, audio)  # 输出形状: (batch_size, 1)

            # 保存预测结果和目标值（转为CPU并numpy化）
            all_predictions.extend(output.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy().flatten())

    # 计算RMSE（均方根误差）和MAE（平均绝对误差）
    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
    mae = mean_absolute_error(all_targets, all_predictions)

    # 打印结果
    print(f"所有样本数: {len(all_targets)}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    return rmse, mae


if __name__ == '__main__':
    args = parse_args()
    print('************************')
    for k, v in vars(args).items():
        print(k, '=', v)
    print('************************')
    rmse, mae = main(args.fold, args)

    print('********* Final Results *********')
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print('*********************************')
