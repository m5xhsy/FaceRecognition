#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import datetime
import os
import time

# linux
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from torchnet import meter

from inception_resnet_v2 import InceptionResnetV2
from utils import get_num_classes, get_dataloader


def net_training(net, num_classes):
    # optimizer = optim.AdamW(net.parameters(), lr=args.learning_rate,)
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    loss_func = torch.nn.CrossEntropyLoss().to(DEVICE)

    # scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=args.decay_interval,gamma=args.decay_rate)
    # scheduler = optim.lr_scheduler.MultiStepLR(
    #     optimizer,
    #     milestones=[int(i) for i in args.decay_range.split(",")],
    #     gamma=args.decay_rate)

    av = meter.AverageValueMeter()
    cf = meter.ConfusionMeter(num_classes)

    train_info = {
        "train": [[], []],
        "verify": [[], []],
        "lr": []
    }
    base_accu = 0.0

    for epoch in range(1, args.epoch + 1):
        out_list = []
        out_list.append("Training Epoch:{}".format(epoch))
        if not args.log and print(out_list[-1]): pass

        start_time = time.time()
        cf.reset()
        av.reset()

        for phase in ["train", "verify"]:
            if phase == "verify":
                net.train()
            else:
                net.eval()

            for step, (x, y) in enumerate(
                    get_dataloader(data_dir=args.data_dir, mode=phase, batch_size=args.batch_size)):
                x, y = x.to(DEVICE), y.to(DEVICE)

                # 前向传播
                pred = net(x)
                loss = loss_func(pred, y)

                # 梯度清零
                optimizer.zero_grad()

                # 反向传播
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                    if (step + 1) % args.print_interval == 0:
                        out_list.append("[INFO] Step:{}  Loss:{:.4f}".format(str(step + 1).rjust(4), loss.item()))
                        if not args.log and print(out_list[-1]): pass
                # 统计损失和准确度
                av.add(loss.item())
                cf.add(pred.cpu().detach(), y.cpu().detach())

            epoch_loss = av.value()[0]
            epoch_accu = sum([cf.value()[i][i] for i in range(num_classes)]) / cf.value().sum()

            # 学习率衰减
            # if phase == "train" and scheduler.step(): pass

            out_list.append("{}:  loss={:.4f}  accuracy={:.4f}".format(phase, epoch_loss, epoch_accu))
            if not args.log and print(out_list[-1]): pass

            # 准备画图的数据
            if phase == "train":
                train_info['lr'].append(optimizer.param_groups[0]["lr"])
            train_info[phase][0].append(epoch_loss)
            train_info[phase][1].append(epoch_accu)

            # 画图
            if phase == "verify" and epoch >= 3 and draw_plt(train_info, epoch): pass

            # 选择性保存
            if phase == "verify" and epoch_accu > base_accu:
                base_accu = epoch_accu
                torch.save(net, "./net_params.pkl")

        out_list.append("Time for each epoch:{:.2f} seconds\n".format(time.time() - start_time))
        if args.log:
            with open(args.log_file,"a",encoding="utf8") as f:
                for item in out_list:
                    f.write(item+"\n")
                f.write("\n")
        else:
            print(out_list[-1])
            

        net.train()

    torch.save(net, "./net_params.pkl")


def test(num_classes):
    net = torch.load('./net_params.pkl').eval()
    loss_func = torch.nn.CrossEntropyLoss()
    test_cf = meter.ConfusionMeter(num_classes)
    test_av = meter.AverageValueMeter()
    test_cf.reset()
    test_av.reset()

    with torch.no_grad():
        for step, (x, y) in enumerate(get_dataloader(data_dir=args.data_dir, mode="test", batch_size=args.batch_size)):
            x, y = x.to(DEVICE), y.to(DEVICE)
            output = net(x)

            test_av.add(loss_func(output, y).item())
            test_cf.add(output.cpu().detach(), y.cpu().detach())

    accuracy = sum([test_cf.value()[i][i] for i in range(num_classes)]) / test_cf.value().sum()
    print("\ntest Loss={:.4f} Accuracy={:.4f}\n".format(test_av.value()[0], accuracy))


def draw_plt(train_info, length):
    # plt.rcParams['font.family'] = ['sans-serif']
    # plt.rcParams['font.sans-serif'] = ['SimHei']

    plt.figure(figsize=(12, 18), dpi=120)
    plt.figure(1)
    ax = plt.subplot(311)
    ax.set_title("Loss")
    ax.set_xlabel("Epoch Size")
    ax.set_ylabel("Accuracy Pate Size")
    ax.plot([i for i in range(1, length + 1)], train_info["train"][0], label="train_loss", color="#FF0000")
    ax.plot([i for i in range(1, length + 1)], train_info["verify"][0], label="verify_loss", color="#0000FF")
    ax.legend()
    plt.tight_layout()

    bx = plt.subplot(312)
    bx.set_title("Accuracy Rate")
    bx.set_xlabel("Epoch Size")
    bx.set_ylabel("Accuracy Pate Size")
    bx.plot([i for i in range(1, length + 1)], train_info["train"][1], label="train_accuracy", color="#FF0000")
    bx.plot([i for i in range(1, length + 1)], train_info["verify"][1], label="verify_accuracy", color="#0000FF")
    bx.legend()
    plt.tight_layout()

    cx = plt.subplot(313)
    cx.set_title("Learning Rate")
    cx.set_xlabel("Epoch Size")
    cx.set_ylabel("Learning Rate Size")
    cx.plot([i for i in range(1, length + 1)], train_info["lr"], label="lr", color="#00FF00")
    cx.legend()
    plt.tight_layout()

    if os.path.isfile(filename(length - 1)):
        os.remove(filename(length - 1))
    plt.savefig(filename(length))

    plt.close()


def filename(length):
    return "./images/{}_{}_{}_{}_{}_{}_{}_{}_loss.jpg".format(
        str(datetime.date.today()),
        args.learning_rate,
        args.decay_rate,
        args.weight_decay,
        args.decay_interval,
        args.dropout_prob,
        args.batch_size,
        length,
    )


def start():
    num_classes = get_num_classes(args.data_dir)

    # net = InceptionResnetV1(classify=True, num_classes=num_classes,dropout_prob=args.dropout_prob).train().to(DEVICE)
    if os.path.isfile(args.model_path):
        net = torch.load(args.model_path)
    else:
        net = InceptionResnetV2(classify=True, num_classes=num_classes, dropout_prob=args.dropout_prob).train().to(DEVICE)

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    net_training(net, num_classes)
    test(num_classes)


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()

    parser.add_argument("--epoch", type=int, default=60, help="Number of training. (default 60)")
    parser.add_argument("--batch-size", type=int, default=64, help="Number of trainings per batch. (default 64)")
    parser.add_argument("--data-dir", type=str, default="./data", help="Training data path. (default ./data)")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Learning rate. (default 0.1)")
    parser.add_argument("--decay-rate", type=float, default=0.1, help="Decay rate. (default 0.1)")
    parser.add_argument("--dropout-prob", type=float, default=0.8, help="Dropout. (default 0.8)")
    parser.add_argument("--print-interval", type=int, default=100, help="Print interval. (default 100)")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="L2 weight decay. (default 1e-5)")
    parser.add_argument("--decay-interval", type=int, default=20, help="Lr_scheduler decay interval. (default 20)")
    parser.add_argument("--log", action='store_true', help="The output log. (default no output)")
    parser.add_argument("--log-file", type=str,default="./output.log", help="The output log file. (default './output.log')")
    parser.add_argument("--model-path",type=str,help="Load model")
    args = parser.parse_args()

    start()
