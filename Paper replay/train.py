import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from model import *

from tqdm import tqdm

from math import cos, pi
import random
import numpy as np

# 便于复现
def same_seeds(seed):
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# warnup
def adjust_learning_rate(optimizer, current_epoch,max_epoch,lr_min=0,lr_max=0.1,warmup=True):
        warmup_epoch = 10 if warmup else 0
        if current_epoch < warmup_epoch:
            lr = lr_max * current_epoch / warmup_epoch
        else:
            lr = lr_min + (lr_max-lr_min)*(1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


same_seeds(2021)



def main(args):
    # 使用GPU
    if args.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(os.path.join(os.getcwd()))  # get data root path
    image_path = os.path.join(data_root, "training_set", "training_set")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path),
                                            transform=data_transform["train"])
    train_num = len(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=args.bs, shuffle=True,
                                                num_workers=0)

    validate_dataset = datasets.ImageFolder(root=os.path.join(data_root+r"\test_set\test_set"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                    batch_size=args.bs, shuffle=False,
                                                    num_workers=0)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                            val_num))

    # 定义模型
    net = VIT_Model(head=16, input_size=256, embedding_size=256, output_size=256,\
                    hidden_size=256 * 4 ,dropout=0, layer_num=2, num_classes=2).to(device)

    # 定义损失函数和优化器
    loss_func = nn.CrossEntropyLoss()
    g = [i for i in net.parameters() if i.requires_grad]
    # optimizer = optim.SGD(g, lr=0.001, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(g, lr=args.lr, weight_decay=args.wd)


    # 打印net参数量
    params_num = sum(p.numel() for p in net.parameters())
    total_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("total params: {}, trainable params: {}".format(params_num, total_num))



    for epoch in range(args.ep):
        net.train()
        progress_bar = tqdm(train_loader)
        accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
        sample_num = 0
        if args.warmup:
            adjust_learning_rate(optimizer=optimizer,current_epoch=epoch,max_epoch=args.ep,lr_min=0.00001,lr_max=0.1,warmup=True)
        for i, (images, labels) in enumerate(progress_bar):
            sample_num += images.shape[0]
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            pred_classes = torch.max(outputs, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels.to(device)).sum()
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            progress_bar.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}"\
                    .format(epoch+1, loss.item(), accu_num.item() / sample_num)
        net.eval()
        correct = 0
        total = 0
        progress_test = tqdm(validate_loader)
        with torch.no_grad():
            for images, labels in progress_test:
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                progress_test.desc = "[validate epoch {}] acc: {:.3f}".format(epoch+1, correct / total)

        if epoch % 10 == 0:
            torch.save(net.state_dict(), "./checkpoint/checkpoint_{}.pth".format(epoch))

if __name__ == '__main__':

    # 接受命令行参数
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--wd', default=5e-4, type=float, help='weight_decay')
    parser.add_argument('--ep', default=100, type=int, help='epochs')
    parser.add_argument('--bs', default=4, type=int, help='batch_size')
    parser.add_argument('--warmup', default=1, type=bool, help='if_warmup')
    parser.add_argument('--gpu', default=1, type=bool, help='gpu_id')
    args = parser.parse_args()

    main(args)
