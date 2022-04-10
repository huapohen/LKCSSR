from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import argparse
from utils import TrainSetLoader
from ssr_kernel import *
from torchvision.transforms import ToTensor
import os
import numpy as np
import torch.nn.functional as F
import time
from torch.cuda.amp import autocast
scaler = torch.cuda.amp.GradScaler()
from timm.scheduler.cosine_lr import CosineLRScheduler



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'



def prt_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'Total: {total_num/1e6:.2f} M  Trainable: {trainable_num/1e6:.2f} M')
    return

def parse_args():
    parser = argparse.ArgumentParser()
    # 要改这里 FIXME
    num_gpus = 1 # 默认读取机器上所有GPU
    num_gpus = 6 # 训练时用的 6块显卡机
    parser.add_argument('--num_gpus', type=int, default=num_gpus)
    parser.add_argument("--scale_factor", type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=16*num_gpus)
    parser.add_argument('--num_workers', type=int, default=6*num_gpus)
    parser.add_argument('--lr_base', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--lr_min', type=float, default=5e-6)
    parser.add_argument('--warmup_lr_init', type=float, default=5e-7)
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
    parser.add_argument('--n_steps', type=int, default=1, help='number of epochs to update learning rate')
    parser.add_argument('--n_epochs', type=int, default=60, help='number of epochs to train')
    parser.add_argument('--gamma', type=float, default=0.87, help='use for lr_scheduler')
    parser.add_argument('--trainset_dir', type=str, default='./data')
    parser.add_argument('--model_name', type=str, default='SSR')
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--testset_dir', type=str, default='./data/test/')
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--clip_grad', type=int, default=5.)
    parser.add_argument('--weight_decay', type=int, default=0.05)
    return parser.parse_args()




def build_scheduler(config, optimizer, n_iter_per_epoch):
    num_steps = int(config.n_epochs * n_iter_per_epoch)
    warmup_steps = int(config.warmup_epochs * n_iter_per_epoch)
    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_steps,
        t_mul=1.,
        lr_min=config.lr_min,
        warmup_t=0,
        warmup_lr_init=0,
        # warmup_t=warmup_steps,
        # warmup_lr_init=config.warmup_lr_init,
        cycle_limit=1,
        t_in_epochs=False,
    )
    return lr_scheduler

def load_pretrain(model, pretrained_dict):
    torch_params =  model.state_dict()
    for k,v in pretrained_dict.items():
        print(k)
    pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if k in torch_params}
    torch_params.update(pretrained_dict_1)
    model.load_state_dict(torch_params)

def train(train_loader, cfg):
    iter_per_epoch = len(train_loader)

    net = SSR(cfg.scale_factor).cuda()
    prt_parameter_number(net)
    cudnn.benchmark = True
    scale = cfg.scale_factor

    net = torch.nn.DataParallel(net)

    if cfg.load_pretrain:
        if os.path.isfile(cfg.model_path):
            model = torch.load(cfg.model_path)
            net.load_state_dict(model['state_dict'])
            cfg.start_epoch = model["epoch"]
        else:
            print("=> no model found at '{}'".format(cfg.load_model))


    # net = torch.nn.DataParallel(net, device_ids=[0, 1])
    criterion_L1 = torch.nn.L1Loss().cuda()
    optimizer = torch.optim.AdamW([paras for paras in net.parameters() if paras.requires_grad == True],
                                  lr=cfg.lr_base,
                                  weight_decay=cfg.weight_decay)
    # scheduler = build_scheduler(cfg, optimizer, iter_per_epoch)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=cfg.n_steps,
                                                gamma=cfg.gamma)

    loss_epoch = []

    train_start_time = time.time()
    for epoch in range(cfg.start_epoch, cfg.n_epochs):
        epoch_start_time = time.time()
        tmp_cost_time = 0
        for iter, (HR_left, HR_right, LR_left, LR_right) in enumerate(train_loader):
            b, c, h, w = LR_left.shape
            _, _, h2, w2 = HR_left.shape
            HR_left, HR_right = Variable(HR_left).cuda(), Variable(HR_right).cuda()
            LR_left, LR_right = Variable(LR_left).cuda(), Variable(LR_right).cuda()

            optimizer.zero_grad()

            with autocast():
                SR_left0, SR_right0, \
                SR_left1, SR_right1, \
                SR_left2, SR_right2, \
                SR_left3, SR_right3, \
                SR_left4, SR_right4, \
                =net(LR_left, LR_right)

                ''' SR Loss '''
                loss_SR0 = criterion_L1(SR_left0, HR_left) + criterion_L1(SR_right0, HR_right)
                loss_SR1 = criterion_L1(SR_left1, HR_left) + criterion_L1(SR_right1, HR_right)
                loss_SR2 = criterion_L1(SR_left2, HR_left) + criterion_L1(SR_right2, HR_right)
                loss_SR3 = criterion_L1(SR_left3, HR_left) + criterion_L1(SR_right3, HR_right)
                loss_SR4 = criterion_L1(SR_left4, HR_left) + criterion_L1(SR_right4, HR_right)

                ''' Total Loss '''
                loss = 0.5*loss_SR0 + 0.5*loss_SR1 + 1*loss_SR2 + 1.5*loss_SR3 + 2*loss_SR4

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            # scheduler.step(epoch * iter_per_epoch + iter)

            loss_epoch.append(loss.data.cpu())

            torch.cuda.synchronize()
            prt_iter = 10
            if iter % prt_iter == 0:
                # lr = optimizer.param_groups[0]['lr']
                lr = scheduler.get_last_lr()[0]
                mem = torch.cuda.max_memory_allocated() / 1e9
                time_per_prt = time.time() - tmp_cost_time
                t = f'{time_per_prt / prt_iter:.2f}'
                t1 = int((iter_per_epoch - iter) * (time_per_prt / prt_iter))
                t1 = f"{t1//3600:02d}:{(t1%3600)//60:02d}:{t1%60:02d}"
                t2 = int((cfg.n_epochs - epoch) * iter_per_epoch* (time_per_prt / prt_iter))
                t2 = f"{t2//3600:02d}:{(t2%3600)//60:02d}:{t2%60:02d}"
                cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                loss_mean = float(np.array(loss_epoch).mean())
                img_per_s = cfg.batch_size * cfg.num_gpus / (time_per_prt / prt_iter)
                prt_str = f"Epoch [{epoch}/{cfg.n_epochs}]  iter: [{iter}/{iter_per_epoch}]  " + \
                          f"lr: {lr:.8f}  loss: {loss:.4f}({loss_mean:.4f})  mem: {mem:.3f}G  " + \
                          f"{cur_time} \n img_t: {img_per_s:.2f}img/s   " + \
                          f"iter_t: {t}s/it   epoch_t: {t1}   train_t: {t2}"
                tmp_cost_time = time.time()
                print(prt_str + '\n')
                with open('./log.txt', 'a') as f:
                    f.write(prt_str + '\n')

        scheduler.step()

        torch.save({'epoch': epoch + 1, 'state_dict': net.state_dict()},
                './checkpoints/' + cfg.model_name + '_' + str(cfg.scale_factor) + 'xSR_epoch' + str(epoch + 1) + '.pth.tar')


def main(cfg):
    train_set = TrainSetLoader(cfg)
    train_loader = DataLoader(dataset=train_set,
                              num_workers=cfg.num_workers,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              prefetch_factor=3)
    train(train_loader, cfg)

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
