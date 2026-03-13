import csv
import os
import argparse
import random

import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import clip
from model.TSANet import MultiModel as create_model
from engine import train_one_epoch, evaluate, create_lr_scheduler
import warnings
from utils.dataset import (MultiTextDataset, Compose, Resize_16, RandomCrop, Resize,
                           RandomVerticalFlip, RandomHorizontalFlip, ToTensor)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
warnings.filterwarnings("ignore", category=UserWarning)


def main(args):
    use_ddp = args.use_ddp
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    world_size = 1
    if use_ddp:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(local_rank)
        world_size = dist.get_world_size()
    is_main_process = (not use_ddp) or dist.get_rank() == 0

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if use_ddp:
        device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    # output file
    filefold_path = args.out_path
    os.makedirs(filefold_path, exist_ok=True)
    file_img_path = os.path.join(filefold_path, "img")
    os.makedirs(file_img_path, exist_ok=True)
    file_weights_path = os.path.join(filefold_path, "weights")
    os.makedirs(file_weights_path, exist_ok=True)
    file_log_path = os.path.join(filefold_path, "log")
    os.makedirs(file_log_path, exist_ok=True)

    # tensorboard
    tb_writer = SummaryWriter(log_dir=file_log_path) if is_main_process else None

    best_val_loss = args.best_val_loss
    start_epoch = 0

    # dataset
    transform = {
        "train": Compose([
            Resize(size=(96, 96)),
            RandomHorizontalFlip(0.5),
            RandomVerticalFlip(0.5),
            ToTensor()
        ]),
        "eval": Compose([
            Resize_16(),
            ToTensor()
        ])
    }
    train_dataset = MultiTextDataset(
        args.root,
        transform,
        "train",
        vis_dir=args.vis_dir,
        ir_dir=args.ir_dir,
        text_vis_dir=args.text_vis_dir,
        text_ir_dir=args.text_ir_dir,
    )
    val_dataset = MultiTextDataset(
        args.root,
        transform,
        "eval",
        vis_dir=args.vis_dir,
        ir_dir=args.ir_dir,
        text_vis_dir=args.text_vis_dir,
        text_ir_dir=args.text_ir_dir,
    )
    train_sampler = None
    if use_ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    if is_main_process:
        print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=train_sampler is None,
                                               sampler=train_sampler,
                                               pin_memory=True,
                                               num_workers=nw,
                                               )
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             )

    # model
    model_clip, _ = clip.load("ViT-B/32", device=device)
    model = create_model(model_clip).to(device)
    for param in model.model_clip.parameters():
        param.requires_grad = False
    if use_ddp:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None, find_unused_parameters=False)

    # checkpoint
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        if is_main_process:
            print(model.load_state_dict(weights_dict, strict=False))
        else:
            model.load_state_dict(weights_dict, strict=False)

    if is_main_process:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model total params：{total_params:,} → {total_params / 1e6:.2f}M")

    # optimizer
    pg = [p for p in model.parameters() if p.requires_grad]
    lr = args.lr * world_size
    if is_main_process:
        print(f"Auto LR scaling enabled: base_lr={args.lr} world_size={world_size} scaled_lr={lr}")
    optimizer = optim.AdamW(pg, lr=lr, weight_decay=5E-2)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        # train
        train_loss, train_ssim_loss, train_max_loss, train_color_loss, train_text_loss, train_consistency_loss, lr = train_one_epoch(
            model=model,
            model_clip=model_clip,
            optimizer=optimizer,
            data_loader=train_loader,
            lr_scheduler=lr_scheduler,
            device=device,
            epoch=epoch)

        if tb_writer is not None:
            tb_writer.add_scalar("train_total_loss", train_loss, epoch)
            tb_writer.add_scalar("train_ssim_loss", train_ssim_loss, epoch)
            tb_writer.add_scalar("train_max_loss", train_max_loss, epoch)
            tb_writer.add_scalar("train_color_loss", train_color_loss, epoch)
            tb_writer.add_scalar("train_text_loss", train_text_loss, epoch)
            tb_writer.add_scalar("train_consistency_loss", train_consistency_loss, epoch)

        if is_main_process and epoch % args.val_every_epcho == 0 and epoch != 0:
            val_loss, val_ssim_loss, val_max_loss, val_color_loss, val_text_loss, val_consistency_loss = evaluate(model=model,
                                                                                            data_loader=val_loader,
                                                                                            device=device,
                                                                                            epoch=epoch, lr=lr,
                                                                                            filefold_path=file_img_path)
            tb_writer.add_scalar("val_total_loss", val_loss, epoch)
            tb_writer.add_scalar("val_ssim_loss", val_ssim_loss, epoch)
            tb_writer.add_scalar("val_max_loss", val_max_loss, epoch)
            tb_writer.add_scalar("val_color_loss", val_color_loss, epoch)
            tb_writer.add_scalar("val_text_loss", val_text_loss, epoch)
            tb_writer.add_scalar("val_consistency_loss", val_consistency_loss, epoch)

            if val_loss < best_val_loss:
                if use_ddp:
                    save_file = {"model": model.module.state_dict(),
                                 "optimizer": optimizer.state_dict(),
                                 "lr_scheduler": lr_scheduler.state_dict(),
                                 "epoch": epoch,
                                 "args": args}
                else:
                    save_file = {"model": model.state_dict(),
                                 "optimizer": optimizer.state_dict(),
                                 "lr_scheduler": lr_scheduler.state_dict(),
                                 "epoch": epoch,
                                 "args": args}
                torch.save(save_file, file_weights_path + "/" + "checkpoint.pth")
                best_val_loss = val_loss

            if use_ddp:
                save_file = {"model": model.module.state_dict(),
                             "optimizer": optimizer.state_dict(),
                             "lr_scheduler": lr_scheduler.state_dict(),
                             "epoch": epoch,
                             "args": args}
            else:
                save_file = {"model": model.state_dict(),
                             "optimizer": optimizer.state_dict(),
                             "lr_scheduler": lr_scheduler.state_dict(),
                             "epoch": epoch,
                             "args": args}
            torch.save(save_file, file_weights_path + "/" + "checkpoint_lastest.pth")
    if use_ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--root', type=str, default="./dataset/ACCV")
    parser.add_argument('--out_path', type=str, default="./experiments/test")
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--val_every_epcho', type=int, default=10, help='val every epcho')
    parser.add_argument('--best_val_loss', type=int, default=1e5, help='')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--device', default='cuda', help='device (i.e. cuda or cpu)')
    parser.add_argument('--use_ddp', default=False, action='store_true', help='use DDP-multigpus')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for DDP')
    parser.add_argument('--seed', type=int, default=1, help='sed')
    parser.add_argument('--vis_dir', type=str, default='vis', help='visible image folder name')
    parser.add_argument('--ir_dir', type=str, default='ir', help='infrared image folder name')
    parser.add_argument('--text_vis_dir', type=str, default='text_vis', help='visible text folder name')
    parser.add_argument('--text_ir_dir', type=str, default='text_ir', help='infrared text folder name')
    opt = parser.parse_args()

    main(opt)
