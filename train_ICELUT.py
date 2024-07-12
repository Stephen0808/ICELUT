import sys
from os.path import join 
import torch.nn as nn
from utils.losses import *
from test_quan import test
from model_ICELUT import *
from datasets import *
from torch.utils.data import DataLoader 
from torch.optim.lr_scheduler import CosineAnnealingLR
from ipdb import set_trace as S

import argparse
import torch
import numpy as np
import os
from time import time

from parameters import *
from torchvision.utils import save_image
# from models import *

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_workers", type=int, default=8, help="for dataloader")
parser.add_argument("--optm", type=str, default="Adam")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--tvmn", default=False, action="store_true", help="whether no use tv and mn constrain")

# --epoch for train:  =1 starts from scratch, >1 load saved checkpoint of <epoch-1>
# --epoch for eval:   load the model of <epoch> and evaluate
parser.add_argument("--epoch", type=int, default=1)

parser.add_argument("--num_epochs", type=int, default=400, help="last epoch of training (include)")
parser.add_argument("--losses", type=str, nargs="+", default=["l1", "cos"], help="one or more loss functions")
parser.add_argument("--model", type=str, nargs="+", default=["ICELUT", "10+05+10"], help="model configuration, [n+s+w, dim]")
parser.add_argument("--name", type=str,default="ICELUT_10+05+10", help="name for this training (if None, use <model> instead)")

parser.add_argument("--save_root", type=str, default=".", help="root path to save images/models/logs")
parser.add_argument("--checkpoint_interval", type=int, default=1)
parser.add_argument("--data_root", type=str, default="/mnt/data/ysd21/", help="root path of data")

parser.add_argument("--dataset", type=str, default="FiveK", help="which dateset class to use (should be implemented first)")



np.set_printoptions(suppress=True)
cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = "cuda" if cuda else 'cpu'


@torch.no_grad()
def test(model, test_dataloader, save_path, best_psnr=None, save_img=False):
    model.eval()
    os.makedirs(save_path, exist_ok=True)  
    avg_psnr_out = 0
    test_start = time()
    for i, batch in enumerate(test_dataloader):
        inputs_msb = batch["input_msb"].to(device)
        inputs_lsb = batch["input_lsb"].to(device)
        inputs_org = batch.get("input_org").to(device)
        targets_org = batch["target_org"].to(device)
        name = os.path.splitext(batch["name"][0])[0]
        # results = model.forward_test(inputs_msb, inputs_lsb, inputs_org, TVMN=None)
        results = model(inputs_msb, inputs_lsb, inputs_org, TVMN=None)
        fakes = results["fakes"]
        psnr_out = psnr(fakes, targets_org).item()
        avg_psnr_out += psnr_out
        if save_img:
            img_ls = [inputs_org.squeeze().data, fakes.squeeze().data, targets_org.squeeze().data]
            if img_ls[0].shape[0] > 3:
                img_ls = [img.permute(2,0,1) for img in img_ls]
            save_image(img_ls, join(save_path, f"{name}_{psnr_out:.2f}.jpg"), nrow=len(img_ls))
        # sys.stdout.write(f"\r{name} {psnr_out:.2f}dB")
        
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    test_cost = time() - test_start

    avg_psnr_out /= len(test_dataloader)
    new_folder_name = save_path + f" {avg_psnr_out:.2f}dB {test_cost:0>5.2f}s"
    if best_psnr is not None and avg_psnr_out > best_psnr:
        new_folder_name += '_best'
    os.rename(save_path,  new_folder_name) 

    return avg_psnr_out, test_cost

if __name__ == "__main__":


    hparams = parser.parse_args()
    hparams.output_dir = join(hparams.save_root, hparams.dataset, hparams.name)
    os.makedirs(hparams.output_dir, exist_ok=True)
    print(f"ckpt will be saved to {hparams.output_dir}")
    hparams.save_models_root = hparams.output_dir
    hparams.save_logs_root = hparams.output_dir
    hparams.save_images_root = hparams.output_dir
    print(*hparams.model[1:])
    model = eval(hparams.model[0])(*hparams.model[1:]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hparams.lr,
    )
    scheduler = None
    train_dataloader = DataLoader(
        eval(hparams.dataset)(hparams.data_root, split="train", model='ICELUT'),
        batch_size=hparams.batch_size,
        shuffle=True,
        num_workers=hparams.num_workers,
    )
    test_dataloader = DataLoader(
        eval(hparams.dataset)(hparams.data_root, split="test", model='ICELUT'),
        batch_size=1,
        shuffle=False,
        num_workers=hparams.num_workers,
    )
    if hparams.tvmn:
        TVMN = TVMN(hparams.model[-1], lambda_smooth=0.0001, lambda_mn=10.0).to(device)
    else:
        TVMN = None
    if hparams.epoch > 1:
        latest_ckpt = torch.load(join(hparams.save_models_root, "latest_ckpt.pth"))
        optimizer.load_state_dict(latest_ckpt['optimizer'])
        best_psnr = latest_ckpt['best_psnr']
        best_epoch = latest_ckpt['best_epoch']
        if scheduler:
            scheduler.load_state_dict(latest_ckpt['scheduler'])
        try:
            model.load_state_dict(torch.load(join(hparams.save_models_root, f"model_{hparams.epoch-1}.pth")), strict=True)
            sys.stdout.write(f"Successfully loading from {hparams.epoch-1} epoch ckpt\n")
        except:
            model.load_state_dict(latest_ckpt['model'], strict=True)
            sys.stdout.write(f"Successfully loading from the latest ckpt\n")
    else:
        best_psnr = 0
        best_epoch = 0
    N = len(train_dataloader)
    interval = N//50
    for epoch in range(hparams.epoch, hparams.num_epochs+1):
        model.train()
        print('starting training ...')
        loss_ls = [0 for loss in hparams.losses] + [0]
        epoch_start_time = time()
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            
            inputs_msb = batch["input_msb"].to(device)
            inputs_lsb = batch["input_lsb"].to(device)
            inputs_org = batch.get("input_org").to(device)
            targets = batch["target"].to(device)
            # flops, params = profile(model, inputs = (inputs, inputs_org, self.TVMN))
            results = model(inputs_msb, inputs_lsb, inputs_org, TVMN=TVMN)
            fakes = results["fakes"]
            loss_ls[-1] = results.get("tvmn_loss", 0)
            
            
            for loss_idx, loss_name in enumerate(hparams.losses):
                loss_ls[loss_idx] = eval(loss_name)(fakes, targets)
            sum(loss_ls).backward()
            optimizer.step()
            
            if i % interval == 0 or i == N-1:
                psnr_result = psnr(fakes, targets).item()
                log_train = f"\rE {epoch:>3d}/{hparams.num_epochs:>3d} B {i+1:>4d} PSNR:{psnr_result:>0.2f}dB "
                for loss_idx, loss_name in enumerate(hparams.losses):
                    log_train += f"{loss_name}:{loss_ls[loss_idx].item():>0.3f} "
                if isinstance(loss_ls[-1], torch.Tensor):
                    log_train += f"tvmn: {loss_ls[-1].item():>0.3f} "
                torch.cuda.synchronize()
                cost_time = (time() - epoch_start_time)/(i+1)
                left_time = cost_time*(N-(i+1))/60
                sys.stdout.write(log_train + f"left={left_time:0>4.2f}m ")
        
        torch.cuda.synchronize()
        cost_time = time() - epoch_start_time
        log_test = " epoch:{:.1f}s ".format(cost_time)

        eval_psnr, test_cost = test(model, test_dataloader, join(hparams.save_images_root, f"{epoch:0>4}"), best_psnr) 
        if eval_psnr > best_psnr:
            best_psnr = eval_psnr
            best_epoch = epoch
            torch.save(model.state_dict(), f"{hparams.save_models_root}/model{epoch:0>4}.pth")

        log_test += f"Test:{eval_psnr:>0.2f}dB {test_cost:0>5.2f}s best:{best_psnr:.2f}dB {best_epoch:3d}. "
        # sys.stdout.write(log_test)
        print(log_test)
        with open(join(hparams.save_logs_root, "log.txt"), "a") as f: # save log
            f.write(log_train + log_test)
        
        ckpt = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_psnr': best_psnr,
            'best_epoch': best_epoch,
        }
        if scheduler is not None:
            scheduler.step()
            ckpt['scheduler'] = scheduler.state_dict()
            
        torch.save(ckpt, f"{hparams.save_models_root}/latest_ckpt.pth")
