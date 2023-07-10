"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from network import *
import argparse
import os
import nibabel as nib
# from visdom import Visdom
# viz = Visdom(port=8850)
import sys
from sets import *
from tqdm import tqdm
from torch.utils.data import DataLoader
import random
sys.path.append(".")
import numpy as np
import numpy
import time
import torch as th
import torch.distributed as dist
from diffusion import dist_util, logger
from diffusion.bratsloader import BRATSDataset
from diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import cv2
import nibabel as nib 


############################################################################
def val(model,dataloader):
    '''
    计算模型在验证集上的准确率等信息
    '''
    model.eval()
    val_losses, dcs = [], []
    for ii, data in enumerate(dataloader):
        input, label = data
        val_input = Variable(input.cuda())
        val_label = Variable(label.cuda())
        if opt.use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()
            model = model.cuda()
        outputs=model(val_input)
        pred = outputs.data.max(1)[1].cpu().numpy().squeeze()
        gt = val_label.data.cpu().numpy().squeeze()

        for i in range(gt.shape[0]):
            #print(i)
            dc,val_loss=calc_dice(gt[i,:,:,:],pred[i,:,:,:])
            dcs.append(dc)
            val_losses.append(val_loss)
    model.train()
    return np.mean(dcs),np.mean(val_losses)
############################################################################

def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    ############################################################################
    print('train:')
    lr = args.lr
    batch_size = args.batch_size

    model = U_Net() 

    #if opt.use_gpu: 
    model.cuda()
    train_data=BRATSDataset('userhome/.../training/',False)
    
    criterion = th.nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    loss_meter=AverageMeter()
    previous_loss = 1e+20

    train_dataloader = DataLoader(train_data,batch_size = batch_size,shuffle=True,num_workers=opt.num_workers)
    optimizer = th.optim.Adam(model.parameters(),lr = lr,weight_decay = opt.weight_decay)

    # train
    for epoch in range(args.max_epoch):

        loss_meter.reset()

        for ii,(data,label) in tqdm(enumerate(train_dataloader),total=len(train_data)):
            
            # train model 
            input = Variable(data)
            labell = label.long()
            target = Variable(labell.long())

            #if opt.use_gpu:
            input = input.cuda()
            target = target.cuda()

            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score,target[:,0,:,:])

            loss.backward()
            optimizer.step()

            # meters update and visualize
            loss_meter.update(loss.item())
    
def create_argparser():
    defaults = dict(
        data_dir="./data/testing",
        clip_denoised=True,
        num_samples=1,
        lr=0.001,
        batch_size=12,
        use_ddim=False,
        max_epoch=1000,
        model_path="",
        num_ensemble=5      #number of samples in the ensemble
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    
    return parser


if __name__ == "__main__":

    main()
