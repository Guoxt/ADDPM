"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from UNet import *
import argparse
import os
import nibabel as nib
# from visdom import Visdom
# viz = Visdom(port=8850)
import sys
import random
sys.path.append(".")
import numpy as np
import time
import torch as th
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import cv2
import nibabel as nib 

seed=2
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def dice_score(pred, targs):
    pred = (pred>=0.5)#.float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()


preseg_model = DSU_Net() 
preseg_model.load_state_dict(torch.load('userhome/....pth'))
preseg_model.eval()
preseg_model.cuda()

de_model = U_Net() 
de_model.load_state_dict(torch.load('userhome/....pth'))
de_model.eval()
de_model.cuda()


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    ds = BRATSDataset(args.data_dir, test_flag=True)
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False)
    data = iter(datal)
    all_images = []
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    while len(all_images) * args.batch_size < args.num_samples:
        b, path = next(data)  #should return an image from the dataloader "data"
        c = th.randn_like(b[:, :1, ...])
        img = th.cat((b, c), dim=1)     #add a noise channel$
        slice_ID=path[0].split("/", -1)[3]

        
        ###
        ### my testing!!!
        ###
        print('Testing model...')
        
        en_num = [4]
        for en_index in en_num:
        
            data_path = 'userhome/.../data/'
            folderlist = os.listdir(data_path)
            for fodername in folderlist[args.file_list]:

                data = np.load(data_path+fodername)
                tru = data[2,:,:,:]

                prob = np.zeros((2,data.shape[1],data.shape[2],data.shape[3]))
                flag = np.zeros((2,data.shape[1],data.shape[2],data.shape[3]))

                vector = data[0:3,:,:,:].astype(float)
                batch_x_img = th.from_numpy(vector.transpose(3,0,1,2)).float()
                input_img = batch_x_img[:,0:2,:,:]

                my_pre = Tmodel(torch.from_numpy(input_img).float().cuda())
                my_pre = torch.nn.Softmax(dim=1)(my_pre).squeeze().detach().cpu().numpy()
                my_pre = np.argmax((my_pre).astype(float),axis=1)
                                        
                pre_seg = my_pre[:,np.newaxis,:,:]
                pre_t = args.pre_t 
                noise = th.randn_like(batch_x_img[:,0:1,:,:])
                pre_seg = diffusion.q_sample(pre_seg, th.tensor([pre_t]* batch_x_img.shape[0]), noise=noise)     #add noise to the segmentation channel                           
                img = th.cat((batch_x_img, pre_seg), dim=1)     #add a noise channel$ 

                num_ensemble = 1
                SCORE = 0
                for k in range(num_ensemble):  #this is for the generation of an ensemble of 5 masks.
                    model_kwargs = {}
                    #start.record()
                    sample_fn = (
                                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
                                )
                    sample, x_noisy, org = sample_fn(
                        pre_t,
                        My_t,
                        pre_seg,
                        model,
                        (args.batch_size, 3, args.image_size, args.image_size), img,
                        clip_denoised=args.clip_denoised,
                        model_kwargs=model_kwargs,
                    )                                    

                    score = TTmodel(sample.float().cuda())
                    SCORE = score + SCORE

                SCORE = SCORE/num_ensemble   
                pred = np.argmax((SCORE).astype(float),axis=0)
                
        print('End while !!!')
        break
    
def create_argparser():
    defaults = dict(
        data_dir="./data/testing",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
        num_ensemble=5      #number of samples in the ensemble
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    
    parser.add_argument('--file_list', type=int, default = "int", help='')
    parser.add_argument('--pre_t', type=int, default = "int", help='')
    
    return parser


if __name__ == "__main__":

    main()
