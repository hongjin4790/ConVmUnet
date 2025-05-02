"""
Our experimental codes are based on 
https://github.com/McGregorWwww/UCTransNet
We thankfully acknowledge the contributions of the authors
"""

import torch.optim
import torch.nn as nn
from tensorboardX import SummaryWriter
import os
import numpy as np
import random
from torch.backends import cudnn
from Load_Dataset import RandomGenerator,ValGenerator,ImageToImage2D


# model 임포트 구간
from nets.UNet_base import UNet_base
from nets.vmunet.vmunet import VMUNet
# from nets.vmunet.vmunet_v2_CIM import VMUNetV2
# from nets.vmunet.vmunet_v2 import VMUNetV2
from nets.vmunet.vmunet_v2_CIM import VMUNetV2
# from nets.vmunet.vmunet_v2_CIM_decoder import VMUNetV2

from nets.SwinUnet import SwinUnet

from nets.conVmunet.ConVmUnet import ConVmUnet
# from nets.conVmunet.convNext_vmunet import ConvNextVMUNet
# from nets.conVmunet.convNext_vmunet_CIM import ConvNextVMUNet # ConvNextVMUNet 에 CIM 추가하고 unetv2의 SDI적용한것
from nets.transunet.vit_seg_modeling import VisionTransformer as ViT_seg
from nets.unet_v2.Unet_v2 import UNetV2

### vss block 1branch
# from nets.vmamba.vmunetV2_1branch import VMUNetV2_1branch

## vmunetV2_adapter
# from nets.vmamba.vmunet_adapter import VMUNetV2_adapter


from torch.utils.data import DataLoader
import logging
import json
import platform
from Train_one_epoch import train_one_epoch
import Config as config
from torchvision import transforms
from utils import CosineAnnealingWarmRestarts, WeightedDiceBCE, BceDiceLoss

def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr

def save_checkpoint(state, save_path):
    '''
        Save the current model.
        If the model is the best model since beginning of the training
        it will be copy
    '''
    logger.info('\t Saving to {}'.format(save_path))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']  # epoch no
    best_model = state['best_model']  # bool
    model = state['model']  # model type

    if best_model:
        filename = save_path + '/' + \
                   'best_model-{}.pth.tar'.format(model)
    else:
        filename = save_path + '/' + \
                   'model-{}-{:02d}.pth.tar'.format(model, epoch)
    torch.save(state, filename)

def worker_init_fn(worker_id):
    random.seed(config.seed + worker_id)

##################################################################################
#=================================================================================
#          Main Loop: load model,
#=================================================================================
##################################################################################
def main_loop(batch_size=config.batch_size, model_type='', tensorboard=True):
    # Load train and val data
    
    train_tf= transforms.Compose([RandomGenerator(output_size=[config.img_size, config.img_size])])
    val_tf = ValGenerator(output_size=[config.img_size, config.img_size])
    train_dataset = ImageToImage2D(config.train_dataset, train_tf,image_size=config.img_size,)
    val_dataset = ImageToImage2D(config.val_dataset, val_tf,image_size=config.img_size)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              worker_init_fn=worker_init_fn,
                              num_workers=8,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            shuffle=True,
                            worker_init_fn=worker_init_fn,
                            num_workers=8,
                            pin_memory=True)

    lr = config.learning_rate
    
    logger.info(model_type)
    # logger.info('n_filts : ' + str(config.n_filts))

    if model_type == 'ACC_UNet':
        config_vit = config.get_CTranS_config()        
        model = ACC_UNet_Lite(n_channels=config.n_channels,n_classes=config.n_labels,n_filts=8)


    elif model_type == 'UNet_base':       
        model = UNet_base(n_channels=config.n_channels,n_classes=config.n_labels)
        lr = 1e-3
    

    elif model_type == 'TransUnet':
        config_vit = config.get_r50_b16_config()
        config_vit.n_classes = 1
        config_vit.patches.grid = (int(config.img_size /16), int(config.img_size / 16))
        model =  ViT_seg(config_vit, img_size=config.img_size, num_classes=config_vit.n_classes)
        model.load_from(weights=np.load(config_vit.pretrained_path))
        lr = 1e-2
        

    elif model_type == 'SwinUnet':            
        model = SwinUnet()
        model.load_from()
        lr = 5e-4
    
    elif model_type == 'vmunet':
        config_vmunet = config.get_vmamba_config()
        model = VMUNet(input_channels=config.n_channels, 
                       num_classes=config.n_labels, 
                       depths= config_vmunet.depths, 
                       depths_decoder=config_vmunet.depths_decoder, 
                       drop_path_rate= config_vmunet.drop_path_rate,
                       load_ckpt_path=config_vmunet.load_ckpt_path,
                       )
        model.load_from()
        lr = 1e-3

    elif model_type == 'vmunet_v2': ## vmunet_v2
        config_vmunetV2 = config.get_vmamba_config()
        model = VMUNetV2(input_channels=config.n_channels, 
                       num_classes=config.n_labels, 
                       depths= config_vmunetV2.depths, 
                       depths_decoder=config_vmunetV2.depths_decoder, 
                       drop_path_rate= config_vmunetV2.drop_path_rate,
                       load_ckpt_path=config_vmunetV2.load_ckpt_path,
                       deep_supervision=config_vmunetV2.deep_supervision,
                       )
        model.load_from()
        lr = 0.001
    
    elif model_type == 'ConVmUnet':
        model = ConVmUnet(num_classes=1)
        lr = 0.001
    
    elif model_type == 'ConvNext_vmunet':
        model = ConvNextVMUNet(num_classes=1)
        lr = 0.001
    else: 
        raise TypeError('Please enter a valid name for the model type')

    if model_type == 'SwinUnet' or model_type == 'TransUnet':            
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    elif model_type =='vmunet' or model_type == 'vmunet_v2' or model_type=='ConvNext_vmunet' or model_type == 'vmunetv2_1branch':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas= (0.9, 0.999), eps= 1e-8, weight_decay=1e-2, amsgrad=False)
    else: # UnetV2 # vmunetv2_adapter
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)  # Choose optimize


    model = model.cuda()
    
    logger.info('Training on ' +str(platform.uname()[1]))
    logger.info('Training using GPU : '+torch.cuda.get_device_name(torch.cuda.current_device()))

    if model_type == 'UNet_base':
        criterion = BceDiceLoss()
    else:
        criterion = WeightedDiceBCE(dice_weight=0.5,BCE_weight=0.5, n_labels=config.n_labels)
    
    if config.cosineLR is True:
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)
    else:
        lr_scheduler =  None
    if tensorboard:
        log_dir = config.tensorboard_folder
        logger.info('log dir: '.format(log_dir))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    max_dice = 0.0
    best_epoch = 1
    for epoch in range(config.epochs):  # loop over the dataset multiple times
        logger.info('\n========= Epoch [{}/{}] ========='.format(epoch + 1, config.epochs + 1))
        logger.info(config.session_name)
        # train for one epoch
        model.train(True)
        logger.info('Training with batch size : {}'.format(batch_size))
        train_one_epoch(train_loader, model, criterion, optimizer, writer, epoch, None, model_type, logger)
        # evaluate on validation set
        logger.info('Validation')
        with torch.no_grad():
            model.eval()
            val_loss, val_dice = train_one_epoch(val_loader, model, criterion,
                                            optimizer, writer, epoch, lr_scheduler,model_type,logger)

        # =============================================================
        #       Save best model
        # =============================================================
        if val_dice > max_dice:
                #if epoch+1 > 5:
                logger.info('\t Saving best model, mean dice increased from: {:.4f} to {:.4f}'.format(max_dice,val_dice))
                max_dice = val_dice
                best_epoch = epoch + 1
                save_checkpoint({'epoch': epoch,
                                 'best_model': True,
                                 'model': model_type,
                                 'state_dict': model.state_dict(),
                                 'val_loss': val_loss,
                                 'optimizer': optimizer.state_dict()}, config.model_path)#+f'_{epoch}')
        else:
            logger.info('\t Mean dice:{:.4f} does not increase, '
                        'the best is still: {:.4f} in epoch {}'.format(val_dice,max_dice, best_epoch))
        early_stopping_count = epoch - best_epoch + 1
        logger.info('\t early_stopping_count: {}/{}'.format(early_stopping_count,config.early_stopping_patience))

        if early_stopping_count > config.early_stopping_patience:
            logger.info('\t early_stopping!')
            break

    return model


if __name__ == '__main__':
    deterministic = True
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    if not os.path.isdir(config.save_path):
        os.makedirs(config.save_path)

    if os.path.isfile(config.logger_path):
        import sys
        sys.exit()
    logger = logger_config(log_path=config.logger_path)
    model = main_loop(model_type=config.model_name, tensorboard=True)
    
    fp = open('log.log','a')
    fp.write(f'{config.model_name} on {config.task_name} completed\n')
    fp.close()