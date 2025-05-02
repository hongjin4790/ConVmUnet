"""
Our experimental codes are based on 
https://github.com/McGregorWwww/UCTransNet
We thankfully acknowledge the contributions of the authors
"""

import torch.optim
from Load_Dataset import ValGenerator, ImageToImage2D
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings("ignore")
import Config as config
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import os
import time


from nets.UNet_base import UNet_base
from nets.vmunet.vmunet import VMUNet
from nets.vmunet.vmunet_v2_CIM import VMUNetV2
from nets.conVmunet.ConVmUnet import ConVmUnet


from nets.SwinUnet import SwinUnet


from nets.conVmunet.convNext_vmunet import ConvNextVMUNet
# from nets.conVmunet.convNext_vmunet_CIM import ConvNextVMUNet

from nets.transunet.vit_seg_modeling import VisionTransformer as ViT_seg

from nets.unet_v2.Unet_v2 import UNetV2

### vss block 1branch
# from nets.vmamba.vmunetV2_1branch import VMUNetV2_1branch

# #adapter
# from nets.vmamba.vmunet_adapter import VMUNetV2_adapter


import json
from utils import *
from sklearn.metrics import recall_score, confusion_matrix
import cv2


def cal_iou_recall_dice(predict_save, labs):

    tmp_lbl = (labs).astype(np.float32)
    tmp_pred = (predict_save).astype(np.float32)
    # iou_pred = jaccard_score(tmp_lbl.reshape(-1),tmp_pred.reshape(-1))
    # dice_pred = 2 * np.sum(tmp_lbl * tmp_pred) / (np.sum(tmp_lbl) + np.sum(tmp_pred) + 1e-5)
    recall_pred = recall_score(tmp_lbl.reshape(-1),tmp_pred.reshape(-1))


    if len(tmp_lbl.reshape(-1)) != len(tmp_pred.reshape(-1)):
        raise ValueError(f"Input variables have inconsistent numbers of samples: {len(tmp_lbl.reshape(-1))} and {len(tmp_pred.reshape(-1))}")

    confusion = confusion_matrix(tmp_lbl.reshape(-1), tmp_pred.reshape(-1))
    TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

    accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
    sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
    specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
    f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
    miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
    log_info = f'test of best model, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
    # print(log_info)

    return accuracy, sensitivity, specificity, f1_or_dsc, miou, recall_pred
    

def vis_and_save_heatmap(model, input_img, labs, vis_save_path):
    model.eval()
    start_time = time.time()
    output = model(input_img.cuda())
    elapsed_time = time.time() - start_time
    pred_class = torch.where(output>0.5,torch.ones_like(output),torch.zeros_like(output))
    predict_save = pred_class[0].cpu().data.numpy()
    predict_save = np.reshape(predict_save, (config.img_size, config.img_size))
    accuracy_tmp, sensitivity_tmp, specificity_tmp, f1_or_dsc_tmp, miou_tmp, recall_pred_tmp = cal_iou_recall_dice(predict_save, labs)
    input_img.to('cpu')


    # input_img = input_img[0].transpose(0,-1).cpu().detach().numpy()
    input_img = input_img[0].permute(1, 2, 0).cpu().detach().numpy()

    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    labs = labs[0]
    output = output[0,0,:,:].cpu().detach().numpy()

   ############# 저장할거면 풀기
    # plt.figure(figsize=(10,3.3))
    # plt.subplot(1,3,1)
    
    # plt.imshow(input_img)
    # plt.subplot(1,3,2)
    # plt.imshow(labs,cmap='gray')
    # plt.subplot(1,3,3)
    # plt.imshow((output>=0.5)*1.0,cmap='gray')    
    # plt.suptitle(f'IoU : {np.round(miou_tmp,3)}\nRecall : {np.round(recall_pred_tmp,3)}\nelapsed_time : {np.round(elapsed_time,3)} \nDice score : {np.round(f1_or_dsc_tmp,3)}')
    # plt.tight_layout()
    # plt.savefig(vis_save_path)
    # plt.close()


    return accuracy_tmp, sensitivity_tmp, specificity_tmp, f1_or_dsc_tmp, miou_tmp, recall_pred_tmp, elapsed_time


if __name__ == '__main__':
    

    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test_session = config.test_session
    if config.task_name =="Skin_pore_exp1":
        test_num = 780
        model_type = config.model_name
        model_path = "./Skin_pore_exp1/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name =="Skin_pore6000_exp1":
        test_num = 608
        model_type = config.model_name
        model_path = "./Skin_pore6000_exp1/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name == 'ISIC17_exp1':
        test_num = 2000
        model_type = config.model_name
        model_path = "./ISIC17_exp1/" + model_type + "/" + test_session + "/models/best_model-" + model_type+".pth.tar"
    
    elif config.task_name == 'ISIC18_exp1':
        test_num = 1000
        model_type = config.model_name
        model_path = "./ISIC18_exp1/" + model_type + "/" + test_session + "/models/best_model-" + model_type+".pth.tar"


    save_path  = config.task_name +'/'+ config.model_name +'/' + test_session + '/'
    vis_path = "./" + config.task_name + '_visualize_test/'
    vis_path = save_path + 'visualize_test/'
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    checkpoint = torch.load(model_path, map_location='cuda')

    fp = open(save_path+'test_result.txt','a')
    fp.write(str(datetime.now())+'\n')

    

    if model_type == 'UNet_base':
        model = UNet_base(n_channels=config.n_channels,n_classes=config.n_labels)
    
    elif model_type == 'UNetV2':
        pretrained_path = './nets/unet_v2/pvt_v2_b2.pth'
        model = UNetV2(n_classes=1, deep_supervision=False, pretrained_path=pretrained_path)

    elif model_type == 'TransUnet':
        config_vit = config.get_r50_b16_config()
        config_vit.n_classes = 1
        config_vit.patches.grid = (int(config.img_size /16), int(config.img_size / 16))
        model =  ViT_seg(config_vit, img_size=config.img_size, num_classes=config_vit.n_classes)
        model.load_from(weights=np.load(config_vit.pretrained_path))

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
        # model.load_from()

    elif model_type == 'vmunetv2_1branch':
        ckpt_path = './pre_trained_weights/vmamba_small_s2l15.pth'
        model = VMUNetV2_1branch(load_ckpt_path=ckpt_path, deep_supervision=True)
        model.load_from()

    elif model_type == 'vmunetv2_adapter':
        ckpt_path = './pre_trained_weights/vmamba_tiny_s1l8.pth'
        model = VMUNetV2_adapter(load_ckpt_path=ckpt_path, deep_supervision=True)
        model.load_from()
        
    elif model_type == 'CvmUnet':
        model = ConVmUnet(num_classes=1)

    elif model_type == 'ConvNext_vmunet':
        model = ConvNextVMUNet(num_classes=1)
    else: raise TypeError('Please enter a valid name for the model type')

    model = model.cuda()
    if torch.cuda.device_count() > 1:
        print ("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=[0,1,2,3])
    model.load_state_dict(checkpoint['state_dict'])
    print('Model loaded !')
    tf_test = ValGenerator(output_size=[config.img_size, config.img_size])
    test_dataset = ImageToImage2D(config.test_dataset, tf_test,image_size=config.img_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    accuracy = 0.0
    sensitivity = 0.0
    specificity = 0.0
    iou_pred = 0.0
    recall_pred = 0.0
    dice_pred = 0.0
    elapsed_time = 0.0

    iou_list = []
    dice_list = []
    with tqdm(total=test_num, desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            test_data, test_label = sampled_batch['image'], sampled_batch['label']
            arr=test_data.numpy()
            arr = arr.astype(np.float32())
            lab=test_label.data.numpy()
            input_img = torch.from_numpy(arr)

            # iou_pred_t, recall_t, elapsed_t, dice_pred_t= vis_and_save_heatmap(model, input_img, lab, vis_path+str(i)+'.png')
            accuracy_t, sensitivity_t, specificity_t, f1_or_dsc_t, miou_t, recall_pred_t, elapsed_t = vis_and_save_heatmap(model, input_img, lab, vis_path+str(i)+'.png')
            
            accuracy += accuracy_t
            sensitivity += sensitivity_t
            specificity += specificity_t
            dice_pred += f1_or_dsc_t
            iou_pred+=miou_t
            recall_pred+= recall_pred_t
            elapsed_time += elapsed_t

            iou_list.append(miou_t)
            dice_list.append(f1_or_dsc_t)

            torch.cuda.empty_cache()
            pbar.update()

    print ("iou_pred",iou_pred/test_num)
    print ("recall",recall_pred/test_num)
    print("dice_pred", dice_pred/ test_num)
    print("accuracy", accuracy/ test_num)
    print("sensitivity", sensitivity/ test_num)
    print("specificity", specificity/ test_num)
    print("elapsed_time", elapsed_time/test_num)
    print('iou_std', np.std(iou_list))
    print('dice_std', np.std(dice_list))
    
    fp.write(f"iou_pred : {iou_pred/test_num}\n")
    fp.write(f"recall : {recall_pred/test_num}\n")
    fp.write(f"dice_pred : {dice_pred/test_num}\n")
    fp.write(f"accuracy : {accuracy/test_num}\n")
    fp.write(f"sensitivity : {sensitivity/test_num}\n")
    fp.write(f"specificity : {specificity/test_num}\n")
    fp.write(f"elapsed_time : {elapsed_time/test_num}\n")
    fp.write(f"iou_std : {np.std(iou_list)}\n")
    fp.write(f"dice_std : {np.std(dice_list)}\n")
    
    


    fp.close()

