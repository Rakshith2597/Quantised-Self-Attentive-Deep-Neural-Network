
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm as tq
import json
import random
import wandb
from utils import dice_coefficient
from dataloader import LiverDataLoader
import augmentation as augs
from losses import DiceLoss
from self_attn_sumnet import FullAttenSUMNet, SUMNet
from unet import UNet
import argparse
import torchmetrics.functional as TF

# For Reproducability
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_model(config):

    data_path = config['datapath']
    save_path = config['savepath']

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    data_split = config['data_split']

    with open(data_split) as f3:
        data_split = json.load(f3)
    
    if config['model'] == 'ssa':
        net_seg = FullAttenSUMNet(in_ch=1,out_ch=config['out_ch'])
        net_seg = net_seg.cuda()
        if os.path.exists(save_path):
            # checkpoint = torch.load(save_path+'model_best_dice.pt')
            # net_seg.load_state_dict(checkpoint['state_dict'])
            optimizer_seg = optim.Adam(net_seg.parameters(), lr=config['seg_lr'], weight_decay= config['w_decay'])
            # optimizer_seg.load_state_dict(checkpoint['optimizer'])
    elif config['model'] == 'sumnet':
        net_seg = SUMNet(in_ch=1,out_ch=config['out_ch'])
        net_seg = net_seg.cuda()
        optimizer_seg = optim.Adam(net_seg.parameters(), lr=config['seg_lr'], weight_decay= config['w_decay']) 
    else:
        net_seg = UNet(n_channels=1, n_classes=config['out_ch'])
        net_seg = net_seg.cuda()
        optimizer_seg = optim.Adam(net_seg.parameters(), lr=config['seg_lr'], weight_decay= config['w_decay'])

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print('CUDA Available and using cuda.')
        # net_seg = net_seg.cuda()

    criterion_seg1 = DiceLoss()

    epochs = config['epochs']
    apply_aug = augs.Compose([augs.RandomRotate(10), augs.RandomVerticallyFlip(0.5), augs.RandomHorizontallyFlip(0.5)])

    bestValidDice = 0.0

    for epoch in range(epochs):
  
        print(f'Epoch: {str(epoch)}')
        trainRunningLoss = 0
        validRunningLoss = 0

        trainBatches = 0
        validBatches = 0
        trainDice_r = 0
        validDice_r = 0

        net_seg.train(True)

        train_split = data_split['train']
        valid_split = data_split['valid']
        # train_split = train_split[:int(0.3*len(train_split))]

        train_d_set = LiverDataLoader(datapath=data_path, file_list=train_split, is_transform=True, augmentation=None)

        train_data_loader = data.DataLoader(
                                    train_d_set, batch_size=config['batch_size'],
                                    shuffle=True, num_workers=8, pin_memory=True,
                                    worker_init_fn=seed_worker,drop_last=True)

        valid_d_set = LiverDataLoader(datapath=data_path, file_list=valid_split, is_transform=True)

        valid_data_loader = data.DataLoader(
                                    valid_d_set, batch_size=config['batch_size'],
                                    shuffle=True, num_workers=8, pin_memory=True,
                                    worker_init_fn=seed_worker,drop_last=True)

        for datasample in tq(train_data_loader):
            img, mask, filename = datasample

            if use_gpu:
                inputs = img.cuda()
                mask = mask.cuda()

            seg_out = net_seg(inputs)
            net_out_sf = F.softmax(seg_out.data, dim=1)
            preds = torch.argmax(net_out_sf, dim=1)

            # SEGMENTATION TRAINING

            net_loss = criterion_seg1(seg_out, mask)

            optimizer_seg.zero_grad()
            net_loss.backward()
            optimizer_seg.step()

            trainRunningLoss += net_loss.item()
            # train_dice = dice_coefficient(preds, mask[:,1])
            train_dice = TF.f1_score(preds.flatten().long(), mask[:,1].flatten().long(), num_classes=2, ignore_index=0)

            # trainDice_r += torch.mean(train_dice).item()
            trainDice_r += train_dice.item()

            trainBatches += 1
            # if trainBatches > 1:
            #     break

        net_seg.eval()

        with torch.no_grad():
            for datasample in tq(valid_data_loader):
                imgs, mask, filename = datasample
        
                if use_gpu:
                    inputs = imgs.cuda()
                    mask = mask.cuda()

                seg_out = net_seg(inputs)
                net_out_sf = F.softmax(seg_out.data, dim=1)

                net_loss = criterion_seg1(seg_out, mask)

                pred_max = torch.argmax(net_out_sf, dim=1)

                val_dice = TF.f1_score(pred_max.flatten().long(), mask[:,1].flatten().long(), num_classes=2, ignore_index=0)
                validDice_r += val_dice.item()
                validRunningLoss += net_loss.item()

                validBatches += 1
                # if validBatches > 1:
                #     break
        # scheduler.step(epoch)
        valid_dice= validDice_r/validBatches

        if (valid_dice > bestValidDice):
            bestValidDice = valid_dice
            state = {'epoch': epoch, 'state_dict': net_seg.state_dict(),
             'optimizer': optimizer_seg.state_dict(), 'best_dice':bestValidDice}
            torch.save(state, save_path+'model_best_dice.pt')

        net_train_loss= trainRunningLoss / trainBatches
        net_valid_loss= validRunningLoss / validBatches
        train_dice= trainDice_r/trainBatches
        valid_dice= validDice_r/validBatches

        log_dict = {
            'net_train_loss': net_train_loss,
            'net_valid_loss': net_valid_loss,
            'train_dice' : train_dice,
            'valid_dice': valid_dice,
            'epoch' : epoch
        }

        wandb.log(log_dict)
    state = {'epoch': epoch, 'state_dict': net_seg.state_dict(),
             'optimizer': optimizer_seg.state_dict(), }
    torch.save(state, save_path+'model_last.pt')

def test(config, model):
    data_path = config['datapath']
    load_path = config['savepath']
    data_split = config['data_split']

    with open(data_split) as f3:
        data_split = json.load(f3)
    file_list = data_split['valid']

    testDset = LiverDataLoader(datapath=data_path, file_list=file_list,is_transform=True)

    testDataLoader = data.DataLoader(
                                testDset, batch_size=1, drop_last=True,
                                shuffle=False, num_workers=2, pin_memory=True,
                                worker_init_fn=seed_worker)

    
    testBatches = 0
    testDice_r = 0

    if config['model'] == 'ssa':
        net_seg = FullAttenSUMNet(in_ch=1,out_ch=config['out_ch'])
        q_flag = False
    elif config['model'] == 'sumnet':
        net_seg = SUMNet(in_ch=1,out_ch=config['out_ch']) 
        q_flag = False
    elif config['model'] == 'unet':
        net_seg = UNet(n_channels=1, n_classes=config['out_ch'])
        q_flag = False
    else:
        net_seg = model
        q_flag = True
    use_gpu = torch.cuda.is_available()

    if use_gpu and not q_flag:
        net_seg = net_seg.cuda()
    
    if not q_flag:
        # checkpoint = torch.load(load_path+'model_best_seg.pt')
        # net_seg.load_state_dict(checkpoint)
        # checkpoint = torch.load(load_path+'model_best_dice.pt')
        checkpoint = torch.load(load_path+'model_last.pt')
        net_seg.load_state_dict(checkpoint['state_dict'])
    dice_list = []
    net_seg.eval()

    for datasample in tq(testDataLoader):
        inputs, mask, _ = datasample

        if use_gpu and not q_flag:
            inputs = inputs.cuda()
            mask = mask.cuda()
        
        with torch.no_grad():
            seg_out = net_seg(inputs)
            net_out_sf = F.softmax(seg_out.data, dim=1)

            preds = torch.argmax(net_out_sf, dim=1)
            test_dice = TF.f1_score(preds.flatten().long(), mask[:,1].flatten().long(), num_classes=2, ignore_index=0)
            print(test_dice)
            testDice_r += test_dice.item()

            dice_list.append(test_dice)

            testBatches += 1
        # if testBatches > 1:
        #     break

    testDiceCoeff = testDice_r/testBatches
    print(testDiceCoeff)
    print(np.mean(dice_list))


    print('Dice Coefficient: {:.3f}|'
          .format(testDiceCoeff))

    results_dict = {
        'dice': testDiceCoeff
    }

    with open('/home/rakshith/miccai_2022/model_weights/segmentation/'+config['model']+'.json', 'w') as f1:
        json.dump(results_dict,f1)


def main_new(args):

    model_type = args.model
    model_name = model_type+'_seg_adam_1e4_e15_beautiful'
    config = {
        "savepath":"/home/rakshith/miccai_2022/model_weights/segmentation/"+model_name+"/",
        "datapath": "/home/rakshith/Datasets/Task03_Liver/np_array_slices/",
        "data_split": "/home/rakshith/miccai_2022/segmentation/train_valid_split.json",
        "seg_lr":1e-4,
        "w_decay":1e-5,
        "batch_size":2,
        "epochs":15,
        "out_ch" : 2,
        "model_name":model_name,
        "model" : model_type
    }

    wandb.init(name=model_name, 
           project='miccai_2022',
           config=config
            )

    train_model(config)
    # test(paths, config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select action to be performed')
    parser.add_argument('--model', default=False, help='Select model ')
    args= parser.parse_args()

    main_new(args)
