import torch
from torch import nn
from torch import optim
from torch.utils import data
from torch.nn import functional as F
import numpy as np
import os
from tqdm import tqdm as tq
import json
import random
from dataloader import ChestDataLoader
from model import AttNet
from torchvision import models
from torchmetrics import AUROC, F1Score
import wandb


# For Reproducability
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def init_weights_he(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight)
    if type(m) == nn.Conv2d:
        nn.init.kaiming_uniform_(m.weight)
def init_weights_xav(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight) 


def train(config):
    img_path = config['imgpath']
    save_path = config['savepath']

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # Details of these json files loaded below are provided in the readme.txt file inside json_files folder.
    with open('cxr8_labels.json') as f1:
        label_json_file = json.load(f1)

    with open('data_split.json') as f2:
        data_split_file = json.load(f2)
    
    with open('problem_files.json') as f3:
        problem_images = json.load(f3)

    with open('no_finding.json') as f4:
        no_findings = json.load(f4)
    
    with open('new_no_findings.json') as f4:
        new_no_findings = json.load(f4)

    train_list = [file for file in data_split_file['train'] if file not in problem_images]
    train_list = [file for file in train_list if file not in no_findings]
    train_list = [file for file in train_list if file not in new_no_findings]
    val_list = [file for file in data_split_file['val'] if file not in problem_images]
    val_list = [file for file in val_list if file not in no_findings]
    val_list = [file for file in val_list if file not in new_no_findings]


    trainDset = ChestDataLoader(img_path=img_path,
                                label_json = label_json_file,
                                name_list=train_list,
                                is_transform=True)
    valDset = ChestDataLoader(img_path=img_path,
                                label_json = label_json_file,
                                name_list=val_list,
                                is_transform=True)
    trainDataLoader = data.DataLoader(
                                trainDset, batch_size=config['bs'], shuffle=True,
                                num_workers=10, pin_memory=True,
                                worker_init_fn=seed_worker)
    validDataLoader = data.DataLoader(
                                valDset, batch_size=config['bs'], shuffle=False,
                                num_workers=10, pin_memory=True,
                                worker_init_fn=seed_worker)

    if config['model'] == 'ssa':
        net = AttNet(in_channels=1, num_classes=14)
    elif config['model'] == 'resnet18':
        net = models.resnet18(pretrained=False)
        net.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, 14)
    else:
        net = models.resnet50(pretrained=False)
        net.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, 14)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        net = net.cuda()
    if config['init']=='he':
        net = net.apply(init_weights_he)
    elif config['init']=='xav':
        net = net.apply(init_weights_xav)
    else:
        pass
 
    optimizer = optim.Adam(net.parameters(), lr=config['lr'], weight_decay=config['w_d'])
    # Class weights calculated based on the distribution of classes.
    class_weights = torch.tensor([0.3016,  1.5003,  0.2904,  0.1968,  0.6161,  0.5397,  2.8818,  0.9920,
         0.8678,  1.7523,  1.7507,  1.8109,  1.1399, 17.8827]).cuda()

    if config['loss'] == 'MLSM':
        criterion = nn.MultiLabelSoftMarginLoss(weight=class_weights)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    epochs = config['epoch']
    best_acc = 0
    best_auroc = 0
    best_fscore = 0
    auroc_score = AUROC(pos_label=1).cuda()
    f1_score = F1Score().cuda()

    for epoch in range(epochs):
        print(f'Epoch: {epoch}')
        # tracker.epoch_start()
        train_running_loss = 0
        valid_running_loss = 0
        train_batches = 0
        valid_batches = 0

        train_A1_correct = 0
        train_A2_correct = 0
        train_A3_correct = 0
        train_A4_correct = 0
        train_A5_correct = 0
        train_A6_correct = 0
        train_A7_correct = 0
        train_A8_correct = 0
        train_A9_correct = 0
        train_A10_correct = 0
        train_A11_correct = 0
        train_A12_correct = 0
        train_A13_correct = 0
        train_A14_correct = 0
        train_mean_correct = 0
        train_auroc_mean_r = 0
        valid_auroc_mean_r = 0
        valid_A1_correct = 0
        valid_A2_correct = 0
        valid_A3_correct = 0
        valid_A4_correct = 0
        valid_A5_correct = 0
        valid_A6_correct = 0
        valid_A7_correct = 0
        valid_A8_correct = 0
        valid_A9_correct = 0
        valid_A10_correct = 0
        valid_A11_correct = 0
        valid_A12_correct = 0
        valid_A13_correct = 0
        valid_A14_correct = 0
        valid_mean_correct = 0
        train_fscore_mean_r = 0
        valid_fscore_mean_r = 0

        net.train(True)

        for data_sample in tq(trainDataLoader):
            img, label = data_sample
            if use_gpu:
                img = img.cuda()
                label = label.cuda()
            
            out = net(img)
            net_out = F.sigmoid(out)
            net_out[net_out > 0.5] = 1
            net_out[net_out <= 0.5] = 0

            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()

            # ATTRIBUTE 1

            A1_out = net_out[:, 0]
            A1_in = label[:, 0]
            train_A1_correct += (A1_out==A1_in).sum()

            # ATTRIBUTE 2

            A2_out = net_out[:, 1]
            A2_in = label[:, 1]
            train_A2_correct += (A2_out==A2_in).sum()

            # ATTRIBUTE 3

            A3_out = net_out[:, 2]
            A3_in = label[:, 2]
            train_A3_correct += (A3_out==A3_in).sum()

            # ATTRIBUTE 4

            A4_out = net_out[:, 3]
            A4_in = label[:, 3]
            train_A4_correct += (A4_out==A4_in).sum()

            # ATTRIBUTE 5

            A5_out = net_out[:, 4]
            A5_in = label[:, 4]
            train_A5_correct += (A5_out==A5_in).sum()

            # ATTRIBUTE 6

            A6_out = net_out[:, 5]
            A6_in = label[:, 5]
            train_A6_correct += (A6_out==A6_in).sum()

            # ATTRIBUTE 7

            A7_out = net_out[:, 6]
            A7_in = label[:, 6]
            train_A7_correct += (A7_out==A7_in).sum()

            # ATTRIBUTE 8

            A8_out = net_out[:, 7]
            A8_in = label[:, 7]
            train_A8_correct += (A8_out==A8_in).sum()

            # ATTRIBUTE 9

            A9_out = net_out[:, 8]
            A9_in = label[:, 8]
            train_A9_correct += (A9_out==A9_in).sum()

            # ATTRIBUTE 10

            A10_out = net_out[:, 9]
            A10_in = label[:, 9]
            train_A10_correct += (A10_out==A10_in).sum()

            # ATTRIBUTE 11

            A11_out = net_out[:, 10]
            A11_in = label[:, 10]
            train_A11_correct += (A11_out==A11_in).sum()

            # ATTRIBUTE 12

            A12_out = net_out[:, 11]
            A12_in = label[:, 11]
            train_A12_correct += (A12_out==A12_in).sum()

            # ATTRIBUTE 13

            A13_out = net_out[:, 12]
            A13_in = label[:, 12]
            train_A13_correct += (A13_out==A13_in).sum()

            # ATTRIBUTE 14

            A14_out = net_out[:, 13]
            A14_in = label[:, 13]
            train_A14_correct += (A14_out==A14_in).sum()

            # ATTRIBUTE MEAN

            A_out = net_out
            A_in = label
            train_mean_correct += (A_out==A_in).sum()
            train_batches += 1

            train_auroc_mean_r += auroc_score(net_out.flatten(),label.long().flatten())
            train_fscore_mean_r += f1_score(net_out.flatten(),label.long().flatten())
            # if train_batches > 1:
            #     break

        img_count = train_batches*config['bs']
        train_A1_accuracy = float(train_A1_correct.item())/img_count
        train_A2_accuracy = float(train_A2_correct.item())/img_count
        train_A3_accuracy = float(train_A3_correct.item())/img_count
        train_A4_accuracy = float(train_A4_correct.item())/img_count
        train_A5_accuracy = float(train_A5_correct.item())/img_count
        train_A6_accuracy = float(train_A6_correct.item())/img_count
        train_A7_accuracy = float(train_A7_correct.item())/img_count
        train_A8_accuracy = float(train_A8_correct.item())/img_count
        train_A9_accuracy = float(train_A9_correct.item())/img_count
        train_A10_accuracy = float(train_A10_correct.item())/img_count
        train_A11_accuracy = float(train_A11_correct.item())/img_count
        train_A12_accuracy = float(train_A12_correct.item())/img_count
        train_A13_accuracy = float(train_A13_correct.item())/img_count
        train_A14_accuracy = float(train_A14_correct.item())/img_count
        train_A_mean_accuracy = float(train_mean_correct.item())/(img_count*15)

        train_loss_mean = train_running_loss/train_batches
        train_auroc_mean = train_auroc_mean_r/train_batches
        train_fscore_mean = train_fscore_mean_r/train_batches

        print(f'Epoch:{epoch}/{epochs} Accuracy:{train_A_mean_accuracy} AUROC:{train_auroc_mean} Fscore:{train_fscore_mean}')
    
        net.eval()
        with torch.no_grad():
            for datasample in tq(validDataLoader):
                img, label = datasample

                if use_gpu:
                    img = img.cuda()
                    label = label.cuda()

                out = net(img)
                net_out = F.sigmoid(out)
                net_out[net_out > 0.5] = 1
                net_out[net_out <= 0.5] = 0
                loss = criterion(out, label)

                valid_running_loss += loss.item()

                # ATTRIBUTE 1

                A1_out = net_out[:, 0]
                A1_in = label[:, 0]
                valid_A1_correct += (A1_out==A1_in).sum()

                # ATTRIBUTE 2

                A2_out = net_out[:, 1]
                A2_in = label[:, 1]
                valid_A2_correct += (A2_out==A2_in).sum()

                # ATTRIBUTE 3

                A3_out = net_out[:, 2]
                A3_in = label[:, 2]
                valid_A3_correct += (A3_out==A3_in).sum()

                # ATTRIBUTE 4

                A4_out = net_out[:, 3]
                A4_in = label[:, 3]
                valid_A4_correct += (A4_out==A4_in).sum()

                # ATTRIBUTE 5

                A5_out = net_out[:, 4]
                A5_in = label[:, 4]
                valid_A5_correct += (A5_out==A5_in).sum()

                # ATTRIBUTE 6

                A6_out = net_out[:, 5]
                A6_in = label[:, 5]
                valid_A6_correct += (A6_out==A6_in).sum()

                # ATTRIBUTE 7

                A7_out = net_out[:, 6]
                A7_in = label[:, 6]
                valid_A7_correct += (A7_out==A7_in).sum()

                # ATTRIBUTE 8

                A8_out = net_out[:, 7]
                A8_in = label[:, 7]
                valid_A8_correct += (A8_out==A8_in).sum()

                # ATTRIBUTE 9

                A9_out = net_out[:, 8]
                A9_in = label[:, 8]
                valid_A9_correct += (A9_out==A9_in).sum()

                # ATTRIBUTE 10

                A10_out = net_out[:, 9]
                A10_in = label[:, 9]
                valid_A10_correct += (A10_out==A10_in).sum()

                # ATTRIBUTE 11

                A11_out = net_out[:, 10]
                A11_in = label[:, 10]
                valid_A11_correct += (A11_out==A11_in).sum()

                # ATTRIBUTE 12

                A12_out = net_out[:, 11]
                A12_in = label[:, 11]
                valid_A12_correct += (A12_out==A12_in).sum()

                # ATTRIBUTE 13

                A13_out = net_out[:, 12]
                A13_in = label[:, 12]
                valid_A13_correct += (A13_out==A13_in).sum()

                # ATTRIBUTE 14

                A14_out = net_out[:, 13]
                A14_in = label[:, 13]
                valid_A14_correct += (A14_out==A14_in).sum()

                # ATTRIBUTE MEAN

                A_out = net_out
                A_in = label
                valid_mean_correct += (A_out==A_in).sum()

                valid_batches += 1
                valid_auroc_mean_r += auroc_score(net_out.flatten(),label.long().flatten())
                valid_fscore_mean_r += f1_score(net_out.flatten(),label.long().flatten())
                # if valid_batches > 1:
                #     break

            img_count = valid_batches*config['bs']
            
            valid_A1_accuracy = float(valid_A1_correct.item())/img_count
            valid_A2_accuracy = float(valid_A2_correct.item())/img_count
            valid_A3_accuracy = float(valid_A3_correct.item())/img_count
            valid_A4_accuracy = float(valid_A4_correct.item())/img_count
            valid_A5_accuracy = float(valid_A5_correct.item())/img_count
            valid_A6_accuracy = float(valid_A6_correct.item())/img_count
            valid_A7_accuracy = float(valid_A7_correct.item())/img_count
            valid_A8_accuracy = float(valid_A8_correct.item())/img_count
            valid_A9_accuracy = float(valid_A9_correct.item())/img_count
            valid_A10_accuracy = float(valid_A10_correct.item())/img_count
            valid_A11_accuracy = float(valid_A11_correct.item())/img_count
            valid_A12_accuracy = float(valid_A12_correct.item())/img_count
            valid_A13_accuracy = float(valid_A13_correct.item())/img_count
            valid_A14_accuracy = float(valid_A14_correct.item())/img_count
            valid_A_mean_accuracy = float(valid_mean_correct.item())/(img_count*15)


            valid_loss_mean = valid_running_loss/valid_batches
            valid_auroc_mean = valid_auroc_mean_r/valid_batches
            valid_fscore_mean = valid_fscore_mean_r/train_batches

            print(f'Epoch:{epoch}/{epochs} Accuracy:{valid_A_mean_accuracy} AUROC:{valid_auroc_mean} Fscore:{valid_fscore_mean}')

            if valid_A_mean_accuracy > best_acc:
                best_acc = valid_A_mean_accuracy
                state = {'epoch': epoch, 'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(), 'best_acc':best_acc}
                torch.save(state, save_path+'model_best_acc_e'+str(epoch)+'_'+str(best_acc)+'.pt')
                # torch.save(net.state_dict(), save_path+'model_best_acc.pt')
            if valid_auroc_mean > best_auroc:
                best_auroc = valid_auroc_mean
                state = {'epoch': epoch, 'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(), 'best_auroc':best_auroc}
                torch.save(state, save_path+'model_best_auroc_e'+str(epoch)+'_'+str(best_auroc)+'.pt')
            if valid_fscore_mean > best_fscore:
                best_fscore = valid_fscore_mean
                state = {'epoch': epoch, 'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(), 'best_fscore':best_fscore}
                torch.save(state, save_path+'model_best_fscore_e'+str(epoch)+'_'+str(best_fscore)+'.pt')
        # tracker.epoch_end()
        state = {'epoch': epoch, 'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(), 'acc':valid_A_mean_accuracy}
        torch.save(state, save_path+'model_epoch'+str(epoch)+'.pt')

        log_dict_acc_train = {
            'acc_train_a1': train_A1_accuracy,
            'acc_train_a2': train_A2_accuracy,
            'acc_train_a3': train_A3_accuracy,
            'acc_train_a4': train_A4_accuracy,
            'acc_train_a5': train_A5_accuracy,
            'acc_train_a6': train_A6_accuracy,
            'acc_train_a7': train_A7_accuracy,
            'acc_train_a8': train_A8_accuracy,
            'acc_train_a9': train_A9_accuracy,
            'acc_train_a10': train_A10_accuracy,
            'acc_train_a11': train_A11_accuracy,
            'acc_train_a12': train_A12_accuracy,
            'acc_train_a13': train_A13_accuracy,
            'acc_train_a14': train_A14_accuracy,
            'acc_valid_a1': valid_A1_accuracy,
            'acc_valid_a2': valid_A2_accuracy,
            'acc_valid_a3': valid_A3_accuracy,
            'acc_valid_a4': valid_A4_accuracy,
            'acc_valid_a5': valid_A5_accuracy,
            'acc_valid_a6': valid_A6_accuracy,
            'acc_valid_a7': valid_A7_accuracy,
            'acc_valid_a8': valid_A8_accuracy,
            'acc_valid_a9': valid_A9_accuracy,
            'acc_valid_a10': valid_A10_accuracy,
            'acc_valid_a11': valid_A11_accuracy,
            'acc_valid_a12': valid_A12_accuracy,
            'acc_valid_a13': valid_A13_accuracy,
            'acc_valid_a14': valid_A14_accuracy,
            'net_train_loss': train_loss_mean,
            'net_valid_loss': valid_loss_mean,
            'train_mean_acc' : train_A_mean_accuracy,
            'valid_mean_acc': valid_A_mean_accuracy,
            'train_mean_auroc' : train_auroc_mean,
            'valid_mean_auroc': valid_auroc_mean,
            'train_mean_fscore' : train_fscore_mean,
            'valid_mean_fscore': valid_fscore_mean,
            'epoch':epoch
        }
        with open(save_path+f'results_{epoch}.json', 'w') as f1:
                json.dump(acc_dict, f1)

        wandb.log(log_dict_acc_train)


def test(config):
    img_path = config['imgpath']
    save_path = config['savepath']

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    with open('cxr8_labels.json') as f1:
        label_json_file = json.load(f1)

    with open('data_split.json') as f2:
        data_split_file = json.load(f2)

    with open('problem_files.json') as f3:
        problem_images = json.load(f3)
    
    with open('no_finding.json') as f4:
        no_findings = json.load(f4)
    
    with open('new_no_findings.json') as f5:
        new_no_findings = json.load(f5)

    test_list = [file for file in data_split_file['test'] if file not in problem_images]
    test_list = [file for file in test_list if file not in no_findings]
    test_list = [file for file in test_list if file not in new_no_findings]

    testDset = ChestDataLoader(img_path=img_path,
                                label_json = label_json_file,
                                name_list=test_list,
                                is_transform=True)

    testDataLoader = data.DataLoader(
                                testDset, batch_size=1, shuffle=False,
                                num_workers=10, pin_memory=True,
                                worker_init_fn=seed_worker)
    
    if config['model'] == 'ssa':
        net = AttNet(in_channels=1, num_classes=15)
        q_flag = False
    elif config['model'] == 'resnet18':
        net = models.resnet18(pretrained=False)
        net.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, 15)
        q_flag = False
    elif config['model'] == 'resnet50':
        net = models.resnet50(pretrained=False)
        net.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, 15)
        q_flag = False
    else:
        net = model
        q_flag = True

    use_gpu = torch.cuda.is_available()
    if use_gpu and not q_flag:
        net = net.cuda()

    
    test_A1_correct = 0
    test_A2_correct = 0
    test_A3_correct = 0
    test_A4_correct = 0
    test_A5_correct = 0
    test_A6_correct = 0
    test_A7_correct = 0
    test_A8_correct = 0
    test_A9_correct = 0
    test_A10_correct = 0
    test_A11_correct = 0
    test_A12_correct = 0
    test_A13_correct = 0
    test_A14_correct = 0
    test_A15_correct = 0
    test_mean_correct = 0
    test_batches = 0


    net.eval()
    with torch.no_grad():
        for datasample in tq(testDataLoader):
            img, label = datasample

            if use_gpu and not q_flag:
                img = img.cuda()
                label = label.cuda()

            out = net(img)
            # net_out = torch.argmax(out, dim=0)
            net_out = F.sigmoid(out)
            net_out[net_out > 0.5] = 1
            net_out[net_out <= 0.5] = 0

            # ATTRIBUTE 1

            A1_out = net_out[:, 0]
            A1_in = label[:, 0]
            test_A1_correct += (A1_out==A1_in).sum()

            # ATTRIBUTE 2

            A2_out = net_out[:, 1]
            A2_in = label[:, 1]
            test_A2_correct += (A2_out==A2_in).sum()

            # ATTRIBUTE 3

            A3_out = net_out[:, 2]
            A3_in = label[:, 2]
            test_A3_correct += (A3_out==A3_in).sum()

            # ATTRIBUTE 4

            A4_out = net_out[:, 3]
            A4_in = label[:, 3]
            test_A4_correct += (A4_out==A4_in).sum()

            # ATTRIBUTE 5

            A5_out = net_out[:, 4]
            A5_in = label[:, 4]
            test_A5_correct += (A5_out==A5_in).sum()

            # ATTRIBUTE 6

            A6_out = net_out[:, 5]
            A6_in = label[:, 5]
            test_A6_correct += (A6_out==A6_in).sum()

            # ATTRIBUTE 7

            A7_out = net_out[:, 6]
            A7_in = label[:, 6]
            test_A7_correct += (A7_out==A7_in).sum()

            # ATTRIBUTE 8

            A8_out = net_out[:, 7]
            A8_in = label[:, 7]
            test_A8_correct += (A8_out==A8_in).sum()

            # ATTRIBUTE 9

            A9_out = net_out[:, 8]
            A9_in = label[:, 8]
            test_A9_correct += (A9_out==A9_in).sum()

            # ATTRIBUTE 10

            A10_out = net_out[:, 9]
            A10_in = label[:, 9]
            test_A10_correct += (A10_out==A10_in).sum()

            # ATTRIBUTE 11

            A11_out = net_out[:, 10]
            A11_in = label[:, 10]
            test_A11_correct += (A11_out==A11_in).sum()

            # ATTRIBUTE 12

            A12_out = net_out[:, 11]
            A12_in = label[:, 11]
            test_A12_correct += (A12_out==A12_in).sum()

            # ATTRIBUTE 13

            A13_out = net_out[:, 12]
            A13_in = label[:, 12]
            test_A13_correct += (A13_out==A13_in).sum()

            # ATTRIBUTE 14

            A14_out = net_out[:, 13]
            A14_in = label[:, 13]
            test_A14_correct += (A14_out==A14_in).sum()

            # ATTRIBUTE MEAN
            A_out = net_out
            A_in = label
            test_mean_correct += (A_out==A_in).sum()

            test_batches += 1
            # if test_batches > 1:
            #     break
        
        img_count = test_batches
        
        test_A1_accuracy = float(test_A1_correct.item())/img_count
        test_A2_accuracy = float(test_A2_correct.item())/img_count
        test_A3_accuracy = float(test_A3_correct.item())/img_count
        test_A4_accuracy = float(test_A4_correct.item())/img_count
        test_A5_accuracy = float(test_A5_correct.item())/img_count
        test_A6_accuracy = float(test_A6_correct.item())/img_count
        test_A7_accuracy = float(test_A7_correct.item())/img_count
        test_A8_accuracy = float(test_A8_correct.item())/img_count
        test_A9_accuracy = float(test_A9_correct.item())/img_count
        test_A10_accuracy = float(test_A10_correct.item())/img_count
        test_A11_accuracy = float(test_A11_correct.item())/img_count
        test_A12_accuracy = float(test_A12_correct.item())/img_count
        test_A13_accuracy = float(test_A13_correct.item())/img_count
        test_A14_accuracy = float(test_A14_correct.item())/img_count
        test_A_mean_accuracy = float(test_mean_correct.item())/(img_count*14)
        
        acc_dict = {
                    1 : test_A1_accuracy,
                    2 : test_A2_accuracy,
                    3 : test_A3_accuracy,
                    4 : test_A4_accuracy,
                    5 : test_A5_accuracy,
                    6 : test_A6_accuracy,
                    7 : test_A7_accuracy,
                    8 : test_A8_accuracy,
                    9 : test_A9_accuracy,
                    10 : test_A10_accuracy,
                    11 : test_A11_accuracy,
                    12 : test_A12_accuracy,
                    13 : test_A13_accuracy,
                    14 : test_A14_accuracy,
                    15 : test_A_mean_accuracy
                    }
        if q_flag:
            with open(save_path+'results_quantized.json', 'w') as f1:
                json.dump(acc_dict, f1)
        else:
            with open(save_path+'results.json', 'w') as f1:
                json.dump(acc_dict, f1)

def main(config):
    wandb.init(name = model_name, 
           project = 'miccai_2022_workshop'
            )
    train(config)
    test(config)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Select action to be performed')
    parser.add_argument('--loss', default=False, help='Select loss')
    parser.add_argument('--model', default=False, help='Select model')
    parser.add_argument('--init', default=None, help='Select Intialisation')
    parser.add_argument('--epoch', type=int, default=15, help='Set number of epochs')
    args = parser.parse_args()

    if args.loss == 'mlsm':
        model_type = args.model
        model_name = model_type+'_WO_nofinding_adam_3e4_e15_mlsm_e'+str(args.epoch)+'_init_'+str(args.init)
        if model_type == 'resnet18':
            batch_size = 64
        elif model_type == 'resnet50':
            batch_size = 32
        else:
            batch_size = 8
        config = {
            "imgpath" : "/home/rakshith/Datasets/CXR8/images/images/",
            "savepath":"/home/rakshith/miccai_2022/model_weights/classification/"+model_name+"/",
            "lr":1e-4,
            "w_d":1e-5,
            "bs":batch_size,
            "epoch":args.epoch,
            "loss":"MLSM",
            "model_name":model_name,
            "init": args.init,
            "model" : model_type
        }
    else:
        model_type = args.model
        if model_type == 'resnet18':
            batch_size = 64
        elif model_type == 'resnet50':
            batch_size = 32
        else:
            batch_size = 8
        model_name = model_type+'_WO_nofinding_adam_3e4_bce_e'+str(args.epoch)+'_init_'+str(args.init)
        config = {
                "imgpath" : "/home/rakshith/Datasets/CXR8/images/images/",
                "savepath":"/home/rakshith/miccai_2022/model_weights/classification/"+model_name+"/",
                "lr":3e-4,
                "w_d":1e-5,
                "bs":batch_size,
                "epoch":args.epoch,
                "init": args.init,
                "loss":"bce",
                "model_name":model_name,
                "model" : model_type
            }

    main(config)
