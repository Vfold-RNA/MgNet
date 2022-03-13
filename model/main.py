import matplotlib
matplotlib.use('Agg')
# from skimage import io, transform
import sys
import matplotlib.pyplot as plt

import datetime

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
from dncon2 import Net
from torch.autograd import Variable
import os

import csv

import torchvision
import torchvision.transforms as transforms

#from utils import progress_bar
from dataset import metalIonDataset
from dataset import NormalizePerImage
from torch.utils.data import Dataset, DataLoader
from dataset import get_mean_and_std

image_resolution = 0.5
num_worker = os.cpu_count()-2
train_batch_size = 32
test_batch_size = 1
validate_batch_size = 1

target_radius = '2.5'

resume = 'ckpt.e0'
channelList = [0,1]
weight_scale = 30.0
test_flag = False
calculateMeanStd = True

cv_index = sys.argv[1]

log_dir = "./log/cv{}/".format(cv_index)
test_result_dir = './test_result/cv{}/'.format(cv_index)

train_data_dir = './image/cv{}/train/'.format(cv_index)
validate_data_dir = './image/cv{}/validate/'.format(cv_index)
test_data_dir = './image/cv{}/validate/'.format(cv_index)

checkpoint_dir = './checkpoint/cv{}/'.format(cv_index)


def write_log(log_folder,file_name,epoch,loss):
    if not os.path.isdir(log_folder):
        os.mkdir(log_folder)
    log_path = log_folder+file_name
    if os.path.exists(log_path) == True and epoch != 1:
        with open(log_path,'a') as log:
            log.write(str(epoch)+','+str(loss)+'\n')
    else:
        with open(log_path,'w') as log:
            log.write(str(epoch)+','+str(loss)+'\n')

train_loss_list = []
validate_loss_list = []

def if_early_stop(validateList,length=40):
    validateList.reverse()
    tail_length = validateList.index(min(validateList))
    validateList.reverse()
    if tail_length >= length:
        print("early stop")
        validateList.clear()
        return True
    else:
        return False

start_epoch = 1
use_cuda = torch.cuda.is_available()
if resume != 'ckpt.e0':
    # Load checkpoint.
    assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
    print('==> Resuming from checkpoint...'+resume)
    checkpoint = torch.load(checkpoint_dir+resume)
    net = checkpoint['net']
    start_epoch = checkpoint['epoch'] + 1
    mean = checkpoint['mean']
    std = checkpoint['std']
else:
    print('==> Building model...')
    net = Net(num_channels=len(channelList))
    preSet = metalIonDataset(root=train_data_dir,channels=channelList,transform=None)
    preloader = DataLoader(preSet, batch_size=1, shuffle=False, num_workers=num_worker)
    if calculateMeanStd:
        print('Calculating Mean and Std...')
        print("Time:", datetime.datetime.now())
        mean, std = get_mean_and_std(dataloader=preloader,channels=channelList)
        print("Time:", datetime.datetime.now())
        print('Saving Mean and Std...')
        MeanStd = {
            'mean': mean,
            'std': std,
        }
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        torch.save(MeanStd, '{}/MeanStd'.format(checkpoint_dir))
    else:
        print('Loading Mean and Std...')
        checkpoint = torch.load('{}/MeanStd'.format(checkpoint_dir))
        mean = checkpoint['mean']
        std = checkpoint['std']

    #mean = np.zeros((1,len(channelList),1,1,1))
    #std = np.ones((1,len(channelList),1,1,1))

transform_train = transforms.Compose([
    NormalizePerImage(mean,std),
    # RandomRotation903D(rotations)
])

transform_validate = transforms.Compose([
    NormalizePerImage(mean,std)
])

transform_test = transforms.Compose([
    NormalizePerImage(mean,std)
])


trainset = metalIonDataset(root=train_data_dir,channels=channelList,transform=transform_train)
trainloader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=
num_worker)
print('trainset--->',len(trainset))
print('trainloader--->',len(trainloader))

validateset = metalIonDataset(root=validate_data_dir,channels=channelList,transform=transform_validate)
validateloader = DataLoader(validateset, batch_size=validate_batch_size, shuffle=False, num_workers=
num_worker)
print('validateset--->',len(validateset))
print('validateloader--->',len(validateloader))

testset = metalIonDataset(root=test_data_dir,channels=channelList,transform=transform_test)
testloader = DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=
num_worker)
print('testset--->',len(testset))
print('testloader--->',len(testloader))



if use_cuda:
    print('GPU--->',range(torch.cuda.device_count()))
    net.cuda()
    print('Use cudnn--->',torch.backends.cudnn.version())
    #range(torch.cuda.device_count())
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True


#optimizer = optim.SGD(net.parameters(), lr=learn_rate, momentum=momentum, weight_decay=weight_decay)
optimizer = optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# Reduce learning rate when a metric has stopped improving
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=100, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[130], gamma=0.1, last_epoch=-1)
# criterion = nn.MSELoss()
criterion = nn.MSELoss(reduction='none')
# criterion = nn.CrossEntropyLoss(weight=loss_weight)
# criterion = nn.KLDivLoss()
#print(net)

params = list(net.parameters())
# print(len(params))
print(net)
# create your optimizer
for param in net.parameters():
    print(type(param.data), param.size())
    
print(optimizer)
print(criterion)
print('image_resolution',image_resolution)
print('num_worker',num_worker)
print('train_batch_size',train_batch_size)
print('test_batch_size',test_batch_size)
print('validate_batch_size',validate_batch_size)

print('channelList',channelList)
print('weight_scale',weight_scale)
print('test_flag',test_flag)
print('calculateMeanStd',calculateMeanStd)

print('log_dir',log_dir)
print('test_result_dir',test_result_dir)

print('train_data_dir',train_data_dir)
print('validate_data_dir',validate_data_dir)
print('test_data_dir',test_data_dir)

def train(epoch):
    net.train()
    train_loss = 0
    for batch_idx, sample_batched in enumerate(trainloader):
        # print(batch_idx, sample_batched['image'].size(), sample_batched['label'].size())
        inputs = sample_batched['image']
        targets = sample_batched['label']
        # masks = 1 - sample_batched['occupied'][:,np.newaxis,:,:,:]
        inputs = inputs.type(torch.FloatTensor)
        targets = targets.type(torch.FloatTensor)
        # masks = masks.type(torch.FloatTensor)

        if use_cuda:
            inputs, targets= inputs.cuda(), targets.cuda()
        # sample = inputs.numpy()
        # print(targets)
        optimizer.zero_grad() # zero the gradient buffers

        #tar = targets.cpu().data.numpy()
        #nonzeroBin = np.nonzero(tar)
        #ratio += len(nonzeroBin[2])/(tar.shape[0]*tar.shape[1]*tar.shape[2]**3)
        weights = weight_scale*targets
        weights += 1.0
        # weights = torch.mul(weights,masks)
        # weights = torch.from_numpy(weights)


        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        # print('outputs--->',outputs.size())
        # print(type(outputs))
        # print('targets--->',targets.size())
        # print(type(targets))
        loss = criterion(outputs, targets)
        # print(type(loss))
        # print(loss.size())
        # print(loss.view(-1).size())
        
        weighted_loss = torch.dot(loss.view(-1),weights.view(-1))/targets.view(-1).size()[0]
        # print(type(weighted_loss))
        # print(weighted_loss.data.item())
        # print(weighted_loss.size())
        train_loss += weighted_loss.data.item()
        # train_loss += loss.data.item()
        # loss.backward()
        weighted_loss.backward()
        optimizer.step()
        # NumBatch = outputs.data.numpy().shape[0]
        # tempCorrcoef = 0.0
        # for batchNum in range(NumBatch):
            # print('out--->',outputs.data.numpy()[batchNum][0].shape)
            # print('tar--->',targets.data.numpy()[batchNum][0].shape)
            # out = outputs.data.numpy()[batchNum][0].flatten()
            # tar = targets.data.numpy()[batchNum][0].flatten()
            # tempCorrcoef += np.corrcoef(out,tar)[0][1]
        
        # corrcoef += tempCorrcoef/float(NumBatch)
        

        # str_progress = 'Epoch: %d | Loss: %.3f | --TRAIN '
        # progress_bar(batch_idx, len(trainloader), str_progress % (epoch, train_loss/(batch_idx+1)))
    print("{}: {:d} | {}: {:.3f} | {}: {} --TRAIN".format("Epoch", epoch, "Loss", train_loss/len(trainloader), "Time", datetime.datetime.now()))
    write_log(log_dir,'train_log.dat',epoch,float(train_loss)/float(len(trainloader)))
    train_loss_list.append(float(train_loss)/float(len(trainloader)))
    # Save checkpoint.
    # print('Regular Saving...')
    state_regular = {
        'net': net.module if use_cuda else net,
        'epoch': epoch,
        'mean': mean,
        'std': std,
    }
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    torch.save(state_regular, checkpoint_dir+'/ckpt.e'+str(epoch))

def validate(epoch):
    net.eval()
    validate_loss = 0
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(validateloader):
            # print(batch_idx, sample_batched['image'].size(), sample_batched['label'].size())
            inputs = sample_batched['image']
            targets = sample_batched['label']
            # masks = 1 - sample_batched['occupied'][:,np.newaxis,:,:,:]
            inputs = inputs.type(torch.FloatTensor)
            targets = targets.type(torch.FloatTensor)
            # masks = masks.type(torch.FloatTensor)

            if use_cuda:
                inputs, targets= inputs.cuda(), targets.cuda()

            #tar = targets.cpu().data.numpy()
            #nonzeroBin = np.nonzero(tar)
            #ratio += len(nonzeroBin[2])/(tar.shape[0]*tar.shape[1]*tar.shape[2]**3)
            weights = weight_scale*targets
            weights += 1.0
            # weights = torch.mul(weights,masks)

            # optimizer.zero_grad() # zero the gradient buffers
            inputs, targets = Variable(inputs, requires_grad=False), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            weighted_loss = torch.dot(loss.view(-1),weights.view(-1))/targets.view(-1).size()[0]
            validate_loss += weighted_loss.data.item()
            
        print("{}: {:d} | {}: {:.3f} | {}: {} --VALIDATE".format("Epoch", epoch, "Loss", validate_loss/len(validateloader), "Time", datetime.datetime.now()))
        write_log(log_dir,'validate_log.dat',epoch,float(validate_loss)/float(len(validateloader)))
        validate_loss_list.append(float(validate_loss)/float(len(validateloader)))

def test(epoch):
    net.eval()
    test_loss = 0
    predicted_dict = {}
    if not os.path.isdir(test_result_dir):
        os.makedirs(test_result_dir)
    if not os.path.isdir(test_result_dir+str(epoch)+'/'):
        os.makedirs(test_result_dir+str(epoch)+'/')
    else:
        os.system('rm -rf '+test_result_dir+str(epoch)+'/*')
    if not os.path.isdir(test_result_dir+str(epoch)+'/raw/'):
        os.makedirs(test_result_dir+str(epoch)+'/raw/')
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(testloader):
            # print(batch_idx, sample_batched['image'].size(), sample_batched['label'].size())
            inputs = sample_batched['image']
            targets = sample_batched['label']
            # masks = 1 - sample_batched['occupied'][:,np.newaxis,:,:,:]
            names = sample_batched['PDB']
            origins = sample_batched['origin']
            inverse_transformation_matrices = sample_batched['inverse_transformation_matrix']

            inputs = inputs.type(torch.FloatTensor)
            targets = targets.type(torch.FloatTensor)
            # masks = masks.type(torch.FloatTensor)
            
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            weights = weight_scale*targets
            weights += 1.0
            # weights = torch.mul(weights,masks)

            # optimizer.zero_grad() # zero the gradient buffers
            inputs, targets = Variable(inputs, requires_grad=False), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            weighted_loss = torch.dot(loss.view(-1),weights.view(-1))/targets.view(-1).size()[0]
            test_loss += weighted_loss.data.item()
            # print('inputs size--->',inputs.size())
            # print('targets size--->',targets.size())
            # print('names len--->',len(names))
            # print('origins size--->',origins.size())
            # print('inverse_transformation_matrices size--->',inverse_transformation_matrices.size())
            # print(sites)
            # outputs = torch.mul(outputs,masks)
            outs = outputs.cpu()
            # print('outs size--->',outs.size())
            # print('outs size--->',outs.size())
            outSize = (outs.size(2),outs.size(3),outs.size(4))
            for i in range(len(names)):
                PDBName = names[i]
                out = outs[i].view(outSize)
                # print('out size--->',out.size())
                indices = np.array(np.nonzero(out.data.numpy())).transpose()
                take_indices = [index[0]*outSize[1]*outSize[2]+index[1]*outSize[2]+index[2] for index in indices]
                values = np.take(out.data.numpy(),take_indices)
                
                coords = (indices-float(outs.size(2)-1)/2.0)*image_resolution
                transformed_coords = np.dot(coords,inverse_transformation_matrices[i])
                # print(type(transformed_coords),type(origins[i]))
                transformed_coords = transformed_coords + origins[i].data.numpy()
                indices = (transformed_coords/image_resolution).astype(int)
             
                #print(len(indices_x),len(indices_y),len(indices_z),len(values))
                if PDBName not in predicted_dict:
                    predicted_dict[PDBName] = [indices,values]
                else:
                    predicted_dict[PDBName][0]=np.concatenate((predicted_dict[PDBName][0], indices), axis=0)
                    predicted_dict[PDBName][1]=np.concatenate((predicted_dict[PDBName][1], values), axis=0)

            #str_progress = 'Epoch %d | Loss: %.3f | Corr: %.3f | ratio: %.3f | --TEST '
            #progress_bar(batch_idx, len(testloader), str_progress
            #% (epoch, test_loss/(batch_idx+1), corrcoef/(batch_idx+1), ratio/(batch_idx+1)))
        
        for k,v in predicted_dict.items():
            with open(test_result_dir+str(epoch)+'/raw/'+str(k)+'_prediction_'+str(epoch)+'.csv', "w") as preFile:
                writer = csv.writer(preFile, delimiter=' ')
                indices = v[0].transpose()
                rows = zip(indices[0],indices[1],indices[2],v[1])
                for row in rows:
                    writer.writerow(row)
        print("{}: {:d} | {}: {:.3f} | {}: {} --TEST".format("Epoch", epoch, "Loss", test_loss/len(testloader), "Time", datetime.datetime.now()))
        write_log(log_dir,'test_log.dat',epoch,float(test_loss)/float(len(testloader)))


if test_flag:
    test(start_epoch-1)
else:
    for epoch in range(start_epoch, 100):
        
        train(epoch)
        validate(epoch)
        if epoch % 20 == 0:
            test(epoch)
        #scheduler.step(validate_loss_list[-1])
        scheduler.step()

        #if if_early_stop(validate_loss_list,length=40):
        #    print("finish fitting, early stop!")
        #    exit(2)
        print('------------------------------------------------------------------------------------------------------------------------')
