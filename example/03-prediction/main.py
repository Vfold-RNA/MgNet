# import matplotlib
# matplotlib.use('Agg')
# from skimage import io, transform
import sys
# import matplotlib.pyplot as plt

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
#train_batch_size = 32
test_batch_size = 1
#validate_batch_size = 1

target_radius = '2.5'

cv_index = sys.argv[1]
image_dir = sys.argv[2]
resume = 'ckpt.e40'
channelList = [0,1]
weight_scale = 30.0
test_flag = True
#calculateMeanStd = True


log_dir = "./log/cv{}/".format(cv_index)
test_result_dir = './test_result/cv{}/'.format(cv_index)
test_data_dir = image_dir
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

test_loss_list = []

use_cuda = torch.cuda.is_available()

# Load checkpoint.
assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
print('==> Resuming from checkpoint...'+resume)
checkpoint = torch.load(checkpoint_dir+resume)
net = checkpoint['net']
start_epoch = checkpoint['epoch']
mean = checkpoint['mean']
std = checkpoint['std']

transform_test = transforms.Compose([
    NormalizePerImage(mean,std)
])


testset = metalIonDataset(root=test_data_dir,channels=channelList,transform=transform_test)
testloader = DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=num_worker)
print('testset--->',len(testset))
print('testloader--->',len(testloader))



if use_cuda:
    print('GPU--->',range(torch.cuda.device_count()))
    net.cuda()
    print('Use cudnn--->',torch.backends.cudnn.version())
    #range(torch.cuda.device_count())
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True


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
    
print(criterion)
print('image_resolution',image_resolution)
print('num_worker',num_worker)
print('test_batch_size',test_batch_size)
print('channelList',channelList)
print('weight_scale',weight_scale)
print('test_flag',test_flag)
print('log_dir',log_dir)
print('test_result_dir',test_result_dir)
print('test_data_dir',test_data_dir)


def test(epoch):
    net.eval()
    test_loss = 0
    predicted_dict = {}
    if not os.path.isdir(test_result_dir):
        os.makedirs(test_result_dir)
    if not os.path.isdir(test_result_dir+str(epoch)+'/'):
        os.makedirs(test_result_dir+str(epoch)+'/')
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
    test(start_epoch)
    print('------------------------------------------------------------------------------------------------------------------------')
