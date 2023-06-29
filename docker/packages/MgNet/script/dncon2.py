import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# class myAvergePool(nn.Module):
#     def __init__(self):
#         super(myAvergePool, self).__init__()
#     def forward(self, x):
#         # print(x.size()[0],x.size()[1],x.size()[2],x.size()[3])
#         # averageSum = torch.sum(x,dim=3)/x.size()[3]
#         averageSum = torch.diagonal(x,dim1=2,dim2=3)
#         # print(averageSum.size())
#         return averageSum

class Block(nn.Module):
    def __init__(self, Channels, kernelSize, stride, padding):
        super(Block, self).__init__()
        self.conv1 = nn.Conv3d(Channels, Channels, kernelSize, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm3d(Channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(Channels, Channels, kernelSize, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm3d(Channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # if self.downsample is not None:
            # residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class Net(nn.Module):

    def __init__(self, num_channels):
        super(Net, self).__init__()

        num_filter = [16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,1]
        filter_size = [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,1]
        strides = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

        paddings = [(f-1)//2 for f in filter_size]

        self.relu = nn.ReLU(inplace=True)

        self.conv_first = nn.Conv3d(num_channels, num_filter[0], filter_size[0], stride = strides[0], padding = paddings[0])
        self.bn_first = nn.BatchNorm3d(num_filter[0])

        self.layer_list = []
        for i in range(1,len(num_filter)-1):
            self.layer_list.append(Block(num_filter[i], filter_size[i], stride = strides[i], padding = paddings[i]))
            #self.layer_list.append(nn.Conv3d(num_filter[i], num_filter[i+1], filter_size[i+1], stride = strides[i+1], padding = paddings[i+1]))
            #self.layer_list.append(nn.BatchNorm3d(num_filter[i+1]))
            #self.layer_list.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*self.layer_list)

        self.conv_last = nn.Conv3d(num_filter[-2], num_filter[-1], filter_size[-1], stride = strides[-1], padding = paddings[-1])
        self.bn_last = nn.BatchNorm3d(num_filter[-1])

        self.sigmoid = nn.Sigmoid()

        # self.myAvergePool = myAvergePool()

        # self.fc = nn.Linear(num_classes * num_classes, num_classes)

        #for m in self.modules():
        #    if isinstance(m, nn.Conv3d):
        #        n = m.kernel_size[0] * m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #        m.weight.data.normal_(0, math.sqrt(2. / n))
        #    elif isinstance(m, nn.BatchNorm3d):
        #        m.weight.data.fill_(1)
        #        m.bias.data.zero_()
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        # print("size x0 ---> ", x.size())
        out = self.conv_first(x)
        # print("conv_first outsize --> ",out.size())
        out = self.bn_first(out)
        out = self.relu(out)
        # print("relu outsize --> ",out.size())
        # print("self.layers len:",len(self.layers))
        for layer in self.layers:
            out = layer(out)
            # print("block outsize --> ",out.size())
        out = self.conv_last(out)
        out = self.bn_last(out)
        out = self.sigmoid(out)
        # out = self.myAvergePool(out)
        # exit()
        # print('after sigmoid--->',out)
        # out = out.view(out.size(2),out.size(3),out.size(4))
        # print('after view--->',out)
        # exit()
        # print('out view--->',out.size())
        # print(out.size(1)**(0.5))
        # pool = nn.AvgPool1d(kernel_size=int(out.size(1)**(0.5)))
        # out = pool(out)
        # out = self.fc(out)

        return out
