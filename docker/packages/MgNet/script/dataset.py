import sys
import os
import os.path
import numpy as np
import torch.utils.data as data
#from utils import progress_bar

IMG_EXTENSIONS = ['.npy']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataset_list(dir):
    images = []
    for root, _, fnames in sorted(os.walk(dir,followlinks=True)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


def get_mean_and_std(dataloader,channels):
    '''Compute the mean and std value of dataset.'''
    print('==> Computing mean...')
    mean = np.zeros((1,len(channels),1,1,1))
    #std = np.zeros((1,len(channels),1,1,1))
    for batch_idx, sample_batched in enumerate(dataloader):
        #print(sample_batched['PDB'])
        #print(sample_batched['image'].shape)
        # print(mean.shape)
        batch_image = sample_batched['image'].numpy()
        #print(batch_image.shape)
        # print(np.sum(sample_batched['image'].numpy(), axis=(2,3), keepdims=True).shape)
        mean += np.mean(batch_image, axis=(0,2,3,4), keepdims=True)
        #std += np.std(batch_image, axis=(0,2,3,4), keepdims=True)
        #progress_bar(batch_idx, len(dataloader))
    mean=np.divide(mean,len(dataloader))
    #std=np.divide(std,len(dataloader))
    #print(mean)
    # print(mean.shape)
    print('mean contains nan: ',np.isnan(mean).any())
    # print('==> Computing std...')
    # std = np.zeros((1,len(channels),1,1,1))
    # for batch_idx, sample_batched in enumerate(dataloader):
    #     # print(sample_batched['image'].shape)
    #     # print(mean.shape)
    #     batch_image = sample_batched['image'].numpy()
    #     # print(np.sum(sample_batched['image'].numpy(), axis=(2,3), keepdims=True).shape)
    #     element_wise_square = (batch_image - mean)*(batch_image - mean)
    #     std += np.mean(element_wise_square, axis=(0,2,3,4), keepdims=True)
    #     #progress_bar(batch_idx, len(dataloader))
    # std=np.divide(std,len(dataloader))
    # std=np.sqrt(std)
    # #print(std)
    # print('std contains nan: ',np.isnan(std).any())
    # #mean = np.zeros((1,len(channelList),1,1,1))
    std = np.ones((1,len(channels),1,1,1))
    return mean, std


class MgNetDataset(data.Dataset):
    def __init__(self, root, channels, transform=None):
        imgs_list = dataset_list(root)
        if len(imgs_list) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.imgs_list = imgs_list
        self.transform = transform
        self.channels = channels
        # self.image_size = image_size

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.imgs_list[index]

        pic = np.load(path, allow_pickle=True).item()

        channel_shape = (1,pic['image'].shape[1],pic['image'].shape[2],pic['image'].shape[3])

        image = pic['image'][self.channels[0]].reshape(channel_shape)
        for channel in self.channels[1:]:
            image = np.concatenate((image, pic['image'][channel].reshape(channel_shape)), axis=0)
        pic['image'] = image
        if self.transform is not None:
            pic = self.transform(pic)
        return pic
    def __len__(self):
        return len(self.imgs_list)

class NormalizePerImage(object):
    """subtract mean. divided by std."""
    def __init__(self, mean, std):
        self.mean = mean[0,:,:,:,:]
        self.std = std[0,:,:,:,:]
    def __call__(self, sample):
        # print('mean type--->',type(self.mean))
        # print('mean shape--->',self.mean.shape)
        # print('sample.type--->',type(sample))
        # print('sample image.type--->',type(sample['image']))
        # print('sample image.shape--->',sample['image'].shape)
        sample['image'] = sample['image'] - self.mean
        sample['image'] = np.divide(sample['image'],self.std)
        if np.isnan(sample['image']).any():
            print('sample after normalize contains nan: ',np.isnan(sample).any())
            exit()
        return sample
