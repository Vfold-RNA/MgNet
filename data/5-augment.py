import numpy as np
#import math
import sys
import os
import os.path
import subprocess

augmentation_folder='./image/16/'

rotations = ((0,0,1), (0,0,2), (0,0,3), (0,1,0), (0,1,1), (0,1,2), (0,1,3), (0,2,0), (0,2,1), (0,2,2), (0,2,3), (0,3,0), (0,3,1), (0,3,2), (0,3,3), (1,0,0), (1,0,1), (1,0,2), (1,0,3), (1,2,0), (1,2,1), (1,2,2), (1,2,3))


def is_image_file(filename):
    return (filename.find('.npy')!=-1)

def dataset_list(dir):
    images = []
    for root, _, fnames in sorted(os.walk(dir,followlinks=True)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


def augment_and_save_image(pic,rotations,save_folder):
    for rot in rotations:
        # clockwise along positive x
        pic['image'] = np.rot90(pic['image'], rot[0], (3,2))
        # clockwise along positive y
        pic['image'] = np.rot90(pic['image'], rot[1], (1,3))
        # clockwise along positive z
        pic['image'] = np.rot90(pic['image'], rot[2], (2,1))
        save_path=save_folder+pic['PDB']+'_'+str(rot[0])+str(rot[1])+str(rot[2])+'_'+str(pic['label'])+'.npy'
        np.save(save_path,pic)
    

if not os.path.isdir(augmentation_folder+'coreset/augmentation/'):
    os.mkdir(augmentation_folder+'coreset/augmentation/')
else:
    os.system('rm -rf '+augmentation_folder+'coreset/augmentation/*')

if not os.path.isdir(augmentation_folder+'refined/augmentation/'):
    os.mkdir(augmentation_folder+'refined/augmentation/')
else:
    os.system('rm -rf '+augmentation_folder+'refined/augmentation/*')

for image_path in dataset_list(augmentation_folder):
    picture = np.load(image_path).item()
    folder=image_path[:image_path.rfind('/')]
    if folder[folder.rfind('/')+1:]=='coreset':
        save_folder=folder+'/augmentation/'
        augment_and_save_image(picture,rotations,save_folder)
    elif folder[folder.rfind('/')+1:]=='refined':
        save_folder=folder+'/augmentation/'
        augment_and_save_image(picture,rotations,save_folder)
    else:
        exit()
