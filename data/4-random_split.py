import os
import numpy as np
import sys
from shutil import copyfile


IMG_EXTENSIONS = ['.npy']
imageFolder='./research/metalIon/data/image/temp/'
saveFolder=["./image/cv1/","./image/cv2/","./image/cv3/","./image/cv4/","./image/cv5/","./image/cv6/","./image/cv7/","./image/cv8/","./image/cv9/","./image/cv10/"]

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

dest=[]
files=dataset_list(imageFolder)
print('images number ---> ',len(files))
PDBList=[path[path.rfind('/')-4:path.rfind('/')] for path in files]
PDBSet=set()
for pdb in PDBList:
    PDBSet.add(pdb)
print('PDB number ---> ',len(PDBSet))
print(PDBSet)

current_work_directory = os.getcwd()    # Return a string representing the current working directory.
print('Current work directory: {}'.format(current_work_directory))
for sf in saveFolder:
    os.system('rm -rf '+sf)
    if not os.path.isdir(sf):
        os.mkdir(sf)

PDBOrder=[]
for pdb in PDBSet:
    fileList=dataset_list(imageFolder+pdb+'/')
    # print(fileList)
    # pos=[fileName for fileName in fileList if fileName.find('positive')!=-1]
    # neg=[fileName for fileName in fileList if fileName.find('negative')!=-1]
    entity={'pdb':pdb,'num':len(fileList)}
    print(pdb,' ---> file ---> ',len(fileList))
    PDBOrder.append(entity)
PDBOrder=sorted(PDBOrder, key=lambda x: x['num'])
print(len(PDBOrder))


fileCount = 0
for pdbEntity in PDBOrder:
    # lucky_number=np.random.randint(10)
    lucky_number = fileCount % 10
    command='ln -s '+imageFolder+pdbEntity['pdb']+' '+saveFolder[lucky_number]+pdbEntity['pdb']
    print(command)
    os.system(command)
    fileCount+=1
