import os
import sys
import numpy as np

def is_image_file(filename,EXTENSIONS):
    return any(filename.endswith(extension) for extension in EXTENSIONS)

def dataset_list(dir,IMG_EXTENSIONS):
    images = []
    for root, _, fnames in sorted(os.walk(dir,followlinks=True)):
        for fname in sorted(fnames):
            if is_image_file(fname,IMG_EXTENSIONS):
                path = os.path.join(root, fname)
                images.append(path)
    return images

def imageStatistics(imageFolder):
    imageFiles=dataset_list(imageFolder,['.npy'])
    print('imageFiles number ---> ',len(imageFiles))
    MGPixel_num = 0
    image_size = 0
    occupied_num = 0
    for image in imageFiles:
        pic = np.load(image, allow_pickle=True).item()
        image_size = pic['image'].shape[-1]
        MGPixel_num += pic['label_non_zero_count']
        #occupied_num += pic['occupied_non_zero_count']
    print('MGPixel_num number ---> ',MGPixel_num)
    totalPixel_num = int(len(imageFiles)*image_size*image_size*image_size)
    print('ratio ---> ',float(MGPixel_num)/float(totalPixel_num))


print("Validate:")
imageStatistics('./image/selected_validate/')

print("Test:")
imageStatistics('./image/selected_test/')

print("non zero Train:")
imageStatistics('./image/non_zero_train/')

print("Train:")
imageStatistics('./image/selected_train/')


#print('check duplicate in train and validation folder...')
#testFiles=dataset_list('./image/cv/validate/',['.csv'])
#trainFiles=dataset_list('./image/cv/train/',['.csv'])
#testFilesSet = set()
#trainFilesSet = set()
#for testF in testFiles:
#    testFilesSet.add(testF.split("/")[-2][0:4]+"/"+testF.split("/")[-1])
#    # print(testF.split("/")[-2][0:4]+"/"+testF.split("/")[-1])
#for trainF in trainFiles:
#    trainFilesSet.add(trainF.split("/")[-2]+"/"+trainF.split("/")[-1])
#for testFile in testFilesSet:
#    if testFile in trainFilesSet:
#        print('duplicate',testFile)
#        exit()
