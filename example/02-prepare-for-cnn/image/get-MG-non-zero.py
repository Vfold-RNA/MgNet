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

print("Train:")
imageFolder='./selected_train/'
os.system('rm -rf ./non_zero_train/*')
        
imageFiles=dataset_list(imageFolder,['.npy'])
print('imageFiles number ---> ',len(imageFiles))
effective_image_num = 0
for image in imageFiles:
    pic = np.load(image, allow_pickle=True).item()
    MGPixel_num = pic['label_non_zero_count']
    #print('MGPixel_num number ---> ',MGPixel_num)
    if MGPixel_num >= 300:
        effective_image_num += 1
        command='ln -s '+'../'+image+' '+'./non_zero_train/'+image.split("/")[-1]
        #print(command)
        os.system(command)
print(effective_image_num)
