import os
import sys
import numpy as np

def is_file(filename,EXTENSIONS):
    return any(filename.endswith(extension) for extension in EXTENSIONS)

def dataset_list(dir,IMG_EXTENSIONS):
    images = []
    for root, _, fnames in sorted(os.walk(dir,followlinks=True)):
        for fname in sorted(fnames):
            if is_file(fname,IMG_EXTENSIONS) and (fname.find("_original.pdb") != -1 or fname.find("_RNA.pdb") != -1):
                path = os.path.join(root, fname)
                images.append(path)
    return images



original_pdbs=dataset_list("./processed/",['.pdb'])
print('original_pdbs number ---> ',len(original_pdbs))
print(original_pdbs)
for pdb in original_pdbs:
    with open(pdb) as f:
        pdblines = f.readlines()
    with open(pdb,'w') as f:
        for l in pdblines:
            record = l[0:6]
            if (record == "ATOM  " or record == "HETATM") and (l[16] == ' ' or l[16] == 'A'):
                s = list(l)
                #print(l)
                s[16] = ' '
                l = "".join(s)
                #print(l)
                f.write(l)
