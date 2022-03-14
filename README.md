# Repository for MgNet

## Requirements

g++ (>= 5.5.0)

VMD (for LINUXAMD64, version 1.9.4a37, August 27, 2019)

mgltools_x86_64Linux2_1.5.6 (https://ccsb.scripps.edu/mgltools/downloads/)

HTMD (1.13.10, https://www.htmd.org/docs/latest/)

pytorch (1.2.0)

numpy (1.19.1)

biopython (1.73)

sklearn (0.21.2)

## 1. **MgNet/example** folder
This folder shows how to run MgNet model.

### 1.1 **MgNet/example/01-input** folder
contains the input structure *example.pdb* and structure list *pdb_list.txt*, 
user can change these files to run MgNet on different structures

### 1.2 run MgNet
use the following command to run the example case:
```
cd MgNet/example
bash run.sh
```

### 1.3 get result
the bound ions predicted by five trained MgNet models 
can be found in **MgNet/example/03-prediction/test_result/cv\*/40/cluster/\*.pdb**.


## 1. **MgNet/data** folder
This folder contains scripts to generate images

## 2. **MgNet/model** folder
This folder contains the code to train the model

