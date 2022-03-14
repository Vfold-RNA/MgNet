# Repository for MgNet

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
can be found in **MgNet/example/03-prediction/test_result/cv*/40/cluster/*.pdb**.


## 1. **MgNet/data** folder
This folder contains scripts to generate images

## 2. **MgNet/model** folder
This folder contains the code to train the model

