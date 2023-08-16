# MgNet - Predicting Mg<sup>2+</sup> ion binding site in RNA structure

## Platform Requirements (Tested)
The following are tested system settings, newer hardware/software could also work but has not been tested.
* CPU (AMD Ryzen 9 7900X 12-Core Processor)
* GPU (NVIDIA GeForce GTX 1080)
* GNU/Linux x86_64 (Ubuntu Server 22.04.2 LTS kernel 5.15.0-75-generic)
* Docker (Docker version 24.0.2, build cb74dfc)
* NVIDIA Driver (Version 530.41.03)

## Installation

### 1. Setup Docker

#### 1.1 Install Docker
> :point_right: Please refer to official [install guide](https://docs.docker.com/engine/install/) for installing Docker.

#### 1.2 Create the docker group and add your user to the group
Create the docker group:
```
sudo groupadd docker
```
Add your user to the docker group:
```
sudo usermod -aG docker $USER
newgrp docker
```
> :point_right: Note: **You may need to start a new session to update the groups.**

#### 1.3 Verify that Docker Engine is installed correctly
```
docker run hello-world
```
This command downloads a test image and runs it in a container. When the container runs, it prints a message and exits.


### 2. Setup NVIDIA driver
The recommended way to install drivers is to use the package manager for your distribution but other installer mechanisms are also available (e.g., by downloading `.run` installers from NVIDIA driver [Downloads](https://www.nvidia.com/Download/index.aspx?lang=en-us)).

#### 2.1 Install NVIDIA driver

> :point_right: installed NVIDIA driver should be compatible with CUDA Toolkit 10.1.

#### 2.2 Reboot your machine and verify the installation with the following command
```
nvidia-smi
```
you should see similar output as the following:
```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 530.41.03              Driver Version: 530.41.03    CUDA Version: 12.1     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                  Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf            Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce GTX 1080         Off| 00000000:01:00.0 Off |                  N/A |
|  0%   37C    P8                8W / 215W|      2MiB /  8192MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```

### 3. Setup Docker to access an NVIDIA GPU

> :point_right: Please also refer to official [install guide 1](https://docs.docker.com/config/containers/resource_constraints/#access-an-nvidia-gpu) and [install guide 2](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#setting-up-nvidia-container-toolkit). And remember to configure the Docker daemon to recognize the NVIDIA Container Runtime and then restart the Docker daemon.

#### 3.1 Setup the package repository and the GPG key

See the first step in Section [Setting up NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#setting-up-nvidia-container-toolkit) to setup the package repo and the GPG key.

#### 3.2 Install NVIDIA-container-runtime
```
sudo apt-get update
sudo apt-get install nvidia-container-runtime
sudo systemctl restart docker
```

#### 3.3 A working setup can be tested by running a base CUDA container
```
docker run -it --rm --gpus all nvidia/cuda:10.1-base-ubuntu18.04 nvidia-smi
```
This should result in a console output similar as the one shown below:
```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 530.41.03              Driver Version: 530.41.03    CUDA Version: 12.1     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                  Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf            Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce GTX 1080         Off| 00000000:01:00.0 Off |                  N/A |
|  0%   37C    P8                8W / 215W|      2MiB /  8192MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```
Now you are ready to run the MgNet in Docker!


## Run MgNet in a Docker container

### 1. Clone this repository on your local machine and run setup script
```
git clone https://github.com/Vfold-RNA/MgNet.git /home/${USER}/MgNet
```
```
cd /home/${USER}/MgNet && ./setup
```
and finally source the .bashrc
```
source /home/${USER}/.bashrc
```

### 2. Using MgNet

#### 2.1 Check MgNet options
```
mgnet -h
```

#### 2.2 Run MgNet for an example case
```
mgnet -i ${MGNET_HOME}/example/example.pdb -o ${MGNET_HOME}/example/
```
The ions predicted by 5 trained models will be saved into `${MGNET_HOME}/example/` as `xxxx_model_y_prediction.pdb`, where `xxxx` and `y` represents name of the input pdb and index of the trained model, respectively.

<!-- > :warning: **CUDA Toolkit: You may need to install CUDA Toolkit 10.1 if the error message contains `RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False`.** -->

#### 2.3 Pull MgNet container image from Docker Hub
```
mgnet -l
```

#### 2.4 Remove loaded MgNet image in Docker
```
mgnet -r
```

## Software References

[1] Humphrey, W., Dalke, A., and Schulten, K. (1996). Vmd: Visual molecular dynamics. Journal of Molecular Graphics, 14(1):33–38.

[2] Cock, P. J. A., Antao, T., Chang, J. T., Chapman, B. A., Cox, C. J., Dalke, A., Friedberg, I., Hamelryck, T., Kauff, F., Wilczynski, B., and de Hoon, M. J. L. (2009). Biopython: freely available Python tools for computational molecular biology and bioinformatics. Bioinformatics, 25(11):1422–1423.

[3] Morris, G. M., Huey, R., Lindstrom, W., Sanner, M. F., Belew, R. K., Goodsell, D. S., and Olson, A. J. (2009). Autodock4 and autodocktools4: Automated docking with selective receptor ﬂexibility. Journal of Computational Chemistry, 30(16):2785–2791.

[4] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., and Chintala, S. (2019). Pytorch: An imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc.

[5] Doerr, S., Harvey, M. J., Noé, F., and De Fabritiis, G. (2016). HTMD: High-throughput molecular dynamics for molecular discovery. Journal of Chemical Theory and Computation, 12(4):18451852. PMID: 26949976.

[6] Zhou, Y., and Chen, S.J. (2022). Graph deep learning locates magnesium ions in RNA. QRB Discovery, pp.1-22. doi:10.1017/qrd.2022.17
