# MgNet - Predicting Mg<sup>2+</sup> ion binding site in RNA structure

## Platform Requirements (Tested)
The following are tested system settings, newer hardware/software could also work but has not been tested.
* GNU/Linux x86_64 (Ubuntu 16.04 kernel 4.15.0-142-generic)
* Docker (version 20.10.7, with NVIDIA Container Toolkit)
* GPU NVIDIA GPU with Architecture >= Kepler or compute capability 3.0 (GeForce GTX 1080 Ti)
* NVIDIA Linux drivers (430.64, NVIDIA Container Toolkit requries driver >= 418.81.07)

## Installation

### 1. Setup Docker

> :point_right: Please refer to official [install guide](https://docs.docker.com/engine/install/) for installing Docker on systems other than the tested ones.

#### 1.1 Update the apt package index and install packages to allow apt to use a repository over HTTPS
```
sudo apt-get update
```
```
sudo apt-get install ca-certificates curl gnupg lsb-release
```

#### 1.2 Add Docker’s official GPG key and set up the repository
```
sudo mkdir -p /etc/apt/keyrings && \
     curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
     echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

#### 1.3 Install Docker Engine, containerd, and Docker Compose
Update the apt package index:
```
sudo apt-get update
```
List the versions available in your repo:
```
apt-cache madison docker-ce
```
Install a specific version using the version string from the second column, here we use version 20.10.7:
```
sudo apt-get install \
     docker-ce=5:20.10.7~3-0~ubuntu-$(lsb_release -cs) \
     docker-ce-cli=5:20.10.7~3-0~ubuntu-$(lsb_release -cs) \
     containerd.io
```
Start the Docker daemon:
```
sudo service docker start
```

#### 1.4 Create the docker group and add your user to the group
Create the docker group:
```
sudo groupadd docker
```
Add your user to the docker group:
```
sudo usermod -aG docker $USER
```
> :point_right: Note: **You need to start a new session to update the groups.**

#### 1.5 Verify that Docker Engine is installed correctly
```
docker run hello-world
```
This command downloads a test image and runs it in a container. When the container runs, it prints a message and exits.


### 2. Setup NVIDIA driver
The recommended way to install drivers is to use the package manager for your distribution but other installer mechanisms are also available (e.g., by downloading `.run` installers from NVIDIA driver [Downloads](https://www.nvidia.com/Download/index.aspx?lang=en-us)).

> :point_right: For instructions on using your package manager to install drivers on system other than Ubuntu 16.04, follow the steps in this [guide](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html).

#### 2.1 Install NVIDIA driver
If NVIDIA driver is not pre-installed with your Ubuntu distribution, you can check your available drivers:
```
ubuntu-drivers devices
```
and install the driver with the following command (here I use nvidia-430):
```
sudo apt-get install nvidia-430
```
`430` is the driver version.
Or you can download the appropriate NVIDIA diver and execute the binary as sudo.

> :point_right: installed NVIDIA driver should be compatible with CUDA Toolkit 10.1.

#### 2.2 Reboot your machine and verify the installation with the following command
```
nvidia-smi
```
you should see similar output as the following:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 430.64       Driver Version: 430.64       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 108...  Off  | 00000000:0A:00.0 Off |                  N/A |
| 20%   40C    P0    57W / 250W |      1MiB / 11176MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   1  GeForce GTX 108...  Off  | 00000000:0B:00.0 Off |                  N/A |
| 16%   35C    P0    51W / 250W |      0MiB / 11178MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   2  GeForce GTX 108...  Off  | 00000000:41:00.0 Off |                  N/A |
| 20%   38C    P0    52W / 250W |      0MiB / 11178MiB |      3%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```
> :warning: **Secure Boot: If you want to install the NVIDIA driver with UEFI Secure Boot enabled, checkout NVIDIA's official guide.**

### 3. Setup NVIDIA Container Toolkit

> :point_right: Please refer to official [install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) for a detailed instruction on systems other than the tested ones.

#### 3.1 Setup the package repository and the GPG key
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
  && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```
#### 3.2 Install the nvidia-docker2 package (and dependencies)
```
sudo apt-get update
```
```
sudo apt-get install -y nvidia-docker2
```
#### 3.3 Restart the Docker daemon
```
sudo systemctl restart docker
```
#### 3.4 Test that NVIDIA runs in Docker
```
docker run --rm --gpus all nvidia/cuda:10.1-base-ubuntu18.04 nvidia-smi
```
This should result in a console output shown below:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 430.64       Driver Version: 430.64       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 108...  Off  | 00000000:0A:00.0 Off |                  N/A |
| 22%   41C    P0    53W / 250W |      0MiB / 11176MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   1  GeForce GTX 108...  Off  | 00000000:0B:00.0 Off |                  N/A |
| 16%   36C    P0    51W / 250W |      0MiB / 11178MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   2  GeForce GTX 108...  Off  | 00000000:41:00.0 Off |                  N/A |
| 22%   40C    P0    52W / 250W |      0MiB / 11178MiB |      3%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```
Now you are ready to run the MgNet in Docker!


## Run MgNet in a Docker container

### 1. Clone this repository on your local machine, and set environment variable
```
git clone https://github.com/Vfold-RNA/MgNet.git /home/${USER}/MgNet
```
```
echo "export MGNET_HOME=/home/${USER}/MgNet/" >> /home/${USER}/.bashrc && \
echo "export PATH=/home/${USER}/MgNet/bin:${PATH}" >> /home/${USER}/.bashrc && \
source /home/${USER}/.bashrc
```

### 2. Download MgNet docker image parts and merge them into a single image
Download parts of the released image and checksum.txt:
```
for n in a b c d ; do wget https://github.com/Vfold-RNA/MgNet/releases/download/stable/MgNet-image.tar.gz.parta${n} -O ${MGNET_HOME}/image/MgNet-image.tar.gz.parta${n} ; done
wget https://github.com/Vfold-RNA/MgNet/releases/download/stable/checksum.txt -O ${MGNET_HOME}/image/checksum.txt
```
Check the integrity of the files:
```
cd ${MGNET_HOME}/image/
sha256sum --check checksum.txt
```
Merge these parts into a single image:
```
cat ${MGNET_HOME}/image/MgNet-image.tar.gz.parta* > ${MGNET_HOME}/image/MgNet-image.tar.gz && rm ${MGNET_HOME}/image/MgNet-image.tar.gz.parta*
```

### 4. Check MgNet options
```
mgnet -h
```

### 3. Load MgNet image into Docker
```
mgnet -l
```

### 4. Run MgNet for an example case
```
mgnet -i ${MGNET_HOME}/example/example.pdb -o ${MGNET_HOME}/example/
```
The ions predicted by 5 trained models will be saved into `${MGNET_HOME}/example/` as `xxxx_model_y_prediction.pdb`, where `xxxx` and `y` represents input pdb name and model number, respectively.

> :warning: **CUDA Toolkit: You may need to install CUDA Toolkit 10.1 if the error message contains `RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False`.**

### 5. Remove loaded MgNet image in Docker
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
