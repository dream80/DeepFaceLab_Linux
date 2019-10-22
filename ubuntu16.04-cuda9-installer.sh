# Ubuntu 16.04 Installer

# Add NVIDIA package repository
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
sudo apt -y install ./cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
sudo apt -y install ./nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
sudo apt update

sudo mkdir /usr/lib/nvidia
sudo apt -y install nvidia-418

sudo apt -y install cuda-toolkit-9-0 cuda-9-0 cuda-cublas-9-0 cuda-cufft-9-0 cuda-curand-9-0 \
    cuda-cusolver-9-0 cuda-cusparse-9-0 libcudnn7=7.3.1.20-1+cuda9.0 \
    libcudnn7-dev=7.3.1.20-1+cuda9.0 libnccl2=2.2.13-1+cuda9.0 cuda-command-line-tools-9-0 

echo "export PATH=/usr/local/cuda-9.0/bin:$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc

# Install Python3.6
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update
sudo apt -y install python3-pip python3.6 python3.6-dev

# Install DeepFaceLab Dependencies
sudo apt -y install ffmpeg cmake build-essential git

# Install DeepFaceLab Python Dependencies
sudo python3.6 -m pip install -r requirements-cuda.txt


echo "Please reboot your system to let the new changes take effect."
