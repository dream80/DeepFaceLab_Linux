# DeepFaceLab_Linux

# 简介

DeepFaceLab的Linux Ubuntu 版本

# 使用

尝试过自己从零开始安装，后来放弃了，还是用Anaconda3比较方便。开始之前先确认下服务器上是否有Git 和FFmpeg。

```shell
#安装 git
apt install git


#安装ffmpeg
apt install ffmpeg
```



```shell

#Anaconda3官网下载地址 ：<https://www.anaconda.com/distribution/#linux>  
#下载后安装，过程中需要多个Enter和yes
wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
./Anaconda3-2019.10-Linux-x86_64.sh
```

```shell

#添加环境变量，初始化conda
export PATH=~/anaconda3/bin:$PATH
conda init bash
```

```shell

创建DeepFaceLab的虚拟环境，并激活。
conda create -y -n deepfacelab python=3.6.6 cudatoolkit=9.0 cudnn=7.3.1
conda activate deepfacelab
```

```shell

#获取DFL源代码，安装python依赖。
git clone https://github.com/lbfs/DeepFaceLab_Linux.git
cd DeepFaceLab_Linux
python -m pip install -r requirements-cuda.txt
```


#图文教程
https://www.deepfakescn.com/?p=1202


