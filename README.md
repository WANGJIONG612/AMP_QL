# 目前是基于之前的青龙框架，加入了AMP的部分
但是整体训练效果还是不理想，去掉AMP部分可以很好的抬腿行走
但是如果加上AMP reward则基本上无法踏步，可能需要对相关参数进行调整


# 强化学习教程 #
使用强化学习训练出的算法能够让青龙人型机器人更加稳定的行走，应对各种复杂地形和外部干扰。
---
## 环境配置 ##
### 硬件需求 ###
使用Isaac gym进行训练***刚需***一张支持***CUDA***功能的Nvidia显卡，为了流畅的进行训练以及可视化训练结果，推荐使用显存16G以上的RTX显卡。
验证：通过运行CUDA提供的示例程序或命令行工具（如nvcc --version）来验证CUDA是否安装。
### 系统配置 ###
Isaac gym的训练需要在Linux系统上进行，我们推荐使用***Ubuntu20.04***，并将Isaac gym配置在conda虚拟环境中运行。
1. 配置好一个带有python（推荐使用***Python3.8***版本）的conda虚拟环境后我们便可以进行后续的环境安装工作。由于本项目的虚拟环境会记录路径信息，不建议将本项目的虚拟环境与其他项目混用。
	- `conda create -n AzureLoong python=3.8`
	- `conda activate AzureLoong`
2. 安装pytorch 1.13.1和cuda-11.7
	- `pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
3. 安装Isaac gym（https://developer.nvidia.com/isaac-gym）
	- `cd isaacgym_lib/python && pip install -e .`
	- 通过运行示例检测是否安装正常
	- `cd examples && python 1080_balls_of_solitude.py`
4. 下载本项目并安装运行本项目需要的依赖文件
	- `cd gpugym && git submodule init && git submodule update`初始化submodule
	- `cd gpu_rl && pip install -e .`安装gpu_rl(强化学习相关文件)
	- `cd .. && pip install -e .`安装gpugym
	- `pip install wandb`安装wandb（用于实验记录）
---
## 训练过程 ##
### 开始训练 ###
在gpugym/scripts路径下打开命令行界面，键入`python train.py --task=pbrs:oghr_v4`开始训练。如果训练正常开始，会弹出gpugym的窗口
按V可以暂停可视化，提升训练的速度
命令行中会显示每轮中奖励的平均数值
### 成果展示 ###
训练结束后键入`python play.py --task=AzureLoong`展示训练的结果

## 已知问题 ##
1. numpy包版本过高会导致训练报错，建议使用numpy1.20.0版本，pillow10.3.0版本，pandas1.40版本(2.0.3版本安装numpy时会有错误提示，但不一定会报错)
2. 如果遇到报错：`ImportError: cannot import name 'LeggedRobotCfg' from partially initialized module 'gpugym.envs'`则需要修改引用路径为直接引用
	- 例如将`from gpugym.envs import LeggedRobot`改为`from gpugym.envs.base.legged_robot import LeggedRobot`
3. 项目中包含一个rsl_rl安装包。如果运行时发现问题可检查一下虚拟环境中的rsl_rl包是否与本环境的一致。
	- 例如anaconda的虚拟环境可以找到`anaconda3/envs/环境名/lib/python3.x/site-packages/rsl-rl.egg-link`文件，查看之中所记述的地址是否与项目路径一致
