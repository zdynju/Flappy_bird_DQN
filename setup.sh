#!/bin/bash

# Flappy Bird DQN训练环境安装脚本
# 创建conda环境并安装兼容的包版本

echo "正在创建Flappy Bird DQN训练环境..."
source $(conda info --base)/etc/profile.d/conda.sh

# 创建新的conda环境
conda create -n flappy-dqn python=3.10 -y
conda activate flappy-dqn

echo "安装基础包..."
# 安装基础科学计算包
pip install numpy==1.26.4


echo "安装TensorFlow和相关包..."
# 安装TensorFlow和GPU支持
pip install tensorflow==2.15.0
conda install -c conda-forge cudatoolkit=11.8 cudnn=8.9
# 查找 libdevice.10.bc
libdevice_path=$(find "$CONDA_PREFIX" -name "libdevice.10.bc" 2>/dev/null | head -1)

if [ -z "$libdevice_path" ]; then
    echo "❌ 未找到 libdevice.10.bc，可能 cudatoolkit 安装不完整"
    exit 1
else
    echo "✅ 找到 libdevice at: $libdevice_path"
fi

# 删除已有软链接或文件
if [ -e ./libdevice.10.bc ]; then
    echo "⚠️  已存在 ./libdevice.10.bc，将其删除重建"
    rm ./libdevice.10.bc
fi

# 创建软链接
ln -s "$libdevice_path" ./libdevice.10.bc
echo "✅ 软链接已创建: ./libdevice.10.bc -> $libdevice_path"

# 安装tensorboard (匹配TensorFlow版本)
pip install tensorboard==2.15.0

echo "安装游戏相关包..."

pip install pygame==2.6.1
pip install opencv-python==4.11.0.86
pip install keyboard==0.13.5

echo "安装完成！"

# 验证安装
echo "验证环境..."

python -c "
import tensorflow as tf
import numpy as np
import cv2
import pygame

print('TensorFlow:', tf.__version__)
print('NumPy:', np.__version__)
print('OpenCV:', cv2.__version__)
print('Pygame:', pygame.__version__)
print("GPU设备列表:", tf.config.list_physical_devices('GPU'))
# 简单张量计算测试
a = tf.constant([1.0, 2.0])
b = tf.constant([3.0, 4.0])
c = a + b
print("计算结果:", c.numpy())
print('GPU可用:', len(tf.config.list_physical_devices('GPU')) > 0)
print('环境安装成功！')
"

echo ""
echo "环境安装完成！"
echo "使用方法:"
echo "1. conda activate flappy-dqn"
echo "2. python deep_q_network_keras.py"
