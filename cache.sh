#!/bin/bash

# 目标文件夹路径
TARGET_DIR="/home/ubuntu/data/datasets/imgnet1k"

find "$TARGET_DIR" -type f | parallel -j4 "dd if={} of=/dev/null bs=4K iflag=direct 2>/dev/null"

echo "所有文件已读取完毕，系统可能已将其缓存至内存中。"
